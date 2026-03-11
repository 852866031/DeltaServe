import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton

from slora.models.llama3.layer_infer.transformer_layer_infer import Llama3TransformerLayerInfer
from slora.models.mixtral.layer_weights.transformer_layer_weight import MixtralTransformerLayerWeight
from slora.models.llama.infer_struct import LlamaInferStateInfo
from slora.models.llama2.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from slora.models.mixtral.triton_kernel.context_flashattention_nopad_swa import context_attention_fwd_swa
from slora.models.mixtral.triton_kernel.token_attention_softmax_and_reducev_swa import token_softmax_reducev_fwd_swa


class MixtralTransformerLayerInfer(Llama3TransformerLayerInfer):
    """
    Transformer layer inference for Mixtral-8x7B.

    Overrides:
      - __init__          : reads sliding_window from network_config
      - _context_attention_kernel    : SWA-aware prefill attention
      - _token_decode_attention_normal: SWA-aware decode attention
      - _ffn              : Sparse MoE routing (TP sharded)

    When sliding_window is None (not in config), falls back to full attention.
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        # None → full attention (no window); int → attend to this many recent tokens
        self.sliding_window_ = network_config.get("sliding_window", None)

    # ------------------------------------------------------------------ #
    #  Prefill attention                                                   #
    # ------------------------------------------------------------------ #

    def _context_attention_kernel(
        self, q, k, v, infer_state: LlamaInferStateInfo, layer_weight
    ) -> torch.Tensor:
        o_tensor = torch.empty_like(q)
        if self.sliding_window_ is not None:
            context_attention_fwd_swa(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                k.view(-1, self.tp_k_head_num_, self.head_dim_),
                v.view(-1, self.tp_v_head_num_, self.head_dim_),
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                sliding_window=self.sliding_window_,
            )
        else:
            from slora.models.llama2.triton_kernel.context_flashattention_nopad import context_attention_fwd
            context_attention_fwd(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                k.view(-1, self.tp_k_head_num_, self.head_dim_),
                v.view(-1, self.tp_v_head_num_, self.head_dim_),
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
            )
        return o_tensor

    # ------------------------------------------------------------------ #
    #  Decode attention                                                    #
    # ------------------------------------------------------------------ #

    def _token_decode_attention_normal(
        self, q, infer_state: LlamaInferStateInfo
    ) -> torch.Tensor:
        total_token_num = infer_state.total_token_num
        batch_size      = infer_state.batch_size
        calcu_shape1    = (batch_size, self.tp_q_head_num_, self.head_dim_)

        # Phase 1: QK dot products — same kernel for both SWA and full attention.
        # Out-of-window logits will be masked in phase 2.
        att_m_tensor = torch.empty(
            (self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda"
        )
        token_att_fwd(
            q.view(calcu_shape1),
            infer_state.mem_manager.key_buffer[self.layer_num_],
            att_m_tensor,
            infer_state.b_loc,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )

        # Phase 2: softmax + value reduction (SWA masking applied here)
        o_tensor = torch.empty_like(q)
        if triton.__version__ >= "2.1.0":
            if self.sliding_window_ is not None:
                token_softmax_reducev_fwd_swa(
                    att_m_tensor,
                    infer_state.mem_manager.value_buffer[self.layer_num_],
                    o_tensor.view(calcu_shape1),
                    infer_state.b_loc,
                    infer_state.b_start_loc,
                    infer_state.b_seq_len,
                    infer_state.max_len_in_batch,
                    infer_state.other_kv_index,
                    sliding_window=self.sliding_window_,
                )
            else:
                from slora.models.llama2.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd
                token_softmax_reducev_fwd(
                    att_m_tensor,
                    infer_state.mem_manager.value_buffer[self.layer_num_],
                    o_tensor.view(calcu_shape1),
                    infer_state.b_loc,
                    infer_state.b_start_loc,
                    infer_state.b_seq_len,
                    infer_state.max_len_in_batch,
                    infer_state.other_kv_index,
                )
            return o_tensor
        else:
            raise Exception("Mixtral attention requires triton >= 2.1.0")

    # ------------------------------------------------------------------ #
    #  MoE FFN                                                            #
    # ------------------------------------------------------------------ #

    def _ffn(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: MixtralTransformerLayerWeight,
    ) -> torch.Tensor:
        hidden = input.view(-1, self.embed_dim_)          # (T, hidden_size)
        num_experts = len(layer_weight.experts_w1_)
        top_k = self.network_config_["num_experts_per_tok"]   # typically 2

        # --- Router (not TP-sharded: gate weight is replicated on all ranks) ---
        router_logits = hidden @ layer_weight.moe_gate_weight_.T   # (T, num_experts)
        # Float32 softmax for numerical stability; convert back afterwards
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = routing_weights.topk(top_k, dim=-1)   # (T, k)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden.dtype)

        # --- Sparse expert dispatch ---
        # final_out accumulates partial sums from each expert.
        # Because expert weights are TP-sharded (row-parallel w2), each rank
        # produces a partial sum; the all_reduce below completes the reduction.
        final_out = torch.zeros_like(hidden)

        for expert_idx in range(num_experts):
            # Which tokens have this expert in their top-k selection?
            expert_mask = (selected_experts == expert_idx)   # (T, top_k)
            if not expert_mask.any():
                continue

            # Unique token indices that visit this expert
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]   # (M,)
            expert_input = hidden[token_indices]   # (M, hidden)

            # FFN through expert j (TP-sharded intermediate dim):
            #   gate_out: (M, split_inter)   up_out: (M, split_inter)
            #   expert_out: (M, hidden)  — partial sum due to row-parallel w2
            w1 = layer_weight.experts_w1_[expert_idx]   # (hidden, split_inter)
            w3 = layer_weight.experts_w3_[expert_idx]   # (hidden, split_inter)
            w2 = layer_weight.experts_w2_[expert_idx]   # (split_inter, hidden)
            gate_out = F.silu(expert_input @ w1)          # (M, split_inter)
            up_out = expert_input @ w3                    # (M, split_inter)
            expert_out = (gate_out * up_out) @ w2         # (M, hidden)  partial

            # Weighted accumulate into final_out
            for local_i, tok in enumerate(token_indices):
                slot_indices = (selected_experts[tok] == expert_idx).nonzero(as_tuple=True)[0]
                weight = routing_weights[tok, slot_indices].sum()
                final_out[tok] += weight * expert_out[local_i]

        # Note: all_reduce across TP ranks is handled by the template's
        # _context_ffn / _token_ffn after this method returns.
        return final_out
