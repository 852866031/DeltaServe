import torch
import torch.nn.functional as F

from slora.models.mixtral.layer_infer.transformer_layer_infer import MixtralTransformerLayerInfer
from slora.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MoeTransformerLayerWeight
from slora.models.llama.infer_struct import LlamaInferStateInfo
from slora.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd


class Qwen3MoeTransformerLayerInfer(MixtralTransformerLayerInfer):
    """
    Transformer layer inference for Qwen3-30B-A3B (TP mode).

    Inherits SWA-capable attention kernels from MixtralTransformerLayerInfer.

    Overrides:
      __init__:    fixes head_dim_ (128 explicit, not 2048//32=64)
      _get_qkv:    applies per-head q_norm / k_norm before RoPE
      _ffn:        uses "num_local_experts" and "moe_intermediate_size" config keys
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        # Override head_dim_ — Qwen3 has explicit head_dim that differs from
        # hidden_size // num_attention_heads (128 vs 64).
        self.head_dim_ = network_config.get(
            "head_dim",
            network_config["hidden_size"] // network_config["num_attention_heads"],
        )

    def _get_qkv(
        self,
        input,
        cache_k,
        cache_v,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MoeTransformerLayerWeight,
    ) -> torch.Tensor:
        # Q projection
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)

        # Per-head q_norm (RMSNorm shared across heads, shape: head_dim)
        if layer_weight.q_norm_weight_ is not None:
            q = rmsnorm_forward(
                q.view(-1, self.head_dim_), weight=layer_weight.q_norm_weight_, eps=self.eps_
            ).view_as(q)

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            infer_state.position_cos,
            infer_state.position_sin,
        )

        # K projection
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.k_weight_,
            out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_),
        )

        # Per-head k_norm
        if layer_weight.k_norm_weight_ is not None:
            normed = rmsnorm_forward(
                cache_k.view(-1, self.head_dim_),
                weight=layer_weight.k_norm_weight_,
                eps=self.eps_,
            )
            cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_).copy_(
                normed.view(-1, self.tp_k_head_num_ * self.head_dim_)
            )

        rotary_emb_fwd(
            cache_k.view(-1, self.tp_k_head_num_, self.head_dim_),
            infer_state.position_cos,
            infer_state.position_sin,
        )

        # V projection
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.v_weight_,
            out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_),
        )

        return q

    def _ffn(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MoeTransformerLayerWeight,
    ) -> torch.Tensor:
        hidden = input.view(-1, self.embed_dim_)
        num_experts = len(layer_weight.experts_w1_)
        top_k = self.network_config_["num_experts_per_tok"]

        router_logits = hidden @ layer_weight.moe_gate_weight_.T
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = routing_weights.topk(top_k, dim=-1)
        # norm_topk_prob=True: re-normalize selected weights to sum to 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden.dtype)

        final_out = torch.zeros_like(hidden)

        for expert_idx in range(num_experts):
            expert_mask = (selected_experts == expert_idx)
            if not expert_mask.any():
                continue
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            expert_input = hidden[token_indices]

            w1 = layer_weight.experts_w1_[expert_idx]
            w3 = layer_weight.experts_w3_[expert_idx]
            w2 = layer_weight.experts_w2_[expert_idx]
            gate_out = F.silu(expert_input @ w1)
            up_out = expert_input @ w3
            expert_out = (gate_out * up_out) @ w2

            for local_i, tok in enumerate(token_indices):
                slot_indices = (selected_experts[tok] == expert_idx).nonzero(as_tuple=True)[0]
                weight = routing_weights[tok, slot_indices].sum()
                final_out[tok] += weight * expert_out[local_i]

        # all_reduce across TP ranks handled by the template's _context_ffn / _token_ffn
        return final_out
