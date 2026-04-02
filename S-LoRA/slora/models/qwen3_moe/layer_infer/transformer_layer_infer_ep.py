import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

from slora.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MoeTransformerLayerInfer
from slora.models.qwen3_moe.layer_weights.transformer_layer_weight_ep import Qwen3MoeEPTransformerLayerWeight
from slora.models.llama.infer_struct import LlamaInferStateInfo

# Routing imbalance logger — same pattern as Mixtral EP for experiment instrumentation.
routing_imbalance_log: list = []

# Per-layer timing logs. Each entry corresponds to one MoE layer in order.
# comm_time_log[i]   = (fwd_comm_ms, bwd_comm_ms) for layer i
# expert_time_log[i] = expert_compute_ms for layer i
# Only populated when timing_enabled = True (set by experiment scripts).
comm_time_log: list = []
expert_time_log: list = []
timing_enabled: bool = False


class Qwen3MoeEPTransformerLayerInfer(Qwen3MoeTransformerLayerInfer):
    """
    Transformer layer inference for Qwen3-30B-A3B with Expert Parallelism.

    EP routing: tokens are dispatched to the rank that owns their selected expert
    via all_to_all_single. Each rank runs only its local experts at full
    moe_intermediate_size. The all_reduce inserted by the TP template is skipped
    by overriding _context_ffn and _token_ffn (same pattern as MixtralEP).
    """

    def _ffn(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MoeEPTransformerLayerWeight,
    ) -> torch.Tensor:
        hidden = input.view(-1, self.embed_dim_)
        T, H = hidden.shape
        K = self.network_config_["num_experts_per_tok"]
        E = self.network_config_["num_local_experts"]
        R = self.world_size_
        epk = E // R

        # Router: replicated on all ranks
        router_logits = hidden @ layer_weight.moe_gate_weight_.T
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        topk_w, topk_e = routing_weights.topk(K, dim=-1)
        # norm_topk_prob=True
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(hidden.dtype)

        if R == 1:
            final_out = torch.zeros_like(hidden)
            for exp_idx in range(epk):
                mask = (topk_e == exp_idx)
                if not mask.any():
                    continue
                tok_ids = mask.any(dim=-1).nonzero(as_tuple=True)[0]
                exp_in = hidden[tok_ids]
                gate = F.silu(exp_in @ layer_weight.experts_w1_[exp_idx])
                up = exp_in @ layer_weight.experts_w3_[exp_idx]
                out = (gate * up) @ layer_weight.experts_w2_[exp_idx]
                for li, tok in enumerate(tok_ids):
                    slots = (topk_e[tok] == exp_idx).nonzero(as_tuple=True)[0]
                    final_out[tok] += topk_w[tok, slots].sum() * out[li]
            return final_out

        # Flatten: each (token, slot) is an independent work item
        flat_e = topk_e.view(-1)
        flat_w = topk_w.view(-1)
        flat_t = hidden.repeat_interleave(K, dim=0)

        dest_rank = flat_e // epk
        local_exp = flat_e % epk

        perm = dest_rank.argsort(stable=True)
        inv_perm = perm.argsort()
        s_dest = dest_rank[perm]
        s_local_exp = local_exp[perm]
        s_tokens = flat_t[perm]

        send_counts_t = torch.zeros(R, dtype=torch.int64, device=hidden.device)
        for r in range(R):
            send_counts_t[r] = (s_dest == r).sum()
        send_counts = send_counts_t.tolist()

        recv_counts_t = torch.empty(R, dtype=torch.int64, device=hidden.device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t.tolist()
        if routing_imbalance_log is not None:
            routing_imbalance_log.append(recv_counts)
        recv_total = sum(recv_counts)

        recv_tokens = torch.empty(recv_total, H, dtype=hidden.dtype, device=hidden.device)

        # --- forward all_to_all: send token embeddings + expert indices to owners ---
        if timing_enabled:
            torch.cuda.synchronize(); _t_fwd0 = time.perf_counter()
        dist.all_to_all_single(
            recv_tokens, s_tokens,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )

        recv_local_exp = torch.empty(recv_total, dtype=torch.long, device=hidden.device)
        dist.all_to_all_single(
            recv_local_exp, s_local_exp,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        if timing_enabled:
            torch.cuda.synchronize(); _t_fwd1 = time.perf_counter()

        # --- expert computation ---
        if timing_enabled:
            _t_exp0 = time.perf_counter()
        expert_out = torch.zeros(recv_total, H, dtype=hidden.dtype, device=hidden.device)
        for exp_idx in range(epk):
            mask = (recv_local_exp == exp_idx)
            if not mask.any():
                continue
            tok = recv_tokens[mask]
            gate = F.silu(tok @ layer_weight.experts_w1_[exp_idx])
            up = tok @ layer_weight.experts_w3_[exp_idx]
            expert_out[mask] = (gate * up) @ layer_weight.experts_w2_[exp_idx]
        if timing_enabled:
            torch.cuda.synchronize(); _t_exp1 = time.perf_counter()

        # --- backward all_to_all: return results to originating ranks ---
        send_out = torch.empty_like(s_tokens)
        if timing_enabled:
            _t_bwd0 = time.perf_counter()
        dist.all_to_all_single(
            send_out, expert_out,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
        )
        if timing_enabled:
            torch.cuda.synchronize(); _t_bwd1 = time.perf_counter()
            comm_time_log.append(((_t_fwd1 - _t_fwd0) * 1000, (_t_bwd1 - _t_bwd0) * 1000))
            expert_time_log.append((_t_exp1 - _t_exp0) * 1000)

        out = send_out[inv_perm]
        final_out = (out * flat_w.unsqueeze(-1)).view(T, K, H).sum(dim=1)
        return final_out

    def _context_ffn(self, input_embdings, infer_state, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    def _token_ffn(self, input_embdings, infer_state, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
