import torch
import torch.distributed as dist
import torch.nn.functional as F

from slora.models.mixtral.layer_infer.transformer_layer_infer import MixtralTransformerLayerInfer
from slora.models.mixtral.layer_weights.transformer_layer_weight_ep import MixtralEPTransformerLayerWeight
from slora.models.llama.infer_struct import LlamaInferStateInfo


class MixtralEPTransformerLayerInfer(MixtralTransformerLayerInfer):
    """
    Transformer layer inference for Mixtral-8x7B with Expert Parallelism.

    EP routing: tokens are dispatched to the rank that owns their selected expert
    via all_to_all_single.  Each rank runs only its local experts at full
    intermediate size.  The all_reduce that the TP template inserts after _ffn
    is skipped by overriding _context_ffn and _token_ffn directly.
    """

    def _ffn(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: MixtralEPTransformerLayerWeight,
    ) -> torch.Tensor:
        hidden = input.view(-1, self.embed_dim_)   # (T, H)
        T, H = hidden.shape
        K = self.network_config_["num_experts_per_tok"]   # typically 2
        E = self.network_config_["num_local_experts"]     # typically 8
        R = self.world_size_
        epk = E // R                                       # experts per rank

        # --- Router: replicated on all ranks, same result everywhere ---
        router_logits = hidden @ layer_weight.moe_gate_weight_.T    # (T, E)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        topk_w, topk_e = routing_weights.topk(K, dim=-1)           # (T, K)
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(hidden.dtype)

        if R == 1:
            # Single GPU: all experts are local, no communication needed
            final_out = torch.zeros_like(hidden)
            for exp_idx in range(epk):
                mask = (topk_e == exp_idx)               # (T, K)
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

        # --- Flatten: each (token, slot) is an independent work item ---
        flat_e = topk_e.view(-1)                        # (T*K,)
        flat_w = topk_w.view(-1)                        # (T*K,)
        flat_t = hidden.repeat_interleave(K, dim=0)     # (T*K, H)

        dest_rank = flat_e // epk                       # which rank owns this expert
        local_exp = flat_e % epk                        # index within that rank

        # Sort by destination rank for contiguous all_to_all buffers
        perm = dest_rank.argsort(stable=True)
        inv_perm = perm.argsort()
        s_dest = dest_rank[perm]
        s_local_exp = local_exp[perm]
        s_tokens = flat_t[perm]

        # Compute send counts per rank
        send_counts_t = torch.zeros(R, dtype=torch.int64, device=hidden.device)
        for r in range(R):
            send_counts_t[r] = (s_dest == r).sum()
        send_counts = send_counts_t.tolist()

        # Exchange counts so every rank knows how many tokens it will receive
        recv_counts_t = torch.empty(R, dtype=torch.int64, device=hidden.device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t.tolist()
        recv_total = sum(recv_counts)

        # Exchange token embeddings
        recv_tokens = torch.empty(recv_total, H, dtype=hidden.dtype, device=hidden.device)
        dist.all_to_all_single(
            recv_tokens, s_tokens,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )

        # Exchange local expert indices
        recv_local_exp = torch.empty(recv_total, dtype=torch.long, device=hidden.device)
        dist.all_to_all_single(
            recv_local_exp, s_local_exp,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )

        # --- Local FFN: process received tokens with this rank's experts ---
        expert_out = torch.zeros(recv_total, H, dtype=hidden.dtype, device=hidden.device)
        for exp_idx in range(epk):
            mask = (recv_local_exp == exp_idx)
            if not mask.any():
                continue
            tok = recv_tokens[mask]
            gate = F.silu(tok @ layer_weight.experts_w1_[exp_idx])
            up = tok @ layer_weight.experts_w3_[exp_idx]
            expert_out[mask] = (gate * up) @ layer_weight.experts_w2_[exp_idx]

        # --- Return results to originating ranks ---
        send_out = torch.empty_like(s_tokens)   # (sum(send_counts), H)
        dist.all_to_all_single(
            send_out, expert_out,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
        )

        # Unsort → restore original (token, slot) order, then weighted sum
        out = send_out[inv_perm]                                        # (T*K, H)
        final_out = (out * flat_w.unsqueeze(-1)).view(T, K, H).sum(dim=1)  # (T, H)
        return final_out

    # Override _context_ffn and _token_ffn to skip the template's all_reduce.
    # EP is already globally reduced via all_to_all — no further reduction needed.

    def _context_ffn(self, input_embdings, infer_state, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    def _token_ffn(self, input_embdings, infer_state, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
