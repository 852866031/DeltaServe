import torch

from slora.models.mixtral.layer_weights.transformer_layer_weight import MixtralTransformerLayerWeight


class MixtralEPTransformerLayerWeight(MixtralTransformerLayerWeight):
    """
    Transformer layer weights for Mixtral-8x7B with Expert Parallelism.

    Attention weights are identical to MixtralTransformerLayerWeight (inherited).

    EP sharding: each rank owns a contiguous slice of experts at full size.
    For world_size R and num_local_experts E:
      - rank r owns experts [r*epk, (r+1)*epk)  where epk = E // R
      - expert weights are stored at full intermediate size (no inter sharding)
    """

    def _load_ffn_weights(self, weights):
        # FFN norm — same as TP
        key = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        if key in weights:
            self.ffn_norm_weight_ = self._cuda(weights[key])

        # Router gate: replicated on all ranks (not sharded)
        key = f"model.layers.{self.layer_num_}.block_sparse_moe.gate.weight"
        if key in weights:
            self.moe_gate_weight_ = self._cuda(weights[key])

        num_experts = self.network_config_["num_local_experts"]
        epk = num_experts // self.world_size_   # experts per rank
        start_exp = epk * self.tp_rank_
        end_exp = start_exp + epk

        self.experts_w1_ = []
        self.experts_w3_ = []
        self.experts_w2_ = []

        for j in range(start_exp, end_exp):
            prefix = f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{j}"

            # w1: gate_proj (inter, hidden) in HF → transpose to (hidden, inter)
            k = f"{prefix}.w1.weight"
            if k in weights:
                self.experts_w1_.append(self._cuda(weights[k].transpose(0, 1)))

            # w3: up_proj — same shape as w1
            k = f"{prefix}.w3.weight"
            if k in weights:
                self.experts_w3_.append(self._cuda(weights[k].transpose(0, 1)))

            # w2: down_proj (hidden, inter) in HF → transpose to (inter, hidden)
            k = f"{prefix}.w2.weight"
            if k in weights:
                self.experts_w2_.append(self._cuda(weights[k].transpose(0, 1)))

    def _load_ffn_dummy_weights(self):
        n_embed = self.network_config_["hidden_size"]
        inter_size = self.network_config_["intermediate_size"]
        num_experts = self.network_config_["num_local_experts"]
        epk = num_experts // self.world_size_

        self.ffn_norm_weight_ = (
            torch.rand((n_embed,), dtype=self.data_type_, device="cuda") * 2 - 1
        ) * 1e-3

        self.moe_gate_weight_ = (
            torch.rand((num_experts, n_embed), dtype=self.data_type_, device="cuda") * 2 - 1
        ) * 1e-3

        self.experts_w1_ = []
        self.experts_w3_ = []
        self.experts_w2_ = []
        for _ in range(epk):
            self.experts_w1_.append(
                (torch.rand((n_embed, inter_size), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
            )
            self.experts_w3_.append(
                (torch.rand((n_embed, inter_size), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
            )
            self.experts_w2_.append(
                (torch.rand((inter_size, n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
            )
