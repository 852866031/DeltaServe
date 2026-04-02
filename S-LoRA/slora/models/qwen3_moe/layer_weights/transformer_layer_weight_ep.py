import torch

from slora.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MoeTransformerLayerWeight


class Qwen3MoeEPTransformerLayerWeight(Qwen3MoeTransformerLayerWeight):
    """
    Transformer layer weights for Qwen3-30B-A3B with Expert Parallelism.

    Attention weights identical to Qwen3MoeTransformerLayerWeight (inherited).

    EP sharding: each rank owns num_experts // world_size experts at full
    moe_intermediate_size (no inter-dim sharding within each expert).
    rank r owns experts [r*epk, (r+1)*epk) where epk = num_experts // world_size.
    """

    def _load_ffn_weights(self, weights):
        key = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        if key in weights:
            self.ffn_norm_weight_ = self._cuda(weights[key])

        key = f"model.layers.{self.layer_num_}.mlp.gate.weight"
        if key in weights:
            self.moe_gate_weight_ = self._cuda(weights[key])

        num_experts = self.network_config_["num_local_experts"]
        moe_inter = self.network_config_["moe_intermediate_size"]
        epk = num_experts // self.world_size_
        start_exp = epk * self.tp_rank_
        end_exp = start_exp + epk

        for j in range(start_exp, end_exp):
            prefix = f"model.layers.{self.layer_num_}.mlp.experts.{j}"

            # Full moe_inter size (no inter-dim split in EP)
            # gate_proj: (moe_inter, hidden) → transpose → (hidden, moe_inter)
            k = f"{prefix}.gate_proj.weight"
            if k in weights:
                self.experts_w1_.append(self._cuda(weights[k].transpose(0, 1)))

            # up_proj: same shape as gate_proj
            k = f"{prefix}.up_proj.weight"
            if k in weights:
                self.experts_w3_.append(self._cuda(weights[k].transpose(0, 1)))

            # down_proj: (hidden, moe_inter) → transpose → (moe_inter, hidden)
            k = f"{prefix}.down_proj.weight"
            if k in weights:
                self.experts_w2_.append(self._cuda(weights[k].transpose(0, 1)))

    def _load_ffn_dummy_weights(self):
        hidden_size = self.network_config_["hidden_size"]
        num_experts = self.network_config_["num_local_experts"]
        moe_inter = self.network_config_["moe_intermediate_size"]
        epk = num_experts // self.world_size_

        def rand(*shape):
            return (torch.rand(shape, dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3

        self.ffn_norm_weight_ = rand(hidden_size)
        self.moe_gate_weight_ = rand(num_experts, hidden_size)

        self.experts_w1_ = []
        self.experts_w3_ = []
        self.experts_w2_ = []
        for _ in range(epk):
            self.experts_w1_.append(rand(hidden_size, moe_inter))
            self.experts_w3_.append(rand(hidden_size, moe_inter))
            self.experts_w2_.append(rand(moe_inter, hidden_size))
