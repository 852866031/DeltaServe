import torch

from slora.models.llama3.layer_weights.transformer_layer_weight import Llama3TransformerLayerWeight


class MixtralTransformerLayerWeight(Llama3TransformerLayerWeight):
    """
    Transformer layer weights for Mixtral-8x7B.

    Attention (Q/K/V/O) is identical to Llama3 GQA — fully reused.

    FFN is replaced with a Sparse MoE block:
      - one router gate matrix (num_experts × hidden_size) — NOT sharded
      - per-expert: w1 (gate_proj), w3 (up_proj), w2 (down_proj)

    TP sharding of expert weights (Megatron-style column/row parallel):
      w1, w3: column-parallel — shard output (inter) dim → each rank holds
              (hidden, inter // world_size) after transpose
      w2:     row-parallel — shard input (inter) dim → each rank holds
              (inter // world_size, hidden) after transpose
    After the MoE FFN computation, an all_reduce sums partial results
    across TP ranks (same as the dense FFN all_reduce in Llama).
    """

    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.moe_gate_weight_ = None
        self.experts_w1_ = []   # gate_proj per expert: (hidden, inter/tp)
        self.experts_w3_ = []   # up_proj per expert:   (hidden, inter/tp)
        self.experts_w2_ = []   # down_proj per expert: (inter/tp, hidden)

    def load_hf_weights(self, weights, dummy=False):
        if dummy:
            self._load_qkvo_dummy_weights()
            self._load_ffn_dummy_weights()
        else:
            self._load_qkvo_weights(weights)
            self._load_ffn_weights(weights)

    def _load_ffn_weights(self, weights):
        # FFN norm — same key as Llama
        key = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        if key in weights:
            self.ffn_norm_weight_ = self._cuda(weights[key])

        # Router gate: shape (num_experts, hidden_size) — not TP-sharded
        key = f"model.layers.{self.layer_num_}.block_sparse_moe.gate.weight"
        if key in weights:
            self.moe_gate_weight_ = self._cuda(weights[key])

        # TP shard of the intermediate dimension
        inter_size = self.network_config_["intermediate_size"]
        split_inter = inter_size // self.world_size_
        start = split_inter * self.tp_rank_
        end = split_inter * (self.tp_rank_ + 1)

        num_experts = self.network_config_["num_local_experts"]
        self.experts_w1_ = []
        self.experts_w3_ = []
        self.experts_w2_ = []

        for j in range(num_experts):
            prefix = f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{j}"

            # w1: gate_proj (inter, hidden) in HF
            #   column-parallel: shard row (inter) → (split_inter, hidden)
            #   transpose → (hidden, split_inter) for matmul
            k = f"{prefix}.w1.weight"
            if k in weights:
                self.experts_w1_.append(self._cuda(weights[k][start:end, :].transpose(0, 1)))

            # w3: up_proj — same sharding as w1
            k = f"{prefix}.w3.weight"
            if k in weights:
                self.experts_w3_.append(self._cuda(weights[k][start:end, :].transpose(0, 1)))

            # w2: down_proj (hidden, inter) in HF
            #   row-parallel: shard column (inter) → (hidden, split_inter)
            #   transpose → (split_inter, hidden) for matmul
            k = f"{prefix}.w2.weight"
            if k in weights:
                self.experts_w2_.append(self._cuda(weights[k][:, start:end].transpose(0, 1)))

    def _load_ffn_dummy_weights(self):
        n_embed = self.network_config_["hidden_size"]
        inter_size = self.network_config_["intermediate_size"]
        num_experts = self.network_config_["num_local_experts"]
        split_inter = inter_size // self.world_size_

        self.ffn_norm_weight_ = (
            torch.rand((n_embed,), dtype=self.data_type_, device="cuda") * 2 - 1
        ) * 1e-3

        # Router gate: full (num_experts, hidden) — no sharding
        self.moe_gate_weight_ = (
            torch.rand((num_experts, n_embed), dtype=self.data_type_, device="cuda") * 2 - 1
        ) * 1e-3

        self.experts_w1_ = []
        self.experts_w3_ = []
        self.experts_w2_ = []
        for _ in range(num_experts):
            # (hidden, split_inter)
            self.experts_w1_.append(
                (torch.rand((n_embed, split_inter), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
            )
            self.experts_w3_.append(
                (torch.rand((n_embed, split_inter), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
            )
            # (split_inter, hidden)
            self.experts_w2_.append(
                (torch.rand((split_inter, n_embed), dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3
            )

    def verify_load(self):
        assert self.att_norm_weight_ is not None, "att_norm_weight_ not loaded"
        assert self.q_weight_ is not None, "q_weight_ not loaded"
        assert self.k_weight_ is not None, "k_weight_ not loaded"
        assert self.v_weight_ is not None, "v_weight_ not loaded"
        assert self.o_weight_ is not None, "o_weight_ not loaded"
        assert self.ffn_norm_weight_ is not None, "ffn_norm_weight_ not loaded"
        assert self.moe_gate_weight_ is not None, "moe_gate_weight_ not loaded"
        assert len(self.experts_w1_) > 0, "expert weights not loaded"
        assert len(self.experts_w1_) == len(self.experts_w2_) == len(self.experts_w3_), \
            "expert weight lists have inconsistent lengths"
