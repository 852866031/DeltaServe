import torch

from slora.models.llama3.layer_weights.transformer_layer_weight import Llama3TransformerLayerWeight


class Qwen3MoeTransformerLayerWeight(Llama3TransformerLayerWeight):
    """
    Transformer layer weights for Qwen3-30B-A3B (TP mode).

    Attention differences from Llama3:
      - Explicit head_dim (128) ≠ hidden_size // num_heads (64)
        → Q out dim = num_heads * head_dim = 4096 (not hidden_size = 2048)
        → K/V out dim = num_kv_heads * head_dim = 512 (not num_kv_heads * 64)
      - Per-head q_norm and k_norm weights (shape: head_dim,)

    FFN differences from Mixtral:
      - Key prefix: "mlp" (not "block_sparse_moe")
      - Router key: "mlp.gate.weight"
      - Expert weights are stacked tensors:
          gate_up_proj: (num_experts, 2*moe_inter, hidden) — gate + up concatenated
          down_proj:    (num_experts, hidden, moe_inter)
      - Config field: "num_experts" (normalised to "num_local_experts" by model.py)
      - Intermediate dim: "moe_intermediate_size" (not "intermediate_size")

    TP sharding of expert weights (same Megatron column/row parallel as Mixtral):
      w1 (gate), w3 (up): column-parallel — shard moe_inter dim
      w2 (down):          row-parallel — shard moe_inter dim
    """

    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.moe_gate_weight_ = None
        self.experts_w1_ = []
        self.experts_w3_ = []
        self.experts_w2_ = []
        self.q_norm_weight_ = None
        self.k_norm_weight_ = None

    def load_hf_weights(self, weights, dummy=False):
        if dummy:
            self._load_qkvo_dummy_weights()
            self._load_ffn_dummy_weights()
        else:
            self._load_qkvo_weights(weights)
            self._load_ffn_weights(weights)

    def _load_qkvo_weights(self, weights):
        key_ln = f"model.layers.{self.layer_num_}.input_layernorm.weight"
        if key_ln in weights:
            self.att_norm_weight_ = self._cuda(weights[key_ln])

        hidden_size = self.network_config_["hidden_size"]
        n_heads = self.network_config_["num_attention_heads"]
        n_kv_heads = self.network_config_.get("num_key_value_heads", n_heads)
        # Qwen3 has an explicit head_dim that differs from hidden_size // n_heads
        head_dim = self.network_config_.get("head_dim", hidden_size // n_heads)

        q_out = n_heads * head_dim          # 32 * 128 = 4096
        kv_out = n_kv_heads * head_dim      # 4  * 128 = 512

        assert q_out % self.world_size_ == 0
        assert kv_out % self.world_size_ == 0

        split_q_out = q_out // self.world_size_
        split_kv_out = kv_out // self.world_size_

        key_q = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        if key_q in weights:
            wq = weights[key_q][split_q_out * self.tp_rank_: split_q_out * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(wq.transpose(0, 1))

        key_k = f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        if key_k in weights:
            wk = weights[key_k][split_kv_out * self.tp_rank_: split_kv_out * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(wk.transpose(0, 1))

        key_v = f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        if key_v in weights:
            wv = weights[key_v][split_kv_out * self.tp_rank_: split_kv_out * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(wv.transpose(0, 1))

        # o_proj: (hidden_size, q_out) → row-parallel split on q_out columns
        key_o = f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        if key_o in weights:
            wo = weights[key_o][:, split_q_out * self.tp_rank_: split_q_out * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(wo.transpose(0, 1))

        # Qwen3-specific: per-head RMSNorm on Q and K (shape: head_dim,)
        key_qn = f"model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        if key_qn in weights:
            self.q_norm_weight_ = self._cuda(weights[key_qn])

        key_kn = f"model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        if key_kn in weights:
            self.k_norm_weight_ = self._cuda(weights[key_kn])

    def _load_qkvo_dummy_weights(self):
        hidden_size = self.network_config_["hidden_size"]
        n_heads = self.network_config_["num_attention_heads"]
        n_kv_heads = self.network_config_.get("num_key_value_heads", n_heads)
        head_dim = self.network_config_.get("head_dim", hidden_size // n_heads)

        q_out = n_heads * head_dim
        kv_out = n_kv_heads * head_dim
        split_q_out = q_out // self.world_size_
        split_kv_out = kv_out // self.world_size_

        def rand(*shape):
            return (torch.rand(shape, dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3

        self.att_norm_weight_ = rand(hidden_size)
        self.q_weight_ = rand(hidden_size, split_q_out)
        self.k_weight_ = rand(hidden_size, split_kv_out)
        self.v_weight_ = rand(hidden_size, split_kv_out)
        self.o_weight_ = rand(split_q_out, hidden_size)
        self.q_norm_weight_ = rand(head_dim)
        self.k_norm_weight_ = rand(head_dim)

    def _load_ffn_weights(self, weights):
        key = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        if key in weights:
            self.ffn_norm_weight_ = self._cuda(weights[key])

        # Router gate: (num_experts, hidden_size) — not TP-sharded
        key = f"model.layers.{self.layer_num_}.mlp.gate.weight"
        if key in weights:
            self.moe_gate_weight_ = self._cuda(weights[key])

        num_experts = self.network_config_["num_local_experts"]
        moe_inter = self.network_config_["moe_intermediate_size"]
        split_inter = moe_inter // self.world_size_
        start = split_inter * self.tp_rank_
        end = split_inter * (self.tp_rank_ + 1)

        # Per-expert weights (same layout as Mixtral, different key names):
        #   gate_proj: (moe_inter, hidden) — column-parallel: shard moe_inter rows → (hidden, split_inter)
        #   up_proj:   (moe_inter, hidden) — same sharding as gate_proj
        #   down_proj: (hidden, moe_inter) — row-parallel: shard moe_inter cols   → (split_inter, hidden)
        for j in range(num_experts):
            prefix = f"model.layers.{self.layer_num_}.mlp.experts.{j}"

            k = f"{prefix}.gate_proj.weight"
            if k in weights:
                self.experts_w1_.append(self._cuda(weights[k][start:end, :].transpose(0, 1)))

            k = f"{prefix}.up_proj.weight"
            if k in weights:
                self.experts_w3_.append(self._cuda(weights[k][start:end, :].transpose(0, 1)))

            k = f"{prefix}.down_proj.weight"
            if k in weights:
                self.experts_w2_.append(self._cuda(weights[k][:, start:end].transpose(0, 1)))

    def _load_ffn_dummy_weights(self):
        hidden_size = self.network_config_["hidden_size"]
        num_experts = self.network_config_["num_local_experts"]
        moe_inter = self.network_config_["moe_intermediate_size"]
        split_inter = moe_inter // self.world_size_

        def rand(*shape):
            return (torch.rand(shape, dtype=self.data_type_, device="cuda") * 2 - 1) * 1e-3

        self.ffn_norm_weight_ = rand(hidden_size)
        self.moe_gate_weight_ = rand(num_experts, hidden_size)

        self.experts_w1_ = []
        self.experts_w3_ = []
        self.experts_w2_ = []
        for _ in range(num_experts):
            self.experts_w1_.append(rand(hidden_size, split_inter))
            self.experts_w3_.append(rand(hidden_size, split_inter))
            self.experts_w2_.append(rand(split_inter, hidden_size))

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
