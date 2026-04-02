from slora.common.build_utils import repair_config
from slora.utils.model_load import hf_load_config
from slora.models.llama3.model import Llama3TpPartModel
from slora.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MoeTransformerLayerWeight
from slora.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MoeTransformerLayerInfer
from slora.models.qwen3_moe.layer_weights.transformer_layer_weight_ep import Qwen3MoeEPTransformerLayerWeight
from slora.models.qwen3_moe.layer_infer.transformer_layer_infer_ep import Qwen3MoeEPTransformerLayerInfer


class Qwen3MoeTpPartModel(Llama3TpPartModel):
    """
    Qwen3-30B-A3B Sparse MoE model (TP mode).

    Differences from Mixtral that require new weight/infer classes:
      - model_type = "qwen3_moe"
      - num_experts (128) stored under "num_experts", not "num_local_experts"
      - moe_intermediate_size (768) separate from dense intermediate_size
      - head_dim (128) is explicit in config; hidden_size/num_heads = 64 ≠ head_dim
      - Expert weights are stacked tensors (gate_up_proj, down_proj) not per-expert files
      - MLP key prefix is "mlp" not "block_sparse_moe"
      - Per-head q_norm and k_norm (RMSNorm before RoPE)
      - num_experts_per_tok = 8 (vs Mixtral's 2)
    """

    transformer_weight_class = Qwen3MoeTransformerLayerWeight
    transformer_layer_infer_class = Qwen3MoeTransformerLayerInfer

    def _init_config(self):
        self.config, self.weight_dir_ = hf_load_config(self.weight_dir_, mode="model")

        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])

        cfg = self.config
        for kv_alias in ("num_key_value_heads", "num_kv_heads"):
            if kv_alias in cfg and "num_key_value_heads" not in cfg:
                cfg["num_key_value_heads"] = cfg[kv_alias]
        cfg.setdefault("num_key_value_heads", cfg["num_attention_heads"])

        # Qwen3 uses "num_experts" (not "num_local_experts")
        assert "num_experts" in cfg, (
            "config.json is missing 'num_experts' — is this actually a Qwen3 MoE checkpoint?"
        )
        cfg.setdefault("num_experts_per_tok", 8)

        # Normalise to "num_local_experts" so shared code paths can use either name
        cfg.setdefault("num_local_experts", cfg["num_experts"])

        assert "moe_intermediate_size" in cfg, (
            "config.json is missing 'moe_intermediate_size'"
        )

    def _init_some_value(self):
        # Qwen3 has an explicit head_dim that differs from hidden_size // num_attention_heads.
        # Override the base class computation so the model-level head_dim_ is correct.
        super()._init_some_value()
        self.head_dim_ = self.config.get(
            "head_dim",
            self.config["hidden_size"] // self.config["num_attention_heads"],
        )

    def _init_mem_manager(self):
        # Use explicit head_dim from config for KV buffer sizing.
        import torch
        head_dim = self.config.get(
            "head_dim",
            self.config["hidden_size"] // self.config["num_attention_heads"],
        )
        tp_kv_head_num = self.config["num_key_value_heads"] // self.world_size_

        if self.enable_unified_mem_manager:
            self.alt_mem_manager = self.alt_memory_manager_class(
                head_num=self.config["num_attention_heads"],
                head_dim=head_dim,
                layer_num=self.config["num_hidden_layers"],
                vocab_size=self.config["vocab_size"],
                dtype=torch.float16,
                max_pool_size=self.unified_mem_manager_max_size,
                log_path=self.mem_manager_log_path,
            )
            self.mem_manager = None
        else:
            self.mem_manager = self.memory_manager_class(
                tot_size=self.max_total_token_num + self.mem_adapter_size,
                cache_size=self.max_total_token_num,
                dtype=torch.float16,
                head_num=tp_kv_head_num,
                head_dim=head_dim,
                layer_num=self.config["num_hidden_layers"],
            )
            self.alt_mem_manager = None


class Qwen3MoeEPTpPartModel(Qwen3MoeTpPartModel):
    """
    Qwen3-30B-A3B with Expert Parallelism.

    Each rank owns num_experts // world_size experts at full moe_intermediate_size.
    Tokens are routed via all_to_all_single (same EP pattern as Mixtral EP).
    """

    transformer_weight_class = Qwen3MoeEPTransformerLayerWeight
    transformer_layer_infer_class = Qwen3MoeEPTransformerLayerInfer
