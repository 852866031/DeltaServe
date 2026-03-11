from slora.common.build_utils import repair_config
from slora.utils.model_load import hf_load_config
from slora.models.llama3.model import Llama3TpPartModel
from slora.models.mixtral.layer_weights.transformer_layer_weight import MixtralTransformerLayerWeight
from slora.models.mixtral.layer_infer.transformer_layer_infer import MixtralTransformerLayerInfer
from slora.models.mixtral.layer_weights.transformer_layer_weight_ep import MixtralEPTransformerLayerWeight
from slora.models.mixtral.layer_infer.transformer_layer_infer_ep import MixtralEPTransformerLayerInfer


class MixtralTpPartModel(Llama3TpPartModel):
    """
    Mixtral-8x7B Sparse MoE model.

    Inherits everything from Llama3TpPartModel (GQA attention, RoPE,
    KV-cache management, unified memory allocator, SFT backward service).
    Only the transformer layer weight and infer classes are swapped out
    to handle the MoE FFN.
    """

    transformer_weight_class = MixtralTransformerLayerWeight
    transformer_layer_infer_class = MixtralTransformerLayerInfer

    def _init_config(self):
        # The base TpPartBaseModel._init_config uses a hardcoded model registry
        # for dummy mode that doesn't know about Mixtral.  Always use hf_load_config
        # (reads from the HF hub cache, works regardless of dummy flag) so we get
        # the real config.json.
        self.config, self.weight_dir_ = hf_load_config(self.weight_dir_, mode="model")

        # Standard field aliasing from TpPartBaseModel._init_config
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])

        # GQA normalisation from Llama3TpPartModel._init_config
        # (num_key_value_heads may be listed under different names in Mixtral config)
        cfg = self.config
        for kv_alias in ("num_key_value_heads", "num_kv_heads"):
            if kv_alias in cfg and "num_key_value_heads" not in cfg:
                cfg["num_key_value_heads"] = cfg[kv_alias]
        cfg.setdefault("num_key_value_heads", cfg["num_attention_heads"])

        # Mixtral-specific fields
        assert "num_local_experts" in cfg, (
            "config.json is missing 'num_local_experts' — "
            "is this actually a Mixtral MoE checkpoint?"
        )
        cfg.setdefault("num_experts_per_tok", 2)


class MixtralEPTpPartModel(MixtralTpPartModel):
    """
    Mixtral-8x7B with Expert Parallelism for the FFN and TP for attention.

    Each rank owns num_local_experts // world_size experts at full intermediate
    size.  Tokens are routed to the correct rank via all_to_all_single inside
    MixtralEPTransformerLayerInfer._ffn; no post-FFN all_reduce is needed.
    """

    transformer_weight_class = MixtralEPTransformerLayerWeight
    transformer_layer_infer_class = MixtralEPTransformerLayerInfer
    # _init_config inherited from MixtralTpPartModel
