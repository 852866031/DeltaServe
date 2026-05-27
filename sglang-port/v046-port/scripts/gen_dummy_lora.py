#!/usr/bin/env python3
"""Generate dummy LoRA weights for DSV-vLLM's toy adapters.

The repo ships adapter_config.json + tokenizer files but no
adapter_model.safetensors. This script generates random bf16 weights
matching adapter_config.json (rank=16 on q/k/v/o for Llama-3-8B) so the
vllm LoRA loader can attach them — it's enough for benchmarking.
"""
import torch, safetensors.torch
from pathlib import Path
import sys

if len(sys.argv) > 1:
    DSV_ROOT = Path(sys.argv[1])
else:
    DSV_ROOT = Path("/tmp/dsv-recon/DeltaServe-vLLM")

D = 4096; L = 32; r = 16
qkv_dim = 4096; kv_dim = 1024
state = {}
for layer in range(L):
    prefix = f"base_model.model.model.layers.{layer}.self_attn"
    for proj, out_dim in [("q_proj", qkv_dim), ("o_proj", qkv_dim),
                          ("k_proj", kv_dim), ("v_proj", kv_dim)]:
        state[f"{prefix}.{proj}.lora_A.weight"] = (torch.randn(r, D) * 0.01).to(torch.bfloat16)
        state[f"{prefix}.{proj}.lora_B.weight"] = torch.zeros(out_dim, r).to(torch.bfloat16)

for adp in ["llama3-toy-lora", "llama3-toy-lora-ft"]:
    out = DSV_ROOT / "adapters" / adp / "adapter_model.safetensors"
    safetensors.torch.save_file(state, str(out))
    print(f"Saved {out}")
