"""Real LoRA-SFT backward for the sglang DeltaServe port.

Adapted from DSV-vLLM's `Llama3BackwardService.process_backward`. Runs
in-process (Path C) — sources base weights directly from the live
model_runner.model. The math layer (layer_forward / layer_backward /
attn_backward_core / head_backward) is unchanged from the vLLM port.

Per backward:
  1. Pull base weight views from model.layers[i] (q/k/v/o/gate/up/down/norms)
  2. Use fp32 LoRA master tensors for q/k/v/o (rank from FT adapter config)
  3. Compute head_backward → grad w.r.t. final residual
  4. Walk layers in reverse: layer_forward (populate cache) → layer_backward
  5. Accumulate grads into LoRA fp32 master; fused AdamW.step()
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from sglang.srt.deltaserve.bwd_services.llama3 import (
    head_backward, layer_forward, layer_backward, rope_cos_sin,
)
from sglang.srt.deltaserve.gpu_grant import gpu_grant

logger = logging.getLogger(__name__)


class _RealBackward:
    """Holds extracted base weights + fp32 LoRA master tensors + optimizer
    for the duration of the process. Constructed once at model_runner.load_model
    end; called from faux_backward.run_faux_backward when real-bwd is enabled."""

    def __init__(self, model: nn.Module, lora_rank: int = 16,
                 lora_alpha: float = 32.0, lr: float = 5e-6,
                 weight_decay: float = 0.01, backward_fp32: bool = False) -> None:
        self.model = model
        self.lora_rank = int(lora_rank)
        self.scaling = float(lora_alpha) / float(lora_rank)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        # Extract dims from config / first layer.
        cfg = model.config if hasattr(model, "config") else getattr(model, "model").config
        def _cfg(name, default=None):
            v = getattr(cfg, name, None)
            if v is None and hasattr(cfg, "to_dict"):
                v = cfg.to_dict().get(name, default)
            return v if v is not None else default
        self.D = int(_cfg("hidden_size"))
        self.L = int(_cfg("num_hidden_layers"))
        self.Hq = int(_cfg("num_attention_heads"))
        self.Hkv = int(_cfg("num_key_value_heads", _cfg("num_attention_heads")))
        head_dim = _cfg("head_dim", self.D // self.Hq)
        self.Hd = int(head_dim)
        self.kv_size = self.Hkv * self.Hd
        self.q_size = self.Hq * self.Hd
        self.inter = int(_cfg("intermediate_size"))
        self.theta = float(_cfg("rope_theta", 10000.0))
        self.eps = float(_cfg("rms_norm_eps", 1e-5))
        self.vocab = int(_cfg("vocab_size"))
        self.dims = (self.Hq, self.Hkv, self.Hd, self.kv_size)
        self.save_attn_qkv = False  # we DON'T capture qh/kh/vh, recompute via layer_forward

        # Reach into the sglang Llama model: model.model.layers[i].self_attn / mlp / norms.
        inner = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
        self.layers = inner.layers
        self.norm_w = inner.norm.weight  # final RMSNorm
        self.lm_w = model.lm_head.weight  # [vocab_pad, D]
        self.base_dtype = self.lm_w.dtype
        self.bwd_dtype = torch.float32 if backward_fp32 else self.base_dtype

        # Slice fused weights once into per-layer views (no copy).
        self.base: List[Dict[str, torch.Tensor]] = []
        for i in range(self.L):
            lyr = self.layers[i]
            attn = lyr.self_attn
            mlp = lyr.mlp
            # With LoRA, qkv_proj might be wrapped as .base_layer.weight. Probe.
            qkv_w = getattr(attn.qkv_proj, "base_layer", attn.qkv_proj).weight
            gate_up_w = getattr(mlp.gate_up_proj, "base_layer", mlp.gate_up_proj).weight
            o_w = getattr(attn.o_proj, "base_layer", attn.o_proj).weight
            down_w = getattr(mlp.down_proj, "base_layer", mlp.down_proj).weight
            self.base.append({
                "q":       qkv_w[:self.q_size],
                "k":       qkv_w[self.q_size:self.q_size + self.kv_size],
                "v":       qkv_w[self.q_size + self.kv_size:],
                "o":       o_w,
                "gate":    gate_up_w[:self.inter],
                "up":      gate_up_w[self.inter:],
                "down":    down_w,
                "in_ln":   lyr.input_layernorm.weight,
                "post_ln": lyr.post_attention_layernorm.weight,
            })

        # Build fp32 LoRA master tensors per layer × {q,k,v,o} × {A,B}.
        # LoRA A: [rank, D]; B: [out_dim, rank] (out_dim = q_size for q/o, kv_size for k/v).
        device = self.lm_w.device
        params: List[nn.Parameter] = []
        self.lora: List[Dict[str, Dict[str, nn.Parameter]]] = []
        for i in range(self.L):
            ld: Dict[str, Dict[str, nn.Parameter]] = {}
            for proj, out_dim in [("q", self.q_size), ("o", self.q_size),
                                  ("k", self.kv_size), ("v", self.kv_size)]:
                A = nn.Parameter(torch.randn(self.lora_rank, self.D, device=device,
                                              dtype=torch.float32) * 0.01)
                B = nn.Parameter(torch.zeros(out_dim, self.lora_rank, device=device,
                                              dtype=torch.float32))
                ld[proj] = {"A": A, "B": B}
                params.append(A); params.append(B)
            self.lora.append(ld)

        self.optimizer = torch.optim.AdamW(params, lr=self.lr,
                                            weight_decay=self.weight_decay, fused=True)
        self._call_count = 0
        self._cum_loss = 0.0
        self._cum_tokens = 0
        logger.warning(
            f"[DeltaServe] real_backward state built: D={self.D} L={self.L} "
            f"Hq={self.Hq} Hkv={self.Hkv} Hd={self.Hd} inter={self.inter} "
            f"vocab={self.vocab} lora_rank={self.lora_rank} "
            f"params={sum(p.numel() for p in params)/1e6:.2f}M"
        )

    def _layer_weights(self, i: int) -> dict:
        lw = dict(self.base[i])
        ld = self.lora[i]
        for proj in ("q", "k", "v", "o"):
            lw[proj + "A"] = ld.get(proj, {}).get("A")
            lw[proj + "B"] = ld.get(proj, {}).get("B")
        return lw

    def process(self, snapshot: Dict[str, Any], sample_lens: List[int]) -> float:
        """Run one real backward over the captured activations. Returns
        elapsed seconds. snapshot is FinetuneAccumulator.pop_step() output."""
        if not snapshot.get("layer_in"):
            return 0.0
        final_in = snapshot.get("final_in")
        if final_in is None:
            return 0.0
        n = final_in.shape[0]
        # Skip decode-step fires that have no shift-by-1 targets (n_valid would
        # be 0). head_backward needs at least 2 tokens per sample to compute
        # one CE term. For 1-token decode steps the backward is pure waste —
        # ~37ms × hundreds of decode steps per FT request adds up.
        if n < 2:
            return 0.0

        # If sample_lens not provided, treat as single sample of length n.
        if not sample_lens or sum(sample_lens) != n:
            sample_lens = [n]
        b_start, acc = [], 0
        for s in sample_lens:
            b_start.append(acc); acc += s

        device = final_in.device
        positions = torch.cat([torch.arange(s, device=device) for s in sample_lens])
        cos, sin = rope_cos_sin(positions, self.Hd, self.theta)
        ids = snapshot.get("concat_input_ids")
        if ids is None:
            return 0.0
        ids = ids[:n].to(device).long()

        grant = gpu_grant()
        self.optimizer.zero_grad(set_to_none=True)

        t0 = time.monotonic()
        # Head: loss + grad w.r.t. final_in
        loss, n_valid, g = head_backward(
            final_in[:n], self.lm_w, self.norm_w, self.eps,
            ids, sample_lens, b_start, self.vocab,
        )
        # Walk layers in reverse, re-computing the per-layer cache via layer_forward
        # then computing grads via layer_backward. Grads accumulate into LoRA params.
        for i in range(self.L - 1, -1, -1):
            grant.maybe_pause()
            x_in = snapshot["layer_in"].get(i)
            if x_in is None:
                continue
            x_in = x_in[:n]
            lw = self._layer_weights(i)
            # layer_forward returns (out, cache). We only need cache for layer_backward.
            with torch.no_grad():
                cache = layer_forward(
                    x_in, lw, self.scaling, cos, sin, sample_lens, b_start,
                    self.dims, self.eps,
                )
            grad_x, grads = layer_backward(
                g, cache, lw, self.scaling, cos, sin, sample_lens, b_start,
                self.dims, self.eps, cdt=self.bwd_dtype,
            )
            # Accumulate per-layer LoRA grads into the fp32 master params.
            ld = self.lora[i]
            for proj in ("q", "k", "v", "o"):
                gA = grads.get(proj + "A"); gB = grads.get(proj + "B")
                if gA is not None and ld[proj]["A"] is not None:
                    pA = ld[proj]["A"]
                    pA.grad = (pA.grad + gA.float()) if pA.grad is not None else gA.float()
                if gB is not None and ld[proj]["B"] is not None:
                    pB = ld[proj]["B"]
                    pB.grad = (pB.grad + gB.float()) if pB.grad is not None else gB.float()
            g = grad_x  # propagate upstream gradient

        self.optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        self._call_count += 1
        self._cum_loss += float(loss)
        self._cum_tokens += int(n_valid)
        logger.warning(
            f"[DeltaServe] real_backward #{self._call_count}: {elapsed*1000:.1f}ms "
            f"loss={loss:.4f} n_valid={n_valid} cum_tokens={self._cum_tokens}"
        )
        return elapsed


_INSTANCE: Optional[_RealBackward] = None


def build_real_backward(model: nn.Module, **kwargs) -> _RealBackward:
    """Module-level singleton accessor."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _RealBackward(model, **kwargs)
    return _INSTANCE


def get_real_backward() -> Optional[_RealBackward]:
    return _INSTANCE


def is_enabled() -> bool:
    """Env var SGLANG_DS_REAL_BACKWARD=1 switches faux → real."""
    return os.environ.get("SGLANG_DS_REAL_BACKWARD", "0") == "1"
