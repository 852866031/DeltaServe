# DeltaServe — Project Context

## What this project is

DeltaServe is an **LLM co-serving framework** that runs **inference** and **LoRA
fine-tuning** concurrently on the **same GPU**. The research goal is to interleave
a backward pass (SFT on LoRA adapters) with ongoing inference serving so the GPU
stays saturated, without letting backward work tank inference TTFT/latency.

Two models are supported:
- `llama` (Llama-1/2 style) — the reference / known-good path.
- `llama3` (Llama-3 style, GQA) — the active target for most optimization work.

## High-level architecture

```
 api_server.py  ── HTTP/CLI entry
      │
      ▼
 router/manager.py  ── scheduler; picks inference batches, interleaves
                        backward micro-batches, enforces co-serving policy
      │
      ▼
 router/model_infer/model_rpc.py  ── owns GPU process; constructs the
                                      forward runner + the backward service
      │
      ├── forward/inference runner (graph-captured, standard flow)
      └── backward service (SFT)  ← most recent work lives here
```

The **backward service** is the interesting part. For each supported model there
is a subclass of the base SFT service:

- `dserve/models/llama/SFT_service.py`   — base class `LlamaSFTBackwardService`.
- `dserve/models/llama3/SFT_service.py`  — subclass `Llama3SFTBackwardService` (GQA).
- `dserve/models/llama/SFT_service_graph.py` — `GraphedBackwardRunner`: CUDA
  graph capture layer for the backward pass. Works for both models.

The graph runner is opt-in via `--enable-bwd-cuda-graph` on `api_server.py`
(threaded down through `manager.py` → `model_rpc.py`).

## Backward pass: what is and isn't graph-captured

Per transformer layer the backward has two regions:

1. **Post-layer + FFN** — shape-stable after padding by
   `max_saved_finetuning_tokens`. **Graph-captured** per layer, one graph per
   layer at the single fixed size. Warmed up once at startup via
   `_warmup_full_backward`.

2. **Attention** — sample-boundary-aware, naturally variable shape. Two paths:
   - **Monolithic** (`_backpop_attention`): per-sample Python loop,
     shape-varying, eager. Always available as a fallback.
   - **Padded** (`_backpop_attention_padded` / `_padded_core`): pads to
     `[ATTN_BN_MAX, ATTN_L_MAX]`, uses a key-padding + causal mask,
     uses `index_put_(accumulate=True)` for shape-stable scatter of
     per-sample grads. **Graph-captured** when enabled. Falls back to
     monolithic if the batch exceeds `ATTN_BN_MAX` or any sample exceeds
     `ATTN_L_MAX`.

Key class constants on `Llama3SFTBackwardService`:
- `USE_GRAPHED_ATTENTION: bool` — dispatch into padded path.
- `ATTN_BN_MAX: int` — padded batch dimension (distinct samples per bwd).
- `ATTN_L_MAX: int` — padded sequence length.

Padded path compute is **quadratic in `L_max`**, so oversizing `L_max` tanks
throughput. Use `eval/llama3/analyze_finetuning_data.py` and
`eval/llama3/keep_p95.py` to size them against the dataset.

## Co-serving contract (load-bearing, do not remove)

- `_maybe_pause()` is called at **every layer boundary** in the backward path.
  It is how backward yields the GPU back to inference. Removing or skipping
  it breaks the whole point of the framework.
- Backward runs on its own CUDA stream (`self.bwd_stream`). Timing must use
  `torch.cuda.synchronize()` before reading the wall clock — otherwise you
  measure host dispatch, not GPU completion (this trap has bitten us).
- MPS partitioning is the mechanism for true concurrent execution.

## Memory / pool gotchas (also load-bearing)

- CUDA graph capture uses a **graph memory pool** (`graph_pool_handle`). Any
  tensor allocated during capture lives in that pool and **can be reused /
  aliased** by later captured graphs sharing the pool.
- LoRA `.grad` buffers and padded-attention context tensors (the `ctx`) MUST
  be allocated **outside** the graph pool as **persistent buffers** before
  capture, then referenced inside the captured graph. Otherwise the optimizer
  reads grads that a later graph has already overwritten — you get NaNs in
  inference softmax that look unrelated. (`_persistent_lora_grads[layer_id]`
  and the persistent attention `ctx` are there for this reason.)
- Before each `run()`, `.grad` is re-attached to the persistent buffer for
  every LoRA weight, because `zero_grad(set_to_none=True)` elsewhere will
  otherwise detach it.

## Precision rule for llama3 attention (recent fix)

GQA attention backward recomputes softmax. Forward and backward **must match
bit-for-bit on `scores`**. In `_backpop_attention` the forward path computes:

```python
scores = (q_blk.float() @ k_rep.float().transpose(-1, -2)) * scale  # fp32 matmul
```

**Do not** downgrade this to a fp16 matmul followed by `.float()` — that was
the bug that caused llama3 loss to plateau while llama1 trained fine.

## Dead code / do not touch

- `dserve/models/llama/SFT_service_backup.py`
- `dserve/models/llama3/SFT_service_backup.py`
- `_backpop_attention_autograd` (if you see it) — experimental, not on any
  live path.
- `eval/llama3/config/emotion_original.txt` — raw dataset; the filtered
  `emotion.txt` is what `launch_llama3.py` actually loads.

## Eval / analysis tooling (`eval/llama3/`)

- `launch_llama3.py` — starts a server with an SFT job attached.
- `auto_benchmark.py` — drives inference load against a running server, logs
  per-request timings and periodic finetuning-token counters.
- `bwd_graph_plot.py` — compares two CSV runs (eager vs. graphed) on a 1×3
  layout: TTFT CDF, E2E latency over time (with avg annotation), cumulative
  finetuning tokens (with tok/s). Uses
  `drop_duplicates(subset="timestamp", keep="last")` so the cumulative line
  can't backtrack.
- `analyze_finetuning_data.py` — tokenizes a dataset, prints percentiles +
  histogram + worst-case greedy-packed distinct-sample count; recommends
  `ATTN_BN_MAX` / `ATTN_L_MAX` and estimates padding blowup.
- `keep_p95.py` — drops the top 5% longest samples (configurable) so you can
  tighten `ATTN_L_MAX` without ever hitting the monolithic fallback.

## Important knobs

- `--enable-bwd-cuda-graph` (api_server.py CLI) — turn on graph capture.
- `--finetuning-params.max_saved_finetuning_tokens` — the single fixed size
  FFN graphs are captured at; also the per-bwd-batch token budget.
- `USE_GRAPHED_ATTENTION` / `ATTN_BN_MAX` / `ATTN_L_MAX` on
  `Llama3SFTBackwardService`.
- `PROFILE_EVERY` on `GraphedBackwardRunner` — GPU-event profile cadence.

## Current state (as of this context)

- FFN graph capture: working, per-layer, fixed size.
- Padded attention graph capture: working, with monolithic fallback on
  `Bn_max` / `L_max` overflow.
- Host syncs: removed from both `_backpop_attention` paths.
- Pool-aliasing NaN: mitigated by persistent buffers; if you see inference
  softmax NaNs, suspect a tensor that snuck back into the graph pool.
- llama3 loss-not-decreasing bug: fixed (fp32 `scores` matmul in forward).
- Backward duration printing: in ms, after `torch.cuda.synchronize()`.

## When opening a fresh session

Good first reads, in order:
1. This file.
2. `dserve/server/api_server.py` + `dserve/server/router/manager.py` for the
   request flow and CLI surface.
3. `dserve/models/llama/SFT_service.py` for the base backward contract.
4. `dserve/models/llama3/SFT_service.py` for the active GQA path
   (`_backpop_attention`, `_backpop_attention_padded*`).
5. `dserve/models/llama/SFT_service_graph.py` for the graph runner and the
   capture/replay lifecycle.
