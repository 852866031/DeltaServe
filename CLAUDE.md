# DeltaServe — Project Context

## What this project is

DeltaServe is an **LLM co-serving framework** that runs **inference** and **LoRA
fine-tuning** concurrently on the **same GPU**. The research goal is to
interleave a backward pass (SFT on LoRA adapters) with ongoing inference
serving so the GPU stays saturated, without letting backward work tank
inference TTFT/latency.

Two models are supported:
- `llama` (Llama-1/2 style) — the reference / known-good path.
- `llama3` (Llama-3 style, GQA) — the active target for most optimization work.

Only two finetune types exist now: **`"SFT"`** (the live path) and
**`"SFT Profile"`** (a profiling-only variant of SFT used to fit cost models).
Alignment / DPO code was removed in full — do not look for `setup_alignment`,
`reference_lora_path`, `live_alignment`, `feedback_collector`, `alpha`/`beta`/
`lambdas`, `print_reset_log`, etc.; none of those exist anymore.

## High-level architecture

```
 api_server.py                   ── HTTP + ServerConfig load
      │
      ▼
 router/manager.py               ── scheduler; picks inference batches,
                                    interleaves backward micro-batches,
                                    enforces co-serving policy
      │
      ▼
 router/model_infer/model_rpc.py ── owns GPU process; constructs the
                                    forward runner + the backward service
      │
      ├── forward/inference runner (graph-captured decode optional)
      └── backward service (SFT)  ← most active development surface
```

Backward services per model:

- `dserve/models/llama/SFT_service.py`   — base class `LlamaSFTBackwardService`.
- `dserve/models/llama3/SFT_service.py`  — subclass `Llama3SFTBackwardService`
  (GQA-aware attention backward).
- `dserve/models/llama/SFT_service_graph.py` — `GraphedBackwardRunner`: CUDA
  graph capture layer for the backward pass. Works for both models.

## Configuration system

Everything runtime-configurable lives in a single YAML loaded at startup. There
is **no flag soup**: `api_server.py` exposes only four CLI args.

### Files

```
dserve/server/config.py                            ← ServerConfig dataclass +
                                                     YAML loader + override parser
eval/llama3/config/serving_config_finetuning.yaml  ← llama3 SFT-enabled
eval/llama3/config/serving_config_no_finetuning.yaml
eval/llama/config/serving_config_finetuning.yaml   ← llama1 SFT-enabled
eval/llama/config/serving_config_no_finetuning.yaml
```

### CLI

`api_server.py` accepts:
- `--config <path>` (required) — YAML to load.
- `--override section.field=value` (repeatable) — YAML-parsed value
  (`true`/`null`/`[a,b,c]` all work).
- `--port N` and `--rank_id N` — direct shortcuts for `server.port` /
  `server.rank_id` (the two knobs that vary every launch).

### Sections (see `dserve/server/config.py` for the dataclasses)

`server`, `model`, `serving`, `lora`, `scheduler`, `memory`, `cuda_graph`,
`finetune`, `slo`, `debug`. Strict validation rejects unknown sections/fields
with a helpful error.

### How cfg flows downstream

- `api_server.main()` builds `cfg`, then spawns subprocesses with `args` (which
  carries `args.cfg`).
- `start_router_process(args, ...)` calls `set_active_config(args.cfg)` and
  builds `InputParams(cfg)`.
- Each model RPC subprocess calls `set_active_config(input_params.cfg)` at the
  top of `exposed_init_model`.
- Hot-path code reads from `dserve/common/configs/config.py`'s
  `get_active_config()` (e.g., `infer_batch.py` for `max_req_total_len`).
- `InputParams` (`dserve/server/input_params.py`) is a **thin view** over cfg
  with legacy attribute names (`self.scheduler`, `self.finetuning_params.X`).
  ~50 downstream call sites still read it; new code should prefer
  `cfg.section.field` directly.

### Launchers (`eval/{llama,llama3}/launch_*.py`)

User-facing launchers expose ~6 flags (`--enable-finetuning`,
`--enable-cuda-graph`, `--enable-bwd-cuda-graph`, `--port`, `--rank_id`,
`--ft_log_path`) and translate them into `--config <yaml> --override ...`
arguments to `api_server.py`. The launcher also resolves relative
`lora.adapter_dirs` entries to absolute paths so they match what the benchmark
client sends as `lora_dir` (the server stores the dir string verbatim in
`lora_ranks` — mismatched keys raise `KeyError` in
`mixed_req_queue._can_add_new_req`).

## Backward pass: what is and isn't graph-captured

Per transformer layer the backward has two regions:

1. **Post-layer + FFN** — shape-stable after padding by
   `cfg.finetune.max_saved_finetuning_tokens`. **Graph-captured** per layer,
   one graph per layer at the single fixed size. Warmed up once at startup via
   `_warmup_full_backward`.

2. **Attention** — sample-boundary-aware, naturally variable shape. Two paths:
   - **Monolithic** (`_backpop_attention`): per-sample Python loop,
     shape-varying, eager. Always available as a fallback.
   - **Padded** (`_backpop_attention_padded` / `_padded_core`): pads to
     `[ATTN_BN_MAX, ATTN_L_MAX]`, uses a key-padding + causal mask, uses
     `index_put_(accumulate=True)` for shape-stable scatter of per-sample
     grads. **Graph-captured** when enabled. Falls back to monolithic if the
     batch exceeds `ATTN_BN_MAX` or any sample exceeds `ATTN_L_MAX`.

Padded-attention sizing comes from `cfg.cuda_graph`:
- `cuda_graph.use_graphed_bwd_attention` — gates dispatch into the padded
  path. Set on `self.USE_GRAPHED_ATTENTION` in the SFT service `__init__`.
- `cuda_graph.attn_bn_max` — padded batch dimension (distinct samples per
  bwd). Set on `self.ATTN_BN_MAX`.
- `cuda_graph.attn_l_max` — padded sequence length. Set on `self.ATTN_L_MAX`.

(These were class constants in earlier revisions; they're now per-instance
attributes set from cfg, but the attribute names on `self` are unchanged so
existing callsites work.)

Padded-path compute is **quadratic in `L_max`**, so oversizing `L_max` tanks
throughput. Use `eval/llama3/analyze_finetuning_data.py` and
`eval/llama3/keep_p95.py` to size them against the dataset.

## Co-serving contract (load-bearing, do not remove)

- `_maybe_pause()` is called at **every layer boundary** in the backward path.
  It is how backward yields the GPU back to inference. Removing or skipping it
  breaks the whole point of the framework.
- Backward runs on its own CUDA stream (`self.bwd_stream`). Timing must use
  `torch.cuda.synchronize()` before reading the wall clock — otherwise you
  measure host dispatch, not GPU completion (this trap has bitten us).
- MPS partitioning is the mechanism for true concurrent execution. The model
  RPC startup briefly sets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10` and
  `CUDA_DEVICE_MAX_CONNECTIONS=1` while spawning the backward subprocess.

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
- `UnifiedMemoryAllocator.max_finetuning_tokens` (from
  `cfg.memory.max_finetuning_tokens`) sizes the shared activation / logit /
  input-id buffers. Do not confuse it with
  `cfg.finetune.max_saved_finetuning_tokens`, which is the per-backward token
  budget and the FFN graph capture size.

## Precision rule for llama3 attention

GQA attention backward recomputes softmax. Forward and backward **must match
bit-for-bit on `scores`**. In `_backpop_attention` the forward path computes:

```python
scores = (q_blk.float() @ k_rep.float().transpose(-1, -2)) * scale  # fp32 matmul
```

**Do not** downgrade this to a fp16 matmul followed by `.float()` — that was
the bug that caused llama3 loss to plateau while llama1 trained fine.

## Dead code / do not touch

- `dserve/models/llama/SFT_service_backup.py` — old reference implementation.
- `dserve/models/llama3/SFT_service_backup.py` — old reference implementation.
- `_backpop_attention_autograd` (if you see it) — experimental, not on any
  live path.
- `eval/llama3/config/emotion_original.txt` — raw dataset; the filtered
  `emotion.txt` is what the launcher actually loads.
- `eval/{llama,llama3}/config/finetuning_config.json` /
  `no_finetuning_config.json` — legacy JSON configs from before the YAML
  migration. No code reads them anymore; safe to ignore.

## Eval / analysis tooling (`eval/llama3/`)

- `launch_llama3.py` — picks
  `serving_config_{finetuning,no_finetuning}.yaml` based on
  `--enable-finetuning`, resolves relative paths to absolute, and execs
  `api_server.py` with `--config + --override`. The llama1 equivalent is
  `eval/llama/launch_server.py` (also handles offline HF-cache mode + MPS
  daemon check).
- `auto_benchmark.py` — drives inference load against a running server, logs
  per-request timings and periodic finetuning-token counters. Calls
  `launch_llama3.py` itself if you pass `--co`. Sends adapter as an absolute
  path; the launcher must resolve `lora.adapter_dirs` to match.
- `bwd_graph_plot.py` — compares two CSV runs (eager vs. graphed) on a 1×3
  layout: TTFT CDF, E2E latency over time (with avg annotation), cumulative
  finetuning tokens (with tok/s). Uses `drop_duplicates(subset="timestamp",
  keep="last")` so the cumulative line can't backtrack.
- `analyze_finetuning_data.py` — tokenizes a dataset, prints percentiles +
  histogram + worst-case greedy-packed distinct-sample count; recommends
  `attn_bn_max` / `attn_l_max` and estimates padding blowup.
- `keep_p95.py` — drops the top 5% longest samples (configurable) so you can
  tighten `attn_l_max` without ever hitting the monolithic fallback.

## Important knobs (all in YAML)

| Knob | YAML path | What it controls |
|---|---|---|
| Enable backward graph | `cuda_graph.enable_bwd_cuda_graph` | FFN graph capture in backward |
| Enable decode graph | `cuda_graph.enable_decode_cuda_graph` | Forward decode graph capture |
| Padded-attn dispatch | `cuda_graph.use_graphed_bwd_attention` | Padded vs monolithic attn bwd |
| Padded-attn shape | `cuda_graph.attn_bn_max`, `cuda_graph.attn_l_max` | Hard-fail-to-fallback thresholds |
| FFN graph size / token budget | `finetune.max_saved_finetuning_tokens` | The single fixed FFN graph shape |
| Allocator buffer size | `memory.max_finetuning_tokens` | Shared activation/logit buffers |
| Unified mem pool | `memory.unified_mem_manager_max_size_gb` | KV+activation pool capacity |
| SLOs | `slo.{ttft_slo, avg_tbt_slo, max_tbt_slo}` | Scheduler trade-off thresholds |
| Scheduler | `scheduler.name` | Currently always `"dserve"` |

`PROFILE_EVERY` on `GraphedBackwardRunner` is still a class constant
(`SFT_service_graph.py`); flip it in source if you want different profiling
cadence.

## When opening a fresh session

Good first reads, in order:
1. This file.
2. `dserve/server/config.py` — the `ServerConfig` shape; every knob lives
   here.
3. `eval/llama3/config/serving_config_finetuning.yaml` — concrete defaults.
4. `dserve/server/api_server.py` + `dserve/server/router/manager.py` — request
   flow and how cfg fans out to subprocesses.
5. `dserve/models/llama/SFT_service.py` — base backward contract,
   `_maybe_pause()`, the LoRA grad lifecycle.
6. `dserve/models/llama3/SFT_service.py` — active GQA path
   (`_backpop_attention`, `_backpop_attention_padded*`).
7. `dserve/models/llama/SFT_service_graph.py` — graph runner and the
   capture/replay lifecycle, persistent-buffer rationale.
