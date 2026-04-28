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

User-facing launchers expose a handful of flags (`--enable-finetuning`,
`--enable-cuda-graph`, `--enable-prefill-cuda-graph`,
`--enable-bwd-cuda-graph`, `--packed-kv`, `--occupancy_log <path>`,
`--port`, `--rank_id`, `--ft_log_path`) and translate them into
`--config <yaml> --override ...` arguments to `api_server.py`. The
`--packed-kv` flag swaps the YAML to
`serving_config_finetuning_packed.yaml` (only meaningful with
`--enable-finetuning`); `--occupancy_log` becomes a
`memory.unified_mem_manager_log_path=...` override and activates the
allocator's CSV tracker. The launcher also resolves relative
`lora.adapter_dirs` entries to absolute paths so they match what the
benchmark client sends as `lora_dir` (the server stores the dir string
verbatim in `lora_ranks` — mismatched keys raise `KeyError` in
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
- **External callers MUST go through pool accessors**, not
  `mem_manager.gpu_pools[layer]` directly. The accessors are
  `get_kv_pool(layer)`, `get_adapter_pool(layer)`, `get_activation_pool(layer)`
  on `UnifiedMemoryAllocator`. Subclasses (e.g. `PackedKVMemoryAllocator`)
  return a different view from `get_kv_pool` while keeping the others on the
  original layout, so the call site stays oblivious to which allocator is in
  use. A grep for `mem_manager.gpu_pools[` in `dserve/` should return zero
  hits — keep it that way.
- **KV freers MUST call `free_kv(sub_ids)`**, not `free(page_ids)`. The base
  allocator aliases `free_kv → free` so this works uniformly. The packed
  allocator implements sub-slot semantics in `free_kv`. Mis-routing KV ids
  through `free()` corrupts the bitmap silently because sub-slot ids in
  `[0, tot_size)` collide with page ids and cannot be disambiguated by range.
  The three KV freers in `infer_batch.py` (`free_self`, the two filter paths)
  already use `free_kv`; preserve that.

## Memory allocator selection (`unified` vs `packed_kv`)

Selected at startup via `cfg.memory.allocator` and dispatched through
`dserve/common/allocator_factory.py`. Both implementations share the bitmap +
`free_bitmap` + `page_type_map` machinery from
`dserve/common/unified_mem_allocator.py`.

- **`unified`** (default): one page = one KV slot. For Llama-3 GQA each page
  holds `[num_attention_heads=32, head_dim=128]` but only the first 8 head
  rows carry K (or V) data — the other 24 rows are zero padding. Simple, but
  the KV portion of the pool is ~4× oversized for GQA.
- **`packed_kv`** (`dserve/common/packed_kv_mem_allocator.py`): subclass of
  `UnifiedMemoryAllocator` that packs `F = num_attention_heads /
  num_key_value_heads` (= 4 for Llama-3) KV sub-slots into each page. The
  trick is a zero-copy reshape view exposed only to KV-reading kernels:

  ```
  gpu_pools[l]      shape [tot_size, 32, 128]   — adapter / activation kernels
  gpu_pools_kv[l]   shape [tot_size * F, 8, 128] — KV kernels (same memory)
  ```

  KV kernel call sites all do `pool[idx, :kv_heads, :]`; on the reshape view
  the middle dim is already `kv_heads`, so the slice is identity and the
  kernel binary doesn't notice. Zero kernel edits.

- **Allocation strategy in `packed_kv`** is *whole-page-per-call* — no
  partial-page draining: `_alloc_kv(n)` calls `super().alloc(ceil(n/F),
  KV_CACHE)` and derives sub-slot ids arithmetically. `alloc_contiguous_kv`
  similarly delegates to the parent with a smaller page count. Each page's
  `kv_refcount` (int64, set on alloc) tracks how many sub-slots the caller
  holds; `free_kv` decrements via `torch.bincount(page_ids,
  minlength=tot_size)` and a global-mask reconcile, page returns to FREE on
  refcount=0. Trade-off: a partial last page's unused sub-slots are
  "wasted" until the page fully drains — fine for low-occupancy workloads,
  worth re-evaluating if pool occupancy ever climbs past ~50%.

- **`free_kv` is the bottleneck-shaped op**. Use `bincount` (1 sync) rather
  than `torch.unique` + boolean indexing (2 syncs). A "global-mask" rewrite
  scanning the whole `kv_sub_mask` was tried and was slower under MPS
  contention because the larger kernel time on the inference stream stalled
  the backward subprocess — the per-call sync cost is less important than
  GPU stream contention with the backward graphs. Keep this in mind before
  optimizing.

- **F=1 fast paths exist in every override** (`_alloc_kv`,
  `alloc_contiguous_kv`, `free_kv`). For Llama-1 MHA the subclass is
  byte-for-byte identical to the parent; any divergence with `packed_kv` on
  Llama-1 is a pure subclass bug. That's the regression-test canary.

- **CPU-side `_used_pages` counter**, maintained on every alloc/free path in
  both allocators. Source of truth for the occupancy tracker (see below) and
  for any allocator query that wants `used_pages` without a GPU sync.

- **Occupancy tracker** (`_OccupancyTracker` in `unified_mem_allocator.py`)
  is a daemon thread that samples `(used_pages, tot_size, occupancy_pct)`
  to a CSV at `cfg.memory.unified_mem_manager_log_path`. Reads
  `self._used_pages` only — **CPU-only**, never `free_bitmap.sum()`. GPU
  work in a background thread can interfere with CUDA-graph capture/replay
  on the inference thread (caching-allocator pool aliasing), causing silent
  NaN logits later. Don't add GPU ops to the tracker.

- The allocator's `log_path` plumbing flows from
  `cfg.memory.unified_mem_manager_log_path` through `manager.py:673` →
  `model_rpc` → model `_init_mem_manager` → `make_allocator(...)`. Setting
  the YAML field activates the tracker; leaving it `null` keeps the daemon
  thread uncreated and the counter overhead negligible (one Python int
  add/sub per alloc/free under the existing `page_table_lock`).

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
  Output filenames are tagged by which knobs are on — order:
  `_decode_prefill_bwd_<tight|loose>_kv` — so the comparison plots find the
  right pair of CSVs. Flags:
  - `--decode_graph` / `--prefill_graph` / `--bwd_graph` — append
    `_decode` / `_prefill` / `_bwd` to the suffix and pass corresponding
    overrides to the launcher.
  - `--packed_kv` — append `_kv` to the suffix; passes `--packed-kv` to the
    launcher.
  - `--track_occupancy` — write `output/occupancy<suffix>.csv` (1 Hz
    sampling); passes `--occupancy_log` to the launcher.
  - `--tight` / `--loose` — pick `timeline_tight.csv` or
    `timeline_loose.csv` instead of the default `timeline_live.csv`;
    appends `_tight` / `_loose` to the suffix. Mutually exclusive.
- `bwd_graph_plot.py` — compares two CSV runs (eager vs. graphed) on a 1×3
  layout: TTFT CDF, E2E latency over time (with avg annotation), cumulative
  finetuning tokens (with tok/s). Uses `drop_duplicates(subset="timestamp",
  keep="last")` so the cumulative line can't backtrack.
- `compare_graphs_plot.py` — graph-ablation comparison plots, four-panel
  layout (request timeline + the three from `bwd_graph_plot`).
- `compare_kv_plot.py` — `unified` vs `packed_kv` comparison. Takes
  `--suffix _decode_prefill_bwd_tight` etc.; looks up `<suffix>` (unified)
  and `<suffix>_kv` (packed) under `output/`.
- `compare_occupancy_plot.py` — `(used_pages / total_pages)` over time,
  unified vs packed_kv, drives off the `occupancy<suffix>.csv` files
  emitted by `--track_occupancy`. Two-panel: occupancy % left, used pages
  right.
- `compare_allocators.py` — sequential, end-to-end correctness check.
  Spawns the server twice (once per YAML), sends N greedy prompts, diffs
  generated text + token counts. Use to verify `packed_kv` produces the
  same outputs as `unified` after any allocator change.
- `analyze_finetuning_data.py` — tokenizes a dataset, prints percentiles +
  histogram + worst-case greedy-packed distinct-sample count; recommends
  `attn_bn_max` / `attn_l_max` and estimates padding blowup.
- `keep_p95.py` — drops the top 5% longest samples (configurable) so you can
  tighten `attn_l_max` without ever hitting the monolithic fallback.

`MEMORY_ANALYSIS.md` (project root) walks through where every GB of GPU
residency goes for the `--co --decode_graph --prefill_graph --bwd_graph`
workload, identifies the wastes (GQA KV oversizing, activation
double-buffer), and quantifies the right pool sizing for a given
throughput. Read it first if you're investigating memory.

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
| Allocator implementation | `memory.allocator` | `"unified"` (default, page=1 KV slot) or `"packed_kv"` (page=F KV sub-slots, GQA-packed) |
| Occupancy tracker | `memory.unified_mem_manager_log_path` | If non-null, allocator daemon writes 1 Hz CSV of (used_pages, occupancy_pct). CPU-only, safe under graph capture. |
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
8. `dserve/common/unified_mem_allocator.py` + `packed_kv_mem_allocator.py`
   + `allocator_factory.py` — pool layout, accessors, the unified vs
   packed_kv split, occupancy tracker.
9. `MEMORY_ANALYSIS.md` (project root) — what consumes GPU memory and
   why; read before any memory-side work.
