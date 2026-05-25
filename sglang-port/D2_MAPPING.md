# D2 — DeltaServe → sglang component mapping

All sglang paths relative to `/tmp/pair_sglang_port/sglang/`. All DeltaServe
paths relative to `/tmp/dsv-recon/DeltaServe-vLLM/dserve-vllm/vllm/`.
Citations cross-reference `D1_SGLANG_RECON.md`.

Ground-truth read of the DeltaServe source confirmed before writing each
section (see turn-12 inspection of `deltaserve/{coordinator,ft_scheduler,
backward_process,accumulate,finetuning_store,ft_injector,estimator}.py`
and `deltaserve/bwd_services/base.py`).

---

## 1. FinetuneCoordinator

**vLLM behavior:** `deltaserve/coordinator.py:55 FinetuneCoordinator` is the
process-wide singleton (`get_coordinator()` at `:46`) that tracks the shared
activation-buffer fill level, gates admission of FT samples into each step
(`reserve:197`, `release_reserve:212`, `note_injection:249`,
`snapshot_admission/restore_admission:227/238`), triggers the backward when
the buffer is full (`_trigger_backward:314`), and brokers the GPU-yield
contract during prefill (`gpu_pause_backward:344`, `gpu_resume_backward:351`,
`poll_backward:357`). It also owns the `FTAborted` sentinel (`:34`) used by
the P6 forward-interruptible path and the `on_backward_done` hook that lets
`FinetuningStore.commit_claimed` close out a sample only after the bwd acks.

**sglang target:** new module
`python/sglang/srt/managers/scheduler_components/finetune_coordinator.py`
(sits next to the existing component-mixins, see D1 §1). The instance is
constructed inside `python/sglang/srt/managers/scheduler.py:296 __init__`
(behind the `FinetuneConfig.enable_finetuning` flag) and held on the
`Scheduler` self. It is *driven* from `event_loop_overlap:1538` (and the
slower `event_loop_normal:1511`) — every loop tick calls
`coord.poll_backward()` and reads the admission snapshot before the next
`schedule()`. The "trigger backward" path sends an IPC request through
`scheduler_components/ipc_channels.py:12 SchedulerIpcChannels` (D1 §8 #6) to
the new BackwardProcess subprocess (see §3 below).

**sglang-specific risks:**
- The Scheduler runs in its own subprocess (D1 §5), so any coordinator state
  accessed by the GPU worker (e.g. `gpu_pause_backward`) must cross the
  Scheduler↔TpWorker IPC seam — in vLLM the worker imports the same Python
  module and shares the singleton, which is not free on sglang.
- D1 §8 #5: `event_loop_overlap` already pipelines schedule↔forward, so
  `note_injection` / `record_capture` need to be invariant under "the
  schedule decision is two steps ahead of the forward result." Mis-sequencing
  will double-count the activation-buffer fill.
- Page-based KV (D1 §6) means `space_remaining` and `next_ft_budget` cannot
  be expressed in raw tokens against a contiguous block range — must be
  re-derived in page units before being compared with `TokenToKVPoolAllocator`
  availability.

## 2. FinetuneScheduler

**vLLM behavior:** `deltaserve/ft_scheduler.py:35 FinetuneScheduler` subclasses
`AsyncScheduler` and overrides `schedule:360`, `has_requests:250`,
`update_from_output:586`, plus adds FT-only predicates
(`would_step_be_ft_only:288`), the SLO budget gate (`_slo_ft_budget:161`), the
estimator-driven graph predicate (`_will_use_graph:232`), feature extraction
for the step (`_current_step_features:129`, `_features_from_output:195`),
profiling-request helpers (`new_profiling_request:320`,
`enqueue_profiling_request:339`, `purge_profiling_requests:343`), and the
tier B/C rollback path (`_rollback_ft_step:520`).

**sglang target:** new mixin class
`python/sglang/srt/managers/scheduler_components/finetune_scheduler_mixin.py`
that gets mixed into `python/sglang/srt/managers/scheduler.py:286 Scheduler`
following the same composition pattern the existing
`scheduler_components/*` mixins use (D1 §8 #1). Methods to override or
extend: `process_input_requests:1625` (intercept FT-side requests),
`handle_generate_request:1796` (annotate `Req` with `_ft_sample` like
`ft_injector.py:85` does), and the batch-selection step inside
`event_loop_normal:1511` / `event_loop_overlap:1538`. SLO-budget gating
plugs into `init_schedule_policy:1062`. Profiling-request injection mirrors
the vLLM `enqueue_profiling_request` path and rides the existing tokenizer→
scheduler IPC channel.

**sglang-specific risks:**
- D1 §8 #1: mixin MRO is real — a new `FinetuneSchedulerMixin` must be added
  to `Scheduler`'s base list **after** the existing helper mixins so its
  overrides win, but **before** anything that finalizes state in `__init__`.
- `init_chunked_prefill:1015` interacts with backward-step admission: a
  chunked-prefill step can leave the model in an in-progress state across
  ticks, so `would_step_be_ft_only` must inspect chunked-prefill flags or
  the FT admit/abort logic will fire mid-prefill.
- vLLM's `AsyncScheduler` is sglang's `event_loop_overlap:1538` analogue —
  reserve-at-inject (`coord.reserve`) must hook into the pipelined queue,
  not the synchronous one, or the reserve count will lag forward execution.

## 3. BackwardProcess + BackwardService

**vLLM behavior:** `deltaserve/backward_process.py:143 BackwardProcess` is a
parent-side handle that forks an MPS-partitioned subprocess via `start:170`
(setting `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`); the child runs
`BackwardService.run:207` and dispatches RPCs:
`_handle_share_weights:272`, `_handle_share_activations:290`,
`_handle_process_activations:301`, plus pause signaling via `set_pause:325`.
The service base (`bwd_services/base.py:69 BackwardService`) drives the
real bwd math (`compute_loss_and_grad:111`, `process_backward:121`,
`_logit_loss_and_grad:138`, the `_maybe_pause:99` GPU-yield checkpoint) and
is subclassed per model family (e.g. `bwd_services/llama3.py`,
`bwd_services/opt.py`).

**sglang target:** new sibling subprocess launched from
`python/sglang/srt/entrypoints/engine.py` near the existing 3-process spawn
block (`engine.py:749`, see D1 §5). New files:
`python/sglang/srt/managers/backward_process.py` (parent-side handle,
mirrors `BackwardProcess`) and
`python/sglang/srt/managers/bwd_services/base.py` (service base) plus
per-model subclasses (`bwd_services/llama3.py`, `bwd_services/opt.py`).
IPC reuses `scheduler_components/ipc_channels.py:12 SchedulerIpcChannels`
(D1 §8 #6) for the Scheduler↔BackwardProcess control channel; weight sync
piggybacks on the existing worker plumbing
(`python/sglang/srt/managers/tp_worker.py:158 update_weights_from_tensor`,
`:169 update_weights_from_ipc`).

**sglang-specific risks:**
- MPS partition is a *process-spawn-time* env-var contract; sglang spawns
  workers from `data_parallel_controller.py:283`/`:444` (D1 §3), so any code
  that mutates `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` must run **before** the
  TpWorker subprocess fork or it inherits the wrong percentage.
- D1 §5: sglang already has 3 long-lived processes (TokenizerManager,
  Scheduler, DetokenizerManager) plus TP workers; adding BackwardProcess
  makes 4+. Process lifecycle (graceful shutdown, crash escalation,
  `is_alive` checks like `backward_process.py:377`) must integrate with
  sglang's existing supervision in `entrypoints/engine.py`.
- Weight sync via `update_weights_from_ipc` (CUDA IPC) crosses TP rank
  boundaries — DeltaServe's single-GPU `share_weights:229` assumption does
  not survive sglang's tp>1 deployments without a fan-out.

## 4. FinetuneAccumulator (forward hooks)

**vLLM behavior:** `deltaserve/accumulate.py:53 FinetuneAccumulator` registers
PyTorch pre/post forward hooks on selected modules (`register_hooks:134`,
`_make_pre_hook:158`, `_make_out_hook:192`) to copy per-layer activations
into preallocated buffers, indexed by the FT mask + offset for the current
step (`begin_step:218`, `accumulate_final:235`, `end_step:255`,
`zero_offset_range:263`). It honors the P6 abort event by raising
`FTAborted` from inside the hook after the copy work is done. The captured
buffers are shared with BackwardProcess via CUDA IPC.

**sglang target:** use the **existing**
`python/sglang/srt/model_executor/hook_manager.py` (D1 §8 #3) instead of
monkey-patching `nn.Module.forward`. Register
`FinetuneAccumulator` against `ModelRunner` (`model_runner.py:335`) just
before the dispatcher at `forward:3159` runs, and have the
accumulator's `begin_step`/`end_step` called from the same point in
`forward_extend:3033` / `forward_decode:2984` where the FT mask is built.
The new accumulator module lives at `dserve-sglang/sglang/deltaserve/
accumulate.py` and is *attached to the TpWorker* (D1 §3) since that is the
process that holds the live model.

**sglang-specific risks:**
- D1 §8 #7: CUDA-graph capture (`cuda_graph_runner.py:533`) bakes the
  forward into a graph and **bypasses Python hooks**, so training-tagged
  batches must route through `breakable_cuda_graph_runner.py:74` or disable
  graph capture entirely. Falling back to eager for FT steps is the safer
  default to start.
- sglang has multiple GraphRunner variants (CudaGraph, BreakableCudaGraph,
  PiecewiseCudaGraph) — the accumulator must declare which variant it is
  compatible with, and `ModelRunner.forward` already chooses among them per
  batch, so the choice must be conditioned on the FT-flag of the batch.
- Hook registration order matters with sglang's existing
  `hook_manager.py` — if it already registers backend-internal hooks
  (e.g. for KV writeback), FinetuneAccumulator hooks must compose, not
  replace, them.

## 5. FinetuneInjector + FinetuningStore

**vLLM behavior:** `deltaserve/ft_injector.py:27 FinetuneInjector` pulls
training samples and turns them into vLLM `Request` objects via
`next_ft_requests:57` and `_make_request:85`, calling `store.claim` at
admit to stake out the sample. `deltaserve/finetuning_store.py:51
FinetuningStore` owns the corpus and the 3-phase claim API
(`claim:185`, `commit_claimed:216`, `release_claimed:234`, plus
`pop_best_under:146`, `pop_next:165`, `advance_epoch:261`,
`has_next:275`, `has_claimed:282`), backed by the
`FinetuningSample` dataclass (`:38`). KV-slot reservation in vLLM piggybacks
on the BlockManager admit path.

**sglang target:** FinetuneInjector lives at
`dserve-sglang/sglang/deltaserve/ft_injector.py` and leverages the
**already-plumbed worker-level LoRA API** (D1 §8 #2):
`python/sglang/srt/managers/tp_worker.py:180 load_lora_adapter`,
`:184 unload_lora_adapter`, `:188 load_lora_adapter_from_tensors`, and the
shared adapter state in
`python/sglang/srt/lora/lora_manager.py:151 load_lora_adapter` and
`:305 prepare_lora_batch`. FinetuningStore re-derives KV reservation against
sglang's 3-level pool (D1 §6):
`python/sglang/srt/mem_cache/memory_pool.py:138 ReqToTokenPool`,
`mem_cache/allocator.py` (`TokenToKVPoolAllocator`), and
`mem_cache/memory_pool.py:700 KVCache` — the store reserves at the
TokenToKVPoolAllocator level, not the BlockManager level.

**sglang-specific risks:**
- D1 §6: page-based KV addressing in sglang invalidates any DeltaServe code
  that treats reservations as contiguous-token ranges; the
  `current_offset` / `space_remaining` arithmetic in
  `FinetuneCoordinator` (see §1) must be reframed in page units.
- D1 §8 #2: sglang's `LoRAManager` already supports a dynamic set of
  adapters with its own eviction policy
  (`lora/eviction_policy.py`, `lora/lora_drainer.py`); the FT adapter must
  be pinned (no eviction) and the trainable buffer must coexist with
  inference adapters in `lora/mem_pool.py LoRAMemoryPool` without being
  evicted mid-step.
- `claim`/`commit_claimed`/`release_claimed` must be safe under
  pipeline-depth-2 overlap (`event_loop_overlap:1538`); the existing 3-phase
  shape (P6) already supports this, but the Scheduler↔TpWorker IPC seam means
  `claim` and `commit_claimed` may run in different processes — the store
  needs to live where commits happen (Scheduler side) and surface remote-callable
  RPCs to the worker for the in-flight checks.

## 6. StepTimeEstimator

**vLLM behavior:** `deltaserve/estimator.py:192 MergedExecutionEstimator` is
the merged-graph/eager step-time predictor backed by
`StepFeatures:59`, `StepParams:91`, and the sliding tracker
`StepExecutionTracker:112` (`add:130`, `_drop:141`, `check_refit:151`,
`write_prediction_stats_csv:159`). It predicts per-step duration
(`predict:219`) and computes the next FT-token admission budget
(`max_next_ft_tokens:236`) used by `FinetuneScheduler._slo_ft_budget`. It
fits via `data_fit:261` and per-regime helper `_fit_regime:287`.

**sglang target:** new module
`python/sglang/srt/managers/scheduler_components/step_time_estimator.py`
(under `scheduler_components/` because it is co-resident with the
Scheduler). Its data source is
`scheduler_components/pool_stats_observer.py:142 SchedulerPoolStatsObserver`
(D1 §1 inventory). Step timing comes from CUDA events in the TpWorker
(`tp_worker.py:65 forward_batch_generation`, D1 §3), piped back to the
estimator over the same IPC channel that returns batch outputs to the
Scheduler.

**sglang-specific risks:**
- D1 §8 #5: `event_loop_overlap:1538` is intentionally pipelined, so the
  estimator's "duration of step N" measurement must distinguish the
  overlapped (forward-runs-while-schedule-runs) regime from a serialized
  step or the regression fits will be biased.
- CUDA-graph dispatcher choice (D1 §8 #7) is per-batch in sglang; the
  estimator's `_will_use_graph` predicate (vLLM `ft_scheduler.py:232`) must
  know which of the GraphRunner variants (`cuda_graph_runner.py:533`,
  `breakable_cuda_graph_runner.py:74`,
  `piecewise_cuda_graph_runner.py`) would handle the batch, not just
  "graph y/n."
- Cross-process timing skew: CUDA-event timestamps captured in the TpWorker
  process must be normalized before regression — wall-clock IPC time is not
  the same as device time.

## 7. FinetuneConfig

**vLLM behavior:** `dserve-vllm/vllm/config/finetune.py:17 FinetuneConfig` is
a dataclass collecting all DeltaServe knobs: `enable_finetuning` master
switch (`:20`), MPS partition (`backward_mps_percentage:26`), adapter and
data paths (`finetuning_lora_path:33`, `data_path:39`), training
hyperparams (`num_epochs:43`, `learning_rate:46`, `weight_decay:50`,
`gamma:53`), plus many later additions across stages P4/P5/P6
(SLO budget, profiling, `forward_interruptible`, `backward_cuda_graph`,
`bwd_log_path`, `start_on_launch`, and the per-batch lifecycle print
gates).

**sglang target:** new dataclass file
`python/sglang/srt/configs/finetune.py` (sibling to the other configs
under `python/sglang/srt/configs/` — see D1 inventory of `srt/configs/`).
Wired into `python/sglang/srt/server_args.py` (cited in D1 §1 inventory)
as opt-in fields and threaded through to
`python/sglang/srt/managers/scheduler.py:296 __init__` and
`python/sglang/srt/managers/tp_worker.py:218 TpModelWorker.__init__`
behind the `enable_finetuning=False` default. The dataclass copy is
preserved verbatim from the vLLM source so phase 2+ ports can swap the
config object in place.

**sglang-specific risks:**
- `ServerArgs` is sglang's public CLI surface — any new field shows up in
  `python -m sglang.launch_server --help`, so naming and defaults must be
  picked carefully and every flag must default to "off / behave like
  upstream sglang."
- DeltaServe's `FinetuneConfig` has accumulated ~20+ fields across stages
  P1→P6; for Phase 1 we only need the early-stage subset
  (`enable_finetuning`, `backward_mps_percentage`, `finetuning_lora_path`,
  `data_path`, `num_epochs`, `learning_rate`, `weight_decay`, `gamma`). The
  rest are added incrementally as the corresponding components are ported.
- The config is read from at least 3 processes in sglang
  (TokenizerManager / Scheduler / TpWorker — D1 §5), so it must be
  serializable (dataclass + JSON / msgpack) and passed via the existing
  process-init plumbing rather than via a process-local singleton.
