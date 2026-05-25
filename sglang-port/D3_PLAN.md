# D3 — Phased implementation plan

All sglang paths relative to /tmp/pair_sglang_port/sglang/.
All port-package paths relative to /tmp/pair_sglang_port/dserve-sglang/.
Phases are gated: each phase's acceptance test must pass before the next begins.
LoC estimates are net-new lines (excluding tests).

---

## Phase 1 — FinetuneConfig + ServerArgs plumbing

**Goal:** Opt-in finetuning flag reaches the Scheduler with a typed config dataclass.

**Files to create:**
 - `dserve-sglang/sglang/config/finetune.py` (dataclass copy from `/tmp/dsv-recon/DeltaServe-vLLM/dserve-vllm/vllm/config/finetune.py:17 FinetuneConfig`)

**Files to modify (upstream sglang):**
 - `python/sglang/srt/server_args.py` — add `--enable-finetuning`, `--backward-mps-percentage`, `--finetune-config` CLI flags + ServerArgs fields (default disabled).
 - `python/sglang/srt/managers/scheduler.py:296` — accept + stash `FinetuneConfig` on `self.finetune_config`; do nothing else when disabled.

**Acceptance test:** `python -c "from sglang.srt.server_args import ServerArgs; a=ServerArgs(model_path='x'); assert hasattr(a,'enable_finetuning') and a.enable_finetuning is False; print('ok')"`

**Estimated LoC:** ~120

---

## Phase 2 — Skeleton package + import smoke

**Goal:** `import sglang.deltaserve.*` works; every concrete method raises `NotImplementedError` so later phases can fill in incrementally.

**Files to create:**
 - `dserve-sglang/sglang/deltaserve/__init__.py`
 - `dserve-sglang/sglang/deltaserve/coordinator.py`
 - `dserve-sglang/sglang/deltaserve/ft_scheduler.py`
 - `dserve-sglang/sglang/deltaserve/backward_process.py`
 - `dserve-sglang/sglang/deltaserve/accumulate.py`
 - `dserve-sglang/sglang/deltaserve/finetuning_store.py`
 - `dserve-sglang/sglang/deltaserve/ft_injector.py`
 - `dserve-sglang/sglang/deltaserve/estimator.py`
 - `dserve-sglang/sglang/deltaserve/bwd_services/__init__.py`
 - `dserve-sglang/sglang/deltaserve/bwd_services/base.py`
 - `dserve-sglang/README.md`

**Files to modify (upstream sglang):** none (additive-only).

**Acceptance test:** `cd /tmp/pair_sglang_port/dserve-sglang && PYTHONPATH=. python -c "import sglang.deltaserve.coordinator, sglang.deltaserve.ft_scheduler, sglang.deltaserve.backward_process; print('imports ok')"`

**Estimated LoC:** ~200

---

## Phase 3 — FinetuneInjector on existing LoRA plumbing

**Goal:** Tag a request as FT-bearing; route adapter load through sglang's already-plumbed worker-level LoRA API instead of reimplementing the load path.

**Files to create:**
 - `dserve-sglang/sglang/deltaserve/ft_injector.py` (concrete impl; replaces the Phase-2 stub).
 - `dserve-sglang/tests/test_injector_loads_adapter.py`

**Files to modify (upstream sglang):**
 - `python/sglang/srt/managers/io_struct.py` — add `is_finetune: bool` and `ft_sample_id: Optional[str]` fields on `GenerateReqInput`.
 - `python/sglang/srt/managers/scheduler.py:1796 handle_generate_request` — when `req.is_finetune`, dispatch to `FinetuneInjector.intake(req)` which calls `tp_worker.py:180 load_lora_adapter` (under the hood: `lora/lora_manager.py:151 load_lora_adapter`).

**Acceptance test:** `pytest dserve-sglang/tests/test_injector_loads_adapter.py -q` — test stubs the adapter on disk, drives one FT request through `handle_generate_request`, asserts `LoRAManager.fetch_new_loras` was invoked and the adapter id is registered.

**Estimated LoC:** ~250

---

## Phase 4 — FinetuneAccumulator via hook_manager + Breakable graph

**Goal:** Pre/post forward hooks capture per-sample activations for FT-tagged requests; CUDA graph capture bypassed for any batch containing an FT sample.

**Files to create:**
 - `dserve-sglang/sglang/deltaserve/accumulate.py` (concrete impl; uses `python/sglang/srt/model_executor/hook_manager.py` to register hooks rather than monkey-patching modules — D1 §8 surprise #3).
 - `dserve-sglang/tests/test_accumulator_captures.py`

**Files to modify (upstream sglang):**
 - `python/sglang/srt/model_executor/model_runner.py:3159 forward` — inspect `forward_batch` for FT-tagged samples; when present, route through `breakable_cuda_graph_runner.py:74 BreakableCudaGraphRunner` instead of the standard `CudaGraphRunner` (D1 §8 surprise #7).
 - `python/sglang/srt/model_executor/hook_manager.py` — register accumulator callbacks (additive registration API).

**Acceptance test:** `pytest dserve-sglang/tests/test_accumulator_captures.py -q` — runs a fake forward over a 2-sample batch (one inference, one FT-tagged), asserts the accumulator's activation dict has exactly one entry keyed by the FT sample id and that no entry exists for the inference-only sample.

**Estimated LoC:** ~300

---

## Phase 5 — FinetuningStore on 3-level KV pool

**Goal:** Reserve KV slots compatible with sglang's `ReqToTokenPool` + `TokenToKVPoolAllocator` + page-based addressing — with zero vLLM block-id leakage (D1 §6, D1 §8 surprise #4).

**Files to create:**
 - `dserve-sglang/sglang/deltaserve/finetuning_store.py` (concrete impl: re-derives the FinetuningStore reservation logic against the 3-level pool).
 - `dserve-sglang/tests/test_store_alloc_release.py`

**Files to modify (upstream sglang):**
 - `python/sglang/srt/mem_cache/memory_pool.py:138 ReqToTokenPool.alloc` — minimal shim allowing FT-reserved indices to be queried before alloc returns (so concurrent inference doesn't grab them).
 - `python/sglang/srt/mem_cache/allocator.py` — 1-line hook to expose the reserved-set predicate to `TokenToKVPoolAllocator`.

**Acceptance test:** `pytest dserve-sglang/tests/test_store_alloc_release.py -q` — allocates N pages via FinetuningStore, commits some, releases others, asserts the pool's free-list invariant (free_count + allocated_count + reserved_count == total) holds at every step.

**Estimated LoC:** ~400

---

## Phase 6 — BackwardProcess subprocess + IPC

**Goal:** Spawn a sibling subprocess to the Scheduler with a constrained CUDA MPS partition; establish IPC for activation tensors (forward → bwd) and weight-grad apply (bwd → forward).

**Files to create:**
 - `dserve-sglang/sglang/deltaserve/backward_process.py` (subprocess entry point + lifecycle, mirroring `/tmp/dsv-recon/.../deltaserve/backward_process.py`).
 - `dserve-sglang/sglang/deltaserve/bwd_services/base.py` (port of `deltaserve/bwd_services/base.py:69 BackwardService`).
 - `dserve-sglang/sglang/deltaserve/bwd_services/llama3.py` (concrete service for llama3).
 - `dserve-sglang/tests/test_bwd_subprocess_roundtrip.py`

**Files to modify (upstream sglang):**
 - `python/sglang/srt/entrypoints/engine.py:749` — when `server_args.enable_finetuning`, also spawn the backward subprocess (sibling to Scheduler) with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set per `FinetuneConfig.backward_mps_percentage`.
 - `python/sglang/srt/managers/scheduler_components/ipc_channels.py:12 SchedulerIpcChannels` — add a `bwd_channel` zmq pair (D1 §8 surprise #6: Scheduler↔bwd IPC).

**Acceptance test:** `pytest dserve-sglang/tests/test_bwd_subprocess_roundtrip.py -q` — launches a stubbed BackwardProcess child, sends a fake activation tensor over the channel, receives a synthetic grad, asserts (a) child PID alive, (b) `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set on the child's env, (c) roundtrip latency < 1s.

**Estimated LoC:** ~600

---

## Phase 7 — FinetuneScheduler mixin + FinetuneCoordinator

**Goal:** Admission gating + backward-step scheduling integrated into the existing `event_loop_overlap` without breaking the mixin MRO (D1 §8 surprises #1, #5).

**Files to create:**
 - `python/sglang/srt/managers/scheduler_components/finetune_scheduler_mixin.py` — new mixin overriding `process_input_requests`, `handle_generate_request`, and the batch-selection step of `event_loop_overlap` / `event_loop_normal`.
 - `python/sglang/srt/managers/scheduler_components/finetune_coordinator.py` — port of `deltaserve/coordinator.py:55 FinetuneCoordinator` (admission gates, GPU-yield contract, on_backward_done hook).
 - `dserve-sglang/sglang/deltaserve/coordinator.py` — thin re-export / wiring that constructs the upstream coordinator.
 - `dserve-sglang/sglang/deltaserve/ft_scheduler.py` — thin re-export of the upstream mixin.
 - `dserve-sglang/tests/test_scheduler_admits_ft_batch.py`

**Files to modify (upstream sglang):**
 - `python/sglang/srt/managers/scheduler.py:286 class Scheduler` — when `enable_finetuning`, add `FinetuneSchedulerMixin` to the MRO; otherwise no behavior change.
 - `python/sglang/srt/managers/scheduler.py:1538 event_loop_overlap` — call `coordinator.gpu_pause_backward()` before prefill, `gpu_resume_backward()` after; consult `coordinator.reserve()` during batch selection.
 - `python/sglang/srt/managers/scheduler.py:1511 event_loop_normal` — same hooks for the non-overlap path.

**Acceptance test:** `pytest dserve-sglang/tests/test_scheduler_admits_ft_batch.py -q` — mocks TpWorker + BackwardService; drives one inference batch and one FT-tagged batch through `event_loop_normal`, asserts both complete and the coordinator's `note_injection` was called for the FT batch.

**Estimated LoC:** ~700

---

## Phase 8 — StepTimeEstimator + end-to-end smoke

**Goal:** Step-timing signal informs admission decisions; full mini end-to-end run survives with one real backward step interleaved with inference.

**Files to create:**
 - `python/sglang/srt/managers/scheduler_components/step_time_estimator.py` — port of `deltaserve/estimator.py` adapted to sglang's stats observer.
 - `dserve-sglang/sglang/deltaserve/estimator.py` — thin re-export / wiring.
 - `dserve-sglang/tests/e2e_smoke.sh`

**Files to modify (upstream sglang):**
 - `python/sglang/srt/managers/scheduler_components/pool_stats_observer.py:142 SchedulerPoolStatsObserver` — publish per-step inference + bwd timings to the estimator.

**Acceptance test:** `bash dserve-sglang/tests/e2e_smoke.sh` — launches sglang with `--enable-finetuning` against a tiny model, sends 4 inference requests and 1 FT request, asserts the process exits 0 and the stdout contains the literal `E2E_OK`.

**Estimated LoC:** ~500

---

**Totals:** Estimated total: ~3,070 LoC + tests; 8 gated phases.
