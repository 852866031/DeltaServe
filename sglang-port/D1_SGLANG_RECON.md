# D1 — sglang reconnaissance

Repo: github.com/sgl-project/sglang (shallow clone at /tmp/pair_sglang_port/sglang/, cloned 2026-05-25).
All paths below are relative to /tmp/pair_sglang_port/sglang/.

## 1. Scheduler

Central class: `python/sglang/srt/managers/scheduler.py:286` — `class Scheduler(...)`.

The Scheduler is **mixin-composed** rather than monolithic. It inherits from multiple `Scheduler*` mixin classes that live under `python/sglang/srt/managers/scheduler_components/` (e.g. `SchedulerBatchResultProcessor`, `SchedulerIpcChannels`, `SchedulerWeightUpdaterManager`, `SchedulerInvariantChecker`, `SchedulerOutputStreamer`, `SchedulerRequestReceiver`, `SchedulerLoadInquirer`, `SchedulerDPAttnAdapter`, `SchedulerPoolStatsObserver`).

Key methods inside `python/sglang/srt/managers/scheduler.py`:
- `:296` `__init__`
- `:847` `init_tp_model_worker` — constructs the TpWorker
- `:912` `init_model_worker`
- `:1015` `init_chunked_prefill`
- `:1062` `init_schedule_policy`
- `:1266` `init_overlap`
- `:1492` `run_event_loop`
- `:1511` `event_loop_normal`
- `:1538` `event_loop_overlap` — pipelined schedule + forward
- `:1625` `process_input_requests`
- `:1796` `handle_generate_request`
- `:3783` `def run_scheduler_process(` — **subprocess bootstrap entrypoint** for the Scheduler

## 2. ModelRunner forward path

Class: `python/sglang/srt/model_executor/model_runner.py:335` — `class ModelRunner(ModelRunnerKVCacheMixin)`. The output dataclass is at `:327` `class ModelRunnerOutput`.

Forward family (main dispatcher + four variants):
- `:3159` `def forward(...)` — main dispatcher
- `:2984` `forward_decode`
- `:3033` `forward_extend`
- `:3101` `forward_idle`
- `:3132` `forward_split_prefill`

KV-cache plumbing is mixed into the runner via `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:61` — `class ModelRunnerKVCacheMixin`.

`python/sglang/srt/model_executor/hook_manager.py` exists alongside the runner — existing forward-hook infrastructure, file present but contents not yet inspected (TODO: verify class names and signatures before wiring FinetuneAccumulator).

CUDA graph variants colocated under `model_executor/`:
- `cuda_graph_runner.py:533` `class CudaGraphRunner`
- `breakable_cuda_graph_runner.py:74` `class BreakableCudaGraphRunner`
- `piecewise_cuda_graph_runner.py:161` `class PiecewiseCudaGraphRunner` (TODO: verify line — derived from the broader grep hit, not a direct file inspection in turn 5)
- `cpu_graph_runner.py:480` `class CPUGraphRunner`

## 3. Worker process bootstrap

Single worker module: `python/sglang/srt/managers/tp_worker.py`.
- `:63` `class BaseTpWorker(ABC)` — abstract base; declares forward, memory-pool, LoRA, and weight-update hooks
- `:218` `class TpModelWorker(BaseTpWorker)` — concrete implementation
- `:65` `forward_batch_generation(self, forward_batch)` — main forward entry into the runner
- `:70` `model_runner` property
- `:90` `get_memory_pool(self) -> Tuple[ReqToTokenPool, BaseTokenToKVPoolAllocator]`
- `:212` `forward_batch_embedding(self, batch: ScheduleBatch)`
- `:221` `__init__` (on `TpModelWorker`)
- `:344` `_init_model_runner` — TpWorker constructs the ModelRunner
- LoRA already plumbed at the worker layer:
  - `:180` `load_lora_adapter`
  - `:184` `unload_lora_adapter`
  - `:188` `load_lora_adapter_from_tensors`
- Weight-update API present at worker level:
  - `:96`  `update_weights_from_disk`
  - `:104` `init_weights_update_group`
  - `:115` `destroy_weights_update_group`
  - `:121` `init_weights_send_group_for_remote_instance`
  - `:136` `send_weights_to_remote_instance`
  - `:146` `update_weights_from_distributed`
  - `:158` `update_weights_from_tensor`
  - `:169` `update_weights_from_ipc`
  - `:174` `get_weights_by_name`

TP launch (data-parallel controller spawns TP groups):
- `python/sglang/srt/managers/data_parallel_controller.py:257` — `target=self.launch_tensor_parallel_group_thread`
- `:283` `def launch_tensor_parallel_group_thread`
- `:291` `self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)`
- `:444` `def launch_tensor_parallel_group`

## 4. LoRA serving / adapter manager

Directory: `python/sglang/srt/lora/` contains `backend/`, `deepseek_mla_correction.py`, `eviction_policy.py`, `layers.py`, `lora.py`, `lora_config.py`, `lora_drainer.py`, `lora_manager.py`, `lora_moe_runner_marlin.py`, `lora_moe_runners.py`, `lora_overlap_loader.py`, `lora_registry.py`, `mem_pool.py`, `torch_ops/`, `triton_ops/`, `utils.py`.

Central manager: `python/sglang/srt/lora/lora_manager.py`
- `:53` `class LoRAManager`
- `:151` `load_lora_adapter`
- `:233` `unload_lora_adapter`
- `:258` `validate_lora_batch`
- `:289` `fetch_new_loras`
- `:305` `prepare_lora_batch(self, forward_batch)` — **critical per-batch LoRA prep, runs immediately before forward**
- `:339` `update_lora_info`
- `:420` `init_state`
- `:457` `init_lora_adapters`
- `:620` `load_lora_weights`
- `:695` `init_memory_pool`
- `:721` `init_lora_modules`

Related abstractions:
- `python/sglang/srt/lora/lora.py:43` `class LoRALayer(nn.Module)`
- `python/sglang/srt/lora/lora.py:53` `class LoRAAdapter(nn.Module)`
- `python/sglang/srt/lora/mem_pool.py:93` `class LoRAMemoryPool` (TODO: verify line — referenced in turn 5 prose, derived from directory listing)
- `python/sglang/srt/lora/layers.py` per-layer `apply_lora` implementations at `:118`, `:337`, `:448`, `:548`, `:627`, `:698`, `:815`
- `python/sglang/srt/lora/backend/lmhead_mixing.py:8` `class LoRABackendLmHeadMixing`

## 5. Process model (multi-process vs single-process)

**Multi-process.** sglang runs as three cooperating processes:

> "TokenizerManager (main process), Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager."
> — `python/sglang/srt/entrypoints/engine.py:184`

Bootstrap evidence:
- `python/sglang/srt/entrypoints/engine.py:120` `class SchedulerInitResult`
- `:569` returns `Tuple[SchedulerInitResult, Optional[List]]` (Scheduler subproc handles)
- `:663` log line: `"Scheduler or DataParallelController {proc.pid} ..."`
- `:749` docstring: "Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess."
- `python/sglang/srt/managers/scheduler.py:3783` `def run_scheduler_process(...)` — the actual subprocess entrypoint

IPC layer:
- `python/sglang/srt/managers/scheduler_components/ipc_channels.py:12` `class SchedulerIpcChannels`
- TP rank plumbing visible throughout `scheduler.py` (`:301`, `:347`, `:357`, `:383`, `:499`, `:851`, `:880`)
- DP controller spawns TP groups in threads: `python/sglang/srt/managers/data_parallel_controller.py:257` `target=self.launch_tensor_parallel_group_thread`

Detokenizer also runs as its own process: `python/sglang/srt/managers/detokenizer_manager.py:76` `class DetokenizerManager(...)`, with event loop at `:145` `def event_loop`.

## 6. KV cache abstraction

Three-level page-based design. Authoritative docstring at `python/sglang/srt/mem_cache/memory_pool.py:18-25` (TODO: verify exact line numbers — turn 5 cited the range but did not paste the exact `head` slice):

> SGLang has two levels of memory pool.
> ReqToTokenPool maps a request to its token locations.
> TokenToKVPoolAllocator manages the indices to kv cache data.
> KVCache actually holds the physical kv cache.

Concrete classes in `python/sglang/srt/mem_cache/memory_pool.py`:
- `:138` `class ReqToTokenPool`
  - `:170` `alloc(self, reqs: list[Req]) -> Optional[List[int]]`
  - `:164` `write(self, indices, values)`
  - `:167` `available_size`
  - `:197` `free(self, req: Req)`
- `:206` `class MambaPool`
- `:499` `class HybridReqToTokenPool(ReqToTokenPool)`
- `:700` `class KVCache(abc.ABC)` (TODO: verify line — turn 5 noted the class lives in this file; exact line not pasted in raw output)

`TokenToKVPoolAllocator` lives in `python/sglang/srt/mem_cache/allocator.py` (file confirmed via dir listing; exact class line not yet inspected — TODO: verify).

Prefix caching:
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/mem_cache/swa_radix_cache.py`

**Page-based.** `page_size` is used throughout the events layer:
- `python/sglang/srt/mem_cache/events.py:36` `# One BlockStored per ``page_size`` chunk.`
- `:45`, `:60`, `:61`, `:96`, `:100` all chunk by `self.page_size`

## 7. Batch scheduling (continuous batching / chunked prefill / overlap)

**Continuous batching** is the default driven by two event loops on the Scheduler:
- `python/sglang/srt/managers/scheduler.py:1511` `def event_loop_normal`
- `:1538` `def event_loop_overlap` — pipelines scheduling and forward execution

**Chunked prefill is first-class:**
- `python/sglang/srt/managers/scheduler.py:1015` `def init_chunked_prefill`
- `python/sglang/srt/managers/schedule_batch.py:2696` references `server_args.chunked_prefill_size` (decision point inside batch construction)
- `python/sglang/srt/managers/schedule_batch.py:2697` `if server_args.chunked_prefill_size > 0`
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:278` `_apply_chunked_prefill_logprobs(...)`
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:287` `req.time_stats.set_last_chunked_prefill_finish_time()`

CUDA graph variants (matter for the backward-pass path, which cannot live inside a captured graph):
- `python/sglang/srt/model_executor/cuda_graph_runner.py:533` `class CudaGraphRunner`
- `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py:74` `class BreakableCudaGraphRunner`
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py:161` `class PiecewiseCudaGraphRunner` (TODO: verify line)

## 8. Surprises that affect the DeltaServe port

1. **Scheduler is heavily mixin-composed**, not a single monolithic class like vLLM's. The DeltaServe `FinetuneScheduler` should subclass `Scheduler` *plus* probably one extra mixin co-located under `python/sglang/srt/managers/scheduler_components/`, rather than monkey-patching methods on the base class. Mixin reference: 10+ helper classes already present in that directory.

2. **TpWorker already exposes LoRA load/unload at the worker layer** (`python/sglang/srt/managers/tp_worker.py:180–212`). DeltaServe's `FinetuneInjector` has a much more natural insertion point in sglang than it has in vLLM — adapter swap is already a first-class TpWorker API.

3. **`python/sglang/srt/model_executor/hook_manager.py` already exists.** This is almost certainly the right place to wire `FinetuneAccumulator`'s forward hooks for activation capture, rather than monkey-patching `nn.Module` instances post-load. (Contents not yet inspected — TODO: confirm in Phase 2.)

4. **Three-level memory pool (ReqToTokenPool + TokenToKVPoolAllocator + KVCache) differs from vLLM's BlockManager v1/v2.** `FinetuningStore`'s KV-reservation logic — which in DeltaServe reserves block ranges for the SFT workload — will need full re-derivation against this three-level page-based design.

5. **`event_loop_overlap` already pipelines schedule + forward** (`scheduler.py:1538`). The DeltaServe `FinetuneCoordinator` pattern of "while the inference forward runs, schedule a bwd step" may map onto this existing overlap loop's structure rather than adding a separate coordination thread. This is a load-bearing architectural opportunity for the port.

6. **The Scheduler runs in a subprocess separate from the workers.** DeltaServe's vLLM Coordinator↔Scheduler interaction happens in-process; in sglang the same interaction crosses a process boundary and will need IPC through the zmq channels owned by `SchedulerIpcChannels` (`python/sglang/srt/managers/scheduler_components/ipc_channels.py:12`).

7. **Multiple `*GraphRunner` variants exist** (CudaGraph, BreakableCudaGraph, PiecewiseCudaGraph). The backward-pass mode almost certainly cannot be captured inside the standard CUDA graph; it likely wants `BreakableCudaGraphRunner` (`breakable_cuda_graph_runner.py:74`) or to bypass graph capture entirely on training steps. Decision should be made before Phase 4 (backward-pass wiring).
