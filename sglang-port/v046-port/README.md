# sglang DeltaServe port — v0.4.6.post5 (real-server validated)

The original port (`sglang-port/upstream-patches/`) targeted a newer
sglang clone that didn't boot in this conda environment because of a
torch version skew (sglang's `pynccl_allocator.py` imports
`torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool`, which the
installed torch 2.6 lacks).

This directory re-applies the same port **against the system-installed
`sglang==0.4.6.post5`** so it can actually run. Two structural
adaptations from the newer-sglang port were required:

| Concern | Newer-sglang port | v0.4.6 port |
|---|---|---|
| Phase 4 FT dispatch target | `BreakableCudaGraphRunner.replay()` | force `can_run_cuda_graph=False` (eager fallback) |
| Phase 5 allocator | `mem_cache/allocator.py:TokenToKVPoolAllocator` | `mem_cache/paged_allocator.py:PagedTokenToKVPoolAllocator` |
| Coordinator + mixin location | `managers/scheduler_components/...` | `managers/finetune_*.py` (no `scheduler_components/` subdir in v0.4.6) |
| Backward subprocess spawn site | `Engine.__init__` | `_launch_subprocesses` (so `python -m sglang.launch_server` triggers it — `Engine` is bypassed by the HTTP path) |

## End-to-end validation

```
$ python sglang-port/v046-port/auto_benchmark_sglang.py --co --tight --ft-fraction 0.1
[bench] gpu=A100 shape=tight rows=224 co=True ft_frac=0.1
[bench] launching: python -m sglang.launch_server --model-path .../Llama-3.2-1B-Instruct
        --enable-finetuning --backward-mps-percentage 20
[bench] server pid=2625943 log=output/server_tight_co.log
[bench] server healthy
[bench] running timeline...
[bench] wrote output/timeline_results_tight_co.csv
[bench] ttft_s:    mean=0.024  p50=0.018  p95=0.022
[bench] latency_s: mean=0.536  p50=0.521  p95=0.654
[bench] reqs ok=224 err=0; ft_tagged=23
```

**224 requests, 0 errors, 23 FT-tagged successfully routed.** Server
also boots cleanly with `--enable-finetuning` and spawns the backward
subprocess with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20`. The FT dispatch
path was confirmed firing per-forward-step via temporary
`logging.warning` calls at the `ForwardBatch.init_new` population site
and the `model_runner._has_ft` check (logs subsequently stripped).

Per-class breakdown (n=201 inf, n=23 FT):

|  | Inference | FT-tagged |
|---|---|---|
| TTFT mean | 23.0 ms | 29.4 ms |
| Latency mean | 537 ms | 532 ms |
| worst TBT mean | 27.0 ms | 13.5 ms |

## What this proves

1. **Boot smoke**: server starts with `--enable-finetuning` and
   passes `/health` ✓
2. **Inference regression**: 201 non-FT requests complete cleanly,
   port did not break the inference fast path ✓
3. **FT plumbing end-to-end**:
   `GenerateReqInput.is_finetuning=True` →
   `TokenizedGenerateReqInput.is_finetuning=True` →
   `Req.is_finetuning=True` →
   `ModelWorkerBatch.is_finetuning_flags=[…,True,…]` →
   `ForwardBatch.is_finetuning_mask=tensor([…,True,…])` →
   `_forward_raw _has_ft=True` → cuda graph disabled → eager forward ✓
4. **Backward subprocess plumbing**: subprocess spawns, MPS env propagated,
   stays alive throughout the benchmark ✓
5. **Phase 7 conditional mixin**: `__class__` swap at runtime doesn't
   break the inference path under sustained load ✓

## What this does NOT prove

- **Real co-serving impact on inference TTFT** — the backward service
  is the Phase 6 `act * 0.01` stub, so FT-tagged requests consume no
  meaningful backward compute. To measure real interference we'd need
  to port the GQA backward kernels from
  `dserve/models/llama3/SFT_service.py` into
  `python/sglang/srt/deltaserve/bwd_services/llama3.py`.
- **Activation capture round-trip** — `FinetuneAccumulator` is wired
  via `mod.register_forward_pre_hook` but the hooks aren't attached at
  startup in this port; doing so requires hooking model load
  (`model_runner.py:load_model`) and tracking which modules get the
  pre-hook on which layers. Phase 4's design exists but isn't wired to
  the real model.
- **Live SLO predictor** — `FinetuneCoordinator.reserve()` always
  returns True; the three-regime estimator from
  `slora-plus/dserve/server/router/tracker.py` would slot here.

## How to reproduce

```bash
# 1. Apply the patches (assumes system sglang at the path below; adjust as needed):
SYS=$(python -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
patch -p1 -d "$SYS/.." < sglang-port/v046-port/sglang-046-port.patch

# 2. Drop in the new files:
cp sglang-port/v046-port/new-files/finetune.py "$SYS/srt/configs/"
cp -r sglang-port/v046-port/new-files/deltaserve "$SYS/srt/"
cp sglang-port/v046-port/new-files/finetune_coordinator.py "$SYS/srt/managers/"
cp sglang-port/v046-port/new-files/finetune_scheduler_mixin.py "$SYS/srt/managers/"
cp sglang-port/v046-port/new-files/step_time_estimator.py "$SYS/srt/managers/"

# 3. Run the benchmark:
python sglang-port/v046-port/auto_benchmark_sglang.py --co --tight --ft-fraction 0.1
```

## Files

| File | Purpose |
|---|---|
| `sglang-046-port.patch` | Unified diff for 10 modified upstream files (256 lines). |
| `new-files/finetune.py` | Phase 1: FinetuneConfig (34 fields). |
| `new-files/deltaserve/{accumulate,backward_process,finetuning_store,ft_injector,finetuning_store_stub}.py` | Phase 3-6: per-component code. |
| `new-files/deltaserve/bwd_services/{base,llama3}.py` | Phase 6: backward service ABC + Llama3 stub. |
| `new-files/{finetune_coordinator,finetune_scheduler_mixin,step_time_estimator}.py` | Phase 7-8: scheduler-level coordination + estimator. |
| `auto_benchmark_sglang.py` | Timeline-driven benchmark adapted from `DeltaServe-vLLM/eval/auto_benchmark.py` (~280 LoC). |
| `output/timeline_results_tight_co.csv` | 224-row per-request metrics from the validation run. |
| `output/server_tight_co.log` | Server stdout from the validation run (boot, ServerArgs, request log). |
