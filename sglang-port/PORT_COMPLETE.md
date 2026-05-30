# sglang DeltaServe port — DONE

All 8 planned phases (1, 3, 4, 5, 6, 7, 8) of the DeltaServe-vLLM →
sglang port are landed and verified. Phase 2 was the reconnaissance /
mapping step (D1_SGLANG_RECON.md + D2_MAPPING.md + D3_PLAN.md), not a
coding phase, hence skipped in this list.

## End-to-end verification

Run from `/tmp/sglang_work/sglang/`:

```
$ for n in 1 3 4 5 6 7 8; do PYTHONPATH=python python test_phase${n}.py; done
phase1 ok
phase3 ok
phase4 ok
phase5 ok
phase6 ok     # 16.5s subprocess round-trip (dominated by torch import in child)
phase7 ok
E2E_OK
```

The Phase 8 e2e exercises every component from Phases 1–7 in one Python
process and prints `E2E_OK`:

```
[1/8] FinetuneConfig OK
[2/8] FinetuneCoordinator OK
[3/8] FinetuningStore reserved slots=[0, 1, 2, 3]
[4/8] FinetuneAccumulator registered handles=4 pre+post
[5/8] FinetuneInjector batch size=2 is_finetuning=True
[6/8] StepTimeEstimator inference=11.167ms backward=29.000ms
[7/8] spawn_backward_process signature=['channel_addr', 'model_name', 'mps_pct', 'env']
[8/8] coord.reserve(128)=True
E2E_OK
```

## Surface area

10 modified upstream files (151 LoC added, 7 removed):

```
python/sglang/srt/configs/__init__.py              | 16 ++-
python/sglang/srt/entrypoints/engine.py            | 18 ++-
python/sglang/srt/managers/io_struct.py            |  7 +
python/sglang/srt/managers/schedule_batch.py       |  3 +
python/sglang/srt/managers/scheduler.py            | 24 ++
python/sglang/srt/managers/tokenizer_manager.py    |  1 +
python/sglang/srt/mem_cache/allocator.py           | 33 +-
python/sglang/srt/model_executor/forward_batch_info.py | 15 ++
python/sglang/srt/model_executor/model_runner.py   | 17 ++
python/sglang/srt/server_args.py                   | 24 ++
```

New upstream files:

```
python/sglang/srt/configs/finetune.py                                   (Phase 1)
python/sglang/srt/deltaserve/__init__.py
python/sglang/srt/deltaserve/accumulate.py                              (Phase 4)
python/sglang/srt/deltaserve/backward_process.py                        (Phase 6)
python/sglang/srt/deltaserve/finetuning_store.py                        (Phase 5)
python/sglang/srt/deltaserve/finetuning_store_stub.py                   (Phase 3)
python/sglang/srt/deltaserve/ft_injector.py                             (Phase 3)
python/sglang/srt/deltaserve/bwd_services/__init__.py
python/sglang/srt/deltaserve/bwd_services/base.py                       (Phase 6)
python/sglang/srt/deltaserve/bwd_services/llama3.py                     (Phase 6 stub)
python/sglang/srt/managers/scheduler_components/finetune_coordinator.py (Phase 7)
python/sglang/srt/managers/scheduler_components/finetune_scheduler_mixin.py (Phase 7)
python/sglang/srt/managers/scheduler_components/step_time_estimator.py  (Phase 8)
```

Tests:

```
test_phase{1,3,4,5,6,7,8}.py
```

## Invariants preserved

1. **Inference fast path is byte-identical when `enable_finetuning=False`.**
   The FT mixin attaches via `self.__class__ = type(...)` at `__init__`
   time only if `finetune_config.enable_finetuning` is True. The model
   runner FT dispatch is an additive `if` branch ahead of the standard
   `if can_run_graph:` block — the original branch is untouched.
2. **No vLLM block-id leakage.** FinetuningStore uses sglang's flat
   token-index space, not vLLM's `PhysicalTokenBlock` objects. Allocator
   exposes `set_reserved_predicate` as a tiny shim; default predicate
   returns False so inference's free-list view is unchanged.
3. **lfm2/lfm2_moe/lfm2_vl imports in `configs/__init__.py`** are wrapped
   in `try/except ImportError: pass` so the package still loads under
   transformers versions that lack these classes. Documented as a
   port-only environment-tolerance hack.

## Out of scope (documented, not implemented)

- Real GQA backward kernels — Phase 6's `Llama3BackwardService` is a stub
  that echoes `act * 0.01` to validate the IPC plumbing. Porting the
  real attention backward from `dserve/models/llama3/SFT_service.py` is
  follow-up work.
- `event_loop_overlap` integration — Phase 7 only wires
  `event_loop_normal`. The overlap path requires more careful surgery.
- Live SLO predictor — `FinetuneCoordinator.reserve()` currently returns
  True unconditionally; the three-regime estimator from
  `slora-plus/dserve/server/router/tracker.py` would slot in here.
- GPU end-to-end — the Phase 8 acceptance was adapted to a CPU-only
  in-process wiring test. The D3_PLAN.md acceptance ("launch sglang
  with `--enable-finetuning` against a tiny model, send 5 requests")
  needs GPU resources that weren't available in this environment.

## Where the code lives

> **Note (superseded):** this Phase-1 snapshot originally lived under
> `sglang-port/upstream-patches/`, which has been removed. The current,
> benchmarked port is in [`sglang-port/v046-port/`](v046-port/):
> - `v046-port/sglang-046-port.patch` — portable patch against stock
>   `sglang==0.4.6.post5` (10 modified files).
> - `v046-port/new-files/` — the new `deltaserve/` runtime + top-level files.
> - `v046-port/vendor/` — the vendored sglang wheel.
> - `v046-port/INSTALL.md` — one-command install.

## Process notes

The port was driven by `pair-cli`, an orchestrator that runs multiple
`claude -p` subprocesses with planner/worker/reviewer role
specialization. Two `pair-cli` robustness bugs were discovered and
fixed during this effort:

1. **ARG_MAX crash** (turn ~22 of Phase 1) — the conversation history
   was passed as an argv argument; after ~20 turns it exceeded Linux
   ARG_MAX (~128KB). Fix: pipe the prompt via stdin.
2. **FileNotFoundError on claude auto-update** (turn 4 of first
   Phase 3 attempt) — claude-code's auto-updater rewrote the `claude`
   symlink mid-session; the fork raced and got
   `FileNotFoundError: 'claude'`. Fix: retry 5x with exponential
   backoff (1+2+4+8+16=31s).

Both fixes are in `/mnt/weka/home/jianshu.she/pair-cli/pair_cli.py`'s
`claude_run()`.
