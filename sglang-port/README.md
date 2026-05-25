# DeltaServe — sglang Port (Phase 1)

This directory holds the **scaffolding + design** for porting DeltaServe to
the [sglang](https://github.com/sgl-project/sglang) serving framework. It is
the parallel of the existing
[DeltaServe-vLLM](https://github.com/852866031/DeltaServe-vLLM) fork, but
targeted at sglang instead of vLLM.

**This is Phase 1 only.** Phase 1 produces a working skeleton package, a
file-by-file integration plan, and a reconnaissance of sglang's
architecture. No real backward pass, scheduler subclass, or hook installation
is implemented yet — those land in Phases 2-8 per `D3_PLAN.md`.

The artefacts in this directory were produced **by `pair-cli`** (a
multi-Claude orchestration harness — see `../pair-cli/` or
https://github.com/852866031/DeltaServe/tree/feature/cuda-graph-prefill-finetuning).
The pair-cli session ran for 18 turns over ~14 minutes with three roles
(planner / worker / reviewer), the reviewer rejected the work 9 times before
finally approving on turn 18. The full session transcript is preserved as
`SESSION_TRANSCRIPT.md`.

---

## Contents

| File | Purpose |
|---|---|
| `D1_SGLANG_RECON.md` | sglang architecture reconnaissance (188 lines, 7 themes). Scheduler/ModelRunner/Worker/LoRA/process-model/KV-cache/batch-scheduling — each with file:line refs and "surprises that affect the port". |
| `D2_MAPPING.md` | Component-by-component mapping (285 lines). For each of the 7 DeltaServe-vLLM components, lists: what it does in vLLM, where in sglang the equivalent should live, and sglang-specific risks. |
| `D3_PLAN.md` | Phased implementation plan (166 lines). 8 numbered phases, each with goal / files to create+modify / acceptance test / LoC estimate. ~3,070 LoC total. |
| `dserve-sglang/` | Skeleton package: 14 files, 375 LoC. Every method body raises `NotImplementedError("Phase N not yet implemented")` with a back-reference to the phase that will fill it in. |
| `SESSION_TRANSCRIPT.md` | Full chat.md of the pair-cli session that produced this work. Audit trail. |
| `phase1_task.yaml` | The exact pair-cli config (delivery criteria + role prompts) that drove the session. Re-run with `pair-cli run phase1_task.yaml` to reproduce. |
| `.gitignore` | Excludes `__pycache__/` and any local sglang clone (the upstream clone is **not** vendored here — see "Working with this branch"). |

---

## Quick verification (smoke test from `D3_PLAN.md` Phase 2)

```bash
cd sglang-port/dserve-sglang && PYTHONPATH=. python -c \
    "import sglang.deltaserve.coordinator; \
     import sglang.deltaserve.ft_scheduler; \
     import sglang.deltaserve.backward_process; \
     print('imports ok')"
# Expected: imports ok
```

All 13 skeleton modules also load cleanly; the `FinetuneConfig` dataclass
has 34 fields mirroring the vLLM port's `config/finetune.py`.

---

## Reading order

1. **`D1_SGLANG_RECON.md`** — orient yourself in sglang's codebase.
2. **`D2_MAPPING.md`** — see which DeltaServe-vLLM component goes where.
3. **`D3_PLAN.md`** — see the build sequence + acceptance tests per phase.
4. **`dserve-sglang/`** — browse the stub package layout.
5. (Optional) **`SESSION_TRANSCRIPT.md`** — read how the planner/worker/reviewer
   triangle arrived at this output. Useful if you want to spin a similar
   session for Phase 2+.

---

## Phase progression (from `D3_PLAN.md`)

| # | Phase | Files | Acceptance test | LoC |
|---|---|---|---|---:|
| 1 | FinetuneConfig + ServerArgs plumbing | 1 new + 2 modified | `python -c "ServerArgs(model_path='x').enable_finetuning"` | ~120 |
| 2 | Skeleton package + import smoke | ~12 new | `python -c "import sglang.deltaserve.*"` | ~200 |
| 3 | FinetuneInjector using existing LoRA plumbing | 1 new + 2 modified | `pytest test_injector_loads_adapter` | ~250 |
| 4 | FinetuneAccumulator via hook_manager + breakable graph | 1 new + 2 modified | `pytest test_accumulator_captures` | ~400 |
| 5 | FinetuningStore on 3-level KV pool | 1 new + 2 modified | `pytest test_store_alloc_release` | ~400 |
| 6 | BackwardProcess subprocess + IPC | 3 new + 2 modified | `pytest test_bwd_subprocess_roundtrip` | ~600 |
| 7 | FinetuneScheduler mixin + FinetuneCoordinator | 4 new + 2 modified | `pytest test_scheduler_admits_ft_batch` | ~700 |
| 8 | StepTimeEstimator + e2e smoke | 2 new + 1 modified | `bash e2e_smoke.sh` | ~500 |
|   |   |   | **Total** | **~3,070** |

Phases 1-2 are landable on this branch. Phases 3+ require vendoring sglang
(either as a git submodule or another fork) so we can modify
`python/sglang/srt/managers/scheduler.py` and friends; that decision is
deferred to a follow-up.

---

## Working with this branch

```bash
git checkout feature/sglang-port-phase1
cd sglang-port

# (optional) clone sglang for cross-reference while reading the docs
git clone --depth 1 https://github.com/sgl-project/sglang.git
# `sglang/` is in .gitignore so it won't be committed.
```

To extend the port:

```bash
# Spin up a new pair-cli session targeted at Phase N (edit phase1_task.yaml first):
pair-cli run sglang-port/phase1_task.yaml
# Or background:
pair-cli start sglang-port/phase1_task.yaml
pair-cli tail <session_id>
```

---

## Why a separate branch (not in DeltaServe-vLLM)?

The vLLM fork is upstream-vendored; the sglang port targets a different
upstream and needs its own deviation history. Keeping it under this repo
(DeltaServe) on a dedicated branch lets the design docs sit next to the
existing co-serving research notes (`CLAUDE.md`, `docs/`) without polluting
either fork's diff against its upstream.
