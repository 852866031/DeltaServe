# DeltaServe — sglang Port

This directory holds the **port of DeltaServe to sglang**. It is the
parallel of the existing
[DeltaServe-vLLM](https://github.com/852866031/DeltaServe-vLLM) fork, but
targeted at [sglang](https://github.com/sgl-project/sglang) instead of vLLM.

**Status: all 8 phases complete. See [`PORT_COMPLETE.md`](PORT_COMPLETE.md)
for the per-phase verification log + surface area + out-of-scope items.**

```
$ for n in 1 3 4 5 6 7 8; do PYTHONPATH=python python test_phase${n}.py; done
phase1 ok / phase3 ok / phase4 ok / phase5 ok / phase6 ok / phase7 ok / E2E_OK
```

The work was driven by `pair-cli` (a multi-Claude orchestration harness —
see `../pair-cli/`). Each phase ran as its own planner/worker/reviewer
session against a YAML config; deliverables and acceptance tests are
pinned by the reviewer role. Phase 1's original session transcript is
preserved as `SESSION_TRANSCRIPT.md`.

---

## Contents

| File | Purpose |
|---|---|
| `PORT_COMPLETE.md` | **Final status doc.** Per-phase test output, surface area against upstream sglang, invariants preserved, out-of-scope items, where the patched code lives. Read this first. |
| `D1_SGLANG_RECON.md` | sglang architecture reconnaissance (188 lines, 7 themes). Scheduler/ModelRunner/Worker/LoRA/process-model/KV-cache/batch-scheduling — each with file:line refs and "surprises that affect the port". |
| `D2_MAPPING.md` | Component-by-component mapping (285 lines). For each of the 7 DeltaServe-vLLM components, lists: what it does in vLLM, where in sglang the equivalent should live, and sglang-specific risks. |
| `D3_PLAN.md` | Original phased implementation plan (166 lines). 8 numbered phases. The actual landed work follows this plan; deviations are documented in `PORT_COMPLETE.md` under "Out of scope". |
| `upstream-patches/` | **The actual port code.** `sglang-deltaserve-port.patch` is the `git diff HEAD` against the upstream sglang clone (10 modified files, +151 LoC). `new-files/` mirrors all new sglang src files + the 7 acceptance tests. |
| `dserve-sglang/` | Original Phase 1 skeleton package (14 files, 375 LoC). Superseded by the real implementation in `upstream-patches/new-files/`. Kept for design-doc reference. |
| `SESSION_TRANSCRIPT.md` | Full chat.md of the original Phase 1 pair-cli session. Audit trail. |
| `phase1_task.yaml` | Original Phase 1 pair-cli config. Subsequent phases used per-phase YAMLs under `/tmp/sglang_phase_0{3,4,5,6,7,8}.yaml`. |
| `.gitignore` | Excludes `__pycache__/` and any local sglang clone (the upstream clone is **not** vendored here — see "Working with this branch"). |

---

## Quick verification (all 7 acceptance tests)

The port lives in a real sglang clone at `/tmp/sglang_work/sglang/`. All
tests are CPU-only (no GPU required for the IPC + wiring tests):

```bash
cd /tmp/sglang_work/sglang
for n in 1 3 4 5 6 7 8; do
  PYTHONPATH=python python test_phase${n}.py | tail -1
done
# Expected:
# phase1 ok
# phase3 ok
# phase4 ok
# phase5 ok
# phase6 ok        # ~16s due to torch import in subprocess
# phase7 ok
# E2E_OK           # exercises all 8 components in one process
```

The acceptance test for Phase 8 prints a checklist of every Phase 1–7
component being touched before emitting `E2E_OK`:

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

---

## Reading order

1. **`PORT_COMPLETE.md`** — what landed, what didn't, where to find the
   patched code. Start here.
2. **`D1_SGLANG_RECON.md`** — orient yourself in sglang's codebase.
3. **`D2_MAPPING.md`** — DeltaServe-vLLM ↔ sglang component map.
4. **`D3_PLAN.md`** — the original phased plan + acceptance tests.
5. **`upstream-patches/sglang-deltaserve-port.patch`** — the actual diff
   against upstream sglang (the 10-file modification set, 338 lines).
6. **`upstream-patches/new-files/python/sglang/srt/deltaserve/`** —
   the new sglang code (FinetuneAccumulator, FinetuningStore,
   FinetuneInjector, BackwardProcess, etc.).
7. (Optional) **`SESSION_TRANSCRIPT.md`** — Phase 1's pair-cli transcript.

---

## Phase status — all done

| # | Phase | Test | Status |
|---|---|---|---|
| 1 | FinetuneConfig + ServerArgs + Scheduler kwarg | `phase1 ok` | ✅ |
| 2 | (Recon + design docs — D1/D2/D3 — not a coding phase) | — | ✅ |
| 3 | FinetuneInjector + is_finetuning routing | `phase3 ok` | ✅ |
| 4 | FinetuneAccumulator hooks + BreakableCudaGraphRunner dispatch | `phase4 ok` | ✅ |
| 5 | FinetuningStore on 3-level KV pool + allocator filter | `phase5 ok` | ✅ |
| 6 | BackwardProcess subprocess + ZMQ IPC + MPS env propagation | `phase6 ok` | ✅ |
| 7 | FinetuneSchedulerMixin + Coordinator + conditional class-swap | `phase7 ok` | ✅ |
| 8 | StepTimeEstimator + e2e wiring (8/8 components) | `E2E_OK` | ✅ |

**Surface area against upstream sglang:** 10 modified files (+151 LoC) and
13 new files. See `PORT_COMPLETE.md` for the detailed breakdown +
preserved invariants + out-of-scope items.

---

## Working with this branch

```bash
git checkout feature/sglang-port-complete
cd sglang-port

# Clone sglang to apply the port patch:
git clone --depth 1 https://github.com/sgl-project/sglang.git
cd sglang
git apply ../upstream-patches/sglang-deltaserve-port.patch
# Then drop in the new files:
cp -r ../upstream-patches/new-files/python/sglang/srt/deltaserve python/sglang/srt/
cp ../upstream-patches/new-files/python/sglang/srt/configs/finetune.py python/sglang/srt/configs/
cp ../upstream-patches/new-files/python/sglang/srt/managers/scheduler_components/{finetune_coordinator,finetune_scheduler_mixin,step_time_estimator}.py python/sglang/srt/managers/scheduler_components/
cp ../upstream-patches/new-files/test_phase*.py .

# Run the acceptance tests:
for n in 1 3 4 5 6 7 8; do PYTHONPATH=python python test_phase${n}.py | tail -1; done
```

To extend the port (e.g. real GQA backward kernels, `event_loop_overlap`
integration, live SLO predictor), spin a new pair-cli session targeted
at the next deliverable:

```bash
pair-cli start /path/to/next_phase.yaml
pair-cli tail <session_id>
```

---

## Why a separate branch (not in DeltaServe-vLLM)?

The vLLM fork is upstream-vendored; the sglang port targets a different
upstream and needs its own deviation history. Keeping it under this repo
(DeltaServe) on a dedicated branch lets the design docs sit next to the
existing co-serving research notes (`CLAUDE.md`, `docs/`) without polluting
either fork's diff against its upstream.
