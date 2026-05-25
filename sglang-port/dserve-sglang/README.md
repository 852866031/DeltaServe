# dserve-sglang

Port of DeltaServe (https://github.com/852866031/DeltaServe-vLLM) onto sglang
(https://github.com/sgl-project/sglang). Phase-1 scaffold: stubs only.

## Layout
- `sglang/deltaserve/` — net-new co-serving components (mirrors `vllm/deltaserve/` in the upstream fork).
  - `coordinator.py` — FinetuneCoordinator (admission + bwd gating)
  - `ft_scheduler.py` — FinetuneScheduler mixin
  - `backward_process.py` / `bwd_services/` — backward subprocess + per-model services
  - `accumulate.py` — forward-hook activation capture
  - `finetuning_store.py` — KV reservation for FT samples
  - `ft_injector.py` — request tagging + adapter wiring
  - `estimator.py` — step-time predictor
- `sglang/config/finetune.py` — `FinetuneConfig` dataclass (copied from DeltaServe).

## Status
All concrete methods raise `NotImplementedError`. See:
- `../D1_SGLANG_RECON.md` — sglang internals reconnaissance
- `../D2_MAPPING.md` — component → sglang-target mapping
- `../D3_PLAN.md` — 8-phase implementation plan

## Smoke
PYTHONPATH=. python -c "import sglang.deltaserve.coordinator, sglang.deltaserve.ft_scheduler, sglang.deltaserve.backward_process; print('imports ok')"
