"""StepTimeEstimator — sglang port of DeltaServe's StepTimeEstimator.

See D2_MAPPING.md §6 for the mapping rationale and D3_PLAN.md Phase 8 for the
implementation phase. Methods raise NotImplementedError until that phase lands.

Predicts per-step execution time so the FinetuneCoordinator can keep admission
within the configured TTFT / TBT SLOs. Data source will be sglang's
``scheduler_components/pool_stats_observer.py`` once Phase 8 lands.
"""
from __future__ import annotations


class StepTimeEstimator:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 8 not yet implemented")

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError
