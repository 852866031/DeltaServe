"""BackwardProcess — sglang port of DeltaServe's BackwardProcess.

See D2_MAPPING.md §3 for the mapping rationale and D3_PLAN.md Phase 6 for the
implementation phase. Methods raise NotImplementedError until that phase lands.

A sibling subprocess to the sglang Scheduler. Runs under a constrained CUDA
MPS partition (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) and consumes activation
tensors handed across via the IPC channel (see ``bwd_services/base.py``).
"""
from __future__ import annotations


class BackwardProcess:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 6 not yet implemented")

    def spawn(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def shutdown(self, *args, **kwargs):
        raise NotImplementedError
