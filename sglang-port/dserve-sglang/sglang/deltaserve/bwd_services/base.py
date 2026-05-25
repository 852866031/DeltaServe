"""BackwardService — sglang port of DeltaServe's BackwardService base class.

See D2_MAPPING.md §3 for the mapping rationale and D3_PLAN.md Phase 6 for the
implementation phase. Methods raise NotImplementedError until that phase lands.

Per-model concrete services (llama3, opt, etc.) will subclass this in
sibling modules. The ``run(conn)`` method is the subprocess main loop.
"""
from __future__ import annotations


class BackwardService:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 6 not yet implemented")

    def compute_loss_and_grad(self, *args, **kwargs):
        raise NotImplementedError

    def process_backward(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
