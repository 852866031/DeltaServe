"""FinetuneCoordinator — sglang port of DeltaServe's FinetuneCoordinator.

See D2_MAPPING.md §1 for the mapping rationale and D3_PLAN.md Phase 7 for the
implementation phase. Methods raise NotImplementedError until that phase lands.

In the upstream fork this is a process-wide singleton owned by the Scheduler
process; it gates admission of FT samples into each step, triggers the
backward when the activation buffer is full, and brokers the GPU-yield
contract during prefill.
"""
from __future__ import annotations


class FinetuneCoordinator:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 7 not yet implemented")

    def reserve(self, *args, **kwargs):
        raise NotImplementedError

    def release_reserve(self, *args, **kwargs):
        raise NotImplementedError

    def note_injection(self, *args, **kwargs):
        raise NotImplementedError

    def trigger_backward(self, *args, **kwargs):
        raise NotImplementedError

    def gpu_pause_backward(self, *args, **kwargs):
        raise NotImplementedError

    def gpu_resume_backward(self, *args, **kwargs):
        raise NotImplementedError

    def poll_backward(self, *args, **kwargs):
        raise NotImplementedError

    def on_backward_done(self, *args, **kwargs):
        raise NotImplementedError
