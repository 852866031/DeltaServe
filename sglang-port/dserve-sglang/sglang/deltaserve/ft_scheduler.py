"""FinetuneScheduler — sglang port of DeltaServe's FinetuneScheduler.

See D2_MAPPING.md §2 for the mapping rationale and D3_PLAN.md Phase 7 for the
implementation phase. Methods raise NotImplementedError until that phase lands.

In sglang the equivalent will live as a mixin under
``python/sglang/srt/managers/scheduler_components/`` and be added to the MRO
of ``managers/scheduler.py:286 class Scheduler`` behind ``enable_finetuning``.
This module is a thin re-export / wiring shim for tests and external callers.
"""
from __future__ import annotations


class FinetuneScheduler:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 7 not yet implemented")

    def process_input_requests(self, *args, **kwargs):
        raise NotImplementedError

    def handle_generate_request(self, *args, **kwargs):
        raise NotImplementedError

    def select_batch(self, *args, **kwargs):
        raise NotImplementedError
