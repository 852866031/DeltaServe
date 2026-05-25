"""FinetuneInjector — sglang port of DeltaServe's FinetuneInjector.

See D2_MAPPING.md §5 for the mapping rationale and D3_PLAN.md Phase 3 for the
implementation phase. Methods raise NotImplementedError until that phase lands.

Routes adapter load/unload through sglang's pre-existing worker-level LoRA
API (``tp_worker.py:180 load_lora_adapter`` and friends) so we don't
re-implement adapter plumbing.
"""
from __future__ import annotations


class FinetuneInjector:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 3 not yet implemented")

    def tag_request(self, *args, **kwargs):
        raise NotImplementedError

    def load_adapter(self, *args, **kwargs):
        raise NotImplementedError

    def unload_adapter(self, *args, **kwargs):
        raise NotImplementedError
