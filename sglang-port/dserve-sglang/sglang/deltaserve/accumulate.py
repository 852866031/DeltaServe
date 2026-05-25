"""FinetuneAccumulator — sglang port of DeltaServe's FinetuneAccumulator.

See D2_MAPPING.md §4 for the mapping rationale and D3_PLAN.md Phase 4 for the
implementation phase. Methods raise NotImplementedError until that phase lands.

Wires into sglang's existing ``model_executor/hook_manager.py`` to register
pre/post-forward hooks that copy per-layer activations for FT-tagged samples
into the shared activation buffers. CUDA graph capture must be disabled or
use the breakable variant for batches carrying FT samples.
"""
from __future__ import annotations


class FinetuneAccumulator:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 4 not yet implemented")

    def register_hooks(self, *args, **kwargs):
        raise NotImplementedError

    def capture(self, *args, **kwargs):
        raise NotImplementedError

    def drain(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError
