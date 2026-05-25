"""FinetuningStore — sglang port of DeltaServe's FinetuningStore.

See D2_MAPPING.md §5 for the mapping rationale and D3_PLAN.md Phase 5 for the
implementation phase. Methods raise NotImplementedError until that phase lands.

Re-derives KV reservation against sglang's three-level memory pool
(``ReqToTokenPool`` → ``TokenToKVPoolAllocator`` → ``KVCache``); no vLLM
block-id assumptions carry over because sglang addressing is page-based.
"""
from __future__ import annotations


class FinetuningStore:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 5 not yet implemented")

    def claim(self, *args, **kwargs):
        raise NotImplementedError

    def commit_claimed(self, *args, **kwargs):
        raise NotImplementedError

    def release(self, *args, **kwargs):
        raise NotImplementedError

    def reserved_indices(self, *args, **kwargs):
        raise NotImplementedError
