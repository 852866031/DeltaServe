import sys

sys.path.insert(0, "/tmp/sglang_work/sglang/python")

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.deltaserve.finetuning_store import FinetuningStore


def main():
    # Note: spec mentioned page_size=8, but TokenToKVPoolAllocator hardcodes
    # page_size=1 internally (see allocator.py:142). Use size=64; slot 0 is
    # reserved as padding so free_pages starts as arange(1, 65).
    allocator = TokenToKVPoolAllocator(
        size=64,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    total = len(allocator.free_pages)  # ground-truth usable slot count
    allocated: set[int] = set()

    def invariant(stage: str, store: FinetuningStore):
        free_in_pool = set(int(i) for i in allocator.free_pages.tolist())
        reserved = set(store._reserved_indices)
        assert reserved.isdisjoint(free_in_pool), (
            f"{stage}: reserved leaked back into free_pages"
        )
        free_count = len(free_in_pool)
        reserved_count = len(reserved)
        allocated_count = len(allocated)
        assert free_count + reserved_count + allocated_count == total, (
            f"{stage}: free={free_count} + reserved={reserved_count} + "
            f"allocated={allocated_count} != total={total}"
        )
        print(
            f"  [{stage}] free={free_count} reserved={reserved_count} "
            f"allocated={allocated_count} total={total}"
        )

    store = FinetuningStore(req_to_token_pool=None, kv_pool_allocator=allocator)
    invariant("init", store)

    reserved = store.reserve(8)
    assert len(reserved) == 8, f"expected 8 reserved indices, got {len(reserved)}"
    for i in reserved:
        assert store.is_reserved(i), f"is_reserved({i}) should be True"
    invariant("after-reserve", store)

    inf_out = allocator.alloc(16)
    assert inf_out is not None, "inference alloc(16) returned None"
    inf_indices = [int(i) for i in inf_out.tolist()]
    overlap = set(inf_indices) & set(reserved)
    assert not overlap, f"reserved leaked into inference alloc: {overlap}"
    allocated.update(inf_indices)
    invariant("after-inference-alloc", store)

    store.release(reserved)
    for i in reserved:
        assert not store.is_reserved(
            i
        ), f"is_reserved({i}) should be False after release"
    invariant("after-release", store)

    # Drain the rest of the pool. With need_sort=False, free() appends to
    # the tail of free_pages, so we must alloc enough to reach the released
    # slots and confirm they are reusable.
    drain = len(allocator.free_pages)
    reuse_out = allocator.alloc(drain)
    assert reuse_out is not None, f"reuse alloc({drain}) returned None"
    reuse_indices = set(int(i) for i in reuse_out.tolist())
    reused = reuse_indices & set(reserved)
    assert reused == set(reserved), (
        f"expected all released slots to be reusable; got {reuse_indices}, "
        f"previously-reserved {set(reserved)}, intersection {reused}"
    )
    allocated.update(reuse_indices)
    invariant("after-reuse", store)

    print("phase5 ok")


if __name__ == "__main__":
    main()
