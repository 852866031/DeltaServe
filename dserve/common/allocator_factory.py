import torch

from dserve.common.unified_mem_allocator import UnifiedMemoryAllocator
from dserve.common.packed_kv_mem_allocator import PackedKVMemoryAllocator


_REGISTRY = {
    "unified":   UnifiedMemoryAllocator,
    "packed_kv": PackedKVMemoryAllocator,
}


def get_allocator_class(name: str):
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown allocator '{name}'. choices: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def make_allocator(name: str, *, head_num, head_dim, vocab_size, layer_num,
                   max_pool_size, dtype=torch.float16, device="cuda",
                   log_path=None, max_finetuning_tokens=1024,
                   num_kv_heads=None):
    """Build the allocator selected by `name`. Subclass-specific kwargs are
    forwarded only when relevant (e.g. num_kv_heads for packed_kv)."""
    cls = get_allocator_class(name)
    extra = {}
    if cls is PackedKVMemoryAllocator:
        if num_kv_heads is None:
            num_kv_heads = head_num   # MHA fallback (F = 1)
        extra["num_kv_heads"] = num_kv_heads
    return cls(
        head_num=head_num,
        head_dim=head_dim,
        vocab_size=vocab_size,
        layer_num=layer_num,
        max_pool_size=max_pool_size,
        dtype=dtype,
        device=device,
        log_path=log_path,
        max_finetuning_tokens=max_finetuning_tokens,
        **extra,
    )
