import torch

from dserve.common.unified_mem_allocator import (
    UnifiedMemoryAllocator,
    PageType,
)


class PackedKVMemoryAllocator(UnifiedMemoryAllocator):
    """
    Packs F = num_attention_heads / num_key_value_heads KV sub-slots into
    each physical page. KV-reading kernels see a reshape view where each
    row is a [num_kv_heads, head_dim] sub-slot; activation, adapter, and
    embedding pages keep the original [num_attention_heads, head_dim]
    layout.

    Allocation strategy: always claim whole pages from the parent
    allocator (`ceil(n / F)` pages), expose the first `n` sub-slots from
    those pages to the caller, and track per-page refcount so the page
    returns to FREE only when every sub-slot the caller received has
    been freed. Trade-off: a partial last page's unused sub-slots are
    "wasted" until the page fully drains — but we keep alloc fast
    (parent's hot path with no extra per-sub-slot scatter) and avoid the
    sync-heavy partial-page scan that the previous design needed.

    Index spaces:
      - alloc(KV_CACHE) / alloc_contiguous_kv -> sub-slot ids in [0, tot_size * F)
      - alloc(other PageType)                 -> page ids in [0, tot_size)

    KV freers MUST call free_kv (sub-slot semantics). free() is inherited
    from the parent and is for PAGE-level frees (activation, adapter,
    embedding); sub-slot ids in [0, tot_size) collide with page ids and
    cannot be safely disambiguated by range.
    """

    def __init__(self, head_num, head_dim, vocab_size, layer_num,
                 max_pool_size, dtype=torch.float16, device="cuda",
                 log_path=None, max_finetuning_tokens=1024,
                 num_kv_heads=None):
        super().__init__(head_num, head_dim, vocab_size, layer_num,
                         max_pool_size, dtype, device, log_path,
                         max_finetuning_tokens)

        kv = num_kv_heads if num_kv_heads is not None else head_num
        assert head_num % kv == 0, (
            f"num_attention_heads={head_num} must be divisible by "
            f"num_key_value_heads={kv}"
        )
        self.F = head_num // kv
        self.num_kv_heads = kv

        # Reshape view — same physical memory, F * tot_size rows of
        # [num_kv_heads, head_dim] each. Free metadata change.
        self._gpu_pools_kv = [
            p.view(self.tot_size * self.F, kv, head_dim)
            for p in self.gpu_pools
        ]

        # Per-page refcount: number of sub-slots from this page currently
        # held by the caller. Set on alloc, decremented in free_kv via
        # `kv_refcount -= bincount(page_ids)`, page returns to FREE when
        # it hits 0. int64 dtype matches torch.bincount's output so the
        # decrement is a single in-place subtract with no cast kernel.
        # Only meaningful when page_type_map[p] == KV_CACHE.
        self.kv_refcount = torch.zeros(
            self.tot_size, dtype=torch.int64, device=device,
        )

        print(f"PackedKVMemoryAllocator: F={self.F}, "
              f"num_kv_heads={kv}, KV view shape={tuple(self._gpu_pools_kv[0].shape)}")

    # ------------------------------------------------------------------
    # Pool accessor — only KV view is overridden; activation and adapter
    # accessors inherit from the parent.
    # ------------------------------------------------------------------
    def get_kv_pool(self, layer_id: int) -> torch.Tensor:
        return self._gpu_pools_kv[layer_id]

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    def alloc(self, num_pages: int, page_type: PageType) -> torch.Tensor:
        if page_type == PageType.KV_CACHE:
            return self._alloc_kv(num_pages)
        return super().alloc(num_pages, page_type)

    def _alloc_kv(self, n: int) -> torch.Tensor:
        """Whole-page allocation: claim ceil(n/F) pages from the parent,
        return the first n sub-slots derived arithmetically from those
        page ids. The last page may be partial (refcount < F)."""
        if self.F == 1:
            return super().alloc(n, PageType.KV_CACHE)
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        with self.page_table_lock:
            need_pages = (n + self.F - 1) // self.F
            page_ids = super().alloc(need_pages, PageType.KV_CACHE)

            # Refcount: write F to every picked page, patch the last one
            # if the trailing page is only partially filled. Aligned
            # allocations (n divisible by F) skip the patch entirely —
            # one kernel total in the fast path.
            self.kv_refcount[page_ids] = self.F
            if n % self.F != 0:
                self.kv_refcount[page_ids[-1]] = n - (need_pages - 1) * self.F

            # Sub-slot ids: each page contributes F consecutive ids
            # starting at page_id*F. Take first n in the flattened view.
            offsets = torch.arange(self.F, device=self.device)
            sub_ids = (page_ids.unsqueeze(1) * self.F + offsets
                       ).flatten()[:n]
            return sub_ids

    def alloc_contiguous_kv(self, need_size: int, page_type: PageType):
        """Returns 2*need_size consecutive sub-slot ids (K half then V
        half) backed by 2 * ceil(need_size / F) consecutive FREE pages,
        or None if no such run exists. The K half and V half each have
        their last page possibly partial."""
        assert page_type == PageType.KV_CACHE
        if self.F == 1:
            return super().alloc_contiguous_kv(need_size, page_type)

        with self.page_table_lock:
            need_pages_per_half = (need_size + self.F - 1) // self.F
            result = super().alloc_contiguous_kv(need_pages_per_half, page_type)
            if result is None:
                return None
            phys_k_pages, sk_pg, ek_pg, phys_v_pages, sv_pg, ev_pg = result

            # Refcount on K and V halves: write F to every picked page,
            # patch the trailing page if need_size isn't divisible by F.
            self.kv_refcount[phys_k_pages] = self.F
            self.kv_refcount[phys_v_pages] = self.F
            if need_size % self.F != 0:
                last_used = need_size - (need_pages_per_half - 1) * self.F
                self.kv_refcount[phys_k_pages[-1]] = last_used
                self.kv_refcount[phys_v_pages[-1]] = last_used

            # Sub-slot ranges. K = [sk_pg*F, sk_pg*F + need_size).
            # V = [sv_pg*F, sv_pg*F + need_size).
            sub_k_start = sk_pg * self.F
            sub_v_start = sv_pg * self.F
            phys_k_subs = torch.arange(
                sub_k_start, sub_k_start + need_size,
                dtype=torch.long, device=self.device,
            )
            phys_v_subs = torch.arange(
                sub_v_start, sub_v_start + need_size,
                dtype=torch.long, device=self.device,
            )
            return (phys_k_subs, sub_k_start, sub_k_start + need_size,
                    phys_v_subs, sub_v_start, sub_v_start + need_size)

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------
    # Note: free() is inherited from UnifiedMemoryAllocator and is for
    # PAGE-level frees (activation, adapter, embedding). KV freers MUST
    # call free_kv() — sub-slot ids in [0, tot_size) collide with page
    # ids and cannot be safely disambiguated by range.

    def free_kv(self, sub_ids):
        if not isinstance(sub_ids, torch.Tensor):
            sub_ids = torch.as_tensor(
                sub_ids, dtype=torch.long, device=self.device
            )
        if sub_ids.numel() == 0:
            return
        if self.F == 1:
            # Sub-slot id == page id; delegate to parent's page-level free
            # which also maintains the counter.
            return super().free(sub_ids)

        with self.page_table_lock:
            page_ids = sub_ids // self.F

            # bincount folds three things into one kernel:
            #   - per-page count of sub-slots being freed (handles
            #     duplicate page_ids correctly via accumulation),
            #   - the "touched pages" mask (decrements > 0),
            #   - source values for the refcount decrement.
            # Output size is fixed at tot_size, so no host sync (unlike
            # torch.unique). int64 dtype matches kv_refcount, no cast.
            decrements = torch.bincount(page_ids, minlength=self.tot_size)
            self.kv_refcount -= decrements

            # A page transitions KV → FREE iff it was touched in this
            # batch AND its refcount is now 0. Pure tensor algebra —
            # no boolean-index sync.
            empty_mask = (decrements > 0) & (self.kv_refcount == 0)

            # Masked scatter writes are no-ops when the mask is empty
            # (typical during steady-state decode); the kernels still
            # launch but write zero positions.
            self.page_type_map[empty_mask] = int(PageType.FREE.value)
            self.free_bitmap |= empty_mask

            # One sync to keep _used_pages coherent on the CPU side.
            self._used_pages -= int(empty_mask.sum().item())
