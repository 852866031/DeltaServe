---
name: Bugs discovered and fixed during predictor research
description: Code bugs found and fixed, with root cause and location, so future sessions don't re-introduce them
type: project
---

## Bug 1 — EP/TP expert weights wiped on every shard load (FIXED 2026-03-16)

**Files**:
- `S-LoRA/slora/models/mixtral/layer_weights/transformer_layer_weight_ep.py` (EP variant)
- `S-LoRA/slora/models/mixtral/layer_weights/transformer_layer_weight.py` (TP variant)

**Symptom**: Model ran but produced garbage outputs (all-zero or near-zero logits).
Expert weights appeared to be unloaded.

**Root cause**: Both `_load_ffn_weights` methods contained:
```python
self.experts_w1_ = []
self.experts_w3_ = []
self.experts_w2_ = []
```
These three lines reset the expert lists unconditionally at the START of each call.
`hf_load_utils.load_hf_weights()` calls `layer.load_hf_weights(weights)` once per
shard file. Mixtral-8x7B has 19 shard files. Layer 0's expert weights are in shard 1.
Shards 2–19 don't contain expert keys for layer 0, so when they're processed, `_load_ffn_weights`
is called, resets the lists, finds no expert keys, and exits — leaving all experts empty.

**Fix**: Removed the three reset lines. Lists are already initialized to `[]` in `__init__`.
Comment added: "# Do NOT reset — initialized in __init__, called once per shard file."

**How to verify**: After loading, call `layer.verify_load()` or check `len(layer.experts_w1_)`.
For EP: should equal `num_local_experts // world_size` (4 for 2-GPU 8-expert model).
For TP: should equal `num_local_experts` (8).

---

## Bug 2 — `reset_all_pool()` OOM in sweep loops (FIXED 2026-03-16)

**Files**:
- `S-LoRA/test/mixtral/exp1_5/sweep_real.py`
- `S-LoRA/test/llama3/exp1_5/sweep_real.py`
- `S-LoRA/test/mixtral/exp2/stress_test.py`

**Symptom**: CUDA OOM crash during the sweep loop, even though MAX_POOL was calculated
to leave enough headroom.

**Root cause**: `MemoryAllocator.reset_all_pool()` (in `slora/common/mem_allocator.py`)
creates ALL new tensors before freeing old ones:
```python
self.key_buffer = [torch.zeros(...) for _ in range(self.layer_num)]  # NEW
# old key_buffer not freed yet → peak doubles
```
At MAX_POOL=35K on Llama3-8B, this temporarily doubles the ~46GB SFT buffer allocation.

**Fix**: Use `model.mem_manager.free_all()` instead. This ONLY resets the logical
KV state bitmap (`mem_state[:] = 1`, `can_use_mem_size = tot_size`) with no tensor
reallocation. All sweep/stress scripts now use `free_all()`.

---

## Bug 3 — Llama3 `load_hf_weights` missing dummy-mode early return (FIXED)

**File**: `S-LoRA/slora/models/llama3/layer_weights/transformer_layer_weight.py`

**Symptom**: When loading with `dummy=True`, the method tried to call `_load_qkvo_weights(weights)`
with a dummy weights dict, causing key errors.

**Fix**: Added early return in `load_hf_weights`:
```python
def load_hf_weights(self, weights, dummy=False):
    if dummy:
        self._load_qkvo_dummy_weights()
        self._load_ffn_dummy_weights()
        return
    ...
```

---

## Stale .pyc cache issue (one-time)

When fixes to `transformer_layer_weight_ep.py` weren't taking effect, the cause was
Python loading a stale `.pyc` file from `__pycache__`. Fixed with:
```bash
find /mnt/nfs/home/ramya/slora-plus/S-LoRA -name "*.pyc" -path "*/mixtral/*" | xargs rm -f
```
Not a recurring issue, but worth knowing if a code fix seems to have no effect.
