# DeltaServe GPU Memory Analysis — Llama-3-8B Co-Serving

**Scope:** `python eval/llama3/auto_benchmark.py --co --bwd_graph --decode_graph`
**Model:** `meta-llama/Meta-Llama-3-8B` (32 layers, `hidden=4096`, 32 Q / 8 KV heads, `head_dim=128`, `intermediate=14336`, `vocab=128256`, fp16)
**Config:** `eval/llama3/config/serving_config_finetuning.yaml`

## 1. Executive summary

Two Python processes show up in `nvidia-smi`:

| PID | VRAM | Role |
|---|---|---|
| 18834 | 25 182 MiB | Router / inference process (owns model weights, KV+activation pool, decode graphs, LoRA adapters) |
| 18980 | 5 546 MiB (with `--bwd_graph`) / 5 154 MiB (without) | Backward / SFT subprocess (spawn, IPC-mapped pool + weights, local optimizer + backward working set) |

Of the ~30 GB total, **only ~0.5–5 GB is actually recoverable**; the rest is structural:
- Model weights (~16 GB fp16) and the 6 GB unified pool are fixed by config.
- Per-backward `lm_head.float()` (~2.1 GB retained by caching allocator) is **required for SFT loss to converge** and must not be removed.
- Two CUDA contexts and the graph pool are inherent to the MPS-partitioned co-serving design.

The two addressable items are:
1. **Activation double-buffer** (~0.5 GB permanent in main process) — a concrete refactor is proposed in §6.
2. **GQA KV oversizing** (~3–4 GB of the 6 GB pool carries zeros) — harder fix because the pool is unified with activation storage.

---

## 2. Execution pipeline recap

```
auto_benchmark.py ── subprocess ──▶  launch_llama3.py  ── os.system ──▶  dserve.server.api_server
                                         │                                       │
                                         │  translates --co / --bwd_graph /      │  loads YAML + overrides,
                                         │  --decode_graph into YAML overrides   │  spawns router + detok
                                         ▼                                       ▼
                                    serving_config_                          ModelRpcServer
                                    finetuning.yaml                         (model_rpc.py)
                                                                                 │
                                                                                 │  exposed_init_model:
                                                                                 │    loads Llama3TpPartModel (weights, pool)
                                                                                 │    loads LoRA adapters
                                                                                 ▼
                                                          Spawn backward_service  (Process, daemon=True)
                                                          under MPS window (10% active thread, 1 connection)
                                                          │
                                                          ▼
                                                    LlamaSFTBackwardService.start_service
                                                    (receives IPC-shared model + activation refs)
```

Code pointers for the pipeline:
- `auto_benchmark.py:414-435` — builds launcher argv
- `launch_llama3.py:127-134` — YAML override translation
- `api_server.py:520-598` — config load, subprocess spawn
- `model_rpc.py:51-181` — model load, adapter load, backward subprocess spawn
- `SFT_service.py:103-123` — backward startup (IPC handle open)
- `SFT_service_graph.py:85-172` — graphed-backward preparation

---

## 3. Main process — 25 GB line-by-line

### 3.1 Base model weights — **~16.0 GB**
`basemodel.py:80-94` (`_init_weights` → `load_hf_weights("fp16", ...)`).

| Component | Size |
|---|---|
| Embedding (`[128256, 4096]` fp16) | 1.05 GB |
| LM head (`[128256, 4096]` fp16, untied) | 1.05 GB |
| Per-layer attention (Q 16.8M + K 4.2M + V 4.2M + O 16.8M params, fp16) | 84 MB × 32 = 2.69 GB |
| Per-layer FFN (gate/up/down `[4096×14336]` fp16) | 336 MB × 32 = 10.75 GB |
| RMSNorm weights | < 1 MB |
| **Total** | **~16.0 GB** |

### 3.2 Unified KV + activation pool — **6.00 GB**
`unified_mem_allocator.py:58-62` (allocator init), sized by `memory.unified_mem_manager_max_size_gb=6` via `llama3/model.py:86-95`.

- Pool shape per layer: `[tot_size, head_num=32, head_dim=128]` fp16 → page size = 8 KB
- `tot_size = 6 GB / 32 layers / 8 KB = 24 576 pages/layer`
- Total: exactly 6.00 GB
- **Pages carry four roles** via `PageType` tag (`unified_mem_allocator.py:23-29`): `KV_CACHE`, `ATTENTION_INPUT_ACTIVATION`, `FFN_INPUT_ACTIVATION`, `EMBEDDING`

### 3.3 Shared activation handoff buffers — **~0.79 GB**
`unified_mem_allocator.py:181-196`, sized by `max_finetuning_tokens=1024`.

| Buffer | Shape | Size |
|---|---|---|
| `shared_transformer_out_activations[32]` | `[1024, 4096]` × 32 | 256 MB |
| `shared_attention_out_activations[32]` | `[1024, 4096]` × 32 | 256 MB |
| `embedding_output` | `[1024, 4096]` | 8 MB |
| `logit_tensor` | `[1024, 128256]` | 262 MB |
| `concat_input_ids` | `[2048]` int64 | 16 KB |

These are IPC-shared to the backward subprocess via `share_activation_dict` (`:198-205`).

### 3.4 LoRA adapters — **~100 MB**
`model_rpc.py:125-145`. With `r=16`, each layer's `w_combined_home` is `[2, 64, 32, 128]` fp16 = 1 MB per layer.

- `adapters/llama3-toy-lora` (inference): 32 MB fp16
- `adapters/llama3-toy-lora-ft` (finetune, appended at `api_server.py:528-532`): 32 MB fp16 + 32 MB fp32 (`w_combined_home_fp32`, lora_layer_weight)

### 3.5 RoPE cos/sin cache — **~18 MB**
`llama/model.py:122-127`. `t = arange(max_seq_len + 65536)` × 64 half-dims × 2 B × 2 tables.

### 3.6 Decode CUDA graphs (enabled via `--decode_graph`) — **~0.5–1.0 GB**
`dserve/common/cuda_graph_runner.py`. Per-bucket `att_m_buffers` of shape `[32 Q-heads, max_total_tokens=25000]` fp16 = ~51 MB per captured graph, plus static input/position/b_loc buffers.

### 3.7 CUDA context, cuBLAS/cuDNN workspaces, PyTorch caching allocator — **~0.8–1.2 GB**
Unavoidable driver / runtime overhead.

### 3.8 Main-process total

| Component | Size |
|---|---|
| Base weights | 16.0 GB |
| Unified pool | 6.0 GB |
| Shared activation buffers | 0.79 GB |
| LoRA (2 adapters × fp16 + 1 × fp32) | 0.10 GB |
| RoPE cache | 0.02 GB |
| Decode graphs | 0.5–1.0 GB |
| CUDA/runtime | 0.8–1.2 GB |
| **Total** | **~24.2–25.1 GB** ✓ matches 25 182 MiB |

---

## 4. Backward subprocess — 5.5 GB line-by-line

The subprocess is spawned via `multiprocessing.Process` with `torch.multiprocessing.set_start_method('spawn')` at `api_server.py:642`, under the MPS partition set briefly at `model_rpc.py:173-178`. Model weights and shared activation buffers are mapped in via CUDA IPC — they are **not** duplicated in VRAM; NVML attributes them to the exporting (main) process.

### 4.1 CUDA primary context + cuBLAS/Triton caches — **~0.5–1.0 GB**
Per-process CUDA context, cuBLAS workspace, Triton JIT cache.

### 4.2 **`lm_head_weight_.float()` reserved block — ~2.1 GB [LOAD-BEARING]**
`SFT_service.py:321` (`_post_layer_backward`):
```python
lm_W   = layer_weight.lm_head_weight_.float()      # [128256, 4096] fp32 = 2.10 GB
norm_W = layer_weight.final_norm_weight_.float()
g_y    = logit_grad @ lm_W
```

This upcast is **required for SFT loss to converge** — a prior experiment confirmed training plateaus without it. The caching allocator retains the 2.1 GB block across calls because the same-sized allocation is requested every backward. This is the single biggest line in the backward process but is structural cost, not waste.

Called from both eager (`_context_backward`, `:235`) and graph (`SFT_service_graph.py:636`) paths — `--bwd_graph` has no effect on it.

### 4.3 Logit-path fp32 conversions — **~0.3–0.5 GB reserved**
- `SFT_service.py:265`: `pred_logits = logits.float()` per request → fp32 `[T, 128256]`
- `SFT_service.py:306-311`: `torch.cat(all_logits)` + `torch.softmax(...)` allocate two fp32 `[N, 128256]` blocks

For `max_saved_finetuning_tokens=256` each block is ~131 MB; the allocator reserves the peak.

### 4.4 Per-layer attention backward transients — **~0.3 GB reserved**
`SFT_service.py:481-622` (`_backpop_attention`). Per call: `q_base/k_base/v_base`, LoRA `proj_lora` outputs, `q_/k_/v_`, `ctx`, per-request `scores/att/mask`, `grad_qh/kh/vh`, `grad_X_from_{q,k,v}`, `grad_X_from_lora_{q,k,v}`, `Zq/Zk/Zv`, rotary intermediates. Peak working set at `s=256` ≈ 60–80 MB; 32-layer reuse leaves ~300 MB reserved after fragmentation.

### 4.5 Per-layer FFN backward transients — **~0.2 GB reserved**
`SFT_service.py:355-387` (`_backprop_ffn`). Per call: `x_norm`, `gate_in`, `gate_out`, `up_out`, `grad_ffn_mid`, `grad_gate_out/up_out`, `grad_x_norm_*`. Intermediates at `[256, 14336]` fp16 (~7 MB each) × several → ~100–200 MB reserved.

### 4.6 AdamW optimizer state — **0.13 GB**
`SFT_service.py:110-114`. Lazy — allocated on first `optimizer.step()`. For 32 layers × `[2, 64, 32, 128]` fp32 params, momentum + variance = 2 × 64 MB = 128 MB.

### 4.7 Graphed-backward persistent buffers (only with `--bwd_graph`) — **~0.1 GB**
`SFT_service_graph.py:111-113, 203-222`. All deliberately allocated **outside** the graph pool (per CLAUDE.md's pool-aliasing NaN invariant):

| Buffer | Size |
|---|---|
| FFN static I/O (3 × `[256, 4096]` fp16) | 6 MB |
| Persistent LoRA fp32 grads (32 × `[2, 64, 32, 128]` fp32) | 64 MB |
| Padded-attn ctx (`flat_in_padded`, `pad_x_prev`, masks, etc.) | ~13 MB |
| **Total** | ~83 MB |

### 4.8 Graph pool (only with `--bwd_graph`) — **~0.3–0.5 GB**
`SFT_service_graph.py:114` (`graph_pool_handle`). Holds the working set of captured FFN + padded-attn graphs (32 + 32 graphs, pool sized to max live set). Mostly overlaps with what eager backward was already reserving, which is why enabling graphs only adds ~400 MB (not 2+ GB).

### 4.9 Caching-allocator fragmentation — **~0.4–0.6 GB**
Net difference between `cuda.memory_reserved()` and `cuda.memory_allocated()`.

### 4.10 Backward-subprocess total

| Component | No `--bwd_graph` | With `--bwd_graph` |
|---|---|---|
| CUDA context + runtime | 0.5–1.0 GB | 0.5–1.0 GB |
| `lm_head.float()` reserved | 2.1 GB | 2.1 GB |
| Logit-path fp32 | 0.3–0.5 GB | 0.3–0.5 GB |
| Attn bwd transients | 0.3 GB | 0.3 GB |
| FFN bwd transients | 0.2 GB | 0.2 GB |
| AdamW | 0.13 GB | 0.13 GB |
| Graph persistent buffers | 0 | 0.08 GB |
| Graph pool | 0 | 0.3–0.5 GB |
| Fragmentation | 0.4–0.6 GB | 0.4–0.6 GB |
| **Total** | **~4.0–5.0 GB** ✓ 5 154 MiB | **~4.5–5.5 GB** ✓ 5 546 MiB |

---

## 5. Wastes identified

### Waste A — Activation double-buffer (RECOVERABLE, ~0.5 GB)
**Where:** `unified_mem_allocator.py:181-196` (permanent allocation) and `:266-278` (per-backward copy).

**What happens:**
1. During finetune-token prefill, each token's attention-input and FFN-input activations are saved to scattered pool pages (`save_activations_by_layer`, `:232-243`).
2. When backward is triggered, `export_requests_info` gathers those scattered pages via `index_select`, concatenates, and **copies** into dense `shared_{transformer,attention}_out_activations` buffers (one `[1024, 4096]` tensor per layer per type).
3. Backward reads the dense buffers via CUDA IPC; reset frees the pool pages (`reset_activation_pool`, `:155-173`).

**Cost:**
- Permanent: 2 × 32 × 1024 × 4096 × 2 B = **512 MB** always resident.
- Transient: ~512 MB of pool pages co-resident with the dense copy during the backward window.

**Why this exists:** pool pages are scattered per-request; backward wants contiguous `[N, D]`. The copy gathers + densifies.

**Fix:** see §6.1.

### Waste B — GQA KV pool oversizing (HARD TO FIX, ~3–4 GB of the 6 GB pool)
**Where:** `llama3/model.py:86-95`.

**What happens:** The docstring at `:78-80` states *"KV cache allocator must use KV head count (num_key_value_heads), not num_attention_heads"*, but the code passes `head_num=self.config["num_attention_heads"]` = 32. Each pool page is sized `[1, 32, 128]` = 8 KB to match full hidden_dim, but a Llama-3 GQA KV entry is only `[1, 8, 128]` = 2 KB. Every KV page carries 6 KB of zero padding that no kernel reads.

**Cost at steady state:** if the 6 GB pool were dedicated to KV (it isn't), ~4.5 GB would be zero padding. In the current mixed-use pool (KV + activation pages), the waste is less clean but still substantial — any page used for KV wastes 3/4 of its bytes.

**Why it's hard:** the pool is unified. Activation pages legitimately need 4096 elements (full hidden_dim). A one-line `head_num` change breaks activation storage. See §6.2 for two fix paths.

### Waste C — Oversized RoPE cache (TRIVIAL, ~16 MB)
`llama/model.py:122-123`: `t = torch.arange(max_seq_len + 65536, ...)`. With `max_req_total_len=1024`, only 1024 positions are ever indexed; the extra 65 536 rows add ~16 MB across cos+sin. Drop the `+ 1024*64` or size to `max_req_total_len`.

### Waste D — Main-process fp16+fp32 LoRA copies (MINOR, ~32 MB)
Main keeps both `w_combined_home` (fp16, used for inference) and `w_combined_home_fp32` (fp32, only to be IPC-shared with backward). Unavoidable given the inference/finetune split, flagging for completeness.

### NOT a waste — `lm_head.float()` / `final_norm.float()` in backward
**Do not touch.** Confirmed with the author: keeping `lm_W` and `norm_W` in fp32 is required for SFT loss to converge. The ~2.1 GB retained block is structural.

---

## 6. Proposed fixes

### 6.1 Fix Waste A — share the pool read-only, gather on the backward side

The target invariants (no new locks):

| Concern | Why safe |
|---|---|
| Inference alloc while backward reads | `alloc()` picks from `free_bitmap==True`; activation pages under read are non-free. Disjoint. |
| Inference writes while backward reads same page | Doesn't happen. Activation pages are write-once-then-read-once. |
| `reset_activation_pool` races with backward | `model_rpc.py:343-347` already blocks on `rpc_recv.recv()` before reset. |
| Cross-stream GPU visibility | One `torch.cuda.synchronize()` in main before `rpc_send.send(...)`. One per backward, ~ms. |

**Change 1 — `unified_mem_allocator.py:198-205`** (IPC-share `gpu_pools`, drop the dense buffers):

```python
def share_activation_dict(self):
    return {
        "gpu_pools": self.gpu_pools,              # NEW
        "logit_tensor": self.logit_tensor,
        "concat_input_ids": self.concat_input_ids,
        "input_layer_output": self.embedding_output,
        # shared_transformer_out_activations / shared_attention_out_activations REMOVED
    }
```

**Change 2 — `unified_mem_allocator.py:266-278`** (send metadata, not data):

```python
def export_requests_info(self):
    self.get_concatenated_finetune_input_ids()
    pages_cpu = [(ffn.cpu(), attn.cpu()) for (ffn, attn) in self.activation_page_indices]
    return {
        "request_token_info": self.request_token_info,
        "activation_page_indices": pages_cpu,     # NEW
    }
    # fill_activations_by_layer calls DELETED
```

**Change 3 — `SFT_service.py:194-215`** (gather on the backward side):

```python
def receive_requests_info(self, req):
    if self.activations is not None:
        del self.activations
    self.activations = Activations()
    request_token_info = req["request_token_info"]
    page_indices = req["activation_page_indices"]
    total = sum(request_token_info)

    # Logits / input_ids — unchanged, still via dense shared buffers
    logit_tensor = self.shared_activations.logit_tensor[:total].detach().clone()
    for n in request_token_info:
        self.activations.logit_list.append(logit_tensor[:n, :])
        logit_tensor = logit_tensor[n:, :]
    self.activations.concat_input_ids = (
        self.shared_activations.concat_input_ids[:total + len(request_token_info)].clone()
    )
    self.activations.input_layer_output = self.shared_activations.input_layer_output[:total]

    # NEW: gather per-layer activations directly from IPC-shared pool
    pools = self.shared_pool                    # list of 32 IPC-imported tensors
    H, Hd = pools[0].shape[1], pools[0].shape[2]
    D = H * Hd
    self.activations.attention_out_activations = []
    self.activations.transformer_out_activations = []
    for layer_id in range(self.num_layers):
        ffn_rows, attn_rows = [], []
        for (ffn_cpu, attn_cpu) in page_indices:
            ffn_rows.append(pools[layer_id].index_select(0, ffn_cpu.cuda()))
            attn_rows.append(pools[layer_id].index_select(0, attn_cpu.cuda()))
        self.activations.attention_out_activations.append(
            torch.cat(ffn_rows, dim=0).reshape(total, D)
        )
        self.activations.transformer_out_activations.append(
            torch.cat(attn_rows, dim=0).reshape(total, D)
        )
```

**Change 4 — `model_rpc.py:340-349`** (stream visibility):

```python
def backward(self):
    requests_info_dict = self.model.mem_manager.export_requests_info()
    requests_info_dict["current_epoch"] = self.current_epoch
    torch.cuda.synchronize()                     # NEW
    self.rpc_send.send(requests_info_dict)
    finished, loss, total_token_processed = self.rpc_recv.recv()
    if finished:
        self.model.mem_manager.reset_activation_pool()
        return True, loss, total_token_processed
```

**Change 5 — `SFT_service.py:186-192` (`receive_activation_addresses`)** — store the pool reference:

```python
def receive_activation_addresses(self, activations_dict):
    self.shared_pool = activations_dict["gpu_pools"]       # NEW
    self.shared_activations = SharedActivations()
    self.shared_activations.logit_tensor = activations_dict["logit_tensor"]
    self.shared_activations.concat_input_ids = activations_dict["concat_input_ids"]
    self.shared_activations.input_layer_output = activations_dict["input_layer_output"]
```

**Delete:** `init_shared_activation_memory`'s two `shared_{trans,attn}_out_activations` allocations and all references to them.

**Expected impact:**
- Main process: **−512 MB permanent**, −512 MB transient per backward.
- Backward process: +~512 MB transient per backward for the gathered tensors (freed via `del self.activations` on next call).
- Same gather work, moved one process over. No new locks. One `cuda.synchronize()` per backward.

### 6.2 Fix Waste B — split the unified pool (two options)

**Option 1 (lightweight, GQA-only):** keep one pool but change page shape to `[head_dim]` instead of `[head_num, head_dim]`. KV pages allocate 8 × 128 / 128 = 8 pages per token per K or V; activation pages allocate 32 × 128 / 128 = 32 pages per token. Page count quadruples, bitmap operations get proportionally more expensive, but total memory is right-sized. Requires updating every site that assumes a page is `[H, Hd]`.

**Option 2 (cleaner, dual pool):** split `UnifiedMemoryAllocator` into `KVPool` (sized for `num_key_value_heads × head_dim`) and `ActivationPool` (sized for `hidden_dim`). Two bitmaps, two `gpu_pools` lists, two sets of `PageType` enums. The cleaner design, larger patch.

Estimated recoverable at 6 GB pool, 32 layers, Llama-3 GQA: ~4.5 GB of zero-padding in KV pages, minus the activation pages that still need the full 4096-element width. Net win ≈ 3 GB. This is the largest single opportunity in the system but also the biggest change.

### 6.3 Fix Waste C — trim RoPE cache
`llama/model.py:123`: change `t = torch.arange(max_seq_len + 1024 * 64, ...)` to `t = torch.arange(max_seq_len, ...)` (or `max(max_seq_len, max_req_total_len)`). Saves ~16 MB. One-line fix.

### 6.4 Fix Waste D — defer LoRA fp32 materialization
Keep `w_combined_home_fp32` on CPU until the moment it's IPC-shared, then `.cuda()` it and let the backward subprocess be the GPU owner. Saves ~32 MB on main. Low priority.

---

## 7. What NOT to change

1. **`lm_head.float()` / `final_norm.float()` at `SFT_service.py:321`.** Required for SFT loss to decrease. Saved as a `feedback` memory; do not re-propose.
2. **Two CUDA contexts.** Inherent to the MPS-based concurrency model documented in CLAUDE.md. Not a waste, it's the mechanism.
3. **`_maybe_pause()` at every layer boundary (`SFT_service.py:160-166`).** Load-bearing for the co-serving contract. Memory-neutral anyway.
4. **fp32 matmul for GQA attention `scores` (`SFT_service.py` analogue in `llama3/SFT_service.py`).** Required for bit-for-bit match between forward-recompute and backward softmax recompute, per CLAUDE.md precision rule.

---

## 8. Priority summary

| Fix | Reclaimable | Risk | Effort |
|---|---|---|---|
| 6.3 Trim RoPE cache | ~16 MB | trivial | 1 line |
| 6.1 Read-only pool sharing | ~512 MB main, simpler handoff | low (no new locks, invariants hold) | ~100 LOC |
| 6.4 Defer LoRA fp32 to CPU | ~32 MB main | low | ~20 LOC |
| 6.2 Split KV vs activation pool | ~3 GB main | medium (pool refactor, allocator state touches every page-indexed site) | ~500 LOC |

Start with 6.3 (free) and 6.1 (highest ratio of impact to risk). 6.2 is a larger patch; prioritize if workloads push `max_total_token_num` higher, since that's where GQA oversizing bites hardest.
