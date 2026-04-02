---
name: Qwen3 routing bottleneck experiment (2026-04-01)
description: Does GPU 0's higher token load actually make it the compute bottleneck per layer? Results from routing_bottleneck.py sweep.
type: project
---

## What Was Done

Ran `test/qwen3/routing_bottleneck.py` — sweeps batch_size × seq_len, collects per-layer
routing imbalance (token-expert pairs per GPU) and expert compute time per GPU, to answer:

> Does GPU 0 getting more tokens actually translate into it being the bottleneck?

**Short answer**: Yes, but only above ~4096 total tokens. And there are two specific layers
(23 and 35) that are structurally pathological regardless of batch size.

---

## Script Details

- **Script**: `test/qwen3/routing_bottleneck.py`
- **Port**: 29528
- **MAX_POOL**: 100,000 (reduced from 200K to avoid OOM — see OOM history below)
- **MAX_TOTAL_TOKENS**: 100,000 (skips configs above this)
- **Batch sizes**: [2, 4, 8, 32, 64]
- **Seq lens**: [256, 512, 1024, 2048, 3072, 4096, 8192]
- **Trials**: 3 per config
- **Output**: `test/qwen3/results/routing_bottleneck/bottleneck.csv` (4176 rows)

**Skipped configs** (total_tokens > 100K):
```
bs=32 sl=4096 (131072), bs=32 sl=8192 (262144)
bs=64 sl=2048 (131072), bs=64 sl=3072/4096/8192
```
29 configs ran successfully.

### How timing works
`timing_enabled = True` is set on `ep_module` before the sweep. Inside `_ffn` in
`transformer_layer_infer_ep.py`, three sections are timed with `torch.cuda.synchronize()`:
- forward all_to_all (send token embeddings + expert indices to owning ranks)
- expert compute (local FFN loops)
- backward all_to_all (return results)

Both ranks record timings independently. The main process merges rank 0 and rank 1 rows
and stores `gpu0_expert_ms`, `gpu1_expert_ms`, `expert_time_ratio = gpu0/gpu1`.

---

## Main Finding: Config-Level Bottleneck

Config-level summary (means over all layers and 3 trials):

```
 bs    sl    total   load_ratio  exp_time_ratio  gpu0_ms  gpu1_ms  bottleneck
  2   256      512       1.403           1.017     11.7     11.6   balanced
  2   512     1024       1.465           1.001     11.7     11.9   balanced
  2  1024     2048       1.491           1.052     12.9     12.4   GPU0
  2  2048     4096       1.542           1.034     13.9     13.6   balanced
  2  3072     6144       1.557           1.050     15.1     14.6   GPU0
  2  4096     8192       1.566           1.070     16.3     15.5   GPU0
  2  8192    16384       1.577           1.098     21.0     20.0   GPU0
  4   256     1024       1.450           1.007     11.7     11.7   balanced
  4   512     2048       1.476           1.001     12.5     12.6   balanced
  4  1024     4096       1.494           1.023     13.9     13.7   balanced
  4  2048     8192       1.535           1.058     16.6     16.0   GPU0
  4  3072    12288       1.555           1.085     20.6     19.4   GPU0
  4  4096    16384       1.573           1.100     21.0     19.7   GPU0
  4  8192    32768       1.582           1.233     31.3     27.6   GPU0
  8   256     2048       1.434           1.096     14.0     12.8   GPU0
  8   512     4096       1.468           1.038     14.4     14.0   balanced
  8  1024     8192       1.504           1.052     16.7     16.1   GPU0
  8  2048    16384       1.527           1.097     21.1     19.8   GPU0
  8  3072    24576       1.537           1.123     25.5     23.6   GPU0
  8  4096    32768       1.569           1.173     29.9     27.1   GPU0
  8  8192    65536       1.629           1.360     50.8     42.6   GPU0
 32   256     8192       1.439           1.054     17.2     16.5   GPU0
 32   512    16384       1.472           1.089     21.6     20.3   GPU0
 32  1024    32768       1.502           1.151     30.4     27.7   GPU0
 32  2048    65536       1.530           1.256     48.4     42.5   GPU0
 32  3072    98304       1.555           1.345     68.8     64.8   GPU0
 64   256    16384       1.439           1.071     21.9     20.9   GPU0
 64   512    32768       1.470           1.144     30.5     28.0   GPU0
 64  1024    65536       1.502           1.265     49.4     42.8   GPU0

corr(load_ratio, expert_time_ratio) = +0.66
```

**Observations:**
- Load ratio (GPU 0 token-expert pairs / GPU 1) is consistently **1.40–1.58** across all configs.
  This is NOT routing noise — it is structural (same experts favored regardless of batch).
- At small total tokens (<4096): expert_time_ratio ≈ 1.0 even though load_ratio ≈ 1.5.
  GPU 0 gets 50% more work but takes the same time. Reason: kernel launch overhead and
  weight loading dominate at low token counts — actual compute is negligible.
- At large total tokens (>4096): expert_time_ratio diverges, reaching **1.36× at bs=8/sl=8192**
  and **1.35× at bs=32/sl=3072**. GPU 0 is a genuine compute bottleneck.
- Correlation = 0.66: moderate. The relationship is non-linear — load imbalance only
  shows up in time when compute cost exceeds kernel-launch floor.

---

## Unexpected Finding: Per-Layer Structural Hotspots

The config-level summary averages over all 48 layers. The per-layer picture is dramatically
different. Two layers are permanently pathological:

### Layer 23 — catastrophically skewed to GPU 0

```
Mean load_ratio:  9.18×  (std=2.06)
Min load_ratio:   3.98×
Max load_ratio:  13.91×
Mean expert_time_ratio: 2.37×
Max expert_time_ratio:  5.58×
```

By batch size:
```
bs=2:  mean=9.33×   bs=4: mean=9.56×
bs=8:  mean=9.67×   bs=32: mean=8.71×
bs=64: mean=7.62×
```

GPU 0 receives **9× more token-expert pairs** than GPU 1 at layer 23 across virtually
every batch configuration. This is consistent, not a spike — minimum is 4×.

### Layer 35 — also heavily skewed

```
Mean load_ratio:  4.76×  (std=0.56)
Min load_ratio:   3.68×
Max load_ratio:   5.93×
```

### Other notably skewed layers (mean load_ratio > 2×, GPU 0 heavier):
- Layer 7: 3.23×
- Layer 11: 3.21×
- Layer 19: 2.84×
- Layer 31: 2.34×
- Layer 46: 1.94×
- Layer 8: 1.92×

### Layers where GPU 1 is heavier (mean load_ratio < 0.7):
- Layer 30: 0.49×   (GPU 1 gets 2× more work)
- Layer 36: 0.53×
- Layer 18: 0.59×
- Layer 24: 0.63×
- Layer 47: 0.68×

**Why the overall ratio looks "only" 1.4–1.6×:** The GPU0-heavy layers (7, 11, 19, 23, 31, 35, 46)
and GPU1-heavy layers (18, 24, 30, 36, 47) partially cancel when averaged. But they do NOT
cancel in time — both GPUs must wait at each layer's all_to_all barrier, so the heavier GPU
at each layer determines the wall-clock time for that layer.

### Why layer 23 is so extreme

This is **not stochastic** — it's a static property of the model's learned expert usage.
At layer 23, the top-8 routing overwhelmingly prefers experts 0–63 (which are assigned to
GPU 0 in 2-GPU EP) over experts 64–127 (GPU 1). The assignment `dest_rank = flat_e // (E // R)`
means experts 0–63 → rank 0, 64–127 → rank 1. If Qwen3's router at layer 23 has learned
to predominantly activate low-index experts, GPU 0 gets the lion's share regardless of input.

Confirmed by: the load_ratio at layer 23 is essentially independent of content (held constant
across 29 different batch configurations and 3 trials each). Random tokens in exp6 showed
random imbalance; real content consistently hits the same hot experts.

**Implication for DeltaServe**: Layer 23 is always the critical-path layer for EP inference.
Any SLO predictor needs to account for this fixed structural bottleneck. A predictor trained
on batches without layer-23 awareness will under-predict GPU 0's compute time.

---

## Per-Layer Distribution (all configs + trials)

```
Per-layer load_ratio distribution (4176 rows × 48 layers):
  p50:  1.16×
  p75:  1.63×
  p90:  2.66×
  p95:  3.29×
  p99:  9.09×
  >2×:  13.2% of all layer observations
  >3×:   8.8%
  >5×:   2.8%  (119 rows — ALL at layers 23 and 35, ALL on GPU 0)
```

All extreme imbalance (>5×) falls exclusively on layers 23 and 35, and exclusively
goes to GPU 0. This is a deterministic property, not noise.

---

## OOM History (important for future runs)

This script failed multiple times before landing on MAX_POOL=100K:

1. **MAX_POOL=50K**: insufficient for BS=8/sl=8192 (`need_size 65536 left_size 50000`)
2. **MAX_POOL=200K + MemoryAllocator.reset_all_pool (unpatched)**:
   OOM during model init — `reset_all_pool` allocates SFT activation buffers
   (`tot_size × 10` per layer × 3 buffers × 48 layers × fp16 × head_dim) → ~80GB+ at 200K pool
3. **MAX_POOL=200K + monkey-patched reset_all_pool** (skip SFT buffers):
   OOM at BS=32/sl=4096 (131072 tokens). Peak memory ≈:
   - Model weights: ~17.7GB
   - KV buffers at P=200K: 48 × [200K, 2, 128] × fp16 × 2 ≈ 9.8GB
   - EP routing tensors at T=1M: 5 tensors × [1M, 2048] × fp16 ≈ 21GB peak
   - Total: ~48GB+ OOM on 80GB A100
4. **MAX_POOL=100K** (current): fits. 29 configs run, 6 skipped (total_tokens > 100K).

The monkey-patch (`_reset_no_sft`) skips allocating `finetune_activation_buffer`,
`input_layer_output`, and `ffn_input_buffer` — these are only needed for SFT co-serving,
not for inference benchmarking. It is applied BEFORE `Qwen3MoeEPTpPartModel(...)` is called.

Memory budget breakdown at MAX_POOL=100K:
- KV buffers: 48 × [100K, 2, 128] × fp16 × 2 ≈ 4.9GB
- EP routing at T=100K×8=800K items: 5 tensors × [800K, 2048] × fp16 ≈ 16.4GB peak
- Model weights: ~17.7GB
- Total peak: ~39GB — well within 80GB

---

## Connection to Previous Experiments

- **exp6 (routing_collision)**: showed corrected mean imbalance ≈ 1.0 across all domains and
  buckets when averaging over all 48 layers. That's still true — the per-batch average is
  ~1.0–1.1. But this averaged away the pathological per-layer structure (layer 23 = 9×,
  layers 18/24/30 = 0.5×). The exp6 conclusion "routing is balanced" is correct at the
  batch level, but misleading at the layer level.

- **routing_counts.py** (earlier, simpler experiment): showed GPU 0 consistently gets
  ~18–19% more total token-expert pairs across seq_lens [64, 128, 256, 512, 1024].
  That 18–19% is the net sum of GPU0-heavy and GPU1-heavy layers averaging out.

- **seq_len_routing.py**: sweep of (n_seqs, seq_len) with per-GPU timings — same story,
  GPU 0 slightly slower at larger configs but similar at small ones.

---

## What Was NOT Answered

1. **Is the layer 23 imbalance due to expert index assignment?** Would assigning experts
   differently (e.g., interleaved: rank 0 gets even indices, rank 1 gets odd) fix it?
   The current assignment is contiguous (0–63 → rank 0, 64–127 → rank 1).
   Interleaved assignment could potentially balance layer 23 if the hot experts span the
   full index range. This would require modifying `dest_rank = flat_e // epk` in `_ffn`.

2. **Is the load_ratio at layer 23 correlated with specific expert activations?**
   Could instrument which specific experts (0–127) are most activated at layer 23 to
   understand if it's 1–2 "always active" experts that drive the skew.

3. **Does the layer 23 bottleneck get worse with LoRA?** LoRA adapters could change
   the routing distribution if the adapter adds to the gate logits.

---

## Files

- **Script**: `test/qwen3/routing_bottleneck.py`
- **Results**: `test/qwen3/results/routing_bottleneck/bottleneck.csv`
- **EP layer infer** (with timing + routing logs): `slora/models/qwen3_moe/layer_infer/transformer_layer_infer_ep.py`
  - `routing_imbalance_log`: list, appended per layer, stores `recv_counts` = [from_gpu0, from_gpu1]
  - `expert_time_log`: list, appended per layer, stores expert_ms (when `timing_enabled=True`)
  - `comm_time_log`: list, stores (fwd_comm_ms, bwd_comm_ms) per layer
  - `timing_enabled`: bool, set by experiment scripts

---

## Pending Work (as of 2026-04-01)

Scripts written but NOT yet run:
- `test/qwen3/exp1_5/sweep_real.py` (port 29515) — seq_len × batch_size predictor accuracy sweep
- `test/qwen3/exp2/stress_test.py` (port 29516) — stress test (hetero, content, convergence)

Scripts not yet written:
- `test/qwen3/exp_refit/timeline_refit.py` (port 29522) — self-correction speed
- `test/qwen3/predictor_gap/fixed_composition_variance.py` (port 29523) — oracle floor
- Update `test/predictor_gap/plot_gap.py` to include Qwen3 as third model alongside Llama3 + Mixtral

SLO threshold reminder: use 700–1200ms for Qwen3 (batches take ~600–700ms). The 200–400ms
thresholds from Mixtral experiments make all gate simulations uninformative for Qwen3.
