---
name: Qwen3-30B-A3B implementation and experiments (2026-03-31)
description: Everything done in this session — model implementation, weight verification, all experiment scripts written, exp3–6 results and analysis
type: project
---

## What Was Done This Session

### 1. Qwen3-30B-A3B Model Implementation (from previous session, verified today)

Full model support added to `slora/models/qwen3_moe/`:
- `model.py` — `Qwen3MoeTpPartModel` and `Qwen3MoeEPTpPartModel`
- `layer_weights/transformer_layer_weight.py` — TP mode weights
- `layer_weights/transformer_layer_weight_ep.py` — EP mode weights
- `layer_infer/transformer_layer_infer.py` — handles head_dim=128 override, q_norm/k_norm
- `layer_infer/transformer_layer_infer_ep.py` — EP all_to_all routing + routing_imbalance_log

Dispatched from `slora/server/router/model_infer/model_rpc.py` when `model_type == "qwen3_moe"`.

**Key Qwen3-30B-A3B quirks that needed special handling:**
- `head_dim=128` ≠ `hidden_size // num_heads = 64` — overrides needed in KV cache alloc and layer infer
- Per-head q_norm and k_norm (RMSNorm, shape: head_dim=128) applied before RoPE
- 128 experts, top-8 routing (`num_experts_per_tok=8`)
- Expert weights are per-expert keys (`mlp.experts.{j}.gate_proj/up_proj/down_proj`) NOT stacked tensors
- Router key: `mlp.gate.weight` (not `block_sparse_moe`)

**Verification:** `test/qwen3/verify_qwen3.py` (NCCL_PORT=29524) — both prompts produced coherent output.

---

### 2. Experiment Scripts Written

All scripts in `test/qwen3/`:

| Script | Port | Status |
|--------|------|--------|
| `ep_predictor_experiment.py` | 29514 | Done (previous session) |
| `exp1_5/sweep_real.py` | 29515 | Written this session |
| `exp2/stress_test.py` | 29516 | Written this session |
| `exp3/variance_profiler.py` | 29518 | Written + run |
| `exp4/domain_experiment.py` | 29519 | Written + run |
| `exp5/mmlu_domain_experiment.py` | 29520 | Written + run |
| `exp6/routing_collision.py` | 29521 | Written + run |
| `exp_refit/timeline_refit.py` | 29522 | **NOT YET WRITTEN** |
| `predictor_gap/fixed_composition_variance.py` | 29523 | **NOT YET WRITTEN** |

Also still needed: update `test/predictor_gap/plot_gap.py` to include Qwen3 as third model.

---

### 3. Experiment Results

#### Exp 3 — Variance Profiler (log-normal trace, 256 calibration + 256 profiling batches)
```
mean=658ms  cv=5.37%  fit_rmse=15.6ms
frac>10% error=0.0%  mean_signed_err=+2.46%
lag1_autocorr=0.042
```
Gate decisions: all 0% because SLO thresholds (200–400ms) are ALL below Qwen3's ~658ms inference time.
**For Qwen3, meaningful SLO thresholds are 700–1200ms.** The exp3 gate simulation result is uninformative.

Fit_rmse comparison:
| Model | fit_rmse |
|-------|---------|
| Llama3-8B | 2.9ms |
| Mixtral-8x7B EP | 13.5ms |
| **Qwen3-30B-A3B EP** | **15.6ms** |

Qwen3 ≈ Mixtral (not dramatically worse despite K=8 vs K=2).

#### Exp 4 — Domain Routing Variance (Dolly-15k)
```
All domains: cv% = 3–9%
No outlier domain (cf. Mixtral closed_qa at 34.9%)
```
| domain | cv% | mean_ms |
|--------|-----|---------|
| general_qa | 8.67 | 638 |
| creative_writing | 6.29 | 589 |
| brainstorming | 6.02 | 598 |
| open_qa | 3.92 | 575 |
| classification | 3.84 | 611 |
| information_extraction | 3.61 | 602 |
| summarization | 3.39 | 586 |
| closed_qa | 3.25 | 601 |

**No content-dependent routing outliers** — unlike Mixtral's closed_qa (2 batches at 217ms and 274ms vs expected 104ms).

Gate decisions misleading (same SLO threshold problem as exp3, all FP because 600ms >> 200-400ms SLO).

#### Exp 5 — MMLU Knowledge-Domain Experiment
```
All domains: cv% = 3–6%
law_ethics NOT elevated vs other domains
```
| domain | cv% | mean_ms | T_in_mean |
|--------|-----|---------|-----------|
| stem_cs | 6.33 | 697 | 360 |
| medical | 5.50 | 660 | 330 |
| law_ethics | 5.48 | 692 | 898 |
| stem_math | 4.82 | 682 | 389 |
| social | 3.55 | 653 | 339 |
| humanities | 3.62 | 632 | 202 |
| stem_sci | 3.39 | 645 | 292 |

law_ethics has high T_in_mean (898) because of long questions, but cv% is not elevated relative to other domains. **No domain-specific routing anomaly** (unlike Mixtral exp5/exp6 where law_ethics/humanities showed high CV at long bucket).

Gate decisions: 100% FP because pred_coserve_ms << SLO (again, SLO thresholds too low for Qwen3).

#### Exp 6 — Routing Collision Dissection

**Formula bug in compute_imbalance:** In the Qwen3/Mixtral EP implementation, BOTH ranks route ALL tokens and both send to rank 0's experts. So rank 0 receives T_in×K items (not T_in×K/2). The formula divided by `TK/2` instead of `TK`, making balanced routing look like "2.0" instead of "1.0".

**Corrected results (raw / 2):**
- All real content: corrected mean_imbalance ≈ 0.99–1.00 across all domains and buckets
- Random ablation: corrected ≈ 1.04–1.09

**Per-layer spread (48 layers per batch):**
- Mean corrected imbalance across layers: 0.998 ± 0.018 (essentially perfect)
- Worst single layer per batch (corrected): 1.272 ± 0.059 on average; up to **1.476× worst case**

This means in the worst layer, one rank gets 47% more token-expert pairs than balanced. The other rank idles. BUT across all 48 layers the imbalance averages out to ~1.0, so total batch time variance is small (5% CV).

**Key open question (Ramya's insight):** Is the per-layer imbalance *correlated* across layers within a batch? If some batches consistently have rank 0 heavy across many layers, those batches would run systematically slower and the predictor can't predict which ones. The current data (only mean and max per batch, not full 48-layer timeseries) doesn't answer this. Would need to store the full per-layer recv_counts to test.

**Conclusions from exp6:**
- H1 (content-driven imbalance) FALSE: law_ethics ≈ STEM domains at matched bucket
- H2 (length confusion) DOES NOT APPLY as main explanation
- Random tokens have slightly MORE imbalance than real content (same as Mixtral)
- Routing variance source: stochastic all_to_all communication latency (not load imbalance)
- Qwen3 K=8 vs Mixtral K=2 does NOT amplify imbalance — still perfectly balanced

---

### 4. Important: What Remains to Run

Exp1_5 (sweep_real) and exp2 (stress_test) were written but NOT run. They use ports 29515 and 29516.

```bash
cd /mnt/nfs/home/ramya/slora-plus/S-LoRA
source /mnt/nfs/home/ramya/slora-plus/.venv/bin/activate
python test/qwen3/exp1_5/sweep_real.py
python test/qwen3/exp2/stress_test.py
python test/qwen3/exp_refit/timeline_refit.py    # not yet written
python test/qwen3/predictor_gap/fixed_composition_variance.py  # not yet written
```

---

### 5. Known Issues / Gotchas

**SLO threshold mismatch:** All exp3–6 gate simulations used SLO thresholds of [200, 250, 300, 350, 400ms] copied from Mixtral. Qwen3 batches take 600–700ms, so ALL SLO thresholds are below actual inference time → gate decisions are all uninformative. For re-runs or future experiments, use SLO thresholds of [700, 800, 900, 1000, 1200ms] for Qwen3.

**Imbalance formula bug:** `compute_imbalance` in exp6 uses `balanced = TK/2` but should use `balanced = TK` for this EP implementation. The raw reported values are 2× what they should be. The corrected imbalance = raw_value / 2.

**Per-layer imbalance not stored:** exp6 only stores mean and max imbalance per batch (across 48 layers). To test whether within-batch layer imbalance is correlated (Ramya's question), would need to modify the script to store the full 48-element recv_counts timeseries per batch.
