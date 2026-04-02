---
name: DeltaServe predictor research — current status
description: What has been done, what the findings mean, and what to work on next for the MoE predictor research thread
type: project
---

## The Research Question

**Can `PrefillExecutionEstimator` and `DecodeExecutionEstimator` reliably gate SFT
admission in DeltaServe's `check_will_starve()` for Mixtral (MoE model)?**

The estimators fit `t = α·Σn² + β·T_in + γ·T_ft + c` (prefill) and `t = δ·B + ε·K + d`
(decode) via linear regression, re-fit every 256 batches. They are used in
`mixed_req_queue.py:check_will_starve()` to decide whether to admit FT work.

---

## What Has Been Run (as of 2026-03-31)

### ep_predictor_experiment.py (dummy weights, 2026-03-12)
Script: `test/mixtral/ep_predictor_experiment.py`
Results: `test/mixtral/results/`
Finding: Irreducible RMSE floor of 0.638ms (equals std of actual times). RMSE flat from
n=5 to n=500. Predictor structurally fails for EP Mixtral with dummy tiny model.

### exp1_5/sweep.py (dummy weights, 2026-03-12)
Script: `test/mixtral/exp1_5/sweep.py`
Results: `test/mixtral/exp1_5/results/`
Finding: Same story. Short contexts (sl≤512) have CV up to 13%. Long contexts fine.

### exp1_5/sweep_real.py (REAL Mixtral-8x7B-v0.1, 2026-03-16)
Script: `test/mixtral/exp1_5/sweep_real.py`
Results: `test/mixtral/exp1_5/results_real/`
Settings: MAX_POOL=15000, 21/48 configs, 2-GPU EP
High-CV configs: bs=4/sl=256 (CV=9.7%), bs=1/sl=1024 (CV=7.0%), bs=16/sl=512 decode (CV=28.7%)
Long contexts: CV <2%, predictor accurate

### exp1_5/sweep_real.py (REAL Meta-Llama-3-8B-HF, 2026-03-16)
Script: `test/llama3/exp1_5/sweep_real.py`
Results: `test/llama3/exp1_5/results_real/`
Settings: MAX_POOL=35000, 33/48 configs, single GPU
Prefill errors: 0.2–2.3% (very clean — dense model, no routing variance)
Decode spikes at bs=1/bs=16 are fit_rmse inflation artifacts, not real variance.
Use as clean baseline vs Mixtral.

### exp2/stress_test.py (REAL Mixtral-8x7B-v0.1, 2026-03-16)
Script: `test/mixtral/exp2/stress_test.py`
Results: `test/mixtral/exp2/results/`
See `project_experiment_results.md` for full numbers.

---

## Key Findings Summary

### What actually breaks the predictor in production

**Heterogeneous batch composition (Exp A) is the critical failure.**
When `check_will_starve()` is called, `pending_inf_token_list` contains real request lengths
from `waiting_req_list` — which in practice are mixed (different users, different prompt lengths).
If the predictor was calibrated on uniform-length batches (e.g., from estimate_finetuning_overhead()
warm-up), it will massively over-predict for any batch with skewed lengths:
- skewed [32,32,32,928] → predicted 3× too slow → +244% error → SFT permanently starved
- bimodal [64,64,448,448] → +63% error
The Σn² feature is dominated by the longest sequence; actual EP cost is not.

### What does NOT break it
- K growth in decode: SWA window=4096 caps attention cost; flat 92–95ms regardless of K
- Token content (random vs constant vs rare tokens): negligible CV difference (<1%)
- RMSE convergence: CV is only 0.8% with 580 samples (earlier 9.7% was small-sample noise)
- SLO gate: FP=0% always (conservative). FN only significant at unrealistically tight slack (<150ms)

### Why Σn² is the wrong feature for heterogeneous batches
The predictor assumes `cost ∝ Σn²` because attention is O(n²) per sequence.
For EP Mixtral: expert dispatch cost ∝ number of tokens (T_in), not Σn².
For a skewed batch: Σn²=861K (928² dominates), T_in=1024 (same as uniform).
Actual time ≈ uniform time because EP dispatch sees the same total token count.
The predictor thinks "huge Σn²" → "slow batch", but Σn² is inflated by ONE long sequence.

---

### Qwen3-30B-A3B (new model added 2026-03-31)
Scripts: `test/qwen3/exp3/`, `exp4/`, `exp5/`, `exp6/` — all run.
exp1_5 and exp2 written but NOT yet run.
exp_refit and predictor_gap NOT yet written.
Key findings: fit_rmse=15.6ms (≈Mixtral), no domain-specific routing outliers, routing balanced.
exp6 re-run (2026-04-01) with full per-layer timeseries — see "Ramya's hypothesis" result below.
Full results in `project_experiment_results.md` and `session_qwen3_experiments_2026-03-31.md`.

---

## What to Work on Next

**Priority 0 — ANSWERED (2026-04-01): Ramya's per-layer correlation question**
Modified exp6 to store full 48-layer recv_counts timeseries per batch (not just mean+max).
Results are definitive — the hypothesis is FALSE for Qwen3:
- lag1_autocorr across layers: mean = **-0.13** (NEGATIVE, mean-reverting)
  → If layer L is rank-0-heavy, layer L+1 tends to be rank-1-heavy. Not systematic.
- frac_r0_heavy: mean=0.47, always in [0.33, 0.63] — no batch is systematically biased
- corr(frac_r0_heavy, actual_ms) = -0.07 — systematic bias does not predict timing
- All corrected mean_imb: 0.982–1.037 (perfectly balanced across all domains/buckets)
**Conclusion: The predictor is NOT failing under the hood. The 5% CV is real hardware
variance (stochastic all_to_all latency), not systematic load imbalance that it misses.
The mean-reverting pattern suggests the routing load alternates ranks across layers,
which is exactly what you'd expect from independent K=8 routing per layer.**
New files: `test/qwen3/exp6/results/layer_timeseries.csv`, updated `per_batch.csv` with
`lag1_autocorr_layers` and `frac_layers_r0_heavy` columns.

**Priority 1 — Fix the feature engineering for heterogeneous batches**
Options (from easiest to most principled):
a) Replace `Σn²` with `(max_n)² + (sum of remaining n²)` — captures that one long seq
   adds attention cost but doesn't inflate EP dispatch cost
b) Add `max_n / mean_n` as a "skewness" feature — lets the model learn to discount Σn²
   when lengths are skewed
c) Two-term model: `α·T_in + β·max_n²` — separates dispatch cost from attention cost
d) Routing-aware feature: estimate per-rank token count from gate scores pre-forward (expensive)
The simplest fix is (c): `T_in` captures dispatch cost, `max_n²` captures attention cost of
the longest sequence (which dominates attention). This requires changing `PrefillExecutionEstimator`
in `slora/server/router/tracker.py`.

**Priority 2 — Validate fix with a re-run of Exp A**
After changing the feature, re-run `stress_test.py` Exp A to verify heterogeneous batch
errors drop to <10%.

**Priority 3 — Re-run Exp D with proper mixed-config training**
The current Exp D design is flawed (all training samples have identical features → predictor
just estimates the mean). Proper design: train on a mixture of (bs, sl) pairs, hold out
one config, measure RMSE on that config. This correctly tests generalization.

---

## Important Implementation Details

### Running sweep/stress scripts
```bash
cd /mnt/nfs/home/ramya/slora-plus/S-LoRA
# Mixtral (2 GPU EP):
python test/mixtral/exp2/stress_test.py
python test/mixtral/exp1_5/sweep_real.py
# Llama3 (1 GPU):
python test/llama3/exp1_5/sweep_real.py
```
All scripts use `mp.spawn` with `dist.init_process_group`. TCP ports used:
- sweep_real.py (Mixtral): 29500
- sweep_real.py (Llama3): 29501
- stress_test.py (Mixtral): 29502
Make sure no other dist jobs are running on those ports.

### Memory budget constraints
- Mixtral-8x7B-v0.1: ~47GB model weights per GPU. MAX_POOL=15000 → ~13GB KV+SFT buffers.
  Total ~60GB < 80GB. Do NOT increase MAX_POOL above 20000.
- Llama3-8B: ~16GB weights + SFT buffers dominate. MAX_POOL=35000 → ~50GB total.
  Do NOT use reset_all_pool() in sweep loops — use free_all() instead.

### check_will_starve() location
`slora/server/router/mixed_req_queue.py` — calls:
- `self.prefill_estimator.predict_inference(pending_inf_token_list)`
- `self.decode_estimator.predict(total_tokens=K_kv, batch_size=bs)`
These are the exact call sites that will benefit from better feature engineering.

### tracker.py predictor details
`slora/server/router/tracker.py`:
- `PrefillExecutionEstimator.fit()` takes `inference_only_tokens` (list of list of seq lengths)
- `PrefillExecutionEstimator.predict_inference(lens)` returns inflated prediction
- Inflation factor: `prediction * (1 + 2 * self.fit_rmse / prediction)` approximately
- Re-fit every 256 batches in `BatchExecutionTracker.maybe_refit()`
