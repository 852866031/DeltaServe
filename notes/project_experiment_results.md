---
name: All experiment numerical results
description: Complete numerical results from every experiment run, for reference in future sessions
type: project
---

## Mixtral exp1_5 sweep_real.py results (2026-03-16)
Model: Mixtral-8x7B-v0.1, EP mode, 2 GPUs, MAX_POOL=15000, 21 configs

### Prefill (high-error configs)
| bs | sl | CV% | mean_ms | mean_err% | max_err% |
|----|-----|-----|---------|-----------|---------|
| 4  | 256 | 9.7 | 214.5   | 1.6       | 46.2    |
| 1  | 1024| 7.0 | 212.8   | 6.8       | 37.3    |
| 1  | 2048| 4.9 | 432.2   | 3.3       | 22.0    |

Long contexts (sl=4096+): CV <2%, error <2%.

### Decode (high-CV configs)
| bs | sl | decode_cv% | decode_mean_ms |
|----|-----|-----------|----------------|
| 1  | 1024| 26.4      | 93.6           |
| 2  | 1024| 25.8      | 94.8           |
| 16 | 512 | 28.7      | 93.1           |
These spikes are fit_rmse inflation artifacts, not real routing variance.

---

## Llama3 exp1_5 sweep_real.py results (2026-03-16)
Model: Meta-Llama-3-8B-HF, single GPU, MAX_POOL=35000, 33 configs

All prefill errors: 0.2–2.3%. Dense model — no routing variance, predictor works well.
Decode spikes at bs=1 and bs=16 are inflation artifacts same as Mixtral.

---

## Mixtral exp2 stress_test.py results (2026-03-16)

### Exp A — Heterogeneous Batch Composition

bs=4, total_tokens=1024 (trained on uniform=[256,256,256,256]):
| family  | lens              | sum_n2  | mean_ms | cv%  | signed_err% | abs_err% |
|---------|-------------------|---------|---------|------|-------------|---------|
| uniform | [256]*4           | 262144  | 212.9   | 0.6  | +5.3        | 5.3     |
| bimodal | [64,64,448,448]   | 409600  | 216.7   | 9.8  | +62.5       | 62.8    |
| skewed  | [32,32,32,928]    | 864256  | 214.9   | 0.5  | +244.0      | 244.0   |

bs=8, total_tokens=2048 (trained on uniform=[256]*8):
| family  | lens                  | mean_ms | cv%  | signed_err% | abs_err% |
|---------|-----------------------|---------|------|-------------|---------|
| uniform | [256]*8               | 346.7   | 0.4  | +3.8        | 3.8     |
| bimodal | [128]*4+[384]*4       | 349.5   | 4.7  | +28.9       | 29.1    |
| skewed  | [32]*7+[1824]         | 355.9   | 4.3  | +543.7      | 543.7   |

Note: signed error is positive = over-predict = SFT starvation. FP=0 always.

### Exp B — Decode Under K Growth
bs=16, sl=256, 5 trials × 100 decode steps. K from 4112 → 5696.
Actual decode time: flat ~92–95ms throughout (SWA window=4096 caps attention).
CV steps 0-9: 3.0%.  CV steps 90-99: 0.9% (stabilizes).
Predictor error at step=99: -7.6% (train_window=10), -3.8% (train_window=50), -3.8% (train_window=50).
Finding: no systematic failure. SWA neutralizes K-growth concern.

### Exp C — Content-Dependent Routing Variance
bs=4, sl=256. Predictor trained on "random".
| regime   | mean_ms | cv%  | signed_err% |
|----------|---------|------|-------------|
| random   | 213.2   | 0.5  | +0.2        |
| constant | 218.6   | 0.3  | -2.3        |
| low_ids  | 221.8   | 0.9  | -3.7        |
| high_ids | 224.3   | 0.5  | -4.7        |
Finding: negligible effect. Content barely affects routing variance for this config.

### Exp D — RMSE Convergence vs n_train
All configs show erratic/non-monotone RMSE curves due to design flaw (see project_deltaserve_status.md).
Actual CV values measured over 580 samples:
- high_cv_bs4_sl256:  CV=0.8% (vs 9.7% from 80-sample sweep — small-sample inflation)
- high_cv_bs1_sl1024: CV=0.9% (vs 7.0% from sweep)
- low_cv_bs32_sl256:  CV=0.4%, RMSE@500=14.7ms >> actual_std=4.3ms (mean drift issue)
- low_cv_bs1_sl8192:  CV=0.5%, RMSE@500=16.5ms >> actual_std=6.2ms

### Exp E — SLO Gate Error Rates
Realistic batches: Poisson(mean=4) bs, log-normal(mean=64,std=48) sl. All prefills 99–155ms.
| slack_ms | FP%  | FN%   |
|----------|------|-------|
| 50       | 0    | 0     |
| 100      | 0    | 100.0 |
| 150      | 0    | 10.1  |
| 200      | 0    | 0     |
| 300      | 0    | 0     |
| 500      | 0    | 0     |
Finding: predictor is conservative (FP always 0). FN spike at 100ms because realistic
Mixtral batches are ≥99ms — right at the threshold. Above 200ms slack, gate is perfect.

---

## Qwen3-30B-A3B exp3 — Variance Profiler (2026-03-31)
Model: Qwen/Qwen3-30B-A3B, EP mode, 2 GPUs, MAX_POOL=20000
Log-normal trace: μ=4.5, σ=1.2, clipped [32,512]. BS=4, 256 calib + 256 profiling batches.

```
mean=658ms  cv=5.37%  fit_rmse=15.6ms
frac>10% error=0.0%  frac>5%=19.1%
mean_signed_err=+2.46%  lag1_autocorr=0.042
```
fit_rmse comparison: Llama3=2.9ms, Mixtral=13.5ms, Qwen3=15.6ms (5.4× Llama3).
Gate decisions: ALL 0% — SLO thresholds (200–400ms) are below Qwen3's inference time (~658ms).
**Use SLO thresholds 700–1200ms for Qwen3 in future gate simulations.**

---

## Qwen3-30B-A3B exp4 — Domain Routing Variance / Dolly-15k (2026-03-31)
Model: Qwen3-30B-A3B EP, 2 GPUs. BS=4, MAX_LEN=512. 256 calib + 25 prof batches/domain.

| domain | cv% | mean_ms | mean_err% |
|--------|-----|---------|-----------|
| general_qa | 8.67 | 638 | +3.72 |
| creative_writing | 6.29 | 589 | +9.59 |
| brainstorming | 6.02 | 598 | +7.07 |
| open_qa | 3.92 | 575 | +7.76 |
| classification | 3.84 | 611 | +7.65 |
| information_extraction | 3.61 | 602 | +4.87 |
| summarization | 3.39 | 586 | +6.08 |
| closed_qa | 3.25 | 601 | +4.34 |

No domain outliers. Contrast with Mixtral closed_qa CV=34.9% (two batches at 2–2.6× expected time).
Gate simulation uninformative (SLO thresholds too low).

---

## Qwen3-30B-A3B exp5 — MMLU Domain Experiment (2026-03-31)
Model: Qwen3-30B-A3B EP, 2 GPUs. BS=4, MAX_LEN=512. 200 calib + 50 prof batches/domain.

| domain | cv% | mean_ms | T_in_mean | T_in_std |
|--------|-----|---------|-----------|----------|
| stem_cs | 6.33 | 697 | 360 | 119 |
| medical | 5.50 | 660 | 330 | 206 |
| law_ethics | 5.48 | 692 | 898 | 269 |
| stem_math | 4.82 | 682 | 389 | 121 |
| humanities | 3.62 | 632 | 202 | 55 |
| social | 3.55 | 653 | 339 | 153 |
| stem_sci | 3.39 | 645 | 292 | 99 |

law_ethics NOT elevated despite high T_in_mean. No domain-specific routing variance.
Contrast with Mixtral exp5 where law_ethics/humanities showed high CV at long bucket.

---

## Qwen3-30B-A3B exp6 — Routing Collision Dissection (2026-03-31)
Model: Qwen3-30B-A3B EP, 2 GPUs. K_EXPERTS=8. Same MMLU setup as exp5.

**Formula bug:** compute_imbalance used balanced=TK/2 but both ranks route all tokens,
so rank 0 receives ~TK items total (not TK/2). Raw values ≈ 2.0 = perfect balance (corrected = raw/2 ≈ 1.0).

Corrected imbalance results (raw / 2):
- All domains, all buckets: corrected mean_imb = 0.98–1.02 (perfectly balanced)
- Random ablation (law_ethics long, random IDs): corrected ≈ 1.04–1.09

Per-layer spread (48 layers stored as mean + max per batch):
- Mean corrected imb across layers: 0.998 ± 0.018
- Mean corrected max imb (worst layer per batch): 1.272 ± 0.059
- Absolute worst layer seen: 1.476× balanced (47% overload on one rank for one layer)

**ANSWERED (2026-04-01): Ramya's per-layer correlation question**
Re-ran exp6 with corrected formula (balanced=TK) and full 48-layer timeseries stored per batch.

Corrected imbalance (balanced=TK, not TK/2):
- All domains, all buckets: mean_imb = 0.982–1.037 (max_imb ≤ 1.05)
- Random ablation: mean_imb=1.085, max_imb=1.090 (slightly more imbalanced, as before)

Per-layer correlation results (Ramya's hypothesis test):
- lag1_autocorr across 48 layers per batch: mean=**-0.13**, std=0.13, p50=-0.14, p95=+0.10
  → NEGATIVE (mean-reverting), not positive. Consecutive layers alternate rank bias.
- frac_layers_r0_heavy: mean=0.47, range [0.33, 0.63] — perfectly symmetric
- Batches with frac<0.3 or >0.7 (systematic bias): **0 out of 152**
- corr(frac_r0_heavy, actual_ms) = -0.07 (near zero — rank bias doesn't predict timing)

**Hypothesis FALSE**: The predictor is NOT hiding systematic per-layer load imbalance.
The 5% CV is irreducible hardware/communication variance, not a predictable pattern.
The negative lag1 suggests independent routing per layer, not correlated overload.

New output files: `test/qwen3/exp6/results/layer_timeseries.csv` (full per-layer timeseries),
`per_batch.csv` now includes `lag1_autocorr_layers` and `frac_layers_r0_heavy` columns.
