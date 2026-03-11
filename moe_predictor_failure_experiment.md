# Experiment: Proving Prefill/Decode Predictors Fail for Mixtral MoE

## Hypothesis

The current `PrefillExecutionEstimator` and `DecodeExecutionEstimator` in
`S-LoRA/slora/server/router/tracker.py` are **fundamentally wrong** for
Mixtral MoE because they model execution time as a function of *total tokens
only*:

```
T_prefill ≈ α·Σn_i² + β·T_in + γ·T_ft + c
T_decode  ≈ δ·B + ε·K + d
```

For dense Transformers (Llama3) the FFN cost per token is constant, so these
linear features are sufficient.  For Mixtral with sparse MoE routing the FFN
cost is **data-dependent and stochastic**:

* The router gates assign each token to its top-2 of 8 experts.
* The runtime of each expert's FFN is proportional to the number of tokens
  it receives (`M_e`), which varies between batches even when total tokens
  are identical.
* Expert load is non-uniform: a popular expert with 80% of tokens takes 4×
  longer than one with 20%.  Two batches with the same `Σn_i²` and `T_in`
  can have completely different `{M_e}` distributions and thus different
  runtimes.

**Expected result**: prediction RMSE for Mixtral will be systematically and
significantly higher than for an equivalent Llama3 batch, and prediction error
will *not* decrease as the predictor refits on more data (unlike Llama3 where
error tightens over time).

---

## Why the Current Model Fails (Mechanistic Argument)

In `transformer_layer_infer.py` the MoE FFN loop is:

```python
for expert_idx in range(num_experts):           # always 8 iterations
    expert_mask = (selected_experts == expert_idx)
    if not expert_mask.any():
        continue
    token_indices = expert_mask.any(dim=-1).nonzero(...)   # M_e tokens
    expert_input  = hidden[token_indices]
    ...                                          # O(M_e) GEMM
    final_out[tok] += weight * expert_out[...]   # scatter-add loop
```

Total FFN time ~ `Σ_e M_e · cost_per_token_per_expert + scatter_overhead`.

With uniform random routing: `E[M_e] = T * top_k / num_experts = T/4`.
Variance of `M_e` is `T * p * (1-p)` where `p = top_k/num_experts`.
Because CUDA GEMM throughput is non-linear at small sizes, an expert that
accidentally receives 0 tokens saves little time relative to one that
receives 2× its expected share.

The predictor has no feature for `max(M_e)` or `std(M_e)`, so it cannot
capture expert load imbalance.

---

## Implementation Plan

### Step 1: Expert-Load Logging Patch

**File to modify**: `S-LoRA/slora/models/mixtral/layer_infer/transformer_layer_infer.py`

Add a thread-local accumulator that counts `M_e` per expert per forward pass.
This does **not** need to be in the hot path—only enabled via an env var flag.

```python
# top of file
import os, collections
_EXPERT_LOG = os.environ.get("LOG_EXPERT_LOAD", "0") == "1"
_expert_load_history = []   # list of dicts {"layer": int, "M_e": [int]*num_experts}
```

Inside `_ffn`, after computing `token_indices`:

```python
if _EXPERT_LOG:
    _expert_load_history.append({
        "layer": self.layer_num_,
        "M_e": [int((selected_experts == e).any(dim=-1).sum()) for e in range(num_experts)]
    })
```

Expose via a module-level function `get_and_clear_expert_load()` that returns
and resets `_expert_load_history`.

### Step 2: Batch-Level Expert Stats Collection in ModelRpcServer

**File to modify**: `S-LoRA/slora/server/router/model_infer/model_rpc.py`

After each `prefill_batch` / `decode_batch` call, if `LOG_EXPERT_LOAD=1`:
1. Call `get_and_clear_expert_load()`.
2. Compute per-batch summary: `max_M_e`, `std_M_e`, `entropy_M_e` (averaged
   across layers).
3. Return these alongside the normal token outputs, or write to a side CSV.

### Step 3: Extended BatchExecutionTracker

**File to modify**: `S-LoRA/slora/server/router/tracker.py`

Add optional `expert_stats` field to `add_batch_stats`:

```python
def add_batch_stats(self, ..., expert_stats: Optional[dict] = None):
    ...
    self.expert_stats_list.append(expert_stats)
```

Write `expert_stats` columns to the CSV in `write_batch_prediction_stats_to_csv`.

### Step 4: Standalone Benchmark Script

**New file**: `S-LoRA/test/mixtral/predictor_benchmark.py`

This script:
1. Launches the server with a Mixtral checkpoint (or a tiny mock MoE if no
   checkpoint is available — see Step 6).
2. Sends N inference requests with **fixed total tokens** but varying
   sequence length distributions (to probe the quadratic attention term
   while keeping `T_in` constant).
3. Sends M inference requests with **fixed sequence lengths** but varying
   batch sizes (to probe the batch-size term).
4. Collects `batch_prediction_stats.csv` with expert load columns.
5. Plots prediction error vs. expert load imbalance.

Key request patterns to generate (designed to expose the failure):

| Experiment | Batch composition | Expected predictor behavior |
|---|---|---|
| A | 1 request × 512 tokens | Good fit (single-request, no variation) |
| B | 8 requests × 64 tokens each | Predictor matches — same total, different batch size |
| C | Mixed lengths: [400,50,30,20,10,2] | Attention term dominates, predictor OK |
| D | Same as C but different input content → different routing | **Predictor fails**: same features, different expert load, different time |
| E | Adversarial: craft inputs that concentrate on 1 expert | Max slowdown vs. prediction |
| F | Adversarial: craft inputs that spread evenly across experts | Min time vs. prediction |

### Step 5: Analysis Script

**New file**: `S-LoRA/test/mixtral/analyze_predictor.py`

```python
import pandas as pd, matplotlib.pyplot as plt, json, numpy as np

df = pd.read_csv("batch_prediction_stats.csv")
df["error_pct"] = (df["predicted_duration"] - df["execution_duration"]).abs() / df["execution_duration"] * 100

# Parse expert stats
df["max_Me"]  = df["expert_stats"].apply(lambda s: json.loads(s)["max_M_e"] if s else None)
df["std_Me"]  = df["expert_stats"].apply(lambda s: json.loads(s)["std_M_e"] if s else None)

# Key plots:
# 1. Error % distribution for Llama3 vs Mixtral (should differ dramatically)
# 2. Scatter: error_pct vs max_Me (should be positively correlated for Mixtral)
# 3. Residual vs prediction time (heteroscedasticity for Mixtral, not Llama3)
# 4. Refit RMSE over time: Llama3 converges, Mixtral stays high

# Regression: does adding max_Me as a feature reduce error?
from sklearn.linear_model import Ridge
X_base = df[["sum_n2", "T_in"]].dropna()
X_aug  = df[["sum_n2", "T_in", "max_Me", "std_Me"]].dropna()
...  # compare R² of base vs augmented model
```

### Step 6: Mock MoE Model for Offline Testing (No GPU Required)

**New file**: `S-LoRA/test/mixtral/mock_moe_timing.py`

If a real Mixtral checkpoint is unavailable, simulate batch execution times
using a synthetic MoE timing model to validate the analysis pipeline:

```python
import numpy as np

def simulate_batch_time(token_counts, num_experts=8, top_k=2, seed=None):
    """
    Simulate Mixtral forward time for a batch.
    True time = attention_cost + moe_cost (expert-load-dependent).

    The predictor only sees: Σn_i², T_in — not expert loads.
    """
    rng = np.random.default_rng(seed)
    T = sum(token_counts)

    # Attention: O(Σn_i²) as in the predictor model
    attn_cost = 1e-6 * sum(n**2 for n in token_counts)

    # MoE: each token routed to top_k experts
    # Expert assignment: multinomial per token
    token_assignments = rng.integers(0, num_experts, size=(T, top_k))
    M_e = np.array([(token_assignments == e).any(axis=1).sum() for e in range(num_experts)])

    # Expert GEMM cost: non-linear (CUDA occupancy cliff below ~32 tokens)
    def expert_cost(m):
        if m == 0: return 0.0
        # Simulate CUDA occupancy: cost is max(base_cost, linear_cost)
        base_cost = 5e-4   # fixed overhead for launching expert kernel
        linear_cost = m * 2e-5
        return max(base_cost, linear_cost)

    moe_cost = sum(expert_cost(m) for m in M_e)
    noise = rng.normal(0, 1e-4)
    return attn_cost + moe_cost + noise, M_e


def run_experiment(n_batches=500):
    """
    Generate batches with fixed Σn_i² and T_in but different expert loads.
    Show that the predictor (which only sees Σn_i², T_in) has high variance.
    """
    results = []
    for i in range(n_batches):
        # Fixed batch: 4 requests × 64 tokens each → T_in=256, Σn²=16384
        token_counts = [64, 64, 64, 64]
        actual_time, M_e = simulate_batch_time(token_counts, seed=i)
        predicted_time = predict_with_linear_model(token_counts)   # fitted separately
        results.append({
            "actual": actual_time,
            "predicted": predicted_time,
            "max_M_e": max(M_e),
            "std_M_e": float(np.std(M_e)),
        })
    return pd.DataFrame(results)
```

---

## Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **Modify** | `slora/models/mixtral/layer_infer/transformer_layer_infer.py` | Add optional expert-load logging |
| **Modify** | `slora/server/router/model_infer/model_rpc.py` | Collect and forward expert stats after each forward pass |
| **Modify** | `slora/server/router/tracker.py` | Add `expert_stats_list`, extend CSV output |
| **Create** | `test/mixtral/predictor_benchmark.py` | End-to-end benchmark that runs fixed-token-count batches with varying content to expose expert load variance |
| **Create** | `test/mixtral/mock_moe_timing.py` | Offline simulation of MoE timing (no GPU required for analysis validation) |
| **Create** | `test/mixtral/analyze_predictor.py` | Load CSV, plot error distributions, scatter vs. expert load, regression comparison |
| **Create** | `test/mixtral/run_predictor_experiment.sh` | Shell wrapper: launches server with `LOG_EXPERT_LOAD=1`, runs benchmark, kills server, runs analysis |

---

## Expected Results / Falsifiable Claims

1. **Claim 1**: Mixtral prefill prediction RMSE will be ≥ 3× the Llama3
   prefill prediction RMSE, measured on batches with the same total token
   count.

2. **Claim 2**: Prediction error for Mixtral is *positively correlated* with
   `max(M_e) / mean(M_e)` (expert load imbalance ratio).  Spearman ρ ≥ 0.4.

3. **Claim 3**: Adding `max_M_e` and `std_M_e` as features to the prefill
   regression reduces Mixtral RMSE by ≥ 30%, while having negligible impact
   (<5% improvement) on Llama3.

4. **Claim 4**: The predictor RMSE for Mixtral does **not** converge as the
   number of observed batches grows (log-log plot of RMSE vs batches seen is
   flat for Mixtral, declining for Llama3).

---

## What "Failure" Looks Like in the SLO-Gating Context

The predictor is used in `Mixed_ReqQueue.check_will_starve()` to decide
whether to admit FT tokens into the next prefill.  If the predictor
systematically **under-predicts** Mixtral prefill time (which it will whenever
a popular expert is overloaded), the SLO gate will admit FT tokens that push
real TTFT past the SLO — defeating the scheduler's correctness guarantee.

Conversely, **over-prediction** (when experts happen to be balanced) causes
unnecessary FT throttling, reducing fine-tuning throughput.

Both failure modes can be demonstrated by plotting:
- `predicted_prefill_time` vs `actual_prefill_time` with the SLO threshold
  marked: count how many batches crossed the SLO line due to prediction error.

---

## Suggested Experiment Execution Order

```
Step A: Validate mock simulation (no GPU needed)
  python test/mixtral/mock_moe_timing.py
  → Produces synthetic CSV; run analyze_predictor.py on it
  → Confirms analysis pipeline works

Step B: Real Mixtral run (requires GPU + checkpoint)
  export LOG_EXPERT_LOAD=1
  bash test/mixtral/run_predictor_experiment.sh
  → Produces batch_prediction_stats_mixtral.csv

Step C: Llama3 baseline (same benchmark script, llama3 server)
  bash test/llama3/run_predictor_experiment.sh
  → Produces batch_prediction_stats_llama3.csv

Step D: Side-by-side analysis
  python test/mixtral/analyze_predictor.py \
    --mixtral batch_prediction_stats_mixtral.csv \
    --llama3  batch_prediction_stats_llama3.csv
```
