# MoE Predictor Failure — Full Implementation Notes

## Why the Predictor Fails (one paragraph)
`PrefillExecutionEstimator` fits `T ≈ α·Σn_i² + β·T_in + γ·T_ft + c`.
In Mixtral's `_ffn` (transformer_layer_infer.py:149) execution time depends on
`M_e` = tokens-per-expert, not just total tokens. Two batches with identical
`Σn_i²`/`T_in` but different input content get different expert routing →
different runtimes → the predictor sees the same features but different labels
→ irreducible high variance. Same argument applies to `DecodeExecutionEstimator`.

---

## Implementation Order (do in this exact sequence)

### STEP 0 — Get Mixtral weights (if needed)
```bash
# Check if weights already present:
ls /mnt/nfs/home/ramya/slora-plus/  # look for a mixtral* folder

# If not, download (needs HF login):
huggingface-cli login   # token is usually already set; just run it
huggingface-cli download mistralai/Mixtral-8x7B-v0.1 \
  --local-dir /mnt/nfs/home/ramya/mixtral-8x7b

# Alternative: use --dummy flag in the launcher to skip real weights.
# The dummy flag still runs the kernel timing, just with random weights.
# That's fine for the predictor experiment.
```

---

### STEP 1 — Patch transformer_layer_infer.py (expert load logging)

**File**: `S-LoRA/slora/models/mixtral/layer_infer/transformer_layer_infer.py`

At the top of the file, after the imports, add:

```python
import os as _os
_LOG_EXPERT_LOAD = _os.environ.get("LOG_EXPERT_LOAD", "0") == "1"
_expert_load_buf: list = []   # list of {"layer": int, "M_e": list[int]}

def get_and_clear_expert_load() -> list:
    """Return accumulated expert load records and reset the buffer."""
    global _expert_load_buf
    out = _expert_load_buf
    _expert_load_buf = []
    return out
```

Inside `_ffn`, at the point where `M_e` per expert is known (after the
`for expert_idx in range(num_experts)` loop, just before `return final_out`),
add:

```python
if _LOG_EXPERT_LOAD:
    M_e_counts = [
        int((selected_experts == e).any(dim=-1).sum())
        for e in range(num_experts)
    ]
    _expert_load_buf.append({"layer": self.layer_num_, "M_e": M_e_counts})
```

Exact insertion point: line 177 (just before `return final_out`).
NOTE: `selected_experts` is already computed at line 139; it's still in scope.

---

### STEP 2 — Expose expert stats from ModelRpcServer

**File**: `S-LoRA/slora/server/router/model_infer/model_rpc.py`

1. Add import at top (near line 14 where other model imports are):
```python
# Add conditionally — only import when Mixtral is loaded
```

2. Modify `exposed_prefill_batch` (line 286) and `exposed_decode_batch`
   (line 291) to optionally return expert stats alongside the normal return:

```python
def exposed_prefill_batch(self, batch_id, prefill_interrupt_event=None):
    result = self.forward(batch_id, is_prefill=True,
                          prefill_interrupt_event=prefill_interrupt_event)
    if _os.environ.get("LOG_EXPERT_LOAD", "0") == "1":
        try:
            from slora.models.mixtral.layer_infer.transformer_layer_infer import get_and_clear_expert_load
            self._last_expert_load = get_and_clear_expert_load()
        except ImportError:
            self._last_expert_load = []
    return result

def exposed_decode_batch(self, batch_id, decode_count=-1):
    result = self.forward(batch_id, is_prefill=False, decode_count=decode_count)
    if _os.environ.get("LOG_EXPERT_LOAD", "0") == "1":
        try:
            from slora.models.mixtral.layer_infer.transformer_layer_infer import get_and_clear_expert_load
            self._last_expert_load = get_and_clear_expert_load()
        except ImportError:
            self._last_expert_load = []
    return result

def exposed_get_expert_load(self):
    """Called by manager after each forward to retrieve expert stats."""
    return getattr(self, "_last_expert_load", [])
```

Add `import os as _os` near the top of model_rpc.py if not already there
(it already has `import os` at line 13).

---

### STEP 3 — Collect expert stats in manager.py

**File**: `S-LoRA/slora/server/router/manager.py`

The manager calls `self.model_rpcs[tp_rank].prefill_batch(...)` and
`decode_batch(...)` via rpyc. After those calls, retrieve and summarize
expert stats.

Add a helper method to the RouterManager class:

```python
def _collect_expert_stats(self) -> dict | None:
    """Retrieve expert load from rank 0 RPC and compute summary stats."""
    import os, json, numpy as np
    if os.environ.get("LOG_EXPERT_LOAD", "0") != "1":
        return None
    try:
        raw = self.model_rpcs[0].get_expert_load()
        if not raw:
            return None
        # raw = [{"layer": int, "M_e": [int]*8}, ...]
        # Average M_e distribution across layers, then compute stats
        all_M_e = np.array([r["M_e"] for r in raw], dtype=float)  # (L, E)
        avg_M_e = all_M_e.mean(axis=0)   # (E,)
        return {
            "max_M_e": float(avg_M_e.max()),
            "min_M_e": float(avg_M_e.min()),
            "std_M_e": float(avg_M_e.std()),
            "imbalance_ratio": float(avg_M_e.max() / max(avg_M_e.mean(), 1e-9)),
            "num_layers": len(raw),
        }
    except Exception:
        return None
```

In `_prefill_batch` (around line 358 where `add_batch_stats` is called),
change to:

```python
expert_stats = self._collect_expert_stats()
self.batch_exec_tracker.add_batch_stats(
    inference_tokens=inference_tokens_list,
    finetuning_tokens=finetuning_tokens_list,
    execution_type=BatchExecutionType.PREFILL,
    execution_duration=duration,
    predicted_duration=self.prefill_estimator.predict_inference(inference_tokens_list),
    expert_stats=expert_stats,   # NEW
)
```

Similarly for `_decode_batch` (around line 431).

---

### STEP 4 — Extend BatchExecutionTracker to store/write expert stats

**File**: `S-LoRA/slora/server/router/tracker.py`

In `BatchExecutionTracker.__init__`, add:
```python
self.expert_stats_list = []
```

In `add_batch_stats`, add parameter `expert_stats: Optional[dict] = None`
and append: `self.expert_stats_list.append(expert_stats)`

In `drop_batch_stats`, add: `del self.expert_stats_list[index]`

In `write_batch_prediction_stats_to_csv`, add `"expert_stats"` to fieldnames
and `"expert_stats": json.dumps(self.expert_stats_list[i]) if self.expert_stats_list[i] else ""` to each row dict.

---

### STEP 5 — Create Mixtral launcher

**New file**: `S-LoRA/test/mixtral/launch_mixtral.py`

Copy `S-LoRA/test/llama3/launch_llama3.py` and change:

```python
CONFIG = {
    "online": {
        "base_model": "mistralai/Mixtral-8x7B-v0.1",
        # No adapter dirs needed for inference-only predictor test
        "adapter_dirs": [],
        "finetuning_config_path": None,   # not used
        "no_finetuning_config_path": "/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/mixtral/config/no_finetuning_config.json",
    },
    "defaults": {
        "half_model": False,
        "enable_unified_mem_manager": True,
        "enable_gpu_profile": False,
        "unified_mem_manager_max_size": 6,
        "num_adapter": 0,
        "num_token": 25000,
        "pool_size_lora": 0,
    }
}
```

Change the cmd building to omit `--lora` args and `--swap` when no adapters.
Change port default to 9001.
Remove the `--finetuning_config_path` line (pass no_finetuning_config_path always).

If using `--dummy` (no real weights):
```python
cmd += " --dummy"
# Also need --model pointing to a valid HF model ID (for config only)
```

**New file**: `S-LoRA/test/mixtral/config/no_finetuning_config.json`

Copy from `S-LoRA/test/llama3/config/no_finetuning_config.json`:
```json
{
    "finetuning_type": "SFT",
    "finetuning_data_path": null,
    "finetuning_lora_path": null,
    "num_epochs": 0,
    "learning_rate": 0,
    "weight_decay": 0,
    "gamma": 1.0,
    "max_saved_finetuning_tokens": 0,
    "max_finetuning_tokens_in_batch": 0,
    "optimizer_threading": false,
    "ttft_slo": 10.0,
    "avg_tbt_slo": 1.0,
    "max_tbt_slo": 2.0,
    "start_on_launch": false
}
```

---

### STEP 6 — Create analysis script

**New file**: `S-LoRA/test/mixtral/analyze_predictor.py`

```python
"""
Loads two batch_prediction_stats CSVs (one Llama3, one Mixtral) and produces:
  1. Error % histogram comparison
  2. Scatter: prediction error vs expert imbalance ratio (Mixtral only)
  3. RMSE over time for both models
  4. Linear regression: base features vs. base+expert features for Mixtral

Usage:
    python analyze_predictor.py \
        --mixtral path/to/mixtral_predictions.csv \
        --llama3  path/to/llama3_predictions.csv  \
        --out     results/
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def load_df(path):
    df = pd.read_csv(path)
    df["error_pct"] = ((df["predicted_duration"] - df["execution_duration"]).abs()
                       / df["execution_duration"].clip(lower=1e-9) * 100)
    # Parse expert stats JSON if present
    if "expert_stats" in df.columns:
        def parse_es(s):
            if pd.isna(s) or s == "":
                return {}
            try: return json.loads(s)
            except: return {}
        stats = df["expert_stats"].apply(parse_es)
        df["imbalance_ratio"] = stats.apply(lambda d: d.get("imbalance_ratio", np.nan))
        df["std_M_e"]         = stats.apply(lambda d: d.get("std_M_e", np.nan))
        df["max_M_e"]         = stats.apply(lambda d: d.get("max_M_e", np.nan))
    return df

def parse_tokens(s):
    """Parse JSON token list string → sum of tokens."""
    try:
        lst = json.loads(s)
        return sum(int(x) for x in lst) if lst else 0
    except:
        return 0

def compute_features(df):
    """Compute Σn_i² and T_in from the inference_tokens column."""
    def sum_n2(s):
        try:
            lst = json.loads(s)
            return sum(int(x)**2 for x in lst)
        except:
            return 0
    df["sum_n2"] = df["inference_tokens"].apply(sum_n2)
    df["T_in"]   = df["inference_tokens"].apply(parse_tokens)
    return df

def plot_error_histograms(df_llama, df_mix, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, df, title in [(axes[0], df_llama, "Llama3"), (axes[1], df_mix, "Mixtral")]:
        prefill = df[df["batch_type"] == "prefill"]["error_pct"].dropna()
        ax.hist(prefill, bins=40, edgecolor="black")
        ax.set_title(f"{title} — Prefill Prediction Error %")
        ax.set_xlabel("Absolute % Error")
        ax.set_ylabel("Count")
        ax.axvline(prefill.median(), color="red", linestyle="--",
                   label=f"Median {prefill.median():.1f}%")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_histogram.png"), dpi=150)
    print(f"Saved error_histogram.png")

def plot_error_vs_imbalance(df_mix, out_dir):
    df = df_mix[df_mix["batch_type"] == "prefill"].dropna(subset=["imbalance_ratio", "error_pct"])
    if df.empty:
        print("No expert stats in Mixtral CSV — skipping imbalance plot")
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(df["imbalance_ratio"], df["error_pct"], alpha=0.4, s=15)
    # Spearman correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(df["imbalance_ratio"], df["error_pct"])
    plt.title(f"Mixtral: Error % vs Expert Imbalance (ρ={rho:.2f}, p={pval:.3f})")
    plt.xlabel("Imbalance Ratio (max_M_e / mean_M_e)")
    plt.ylabel("Absolute % Error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_vs_imbalance.png"), dpi=150)
    print(f"Saved error_vs_imbalance.png  (Spearman ρ={rho:.3f})")

def regression_comparison(df_mix, out_dir):
    """Base features vs. base+expert features for Mixtral prefill."""
    df = compute_features(df_mix[df_mix["batch_type"] == "prefill"].copy()).dropna(
        subset=["sum_n2", "T_in", "execution_duration"])
    if df.empty:
        print("Not enough prefill rows for regression")
        return
    y = df["execution_duration"].values
    X_base = np.column_stack([df["sum_n2"], df["T_in"], np.ones(len(df))])

    # If expert stats present, try augmented model
    has_exp = "imbalance_ratio" in df.columns and not df["imbalance_ratio"].isna().all()
    results = {}
    for label, X in [("base", X_base)]:
        model = Ridge(alpha=1e-3).fit(X, y)
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        results[label] = rmse
        print(f"  {label:20s}: RMSE={rmse:.4f}s")

    if has_exp:
        df2 = df.dropna(subset=["imbalance_ratio", "std_M_e"])
        if len(df2) > 10:
            X_aug = np.column_stack([df2["sum_n2"], df2["T_in"],
                                     df2["imbalance_ratio"], df2["std_M_e"],
                                     np.ones(len(df2))])
            y2 = df2["execution_duration"].values
            m2 = Ridge(alpha=1e-3).fit(X_aug, y2)
            rmse2 = np.sqrt(mean_squared_error(y2, m2.predict(X_aug)))
            results["base+expert"] = rmse2
            print(f"  {'base+expert':20s}: RMSE={rmse2:.4f}s  "
                  f"(improvement {(results['base']-rmse2)/results['base']*100:.1f}%)")

    with open(os.path.join(out_dir, "regression_results.json"), "w") as f:
        json.dump(results, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixtral", required=True)
    ap.add_argument("--llama3",  required=True)
    ap.add_argument("--out",     default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading CSVs...")
    df_mix   = load_df(args.mixtral)
    df_llama = load_df(args.llama3)

    print(f"Mixtral rows: {len(df_mix)}, Llama3 rows: {len(df_llama)}")

    # Claim 1: RMSE comparison
    mix_rmse   = df_mix[df_mix["batch_type"]=="prefill"]["error_pct"].std()
    llama_rmse = df_llama[df_llama["batch_type"]=="prefill"]["error_pct"].std()
    print(f"\nClaim 1 — Prefill error std:  Mixtral={mix_rmse:.1f}%  Llama3={llama_rmse:.1f}%  "
          f"ratio={mix_rmse/max(llama_rmse,1e-9):.1f}x")

    plot_error_histograms(df_llama, df_mix, args.out)
    plot_error_vs_imbalance(df_mix, args.out)

    print("\nClaim 3 — Regression comparison:")
    regression_comparison(df_mix, args.out)

if __name__ == "__main__":
    main()
```

---

### STEP 7 — Create the shell wrapper

**New file**: `S-LoRA/test/mixtral/run_predictor_experiment.sh`

```bash
#!/usr/bin/env bash
# Runs the full predictor failure experiment.
# Usage: bash run_predictor_experiment.sh [--dummy]
set -euo pipefail

REPO=/mnt/nfs/home/ramya/slora-plus/S-LoRA
MIXTRAL_MODEL=${MIXTRAL_MODEL:-"mistralai/Mixtral-8x7B-v0.1"}
LLAMA_MODEL=${LLAMA_MODEL:-"meta-llama/Meta-Llama-3-8B"}
DUMMY=${1:-""}   # pass "--dummy" for weight-free run

cd $REPO

echo "=== Phase 1: Mixtral predictor benchmark ==="
export LOG_EXPERT_LOAD=1
python test/mixtral/launch_mixtral.py --port 9001 $DUMMY &
SERVER_PID=$!
sleep 30   # wait for server to start (adjust for real weights: ~120s)
python test/llama3/predictor_benchmark.py --port 9001 --out mixtral_predictions.csv
kill $SERVER_PID 2>/dev/null || true
# Server writes batch_prediction_stats.csv in its CWD (S-LoRA/)
mv batch_prediction_stats.csv mixtral_predictions.csv 2>/dev/null || true
unset LOG_EXPERT_LOAD

echo "=== Phase 2: Llama3 predictor benchmark (baseline) ==="
python test/llama3/launch_llama3.py --port 9000 &
SERVER_PID=$!
sleep 30
python test/llama3/predictor_benchmark.py --port 9000 --out llama3_predictions.csv
kill $SERVER_PID 2>/dev/null || true
mv batch_prediction_stats.csv llama3_predictions.csv 2>/dev/null || true

echo "=== Phase 3: Analysis ==="
mkdir -p results/predictor_experiment
python test/mixtral/analyze_predictor.py \
    --mixtral mixtral_predictions.csv \
    --llama3  llama3_predictions.csv \
    --out     results/predictor_experiment/

echo "Done. Results in results/predictor_experiment/"
```

---

## Critical Code Context (to avoid re-reading files)

### tracker.py — current `add_batch_stats` signature (line 41):
```python
def add_batch_stats(self, inference_tokens, finetuning_tokens,
                    execution_type, execution_duration,
                    predicted_duration=None) -> None:
```
Add `expert_stats: Optional[dict] = None` at the end.

### manager.py — prefill tracker call (line 358):
```python
self.batch_exec_tracker.add_batch_stats(
    inference_tokens=inference_tokens_list,
    finetuning_tokens=finetuning_tokens_list,
    execution_type=BatchExecutionType.PREFILL,
    execution_duration=duration,
    predicted_duration = self.prefill_estimator.predict_inference(inference_tokens_list))
```

### manager.py — decode tracker call (line ~431):
```python
self.batch_exec_tracker.add_batch_stats(
    inference_tokens=inference_tokens_list,
    finetuning_tokens=finetuning_tokens_list,
    execution_type=BatchExecutionType.DECODE,
    execution_duration=duration,
    predicted_duration = self.decode_estimator.predict(sum(inference_tokens_list), len(inference_tokens_list)))
```

### transformer_layer_infer.py — `_ffn` method structure:
- Line 131: `hidden = input.view(-1, self.embed_dim_)`
- Line 132: `num_experts = len(layer_weight.experts_w1_)`
- Line 133: `top_k = self.network_config_["num_experts_per_tok"]`
- Line 139: `routing_weights, selected_experts = routing_weights.topk(top_k, dim=-1)`
  — `selected_experts` shape: `(T, top_k)`, values 0..num_experts-1
- Line 149: `for expert_idx in range(num_experts):`
- Line 177: `return final_out`  ← INSERT logging just before this

### model_rpc.py — key lines:
- Line 286: `def exposed_prefill_batch(self, batch_id, prefill_interrupt_event=None):`
- Line 291: `def exposed_decode_batch(self, batch_id, decode_count=-1):`
- Both are one-liners delegating to `self.forward(...)`.

---

## Directory Structure to Create
```
S-LoRA/test/mixtral/
├── __init__.py           (empty)
├── launch_mixtral.py
├── run_predictor_experiment.sh
├── analyze_predictor.py
└── config/
    └── no_finetuning_config.json
```

---

## Gotchas / Warnings

1. **rpyc serialization**: `get_expert_load()` returns a list of dicts with
   Python lists inside. rpyc may wrap these in netref proxies. Call
   `from rpyc.utils.classic import obtain` and wrap the return value:
   `raw = obtain(self.model_rpcs[0].get_expert_load())`.

2. **World size > 1**: If running multi-GPU TP, expert stats accumulate on
   each rank independently. Only collect from rank 0 (they'll be identical
   for TP since all ranks process the same tokens in the attention/FFN).

3. **CSV write path**: `write_batch_prediction_stats_to_csv` in manager.py
   is called at `/exit_finetuning` endpoint (line ~539) and on shutdown
   (line ~556). The default path is `"batch_prediction_stats.csv"` relative
   to the server's CWD (which is wherever `api_server.py` was launched from,
   i.e. the `S-LoRA/` directory if launched with `python -m slora.server...`).

4. **`--dummy` mode**: model_rpc.py passes `dummy=input_params.dummy` to the
   model constructor. Mixtral's `_ffn` will still run with random weights on
   GPU — expert routing will be random but valid. This is fine for timing
   experiments.

5. **No LoRA adapters for Mixtral**: SFT backward is not implemented for
   Mixtral (noted in CLAUDE.md). Pass empty `adapter_dirs` and a
   no_finetuning_config to avoid the backward code path entirely.

6. **predictor_benchmark.py already exists** at
   `S-LoRA/test/llama3/predictor_benchmark.py` — it can be reused as-is for
   Mixtral by just pointing it at port 9001. No need to copy it.
