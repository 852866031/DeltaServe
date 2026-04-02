# Mixtral Serving Experiment — Session Notes
*Written 2026-03-16. Goal: run Mixtral-8x7B inference server with the full predictor
pipeline active (no co-serving), drive 3-phase traffic (uniform → skewed → bimodal),
and observe how PrefillExecutionEstimator adapts in real time.*

---

## Background / Why We Care

`PrefillExecutionEstimator` fits `T ≈ α·Σn² + β·T_in + γ·T_ft + c` from live batch data.
Exp A (stress_test.py) showed it over-predicts skewed batches by **+244%** (trained on
uniform, tested on [32,32,32,928]). But in a live server the predictor refits online
every 256 batches — so the 244% error may self-correct once heterogeneous traffic arrives.

This experiment runs the actual serving stack and measures:
1. What the startup predictor looks like (calibrated on ~50-tok uniform profiling batches)
2. How TTFT changes when skewed requests arrive (initial period: bad predictions → over-queuing)
3. How fast the online refit corrects things (should improve after batch 256 and 512)

The `batch_prediction_stats_*.csv` written on server exit has every prediction vs actual.

---

## Files Created

All in `S-LoRA/test/mixtral/`:
```
launch_mixtral.py              — starts the server (2-GPU EP, no FT, predictor active)
auto_benchmark.py              — drives 3-phase traffic; writes benchmark_results.csv
config/inference_config.json   — minimal slora_plus config, no FT
```

---

## CRITICAL KNOWN ISSUES — Read Before Running

### Issue 1: `inference_config.json` is missing `num_epochs` (will crash)

`api_server.py` line 615 asserts ALL of these keys exist:
```
finetuning_data_path, finetuning_lora_path, num_epochs,
finetuning_type, ttft_slo, avg_tbt_slo, max_tbt_slo
```
My `inference_config.json` is missing `num_epochs`. **Fix before running:**
```bash
cat > /mnt/nfs/home/ramya/slora-plus/S-LoRA/test/mixtral/config/inference_config.json << 'EOF'
{
  "finetuning_type": "SFT",
  "finetuning_data_path": "/dev/null",
  "finetuning_lora_path": null,
  "num_epochs": 1,
  "start_on_launch": false,
  "ttft_slo": 0.5,
  "avg_tbt_slo": 0.3,
  "max_tbt_slo": 0.6,
  "max_finetuning_tokens_in_batch": 128,
  "max_saved_finetuning_tokens": 128
}
EOF
```

### Issue 2: `finetuning_lora_path: null` appended to `lora_dirs` (may crash)

`api_server.py` line 651 does `args.lora_dirs.append(config_data['finetuning_lora_path'])`.
With `null`, `None` gets appended to `lora_dirs`. Later `print_mem_stats` iterates over
`lora_dirs` and calls `lora_dir.split("/")` — will crash with `AttributeError: 'NoneType'`.

**Two options:**

**Option A (preferred) — skip the config entirely:**
Remove `--finetuning_config_path` from `launch_mixtral.py`. The default scheduler is
already `slora_plus`. The predictor pipeline still runs. The profiling batches still run
(with `adapter_dir=None` = base model). No crash from `print_mem_stats`.

Edit `launch_mixtral.py`, remove the `--finetuning_config_path {CONFIG}` line:
```bash
# In launch_mixtral.py, change the cmd string to NOT include:
#   f" --finetuning_config_path {CONFIG}"
# The slora_plus scheduler is the default — predictor runs without a config
```

**Option B — create a real Mixtral adapter:**
See "Creating a Dummy Mixtral Adapter" section below.

### Issue 3: `estimate_finetuning_overhead()` with no adapters

Without any `--lora` dir, `adapter_dirs = []` and all profiling batches have `adapter_dir=None`
(base model). Whether `load_adapters({None})` is a no-op depends on `model_rpc.py`.
**If you see this error:**
```
AttributeError: 'NoneType' object has no attribute ...
# OR
KeyError: None
```
It means None adapter is not handled. Fix: create a dummy Mixtral adapter (Option B above)
and pass it via `--lora` in `launch_mixtral.py`.

### Issue 4: auto_benchmark.py doesn't send `lora_dir`

The generate endpoint accepts `lora_dir` as optional (defaults to None = base model).
This is fine for `--no-lora` mode. But if you do pass an adapter, add to the payload:
```python
"lora_dir": "/path/to/mixtral-adapter"
```

### Issue 5: NCCL port conflicts

`launch_mixtral.py` uses `--nccl_port 28766` (sets 28766 and 28767 for tp=2).
`sweep_real.py` uses 29500. `stress_test.py` uses 29502. No conflict.
But if another job is running, check: `ss -ltnp | grep '2876[6-7]'`

---

## Step-by-Step Run Instructions

### Step 0: Environment check

```bash
conda activate dserve   # or your env
cd /mnt/nfs/home/ramya/slora-plus/S-LoRA

# Check GPUs are free
nvidia-smi

# Check model exists
ls /mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1/config.json

# Check no stale NCCL ports
ss -ltnp | grep '2876[6-7]'

# Check MPS is running (required for multi-GPU)
nvidia-cuda-mps-control get_server_list 2>/dev/null || echo "MPS not running"
# If not running:
# sudo nvidia-cuda-mps-control -d
```

### Step 1: Fix the config (do this once)

```bash
cat > /mnt/nfs/home/ramya/slora-plus/S-LoRA/test/mixtral/config/inference_config.json << 'EOF'
{
  "finetuning_type": "SFT",
  "finetuning_data_path": "/dev/null",
  "finetuning_lora_path": null,
  "num_epochs": 1,
  "start_on_launch": false,
  "ttft_slo": 0.5,
  "avg_tbt_slo": 0.3,
  "max_tbt_slo": 0.6,
  "max_finetuning_tokens_in_batch": 128,
  "max_saved_finetuning_tokens": 128
}
EOF
```

Then edit `launch_mixtral.py` to remove the config path line if you hit Issue 2.

### Step 2: Start the server (Terminal 1)

```bash
cd /mnt/nfs/home/ramya/slora-plus/S-LoRA
python test/mixtral/launch_mixtral.py --port 8000
```

**Expected startup output** (takes ~2-3 min to load Mixtral):
```
Loading model from /mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1 ...
Load weight from ...
Start modeling prefill/decode time          ← estimate_finetuning_overhead() starting
Running inference-only batch 1/18 ...       ← profiling batches
Running inference-only batch 2/18 ...
...
Error for prefill estimator: <value>        ← initial fit done
Error for decode estimator: <value>
init ok                                     ← server ready for requests
```

**If startup hangs** at "Load weight from": weights are loading over NFS, give it 3+ min.

**If startup crashes** with `AttributeError: 'NoneType'`: Issue 2 or 3 above. See fixes.

**Verify server is up:**
```bash
curl http://localhost:8000/health
# Expected: "OK"
```

### Step 3: Quick smoke test (Terminal 2)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello world", "parameters": {"max_new_tokens": 1}}'
# Expected: {"generated_text": "..."}
```

### Step 4: Run the benchmark (Terminal 2)

```bash
cd /mnt/nfs/home/ramya/slora-plus/S-LoRA
python test/mixtral/auto_benchmark.py --port 8000 --bs 4 --n_per_phase 200
```

**Traffic phases:**
- Phase 1 `uniform_256`: 200 requests, all 256 tokens → predictor calibrated on this
- Phase 2 `skewed_32x3_800`: 200 requests, [32,32,32,800] per batch → **Exp A scenario**
- Phase 3 `bimodal_64_448`: 200 requests, [64,64,448,448] → moderate skew

**Watch in the SERVER log (Terminal 1) for refit events:**
```
[Router]: Error for prefill estimator: X.XXX   ← fires every 256 batches
[Router]: Error for decode estimator: X.XXX
```
The RMSE should spike when the transition from uniform → skewed happens, then
may decrease after the refit incorporates the mixed data.

### Step 5: Stop server and collect data (Terminal 1)

```bash
# Ctrl+C in Terminal 1
# This triggers RouterManager.clean_up() which writes:
# prediction_stats_<model_path>.csv   ← every prediction vs actual
```

The CSV has columns: `timestamp, batch_index, batch_type, inference_tokens,
finetuning_tokens, execution_duration, predicted_duration`.

### Step 6: Analyze results

```bash
# Benchmark results (per request TTFT by phase):
cat test/mixtral/benchmark_results.csv

# Per-batch prediction accuracy:
python -c "
import csv, ast, statistics
rows = list(csv.DictReader(open('prediction_stats_*.csv')))
# ... analyze pred vs actual by batch range
"
```

---

## Creating a Dummy Mixtral Adapter (Option B fallback)

If you need an actual adapter for the server to load:

```python
# test/mixtral/create_adapter.py
import torch, os, json
from pathlib import Path

MIXTRAL = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
SAVE_DIR = "/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/mixtral/adapters/mixtral-toy-lora"
RANK = 8

# Read model config to get hidden_size
with open(f"{MIXTRAL}/config.json") as f:
    cfg = json.load(f)
hidden = cfg["hidden_size"]  # 4096 for Mixtral-8x7B

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# adapter_config.json (PEFT format)
adapter_cfg = {
    "base_model_name_or_path": MIXTRAL,
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": RANK,
    "target_modules": ["q_proj", "v_proj"],
    "task_type": "CAUSAL_LM"
}
with open(f"{SAVE_DIR}/adapter_config.json", "w") as f:
    json.dump(adapter_cfg, f, indent=2)

# Create zero-valued adapter weights for all 32 layers
state_dict = {}
num_layers = cfg["num_hidden_layers"]  # 32
for i in range(num_layers):
    prefix = f"base_model.model.model.layers.{i}"
    # q_proj: hidden → hidden, split by r
    state_dict[f"{prefix}.self_attn.q_proj.lora_A.weight"] = torch.zeros(RANK, hidden)
    state_dict[f"{prefix}.self_attn.q_proj.lora_B.weight"] = torch.zeros(hidden, RANK)
    # v_proj: hidden → hidden (GQA: num_key_value_heads * head_dim)
    num_kv_heads = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 32))
    kv_dim = num_kv_heads * (hidden // cfg.get("num_attention_heads", 32))
    state_dict[f"{prefix}.self_attn.v_proj.lora_A.weight"] = torch.zeros(RANK, hidden)
    state_dict[f"{prefix}.self_attn.v_proj.lora_B.weight"] = torch.zeros(kv_dim, RANK)

torch.save(state_dict, f"{SAVE_DIR}/adapter_model.bin")
print(f"Saved dummy adapter to {SAVE_DIR}")
print(f"  hidden_size={hidden}, num_layers={num_layers}, rank={RANK}")
```

Run it:
```bash
cd /mnt/nfs/home/ramya/slora-plus/S-LoRA
python test/mixtral/create_adapter.py
```

Then update `launch_mixtral.py` to add `--lora test/mixtral/adapters/mixtral-toy-lora`
and update `inference_config.json` `finetuning_lora_path` accordingly.

---

## What to Observe and How to Interpret

### In the server log (Terminal 1)

**Predictor refit lines** (every 256 batches):
```
[Router]: Error for prefill estimator: 0.015
```
This is `fit_rmse` in seconds. During startup profiling (uniform 50-tok batches): expect
low value (~0.001–0.010). After uniform_256 traffic: might increase slightly (different
distribution from profiling batches). After skewed traffic: should spike, then drop
at next refit as the model learns α≈0.

**TTFT lines** (every prefill):
```
[Router]: Prefill Duration: 0.213, worst TTFT: 0.219
```
Worst TTFT = time from request arrival to first token. Compare across phases:
- uniform_256: should be ~200–220ms
- skewed_32x3_800 (first 256 batches, before refit): expect TTFT similar but the scheduler
  might delay FT admission unnecessarily (if FT were running — in our case no FT so
  this doesn't matter for throughput, but the prediction error is visible in the CSV)
- after refit: prediction error should drop

### In benchmark_results.csv

Compare `mean_ttft_ms` and `p90_ttft_ms` across phases. In pure inference (no FT),
TTFT is mostly the prefill time. The phases have similar total tokens (~1024) so actual
prefill times should be close (~210–220ms for all three). Any large differences
indicate queuing or scheduling artifacts.

### In prediction_stats_*.csv (after server exit)

Key columns: `execution_duration` vs `predicted_duration`, split by `batch_type=prefill`.
- Compute `signed_err = (predicted - actual) / actual * 100` per row
- Group by batch index ranges: [0–17]=warmup, [18–273]=phase1, [274–529]=phase2, etc.
- The transition from uniform to skewed should show a spike in positive error (over-prediction)
- After batch 256, the refit fires; check if error drops in subsequent batches

Quick analysis snippet:
```python
import csv, ast, numpy as np

rows = list(csv.DictReader(open('prediction_stats_<model>.csv')))
prefill = [r for r in rows if r['batch_type'] == 'prefill']
for r in prefill:
    actual = float(r['execution_duration'])
    pred   = float(r['predicted_duration'])
    r['err_pct'] = (pred - actual) / actual * 100

# Group by 50-batch windows
for start in range(0, len(prefill), 50):
    window = prefill[start:start+50]
    errs = [r['err_pct'] for r in window]
    print(f"Batches {start}–{start+len(window)-1}: mean_err={np.mean(errs):+.1f}% std={np.std(errs):.1f}%")
```

---

## Predictor Pipeline Recap (for reference)

**Startup** (`estimate_finetuning_overhead`):
- Runs 18 inference-only profiling batches: total token targets [2000,100,200,...,5000]
- Each batch has `total_tok // 50` requests of ~50 tokens each (UNIFORM)
- Also runs 9 co-serving profiling batches (inf+FT)
- Initial fit: predictor calibrated on uniform ~50-tok requests
- Refit trigger: `BatchExecutionTracker.check_refit()` fires when `size() % 256 == 0`
- Decode tracked: only first 8 decode steps per sequence (`decode_step_count < 8`)

**Online** (`_co_serving_step`):
- Every completed prefill/decode → `add_batch_stats(inference_tokens, ft_tokens, duration)`
- Every 256 batches → `data_fit()` on all accumulated data (max 10240 entries)
- `fit_rmse` printed after each refit

**Implication for Exp A**:
- First 256 real batches served with STALE predictor (startup calibration)
- After 256 batches: refit with startup batches + 256 real batches → adapts to actual distribution
- If real traffic is all uniform → refit never corrects the skewed-batch failure
- If real traffic is mixed → refit learns α≈0, errors drop to <10%

---

## Next Steps After This Experiment

If the online refit successfully corrects the skewed-batch error, the implication is:
**the predictor failure is a cold-start problem**, not a permanent structural one.
The fix (two-term model `α·T_in + β·max_n²`) would just make it correct from the start,
without needing 256 skewed batches to self-calibrate.

If the online refit does NOT correct it even after 500+ skewed batches:
Then `Σn²` is structurally wrong and the feature engineering fix is mandatory.
Priority: edit `slora/server/router/tracker.py` `PrefillExecutionEstimator.fit()` and
`predict_inference()` to use `max(n)²` instead of `Σn²`.

---

## Port Reference

| Script | NCCL ports | TCP dist port |
|--------|-----------|--------------|
| launch_mixtral.py | 28766, 28767 | — |
| test/mixtral/exp2/stress_test.py | — | 29502 |
| test/mixtral/exp1_5/sweep_real.py | — | 29500 |
| test/llama3/exp1_5/sweep_real.py | — | 29501 |

**Never run two of these simultaneously on the same machine.**
