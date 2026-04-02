#!/usr/bin/env python3
"""
BurstGPT Distribution Shift — Llama3-8B control
================================================
Identical design to test/mixtral/exp7/burstgpt_shift.py but on Llama3-8B
(1 GPU, dense, no all_to_all). Used as a control to compare how much better
a dense model's predictor tracks the same real-world distribution shifts.

Usage:
    cd S-LoRA
    python test/llama3/exp7/burstgpt_shift.py
"""

import os, sys, csv, time, random
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR   = "/mnt/nfs/home/ramya/models/meta-llama/Meta-Llama-3-8B-HF"
MAX_POOL    = 35_000
OUT_DIR     = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT    = "29517"

BS           = 4
MAX_LEN      = 512        # match Mixtral cap for direct comparison
N_REQUESTS   = 2000
N_WARMUP     = 20
REFIT_EVERY  = 32
TRAIN_WINDOW = 256


# ---------------------------------------------------------------------------
# BurstGPT loader
# ---------------------------------------------------------------------------

def load_burstgpt(n_requests, max_len, vocab_size, seed=42):
    print(f"  Loading BurstGPT trace (first {n_requests} requests)...")
    from datasets import load_dataset
    ds = load_dataset("lzzmm/BurstGPT", split="train")
    df = ds.to_pandas().sort_values("Timestamp").head(n_requests)
    lengths = df["Request tokens"].clip(upper=max_len).clip(lower=1).astype(int).tolist()
    lengths = [l for l in lengths if l > 0]

    rng = random.Random(seed)
    batches = []
    for i in range(0, len(lengths) - BS + 1, BS):
        seq_lens  = lengths[i:i+BS]
        token_ids = [rng.choices(range(vocab_size), k=sl) for sl in seq_lens]
        batches.append((seq_lens, token_ids))

    print(f"  Loaded {len(batches)} batches. T_in range: "
          f"[{min(sum(s) for s,_ in batches)}, {max(sum(s) for s,_ in batches)}]")
    return batches


# ---------------------------------------------------------------------------
# Forward-pass helper
# ---------------------------------------------------------------------------

def run_prefill(model, token_ids_list):
    bs       = len(token_ids_list)
    seq_lens = [len(x) for x in token_ids_list]
    total_p  = sum(seq_lens)
    max_len  = max(seq_lens)

    flat        = np.concatenate([np.asarray(x, np.int64) for x in token_ids_list])
    input_ids_p = torch.from_numpy(flat).cuda()
    b_seq_len   = torch.tensor(seq_lens, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(bs, dtype=torch.long, device="cuda")
    for i in range(1, bs):
        b_start_loc[i] = b_start_loc[i-1] + seq_lens[i-1]
    b_loc = torch.zeros(bs, max_len, dtype=torch.long, device="cuda")

    model.mem_manager.free_all()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_p, max_len_in_batch=max_len,
                      input_ids=input_ids_p, b_loc=b_loc, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len, is_prefill=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def warmup(model, vocab_size, n=20):
    print(f"  Warming up ({n} batches)...")
    rng = np.random.default_rng(0)
    for _ in range(n):
        batch = [rng.integers(0, vocab_size, size=64) for _ in range(BS)]
        run_prefill(model, batch)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(model, vocab_size):
    from slora.server.router.tracker import PrefillExecutionEstimator

    batches = load_burstgpt(N_REQUESTS, MAX_LEN, vocab_size)

    warmup(model, vocab_size, N_WARMUP)

    history_seq_lens = []
    history_times    = []
    est              = PrefillExecutionEstimator()
    est_fitted       = False
    refit_id         = 0

    all_rows = []

    for batch_id, (seq_lens, token_ids) in enumerate(batches):
        actual_s  = run_prefill(model, token_ids)
        actual_ms = actual_s * 1000
        T_in      = sum(seq_lens)

        history_seq_lens.append(seq_lens)
        history_times.append(actual_s)

        pred_ms  = float("nan")
        fit_rmse = float("nan")

        if batch_id > 0 and batch_id % REFIT_EVERY == 0:
            window_tokens = history_seq_lens[-TRAIN_WINDOW:]
            window_times  = history_times[-TRAIN_WINDOW:]
            est = PrefillExecutionEstimator()
            est.fit(
                inference_only_tokens=window_tokens,
                inference_only_times=window_times,
                coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
            )
            est_fitted = True
            refit_id  += 1
            fit_rmse   = est.fit_rmse * 1000
            if (batch_id // REFIT_EVERY) % 4 == 0:
                print(f"  batch={batch_id:4d}  refit={refit_id}  "
                      f"fit_rmse={fit_rmse:.2f}ms  "
                      f"T_in_mean={np.mean([sum(s) for s in window_tokens]):.0f}")

        if est_fitted:
            pred_ms = est.predict_inference(seq_lens) * 1000

        t_in_mean_last = float(np.mean([sum(s) for s in history_seq_lens[-TRAIN_WINDOW:]]))

        all_rows.append({
            "batch_id":          batch_id,
            "T_in":              T_in,
            "actual_ms":         float(actual_ms),
            "pred_ms":           float(pred_ms) if not np.isnan(pred_ms) else "",
            "fit_rmse_ms":       float(fit_rmse) if not np.isnan(fit_rmse) else "",
            "refit_id":          refit_id,
            "t_in_mean_last256": float(t_in_mean_last),
        })

    return all_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "burstgpt_shift.csv")
    fields = ["batch_id", "T_in", "actual_ms", "pred_ms",
              "fit_rmse_ms", "refit_id", "t_in_mean_last256"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {path}")


def print_summary(rows):
    fitted_rows = [r for r in rows if r["fit_rmse_ms"] != ""]
    if not fitted_rows:
        print("No fitted rows."); return
    rmse_vals = [float(r["fit_rmse_ms"]) for r in fitted_rows]
    print("\n" + "=" * 55)
    print("BURSTGPT DISTRIBUTION SHIFT — Llama3-8B (1 GPU, dense)")
    print("=" * 55)
    print(f"  Total batches:   {len(rows)}")
    print(f"  Refits:          {max(int(r['refit_id']) for r in rows)}")
    print(f"  fit_rmse: min={min(rmse_vals):.2f}ms  "
          f"mean={np.mean(rmse_vals):.2f}ms  max={max(rmse_vals):.2f}ms")
    print(f"\n  {'refit_id':>8}  {'fit_rmse_ms':>12}  {'t_in_mean':>10}")
    seen = set()
    for r in rows:
        rid = r["refit_id"]
        if rid not in seen and r["fit_rmse_ms"] != "":
            seen.add(rid)
            print(f"  {int(rid):>8}  {float(r['fit_rmse_ms']):>12.2f}  "
                  f"{float(r['t_in_mean_last256']):>10.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{TCP_PORT}",
                            world_size=1, rank=0)

    from slora.models.llama3.model import Llama3TpPartModel
    print(f"Loading model from {MODEL_DIR}...")
    model = Llama3TpPartModel(tp_rank=0, world_size=1, weight_dir=MODEL_DIR,
                               max_total_token_num=MAX_POOL, mem_adapter_size=0,
                               dummy=False)
    vocab_size = model.config["vocab_size"]
    print(f"Model loaded. vocab_size={vocab_size}\n")

    rows = run_experiment(model, vocab_size)
    save_results(rows)
    print_summary(rows)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
