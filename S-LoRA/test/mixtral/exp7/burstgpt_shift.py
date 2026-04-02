#!/usr/bin/env python3
"""
BurstGPT Distribution Shift — Predictor Tracking Under Real Workload
=====================================================================
Replays a truncated slice of the BurstGPT trace (lzzmm/BurstGPT) through
Mixtral-8x7B EP. Uses real prompt token lengths from the trace, sorted by
timestamp, to preserve the natural distribution shifts over time.

The predictor refits every REFIT_EVERY batches (same cadence as DeltaServe).
We track rolling fit_rmse and the distribution of T_in to show:
  - When the distribution shifts, fit_rmse spikes
  - For Mixtral EP, fit_rmse never falls below the all_to_all noise floor
    (~13ms) even after the predictor has seen enough data to converge

Compare with test/llama3/exp7/burstgpt_shift.py — Llama3 tracks the
distribution shifts with much lower fit_rmse because it has no all_to_all.

Output:
  results/burstgpt_shift.csv  — per-batch: batch_id, T_in, actual_ms, pred_ms,
                                 fit_rmse_ms, refit_id, t_in_mean_last256

Usage:
    cd S-LoRA
    python test/mixtral/exp7/burstgpt_shift.py
"""

import os, sys, csv, time, random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR   = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
MAX_POOL    = 15_000
OUT_DIR     = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT    = "29516"

BS           = 4
MAX_LEN      = 512        # cap prompt tokens for Mixtral
N_REQUESTS   = 2000       # take first N_REQUESTS from trace (sorted by timestamp)
N_WARMUP     = 20
REFIT_EVERY  = 32         # how often predictor refits (DeltaServe default: 256, use 32 here for more resolution)
TRAIN_WINDOW = 256        # rolling window for predictor fit


# ---------------------------------------------------------------------------
# BurstGPT loader
# ---------------------------------------------------------------------------

def load_burstgpt(n_requests, max_len, vocab_size, seed=42):
    """
    Load BurstGPT trace, take first n_requests sorted by timestamp,
    cap prompt tokens at max_len, group into BS=4 batches.
    Returns list of (seq_lens, token_ids_list) per batch.
    """
    print(f"  Loading BurstGPT trace (first {n_requests} requests)...")
    from datasets import load_dataset
    ds = load_dataset("lzzmm/BurstGPT", split="train")
    df = ds.to_pandas().sort_values("Timestamp").head(n_requests)

    # Cap and filter
    lengths = df["Request tokens"].clip(upper=max_len).clip(lower=1).astype(int).tolist()
    lengths = [l for l in lengths if l > 0]

    # Group into batches
    rng = random.Random(seed)
    batches = []
    for i in range(0, len(lengths) - BS + 1, BS):
        seq_lens = lengths[i:i+BS]
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
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_p, max_len_in_batch=max_len,
                      input_ids=input_ids_p, b_loc=b_loc, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len, is_prefill=True)
    torch.cuda.synchronize(); dist.barrier()
    return time.perf_counter() - t0


def warmup(model, vocab_size, rank, n=20):
    if rank == 0:
        print(f"  Warming up ({n} batches)...")
    rng = np.random.default_rng(0)
    for _ in range(n):
        batch = [rng.integers(0, vocab_size, size=64) for _ in range(BS)]
        run_prefill(model, batch)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(model, vocab_size, rank):
    from slora.server.router.tracker import PrefillExecutionEstimator

    batches = None
    if rank == 0:
        batches = load_burstgpt(N_REQUESTS, MAX_LEN, vocab_size)

    # Broadcast batch count
    n_batches_t = torch.zeros(1, dtype=torch.long, device="cuda")
    if rank == 0:
        n_batches_t[0] = len(batches)
    dist.broadcast(n_batches_t, src=0)
    n_batches = int(n_batches_t.item())

    warmup(model, vocab_size, rank, N_WARMUP)

    history_seq_lens = []
    history_times    = []
    est              = PrefillExecutionEstimator()
    est_fitted       = False
    refit_id         = 0

    all_rows = []

    for batch_id in range(n_batches):
        if rank == 0:
            seq_lens, token_ids = batches[batch_id]
        else:
            # Rank 1 needs to run the forward pass with some batch;
            # token IDs don't affect timing for rank 1 in EP mode
            seq_lens  = [64] * BS
            token_ids = [[0] * 64 for _ in range(BS)]

        # Broadcast seq_lens so rank 1 allocates correct b_loc
        seq_lens_t = torch.zeros(BS, dtype=torch.long, device="cuda")
        if rank == 0:
            seq_lens_t[:] = torch.tensor(seq_lens, dtype=torch.long)
        dist.broadcast(seq_lens_t, src=0)
        seq_lens_broadcast = seq_lens_t.tolist()

        # Build token_ids for rank 1 using broadcast seq_lens
        if rank != 0:
            token_ids = [[0] * int(sl) for sl in seq_lens_broadcast]

        actual_s = run_prefill(model, token_ids)
        actual_ms = actual_s * 1000

        if rank == 0:
            T_in    = sum(seq_lens)
            history_seq_lens.append(seq_lens)
            history_times.append(actual_s)

            # Refit predictor periodically using rolling window
            pred_ms   = float("nan")
            fit_rmse  = float("nan")
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
                "batch_id":         batch_id,
                "T_in":             T_in,
                "actual_ms":        float(actual_ms),
                "pred_ms":          float(pred_ms) if not np.isnan(pred_ms) else "",
                "fit_rmse_ms":      float(fit_rmse) if not np.isnan(fit_rmse) else "",
                "refit_id":         refit_id,
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
        print("No fitted rows yet.")
        return
    rmse_vals = [float(r["fit_rmse_ms"]) for r in fitted_rows]
    print("\n" + "=" * 55)
    print("BURSTGPT DISTRIBUTION SHIFT — Mixtral-8x7B EP")
    print("=" * 55)
    print(f"  Total batches:   {len(rows)}")
    print(f"  Refits:          {max(int(r['refit_id']) for r in rows)}")
    print(f"  fit_rmse: min={min(rmse_vals):.2f}ms  "
          f"mean={np.mean(rmse_vals):.2f}ms  max={max(rmse_vals):.2f}ms")
    # Per-refit summary
    print(f"\n  {'refit_id':>8}  {'fit_rmse_ms':>12}  {'t_in_mean':>10}")
    seen = set()
    for r in rows:
        rid = r["refit_id"]
        if rid not in seen and r["fit_rmse_ms"] != "":
            seen.add(rid)
            print(f"  {int(rid):>8}  {float(r['fit_rmse_ms']):>12.2f}  "
                  f"{float(r['t_in_mean_last256']):>10.1f}")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{TCP_PORT}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    from slora.models.mixtral.model import MixtralEPTpPartModel
    if rank == 0:
        print(f"Loading model from {MODEL_DIR}...")
    model = MixtralEPTpPartModel(tp_rank=rank, world_size=world_size,
                                  weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                  mem_adapter_size=0, dummy=False)
    vocab_size = model.config["vocab_size"]
    if rank == 0:
        print(f"Model loaded. vocab_size={vocab_size}\n")

    rows = run_experiment(model, vocab_size, rank)
    if rank == 0:
        mp_results["rows"] = rows
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, mp_results), nprocs=2, join=True)

    rows = mp_results["rows"]
    save_results(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
