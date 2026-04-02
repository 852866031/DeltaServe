#!/usr/bin/env python3
"""
Predictor Gap — Fixed Composition Variance (Mixtral-8x7B EP, 2 GPUs)
=====================================================================
Runs 100 trials at each of 4 fixed T_in values (256, 512, 1024, 2048).
Every trial within a T_in group uses IDENTICAL batch composition (same
sequence lengths, same token IDs). The only source of timing variance is
hardware — stochastic all_to_all communication latency.

This directly measures the "oracle floor": the minimum RMSE achievable by
any static predictor that uses batch-composition features (T_in, Σn²),
regardless of how well trained it is.

Also fits PrefillExecutionEstimator on the first N_TRAIN trials and records
pred_ms, showing the predictor tracks the mean but cannot track the variance.

Usage:
    cd S-LoRA
    python test/mixtral/predictor_gap/fixed_composition_variance.py
"""

import os, sys, csv, time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
MAX_POOL  = 15_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT  = "29512"

BS        = 4
T_IN_VALUES = [256, 512, 1024, 2048]
N_WARMUP  = 20     # warmup before each T_in group
N_TRIALS  = 100    # timing trials per T_in
N_TRAIN   = 50     # first N_TRAIN trials used to fit predictor; rest are test


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

    all_rows = []

    for T_in in T_IN_VALUES:
        seq_len = T_in // BS   # uniform sequence length per batch
        # Fixed batch: same token IDs every trial (seed=T_in for reproducibility)
        rng = np.random.default_rng(T_in)
        fixed_batch = [rng.integers(0, vocab_size, size=seq_len).tolist()
                       for _ in range(BS)]
        seq_lens = [seq_len] * BS

        warmup(model, vocab_size, rank, N_WARMUP)

        if rank == 0:
            print(f"  T_in={T_in} (seq_len={seq_len}×{BS}): {N_TRIALS} trials...")

        times = []
        for trial in range(N_TRIALS):
            t = run_prefill(model, fixed_batch)
            if rank == 0:
                times.append(t * 1000)

        if rank == 0:
            # Fit predictor on training split
            est = PrefillExecutionEstimator()
            train_tokens = [seq_lens] * N_TRAIN
            train_times  = [t / 1000 for t in times[:N_TRAIN]]
            est.fit(
                inference_only_tokens=train_tokens,
                inference_only_times=train_times,
                coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
            )
            pred_s = est.predict_inference(seq_lens)
            pred_ms = pred_s * 1000

            train_arr = np.array(times[:N_TRAIN])
            test_arr  = np.array(times[N_TRAIN:])
            print(f"    train: mean={np.mean(train_arr):.2f}ms  std={np.std(train_arr):.2f}ms")
            print(f"    test:  mean={np.mean(test_arr):.2f}ms   std={np.std(test_arr):.2f}ms")
            print(f"    pred:  {pred_ms:.2f}ms   fit_rmse={est.fit_rmse*1000:.3f}ms")

            for trial, actual_ms in enumerate(times):
                all_rows.append({
                    "T_in":      T_in,
                    "trial_id":  trial,
                    "actual_ms": float(actual_ms),
                    "pred_ms":   float(pred_ms),
                    "is_train":  int(trial < N_TRAIN),
                    "fit_rmse_ms": float(est.fit_rmse * 1000),
                })

    return all_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "fixed_composition.csv")
    fields = ["T_in", "trial_id", "actual_ms", "pred_ms", "is_train", "fit_rmse_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {path}")


def print_summary(rows):
    print("\n" + "=" * 60)
    print("FIXED COMPOSITION VARIANCE — Mixtral-8x7B EP (2 GPUs)")
    print("=" * 60)
    print(f"  {'T_in':>6}  {'mean_ms':>8}  {'std_ms':>7}  {'cv%':>6}  {'pred_ms':>8}  {'fit_rmse':>9}")
    for T_in in T_IN_VALUES:
        sub = [r for r in rows if r["T_in"] == T_in and r["is_train"] == 0]
        if not sub:
            continue
        times = np.array([r["actual_ms"] for r in sub])
        pred  = sub[0]["pred_ms"]
        rmse  = sub[0]["fit_rmse_ms"]
        print(f"  {T_in:>6}  {np.mean(times):>8.2f}  {np.std(times):>7.3f}  "
              f"{np.std(times)/np.mean(times)*100:>6.2f}  {pred:>8.2f}  {rmse:>9.3f}")


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
