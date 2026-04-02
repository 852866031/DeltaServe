#!/usr/bin/env python3
"""
Predictor Gap — Fixed Composition Variance (Llama3-8B, 1 GPU — control)
========================================================================
Same design as the Mixtral version: 100 trials at each of 4 fixed T_in
values (256, 512, 1024, 2048) with identical batch composition per group.

For Llama3 (dense, no all_to_all), the execution time at fixed composition
should have very low variance (~0.7ms std) because there is no stochastic
inter-GPU communication. This is the oracle floor for dense models.

Compare the output directly against Mixtral's fixed_composition.csv to see
the irreducible gap: Llama3 std ≈ 0.7ms vs Mixtral std ≈ 9ms at T_in=1024.

Usage:
    cd S-LoRA
    python test/llama3/predictor_gap/fixed_composition_variance.py
"""

import os, sys, csv, time
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/meta-llama/Meta-Llama-3-8B-HF"
MAX_POOL  = 35_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT  = "29513"

BS          = 4
T_IN_VALUES = [256, 512, 1024, 2048]
N_WARMUP    = 20
N_TRIALS    = 100
N_TRAIN     = 50


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

    all_rows = []

    for T_in in T_IN_VALUES:
        seq_len = T_in // BS
        rng = np.random.default_rng(T_in)
        fixed_batch = [rng.integers(0, vocab_size, size=seq_len).tolist()
                       for _ in range(BS)]
        seq_lens = [seq_len] * BS

        warmup(model, vocab_size, N_WARMUP)
        print(f"  T_in={T_in} (seq_len={seq_len}×{BS}): {N_TRIALS} trials...")

        times = []
        for trial in range(N_TRIALS):
            t = run_prefill(model, fixed_batch)
            times.append(t * 1000)

        est = PrefillExecutionEstimator()
        train_tokens = [seq_lens] * N_TRAIN
        train_times  = [t / 1000 for t in times[:N_TRAIN]]
        est.fit(
            inference_only_tokens=train_tokens,
            inference_only_times=train_times,
            coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
        )
        pred_ms = est.predict_inference(seq_lens) * 1000

        train_arr = np.array(times[:N_TRAIN])
        test_arr  = np.array(times[N_TRAIN:])
        print(f"    train: mean={np.mean(train_arr):.2f}ms  std={np.std(train_arr):.2f}ms")
        print(f"    test:  mean={np.mean(test_arr):.2f}ms   std={np.std(test_arr):.2f}ms")
        print(f"    pred:  {pred_ms:.2f}ms   fit_rmse={est.fit_rmse*1000:.3f}ms")

        for trial, actual_ms in enumerate(times):
            all_rows.append({
                "T_in":        T_in,
                "trial_id":    trial,
                "actual_ms":   float(actual_ms),
                "pred_ms":     float(pred_ms),
                "is_train":    int(trial < N_TRAIN),
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
    print("FIXED COMPOSITION VARIANCE — Llama3-8B (1 GPU, dense)")
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
