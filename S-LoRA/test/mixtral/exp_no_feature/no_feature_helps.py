#!/usr/bin/env python3
"""
Exp 5 — "No Feature Helps" Proof (Mixtral-8x7B EP)
====================================================
Directly tests whether adding richer batch-composition features to
PrefillExecutionEstimator reduces RMSE for Mixtral EP.

Current features: [Σn², T_in, 1]
Extended features: [Σn², T_in, max_n², skew (max_n/mean_n), B, T_in², 1]

Design:
  - Run N_TRAIN heterogeneous prefill batches (mixed batch sizes and length
    distributions) to give both models real feature variance to learn from.
  - Run N_TEST held-out batches of the same distribution.
  - Fit baseline (2-feature) and extended (6-feature) OLS models on train.
  - Report train RMSE and test RMSE for each.

Expected result:
  - Mixtral EP: extended ≈ baseline RMSE (features don't help — variance is
    from stochastic all_to_all communication, not batch composition).
  - Compare with test/llama3/exp_no_feature/no_feature_helps.py where Llama3
    (dense, no all_to_all) should also show low RMSE for both, confirming
    there's nothing to gain feature-wise in the dense case either.

Usage:
    cd S-LoRA
    python test/mixtral/exp_no_feature/no_feature_helps.py

Output:
    results/no_feature_helps.csv  — per-batch features + actual_ms +
                                    pred_baseline_ms + pred_extended_ms
    Prints: RMSE comparison table for train and test splits.
"""

import os, sys, csv, time, random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
MAX_POOL  = 15_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT  = "29518"

N_WARMUP = 20
N_TRAIN  = 300   # batches used to fit both models
N_TEST   = 100   # held-out evaluation batches

# Batch shapes to sample from — deliberately heterogeneous so features vary
# Each entry: (batch_size, seq_len_distribution)
SHAPE_POOL = [
    (1,  [256]),
    (2,  [128, 384]),
    (4,  [64,  64,  64,  64]),
    (4,  [32,  32,  32,  416]),   # skewed
    (4,  [64,  128, 256, 512]),   # spread
    (8,  [64]  * 8),
    (8,  [32,  32,  32,  32, 32,  32,  32,  928]),  # skewed bs=8
    (4,  [512] * 4),
    (2,  [512, 512]),
    (1,  [1024]),
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def batch_features(seq_lens):
    """Return feature dict from a list of sequence lengths."""
    n  = np.array(seq_lens, dtype=float)
    B  = float(len(n))
    T  = float(n.sum())
    S  = float((n ** 2).sum())
    mx = float(n.max())
    mn = float(n.mean())
    return {
        "B":       B,
        "T_in":    T,
        "sum_n2":  S,
        "max_n2":  mx ** 2,
        "skew":    mx / mn if mn > 0 else 1.0,
        "T_in2":   T ** 2,
    }


def baseline_row(feat):
    """Feature vector for baseline model: [Σn², T_in, 1]"""
    return [feat["sum_n2"], feat["T_in"], 1.0]


def extended_row(feat):
    """Feature vector for extended model: [Σn², T_in, max_n², skew, B, T_in², 1]"""
    return [feat["sum_n2"], feat["T_in"], feat["max_n2"],
            feat["skew"],   feat["B"],    feat["T_in2"], 1.0]


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
        batch = [rng.integers(0, vocab_size, size=64) for _ in range(4)]
        run_prefill(model, batch)


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def make_batch(vocab_size, rng):
    """Pick a random shape and generate token IDs."""
    bs, lens_template = random.choice(SHAPE_POOL)
    seq_lens  = lens_template
    token_ids = [rng.integers(0, vocab_size, size=sl).tolist() for sl in seq_lens]
    return seq_lens, token_ids


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(model, vocab_size, rank):
    rng = np.random.default_rng(42)
    random.seed(42)

    N_TOTAL = N_TRAIN + N_TEST
    all_records = []

    if rank == 0:
        print(f"  Running {N_TOTAL} batches ({N_TRAIN} train + {N_TEST} test)...")

    for i in range(N_TOTAL):
        # Rank 0 decides batch shape, broadcasts seq_lens to rank 1
        if rank == 0:
            seq_lens, token_ids = make_batch(vocab_size, rng)
        else:
            seq_lens  = None
            token_ids = None

        # Broadcast batch size and seq_lens
        meta = torch.zeros(9, dtype=torch.long, device="cuda")  # max 8 seqs + length
        if rank == 0:
            meta[0] = len(seq_lens)
            for j, sl in enumerate(seq_lens):
                meta[j + 1] = sl
        dist.broadcast(meta, src=0)

        bs_broad  = int(meta[0].item())
        sl_broad  = [int(meta[j + 1].item()) for j in range(bs_broad)]
        if rank != 0:
            token_ids = [rng.integers(0, vocab_size, size=sl).tolist() for sl in sl_broad]
            seq_lens  = sl_broad

        actual_s  = run_prefill(model, token_ids)
        actual_ms = actual_s * 1000

        if rank == 0:
            feat = batch_features(seq_lens)
            feat["actual_ms"] = actual_ms
            feat["split"]     = "train" if i < N_TRAIN else "test"
            all_records.append(feat)

        if rank == 0 and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N_TOTAL}] last actual={actual_ms:.1f}ms")

    return all_records


# ---------------------------------------------------------------------------
# Fit and evaluate
# ---------------------------------------------------------------------------

def fit_and_report(records):
    train = [r for r in records if r["split"] == "train"]
    test  = [r for r in records if r["split"] == "test"]

    X_tr_base = np.array([baseline_row(r) for r in train])
    X_tr_ext  = np.array([extended_row(r) for r in train])
    y_tr      = np.array([r["actual_ms"]  for r in train])

    X_te_base = np.array([baseline_row(r) for r in test])
    X_te_ext  = np.array([extended_row(r) for r in test])
    y_te      = np.array([r["actual_ms"]  for r in test])

    def fit_rmse(X_tr, y_tr, X_te, y_te):
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        pred_tr  = X_tr @ coef
        pred_te  = X_te @ coef
        rmse_tr  = float(np.sqrt(np.mean((pred_tr - y_tr) ** 2)))
        rmse_te  = float(np.sqrt(np.mean((pred_te - y_te) ** 2)))
        return coef, rmse_tr, rmse_te, pred_tr, pred_te

    coef_b, rmse_tr_b, rmse_te_b, pred_tr_b, pred_te_b = fit_rmse(X_tr_base, y_tr, X_te_base, y_te)
    coef_e, rmse_tr_e, rmse_te_e, pred_tr_e, pred_te_e = fit_rmse(X_tr_ext,  y_tr, X_te_ext,  y_te)

    print("\n" + "=" * 60)
    print("FEATURE ABLATION — Mixtral-8x7B EP")
    print("=" * 60)
    print(f"  {'Model':<18}  {'Train RMSE':>12}  {'Test RMSE':>12}")
    print(f"  {'-'*18}  {'-'*12}  {'-'*12}")
    print(f"  {'Baseline [2-feat]':<18}  {rmse_tr_b:>11.2f}ms  {rmse_te_b:>11.2f}ms")
    print(f"  {'Extended [6-feat]':<18}  {rmse_tr_e:>11.2f}ms  {rmse_te_e:>11.2f}ms")
    print(f"\n  Reduction from extended: train={rmse_tr_b - rmse_tr_e:+.2f}ms  "
          f"test={rmse_te_b - rmse_te_e:+.2f}ms")
    print(f"\n  If test RMSE barely changes, the variance is irreducible —")
    print(f"  no static feature can capture stochastic all_to_all jitter.")

    # Per-batch output
    all_out = []
    for i, r in enumerate(train):
        all_out.append({**r,
                        "pred_baseline_ms": pred_tr_b[i],
                        "pred_extended_ms": pred_tr_e[i]})
    for i, r in enumerate(test):
        all_out.append({**r,
                        "pred_baseline_ms": pred_te_b[i],
                        "pred_extended_ms": pred_te_e[i]})

    return all_out, {
        "rmse_train_baseline": rmse_tr_b,
        "rmse_test_baseline":  rmse_te_b,
        "rmse_train_extended": rmse_tr_e,
        "rmse_test_extended":  rmse_te_e,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(rows, summary):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "no_feature_helps.csv")
    fields = ["split", "B", "T_in", "sum_n2", "max_n2", "skew", "T_in2",
              "actual_ms", "pred_baseline_ms", "pred_extended_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in rows])
    print(f"\nSaved: {path}")

    import json
    spath = os.path.join(OUT_DIR, "summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {spath}")


# ---------------------------------------------------------------------------
# Worker / Main
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

    warmup(model, vocab_size, rank, N_WARMUP)
    records = run_experiment(model, vocab_size, rank)
    if rank == 0:
        mp_results["records"] = records
    dist.destroy_process_group()


def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, mp_results), nprocs=2, join=True)

    records = mp_results["records"]
    rows, summary = fit_and_report(records)
    save_results(rows, summary)


if __name__ == "__main__":
    main()
