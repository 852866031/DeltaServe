#!/usr/bin/env python3
"""
Exp 5 — "No Feature Helps" Proof (Llama3-8B, dense control)
=============================================================
Identical design to test/mixtral/exp_no_feature/no_feature_helps.py.
Dense model control: no all_to_all, so variance is hardware-only (~0.5ms).
Both baseline and extended models should show low RMSE, confirming that
for dense models there's nothing to gain — and for Mixtral EP, the gap
between baseline and extended is also near-zero, proving the residual
is irreducible all_to_all jitter rather than a missing feature.

Usage:
    cd S-LoRA
    python test/llama3/exp_no_feature/no_feature_helps.py

Output:
    results/no_feature_helps.csv
    results/summary.json
"""

import os, sys, csv, time, random
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/meta-llama/Meta-Llama-3-8B-HF"
MAX_POOL  = 35_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT  = "29519"

N_WARMUP = 20
N_TRAIN  = 300
N_TEST   = 100

SHAPE_POOL = [
    (1,  [256]),
    (2,  [128, 384]),
    (4,  [64,  64,  64,  64]),
    (4,  [32,  32,  32,  416]),
    (4,  [64,  128, 256, 512]),
    (8,  [64]  * 8),
    (8,  [32,  32,  32,  32, 32,  32,  32,  928]),
    (4,  [512] * 4),
    (2,  [512, 512]),
    (1,  [1024]),
]


def batch_features(seq_lens):
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
    return [feat["sum_n2"], feat["T_in"], 1.0]


def extended_row(feat):
    return [feat["sum_n2"], feat["T_in"], feat["max_n2"],
            feat["skew"],   feat["B"],    feat["T_in2"], 1.0]


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
        batch = [rng.integers(0, vocab_size, size=64) for _ in range(4)]
        run_prefill(model, batch)


def make_batch(vocab_size, rng):
    bs, lens_template = random.choice(SHAPE_POOL)
    token_ids = [rng.integers(0, vocab_size, size=sl).tolist() for sl in lens_template]
    return lens_template, token_ids


def run_experiment(model, vocab_size):
    rng = np.random.default_rng(42)
    random.seed(42)
    N_TOTAL = N_TRAIN + N_TEST
    all_records = []
    print(f"  Running {N_TOTAL} batches ({N_TRAIN} train + {N_TEST} test)...")

    for i in range(N_TOTAL):
        seq_lens, token_ids = make_batch(vocab_size, rng)
        actual_s  = run_prefill(model, token_ids)
        actual_ms = actual_s * 1000
        feat = batch_features(seq_lens)
        feat["actual_ms"] = actual_ms
        feat["split"]     = "train" if i < N_TRAIN else "test"
        all_records.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N_TOTAL}] last actual={actual_ms:.1f}ms")

    return all_records


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
        return coef, float(np.sqrt(np.mean((pred_tr - y_tr)**2))), \
               float(np.sqrt(np.mean((pred_te - y_te)**2))), pred_tr, pred_te

    coef_b, rmse_tr_b, rmse_te_b, pred_tr_b, pred_te_b = fit_rmse(X_tr_base, y_tr, X_te_base, y_te)
    coef_e, rmse_tr_e, rmse_te_e, pred_tr_e, pred_te_e = fit_rmse(X_tr_ext,  y_tr, X_te_ext,  y_te)

    print("\n" + "=" * 60)
    print("FEATURE ABLATION — Llama3-8B (dense, 1 GPU)")
    print("=" * 60)
    print(f"  {'Model':<18}  {'Train RMSE':>12}  {'Test RMSE':>12}")
    print(f"  {'-'*18}  {'-'*12}  {'-'*12}")
    print(f"  {'Baseline [2-feat]':<18}  {rmse_tr_b:>11.2f}ms  {rmse_te_b:>11.2f}ms")
    print(f"  {'Extended [6-feat]':<18}  {rmse_tr_e:>11.2f}ms  {rmse_te_e:>11.2f}ms")
    print(f"\n  Reduction from extended: train={rmse_tr_b - rmse_tr_e:+.2f}ms  "
          f"test={rmse_te_b - rmse_te_e:+.2f}ms")

    all_out = []
    for i, r in enumerate(train):
        all_out.append({**r, "pred_baseline_ms": pred_tr_b[i], "pred_extended_ms": pred_tr_e[i]})
    for i, r in enumerate(test):
        all_out.append({**r, "pred_baseline_ms": pred_te_b[i], "pred_extended_ms": pred_te_e[i]})

    return all_out, {
        "rmse_train_baseline": rmse_tr_b, "rmse_test_baseline":  rmse_te_b,
        "rmse_train_extended": rmse_tr_e, "rmse_test_extended":  rmse_te_e,
    }


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
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {os.path.join(OUT_DIR, 'summary.json')}")


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

    warmup(model, vocab_size, N_WARMUP)
    records = run_experiment(model, vocab_size)
    rows, summary = fit_and_report(records)
    save_results(rows, summary)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
