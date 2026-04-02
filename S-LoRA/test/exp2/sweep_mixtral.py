#!/usr/bin/env python3
"""
Exp 2 — Predictor Sweep: Mixtral with REAL Weights
====================================================
Replicates the exp1_5 sweep structure but loads an actual Mixtral-8x7B
checkpoint (dummy=False).  Uses Expert Parallelism on 2 GPUs to match the
predictor-experiment setup.

Key differences from exp1_5/sweep.py
--------------------------------------
* dummy=False  — real checkpoint, real routing logits
* Random token IDs still used so routing variance is the *only* variable
  (content-driven routing experiments are separate)
* N_TRAIN/N_TEST reduced (real model is much slower per batch)
* MAX_TOKENS tightened to avoid OOM on 2×A100-80GB

Usage:
    cd S-LoRA
    python test/exp2/sweep_mixtral.py \\
        --model_dir mistralai/Mixtral-8x7B-v0.1
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
CONTEXT_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096]
BATCH_SIZES     = [1, 2, 4, 8, 16, 32, 64, 128]

# Skip configs whose total token count exceeds this (avoid OOM / very slow runs)
MAX_TOKENS = 200_000
# KV cache pool (tokens).  2 GPUs × 80 GB, model ~94 GB → ~66 GB spare.
# KV slot for Mixtral-8x7B on 2-GPU EP: 32 layers × 4 KV heads × 128 dim × fp16
#   = 32 × 4 × 128 × 2 = 32 768 bytes ≈ 32 KB / token → 2 GB for 65 536 tokens.
MAX_POOL = 80_000

N_WARMUP = 10
N_TRAIN  = 50
N_TEST   = 50

WORLD_SIZE = 2   # EP requires ≥ 2; must divide num_local_experts (8 for 8x7B)

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Forward-pass helpers  (identical logic to exp1_5)
# ---------------------------------------------------------------------------

def make_batch(seed, bs, sl, vocab_size):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, vocab_size, size=sl) for _ in range(bs)]


def run_prefill_then_decode(model, token_ids_list):
    bs       = len(token_ids_list)
    seq_lens = [len(x) for x in token_ids_list]
    sl       = seq_lens[0]
    total_p  = sum(seq_lens)
    max_len  = max(seq_lens)

    flat        = np.concatenate([np.asarray(x, np.int64) for x in token_ids_list])
    input_ids_p = torch.from_numpy(flat).cuda()
    b_seq_len   = torch.tensor(seq_lens, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(bs, dtype=torch.long, device="cuda")
    for i in range(1, bs):
        b_start_loc[i] = b_start_loc[i - 1] + seq_lens[i - 1]
    b_loc = torch.zeros(bs, max_len, dtype=torch.long, device="cuda")

    # Prefill
    model.mem_manager.reset_all_pool()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_p, max_len_in_batch=max_len,
                      input_ids=input_ids_p, b_loc=b_loc,
                      b_start_loc=b_start_loc, b_seq_len=b_seq_len, is_prefill=True)
    torch.cuda.synchronize(); dist.barrier()
    prefill_time = time.perf_counter() - t0

    # Decode (one step)
    b_loc_d     = torch.cat([b_loc, torch.zeros(bs, 1, dtype=torch.long, device="cuda")], dim=1)
    b_seq_len_d = b_seq_len + 1
    total_d     = int(b_seq_len_d.sum().item())
    max_len_d   = int(b_seq_len_d.max().item())
    input_ids_d = torch.randint(0, model.vocab_size, (bs,), dtype=torch.long, device="cuda")

    torch.cuda.synchronize(); dist.barrier()
    t1 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_d, max_len_in_batch=max_len_d,
                      input_ids=input_ids_d, b_loc=b_loc_d,
                      b_start_loc=b_start_loc, b_seq_len=b_seq_len_d, is_prefill=False)
    torch.cuda.synchronize(); dist.barrier()
    decode_time = time.perf_counter() - t1

    return prefill_time, decode_time


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, model_dir, results):
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29503",
                            world_size=WORLD_SIZE, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.mixtral.model import MixtralEPTpPartModel

    if rank == 0:
        print(f"Loading real Mixtral weights from: {model_dir}")

    model = MixtralEPTpPartModel(
        tp_rank=rank, world_size=WORLD_SIZE,
        weight_dir=model_dir,
        max_total_token_num=MAX_POOL,
        mem_adapter_size=0,
        dummy=False,      # ← real weights
    )

    vocab_size = model.vocab_size

    configs = [(bs, sl) for sl in CONTEXT_LENGTHS for bs in BATCH_SIZES
               if bs * sl <= MAX_TOKENS]

    if rank == 0:
        skipped = [(bs, sl) for sl in CONTEXT_LENGTHS for bs in BATCH_SIZES
                   if bs * sl > MAX_TOKENS]
        print(f"Model loaded.  {len(configs)} configs, {len(skipped)} skipped.")
        print(f"Warmup ({N_WARMUP} batches) ...")

    # Warmup
    for i in range(N_WARMUP):
        run_prefill_then_decode(model, make_batch(i, 4, 64, vocab_size))

    if rank == 0:
        print("Warmup done.  Running sweep ...\n")

    exp_results = {}
    global_seed = N_WARMUP

    for cfg_idx, (bs, sl) in enumerate(configs):
        prefill_times, decode_times = [], []

        for i in range(N_TRAIN + N_TEST):
            pt, dt = run_prefill_then_decode(
                model, make_batch(global_seed + i, bs, sl, vocab_size))
            prefill_times.append(pt)
            decode_times.append(dt)
        global_seed += N_TRAIN + N_TEST

        if rank == 0:
            from slora.server.router.tracker import (
                PrefillExecutionEstimator, DecodeExecutionEstimator)

            lens   = [sl] * bs
            sum_n2 = bs * sl * sl
            T_in   = bs * sl
            K_kv   = bs * (sl + 1)

            # Prefill predictor
            p_est = PrefillExecutionEstimator()
            p_est.fit(
                inference_only_tokens=[lens] * N_TRAIN,
                inference_only_times=prefill_times[:N_TRAIN],
                coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
            )
            p_pred = p_est.predict_inference(lens)
            p_test = prefill_times[N_TRAIN:]
            p_errs = [abs(p_pred - a) / a * 100 for a in p_test]

            # Decode predictor
            d_est = DecodeExecutionEstimator()
            d_est.fit(
                total_tokens=[K_kv] * N_TRAIN,
                batch_sizes=[bs] * N_TRAIN,
                times=decode_times[:N_TRAIN],
            )
            d_pred = d_est.predict(total_tokens=K_kv, batch_size=bs)
            d_test = decode_times[N_TRAIN:]
            d_errs = [abs(d_pred - a) / a * 100 for a in d_test]

            label = f"bs={bs:3d} sl={sl:4d}"
            exp_results[label] = {
                "batch_size": bs, "seq_len": sl,
                "sum_n2": sum_n2, "T_in": T_in, "K_kv": K_kv,
                "prefill_mean_ms":     float(np.mean(p_test) * 1000),
                "prefill_std_ms":      float(np.std(p_test) * 1000),
                "prefill_cv_pct":      float(np.std(p_test) / np.mean(p_test) * 100),
                "prefill_p90_ms":      float(np.percentile(p_test, 90) * 1000),
                "prefill_pred_ms":     p_pred * 1000,
                "prefill_mean_err_pct": float(np.mean(p_errs)),
                "prefill_max_err_pct":  float(np.max(p_errs)),
                "prefill_fit_rmse_ms":  float((p_est.fit_rmse or 0) * 1000),
                "decode_mean_ms":      float(np.mean(d_test) * 1000),
                "decode_std_ms":       float(np.std(d_test) * 1000),
                "decode_cv_pct":       float(np.std(d_test) / np.mean(d_test) * 100),
                "decode_p90_ms":       float(np.percentile(d_test, 90) * 1000),
                "decode_pred_ms":      d_pred * 1000,
                "decode_mean_err_pct": float(np.mean(d_errs)),
                "decode_max_err_pct":  float(np.max(d_errs)),
                "decode_fit_rmse_ms":  float((d_est.fit_rmse or 0) * 1000),
            }

            r = exp_results[label]
            print(f"[{cfg_idx+1:2d}/{len(configs)}] {label} | "
                  f"prefill: {r['prefill_mean_ms']:7.2f}±{r['prefill_std_ms']:.2f}ms "
                  f"err={r['prefill_mean_err_pct']:.1f}% | "
                  f"decode: {r['decode_mean_ms']:6.2f}±{r['decode_std_ms']:.2f}ms "
                  f"err={r['decode_mean_err_pct']:.1f}%")

    if rank == 0:
        results["sweep"] = exp_results

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Tables + CSV output  (same format as exp1_5)
# ---------------------------------------------------------------------------

def print_tables(sweep):
    sls = sorted(set(v["seq_len"]   for v in sweep.values()))
    bss = sorted(set(v["batch_size"] for v in sweep.values()))
    hdr = f"{'bs\\sl':>8}" + "".join(f"{sl:>10}" for sl in sls)

    for title, key in [
        ("PREFILL PREDICTOR — mean abs error %",  "prefill_mean_err_pct"),
        ("DECODE  PREDICTOR — mean abs error %",  "decode_mean_err_pct"),
        ("CV% OF ACTUAL PREFILL TIMES",           "prefill_cv_pct"),
    ]:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(hdr)
        for bs in bss:
            row = f"{bs:>8}"
            for sl in sls:
                k = f"bs={bs:3d} sl={sl:4d}"
                row += f"{sweep[k][key]:>9.1f}%" if k in sweep else f"{'skip':>10}"
            print(row)


def save_results(sweep, out_dir, model_tag):
    os.makedirs(out_dir, exist_ok=True)
    prefill_fields = [
        "batch_size", "seq_len", "sum_n2", "T_in",
        "prefill_mean_ms", "prefill_std_ms", "prefill_cv_pct", "prefill_p90_ms",
        "prefill_pred_ms", "prefill_mean_err_pct", "prefill_max_err_pct", "prefill_fit_rmse_ms",
    ]
    decode_fields = [
        "batch_size", "seq_len", "K_kv",
        "decode_mean_ms", "decode_std_ms", "decode_cv_pct", "decode_p90_ms",
        "decode_pred_ms", "decode_mean_err_pct", "decode_max_err_pct", "decode_fit_rmse_ms",
    ]
    rows = sorted(sweep.values(), key=lambda r: (r["seq_len"], r["batch_size"]))
    for fname, fields in [
        (f"{model_tag}_prefill.csv", prefill_fields),
        (f"{model_tag}_decode.csv",  decode_fields),
    ]:
        path = os.path.join(out_dir, fname)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {path}")

    jpath = os.path.join(out_dir, f"{model_tag}_sweep.json")
    with open(jpath, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"Saved: {jpath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="mistralai/Mixtral-8x7B-v0.1",
                        help="HuggingFace repo id or local path to Mixtral checkpoint")
    args = parser.parse_args()

    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"Need {WORLD_SIZE} GPUs, found {torch.cuda.device_count()}"); sys.exit(1)

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(worker, args=(args.model_dir, results), nprocs=WORLD_SIZE, join=True)

    sweep = results["sweep"]
    print_tables(sweep)
    save_results(sweep, OUT_DIR, model_tag="mixtral_real")
    print("\nDone.")


if __name__ == "__main__":
    main()
