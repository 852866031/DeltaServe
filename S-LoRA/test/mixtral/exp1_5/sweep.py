#!/usr/bin/env python3
"""
Experiment 1.5 — Extended Uniform Config Sweep
===============================================
Tests PrefillExecutionEstimator AND DecodeExecutionEstimator across:
  - Context lengths: 256, 512, 1024, 2048, 4096, 8192
  - Batch sizes:     1, 2, 4, 8, 16, 32, 64, 128

For each (batch_size, context_len) pair:
  1. Run N prefill passes → different input_ids each time → different routing → time varies
  2. Immediately follow each prefill with one decode step (KV cache intact)
  3. Fit PrefillExecutionEstimator on train prefill times, evaluate on test
  4. Fit DecodeExecutionEstimator  on train decode  times, evaluate on test

Since all batches within a config have IDENTICAL predictor features (same Σn², T_in,
B, K), the predictor outputs ONE number for the whole test set. Any spread in actual
times is purely from expert routing variance — irreducible from the predictor's view.

Skips combos where bs * sl > MAX_TOKENS (memory limit).

Usage:
    cd S-LoRA
    python test/mixtral/exp1_5/sweep.py
"""

import os, sys, json, csv, time, tempfile
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ---------------------------------------------------------------------------
# Config — larger max_position_embeddings to support sl=8192
# ---------------------------------------------------------------------------
TINY_CONFIG = {
    "model_type": "mixtral",
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 4,
    "vocab_size": 4096,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 8192,
}

CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZES     = [1, 2, 4, 8, 16, 32, 64, 128]
MAX_TOKENS      = 600_000   # skip if bs * sl exceeds this
MAX_POOL        = 700_000   # KV cache pool size (tokens)

N_WARMUP = 20
N_TRAIN  = 80
N_TEST   = 80

VOCAB_SIZE = TINY_CONFIG["vocab_size"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------

def make_batch(seed, bs, sl):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, VOCAB_SIZE, size=sl) for _ in range(bs)]


def run_prefill_then_decode(model, token_ids_list):
    """
    Run one prefill pass, then immediately one decode step on the same KV cache.
    Returns (prefill_time_s, decode_time_s).

    The KV cache is reset before the prefill and left allocated through the decode.
    """
    bs       = len(token_ids_list)
    seq_lens = [len(x) for x in token_ids_list]
    sl       = seq_lens[0]   # uniform within a batch
    total_p  = sum(seq_lens)
    max_len  = max(seq_lens)

    flat        = np.concatenate([np.asarray(x, np.int64) for x in token_ids_list])
    input_ids_p = torch.from_numpy(flat).cuda()
    b_seq_len   = torch.tensor(seq_lens, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(bs, dtype=torch.long, device="cuda")
    for i in range(1, bs):
        b_start_loc[i] = b_start_loc[i-1] + seq_lens[i-1]
    b_loc = torch.zeros(bs, max_len, dtype=torch.long, device="cuda")

    # --- Prefill ---
    model.mem_manager.reset_all_pool()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_p, max_len_in_batch=max_len,
                      input_ids=input_ids_p, b_loc=b_loc, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len, is_prefill=True)
    torch.cuda.synchronize(); dist.barrier()
    prefill_time = time.perf_counter() - t0

    # --- Decode (one step, KV from prefill still allocated) ---
    # b_loc now has the prefill KV indices; extend by 1 col for the new token slot
    b_loc_d      = torch.cat([b_loc, torch.zeros(bs, 1, dtype=torch.long, device="cuda")], dim=1)
    b_seq_len_d  = b_seq_len + 1
    total_d      = int(b_seq_len_d.sum().item())
    max_len_d    = int(b_seq_len_d.max().item())
    input_ids_d  = torch.randint(0, VOCAB_SIZE, (bs,), dtype=torch.long, device="cuda")

    torch.cuda.synchronize(); dist.barrier()
    t1 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_d, max_len_in_batch=max_len_d,
                      input_ids=input_ids_d, b_loc=b_loc_d, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len_d, is_prefill=False)
    torch.cuda.synchronize(); dist.barrier()
    decode_time = time.perf_counter() - t1

    return prefill_time, decode_time


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, config_dir, results):
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29500",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.mixtral.model import MixtralEPTpPartModel
    model = MixtralEPTpPartModel(tp_rank=rank, world_size=world_size,
                                  weight_dir=config_dir, max_total_token_num=MAX_POOL,
                                  mem_adapter_size=0, dummy=True)

    configs = [(bs, sl) for sl in CONTEXT_LENGTHS for bs in BATCH_SIZES
               if bs * sl <= MAX_TOKENS]

    if rank == 0:
        skipped = [(bs, sl) for sl in CONTEXT_LENGTHS for bs in BATCH_SIZES
                   if bs * sl > MAX_TOKENS]
        print(f"Model ready. {len(configs)} configs, {len(skipped)} skipped.")
        print(f"Warming up ({N_WARMUP} batches)...")

    for i in range(N_WARMUP):
        run_prefill_then_decode(model, make_batch(i, 4, 64))

    if rank == 0:
        print("Warmup done. Running sweep...\n")

    exp_results = {}
    global_seed = N_WARMUP

    for cfg_idx, (bs, sl) in enumerate(configs):
        prefill_times, decode_times = [], []

        for i in range(N_TRAIN + N_TEST):
            pt, dt = run_prefill_then_decode(model, make_batch(global_seed + i, bs, sl))
            prefill_times.append(pt)
            decode_times.append(dt)
        global_seed += N_TRAIN + N_TEST

        if rank == 0:
            from slora.server.router.tracker import (
                PrefillExecutionEstimator, DecodeExecutionEstimator
            )

            lens = [sl] * bs  # uniform
            sum_n2 = bs * sl * sl
            T_in   = bs * sl
            K_kv   = bs * (sl + 1)   # KV tokens seen during decode step

            # --- Fit prefill predictor ---
            p_est = PrefillExecutionEstimator()
            p_est.fit(
                inference_only_tokens=[lens] * N_TRAIN,
                inference_only_times=prefill_times[:N_TRAIN],
                coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
            )
            p_pred  = p_est.predict_inference(lens)
            p_test  = prefill_times[N_TRAIN:]
            p_errs  = [abs(p_pred - a) / a * 100 for a in p_test]

            # --- Fit decode predictor ---
            d_est = DecodeExecutionEstimator()
            d_est.fit(
                total_tokens=[bs * (sl + 1)] * N_TRAIN,
                batch_sizes=[bs] * N_TRAIN,
                times=decode_times[:N_TRAIN],
            )
            d_pred  = d_est.predict(total_tokens=K_kv, batch_size=bs)
            d_test  = decode_times[N_TRAIN:]
            d_errs  = [abs(d_pred - a) / a * 100 for a in d_test]

            label = f"bs={bs:3d} sl={sl:4d}"
            exp_results[label] = {
                # identifiers
                "batch_size": bs, "seq_len": sl,
                "sum_n2": sum_n2, "T_in": T_in, "K_kv": K_kv,
                # prefill actuals
                "prefill_mean_ms":  float(np.mean(p_test) * 1000),
                "prefill_std_ms":   float(np.std(p_test) * 1000),
                "prefill_cv_pct":   float(np.std(p_test) / np.mean(p_test) * 100),
                "prefill_p90_ms":   float(np.percentile(p_test, 90) * 1000),
                # prefill predictor
                "prefill_pred_ms":       p_pred * 1000,
                "prefill_mean_err_pct":  float(np.mean(p_errs)),
                "prefill_max_err_pct":   float(np.max(p_errs)),
                "prefill_fit_rmse_ms":   float((p_est.fit_rmse or 0) * 1000),
                # decode actuals
                "decode_mean_ms":   float(np.mean(d_test) * 1000),
                "decode_std_ms":    float(np.std(d_test) * 1000),
                "decode_cv_pct":    float(np.std(d_test) / np.mean(d_test) * 100),
                "decode_p90_ms":    float(np.percentile(d_test, 90) * 1000),
                # decode predictor
                "decode_pred_ms":        d_pred * 1000,
                "decode_mean_err_pct":   float(np.mean(d_errs)),
                "decode_max_err_pct":    float(np.max(d_errs)),
                "decode_fit_rmse_ms":    float((d_est.fit_rmse or 0) * 1000),
            }

            r = exp_results[label]
            print(f"[{cfg_idx+1:2d}/{len(configs)}] {label} | "
                  f"prefill: {r['prefill_mean_ms']:6.2f}±{r['prefill_std_ms']:.2f}ms "
                  f"err={r['prefill_mean_err_pct']:.1f}% | "
                  f"decode: {r['decode_mean_ms']:6.2f}±{r['decode_std_ms']:.2f}ms "
                  f"err={r['decode_mean_err_pct']:.1f}%")

    if rank == 0:
        results["sweep"] = exp_results

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_tables(sweep):
    print("\n" + "=" * 90)
    print("PREFILL PREDICTOR — error by (batch_size × context_len)")
    print("Each cell = mean absolute error %. Predictor outputs ONE value per config.")
    print("=" * 90)

    sls = sorted(set(v["seq_len"] for v in sweep.values()))
    bss = sorted(set(v["batch_size"] for v in sweep.values()))

    # Header
    header = f"{'bs\\sl':>8}" + "".join(f"{sl:>10}" for sl in sls)
    print(header)
    for bs in bss:
        row = f"{bs:>8}"
        for sl in sls:
            key = f"bs={bs:3d} sl={sl:4d}"
            if key in sweep:
                row += f"{sweep[key]['prefill_mean_err_pct']:>9.1f}%"
            else:
                row += f"{'skip':>10}"
        print(row)

    print()
    print("=" * 90)
    print("DECODE PREDICTOR — error by (batch_size × context_len)")
    print("=" * 90)
    print(header)
    for bs in bss:
        row = f"{bs:>8}"
        for sl in sls:
            key = f"bs={bs:3d} sl={sl:4d}"
            if key in sweep:
                row += f"{sweep[key]['decode_mean_err_pct']:>9.1f}%"
            else:
                row += f"{'skip':>10}"
        print(row)

    print()
    print("=" * 90)
    print("CV% OF ACTUAL PREFILL TIMES — routing variance across configs")
    print("=" * 90)
    print(header)
    for bs in bss:
        row = f"{bs:>8}"
        for sl in sls:
            key = f"bs={bs:3d} sl={sl:4d}"
            if key in sweep:
                row += f"{sweep[key]['prefill_cv_pct']:>9.1f}%"
            else:
                row += f"{'skip':>10}"
        print(row)


def save_csvs(sweep, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    prefill_fields = ["batch_size","seq_len","sum_n2","T_in",
                      "prefill_mean_ms","prefill_std_ms","prefill_cv_pct","prefill_p90_ms",
                      "prefill_pred_ms","prefill_mean_err_pct","prefill_max_err_pct","prefill_fit_rmse_ms"]
    decode_fields  = ["batch_size","seq_len","K_kv",
                      "decode_mean_ms","decode_std_ms","decode_cv_pct","decode_p90_ms",
                      "decode_pred_ms","decode_mean_err_pct","decode_max_err_pct","decode_fit_rmse_ms"]

    for fname, fields in [("prefill.csv", prefill_fields), ("decode.csv", decode_fields)]:
        path = os.path.join(out_dir, fname)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            rows = sorted(sweep.values(), key=lambda r: (r["seq_len"], r["batch_size"]))
            w.writerows(rows)
        print(f"Saved: {path}")

    # Combined JSON
    jpath = os.path.join(out_dir, "sweep.json")
    with open(jpath, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"Saved: {jpath}")


def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)

    config_dir = tempfile.mkdtemp(prefix="mixtral_exp15_")
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f)

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(worker, args=(2, config_dir, results), nprocs=2, join=True)

    sweep = results["sweep"]
    print_tables(sweep)
    save_csvs(sweep, OUT_DIR)


if __name__ == "__main__":
    main()
