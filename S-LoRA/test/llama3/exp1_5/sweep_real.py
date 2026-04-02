#!/usr/bin/env python3
"""
Experiment 1.5 — Extended Uniform Config Sweep (Llama3, REAL WEIGHTS)
======================================================================
Same grid as sweep.py but uses the actual Meta-Llama-3-8B weights instead
of a tiny dummy model.  Results go to results_real/ to avoid overwriting
the dummy-model baseline.

Usage:
    cd S-LoRA
    python test/llama3/exp1_5/sweep_real.py
"""

import os, sys, json, csv, time
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR   = "/mnt/nfs/home/ramya/models/meta-llama/Meta-Llama-3-8B-HF"

CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZES     = [1, 2, 4, 8, 16, 32, 64, 128]
MAX_TOKENS      = 35_000    # skip if bs * sl exceeds this
MAX_POOL        = 35_000    # KV cache pool size — SFT buffers in MemoryAllocator are tot_size*10*layers,
                            # so 35K fits in ~70GB at init; free_all() is used for per-batch resets

N_WARMUP = 20
N_TRAIN  = 80
N_TEST   = 80

OUT_DIR = os.path.join(os.path.dirname(__file__), "results_real")


# ---------------------------------------------------------------------------
# Forward-pass helpers  (identical to sweep.py)
# ---------------------------------------------------------------------------

def make_batch(seed, bs, sl, vocab_size):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, vocab_size, size=sl) for _ in range(bs)]


def run_prefill_then_decode(model, token_ids_list, vocab_size):
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
        b_start_loc[i] = b_start_loc[i-1] + seq_lens[i-1]
    b_loc = torch.zeros(bs, max_len, dtype=torch.long, device="cuda")

    # --- Prefill ---
    # Use free_all() to reset only the logical KV state, not reallocate tensors.
    # reset_all_pool() doubles peak memory by creating new tensors before freeing old ones.
    model.mem_manager.free_all()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_p, max_len_in_batch=max_len,
                      input_ids=input_ids_p, b_loc=b_loc, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len, is_prefill=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    # --- Decode (one step, KV from prefill still allocated) ---
    b_loc_d     = torch.cat([b_loc, torch.zeros(bs, 1, dtype=torch.long, device="cuda")], dim=1)
    b_seq_len_d = b_seq_len + 1
    total_d     = int(b_seq_len_d.sum().item())
    max_len_d   = int(b_seq_len_d.max().item())
    input_ids_d = torch.randint(0, vocab_size, (bs,), dtype=torch.long, device="cuda")

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_d, max_len_in_batch=max_len_d,
                      input_ids=input_ids_d, b_loc=b_loc_d, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len_d, is_prefill=False)
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t1

    return prefill_time, decode_time


# ---------------------------------------------------------------------------
# Main sweep (single GPU)
# ---------------------------------------------------------------------------

def run_sweep():
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29501",
                                world_size=1, rank=0)

    from slora.models.llama3.model import Llama3TpPartModel

    print(f"Loading model from {MODEL_DIR} ...")
    model = Llama3TpPartModel(
        tp_rank=0, world_size=1,
        weight_dir=MODEL_DIR,
        max_total_token_num=MAX_POOL,
        mem_adapter_size=0,
        dummy=False,
    )
    vocab_size = model.config["vocab_size"]
    print(f"Model loaded. vocab_size={vocab_size}")

    configs = [(bs, sl) for sl in CONTEXT_LENGTHS for bs in BATCH_SIZES
               if bs * sl <= MAX_TOKENS]
    skipped = [(bs, sl) for sl in CONTEXT_LENGTHS for bs in BATCH_SIZES
               if bs * sl > MAX_TOKENS]
    print(f"{len(configs)} configs, {len(skipped)} skipped.")
    print(f"Warming up ({N_WARMUP} batches)...")

    for i in range(N_WARMUP):
        run_prefill_then_decode(model, make_batch(i, 4, 64, vocab_size), vocab_size)

    print("Warmup done. Running sweep...\n")

    from slora.server.router.tracker import PrefillExecutionEstimator, DecodeExecutionEstimator

    exp_results = {}
    global_seed = N_WARMUP

    for cfg_idx, (bs, sl) in enumerate(configs):
        prefill_times, decode_times = [], []

        for i in range(N_TRAIN + N_TEST):
            pt, dt = run_prefill_then_decode(model, make_batch(global_seed + i, bs, sl, vocab_size), vocab_size)
            prefill_times.append(pt)
            decode_times.append(dt)
        global_seed += N_TRAIN + N_TEST

        lens   = [sl] * bs
        sum_n2 = bs * sl * sl
        T_in   = bs * sl
        K_kv   = bs * (sl + 1)

        # --- Fit prefill predictor ---
        p_est = PrefillExecutionEstimator()
        p_est.fit(
            inference_only_tokens=[lens] * N_TRAIN,
            inference_only_times=prefill_times[:N_TRAIN],
            coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
        )
        p_pred = p_est.predict_inference(lens)
        p_test = prefill_times[N_TRAIN:]
        p_errs = [abs(p_pred - a) / a * 100 for a in p_test]

        # --- Fit decode predictor ---
        d_est = DecodeExecutionEstimator()
        d_est.fit(
            total_tokens=[bs * (sl + 1)] * N_TRAIN,
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
            "prefill_mean_ms":      float(np.mean(p_test) * 1000),
            "prefill_std_ms":       float(np.std(p_test) * 1000),
            "prefill_cv_pct":       float(np.std(p_test) / np.mean(p_test) * 100),
            "prefill_p90_ms":       float(np.percentile(p_test, 90) * 1000),
            "prefill_pred_ms":      p_pred * 1000,
            "prefill_mean_err_pct": float(np.mean(p_errs)),
            "prefill_max_err_pct":  float(np.max(p_errs)),
            "prefill_fit_rmse_ms":  float((p_est.fit_rmse or 0) * 1000),
            "decode_mean_ms":       float(np.mean(d_test) * 1000),
            "decode_std_ms":        float(np.std(d_test) * 1000),
            "decode_cv_pct":        float(np.std(d_test) / np.mean(d_test) * 100),
            "decode_p90_ms":        float(np.percentile(d_test, 90) * 1000),
            "decode_pred_ms":       d_pred * 1000,
            "decode_mean_err_pct":  float(np.mean(d_errs)),
            "decode_max_err_pct":   float(np.max(d_errs)),
            "decode_fit_rmse_ms":   float((d_est.fit_rmse or 0) * 1000),
        }

        r = exp_results[label]
        print(f"[{cfg_idx+1:2d}/{len(configs)}] {label} | "
              f"prefill: {r['prefill_mean_ms']:6.2f}±{r['prefill_std_ms']:.2f}ms "
              f"err={r['prefill_mean_err_pct']:.1f}% | "
              f"decode: {r['decode_mean_ms']:6.2f}±{r['decode_std_ms']:.2f}ms "
              f"err={r['decode_mean_err_pct']:.1f}%")

    return exp_results


# ---------------------------------------------------------------------------
# Output  (identical to sweep.py)
# ---------------------------------------------------------------------------

def print_tables(sweep):
    print("\n" + "=" * 90)
    print("PREFILL PREDICTOR — error by (batch_size × context_len)")
    print("Each cell = mean absolute error %. Predictor outputs ONE value per config.")
    print("=" * 90)

    sls = sorted(set(v["seq_len"] for v in sweep.values()))
    bss = sorted(set(v["batch_size"] for v in sweep.values()))

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
    print("CV% OF ACTUAL PREFILL TIMES — variance across configs")
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

    prefill_fields = ["batch_size", "seq_len", "sum_n2", "T_in",
                      "prefill_mean_ms", "prefill_std_ms", "prefill_cv_pct", "prefill_p90_ms",
                      "prefill_pred_ms", "prefill_mean_err_pct", "prefill_max_err_pct", "prefill_fit_rmse_ms"]
    decode_fields  = ["batch_size", "seq_len", "K_kv",
                      "decode_mean_ms", "decode_std_ms", "decode_cv_pct", "decode_p90_ms",
                      "decode_pred_ms", "decode_mean_err_pct", "decode_max_err_pct", "decode_fit_rmse_ms"]

    for fname, fields in [("prefill.csv", prefill_fields), ("decode.csv", decode_fields)]:
        path = os.path.join(out_dir, fname)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            rows = sorted(sweep.values(), key=lambda r: (r["seq_len"], r["batch_size"]))
            w.writerows(rows)
        print(f"Saved: {path}")

    jpath = os.path.join(out_dir, "sweep.json")
    with open(jpath, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"Saved: {jpath}")


def main():
    if torch.cuda.device_count() < 1:
        print("Need at least 1 GPU"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    sweep = run_sweep()
    print_tables(sweep)
    save_csvs(sweep, OUT_DIR)


if __name__ == "__main__":
    main()
