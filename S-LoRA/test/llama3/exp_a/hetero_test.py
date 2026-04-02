#!/usr/bin/env python3
"""
Experiment A (Llama3 control) — Heterogeneous Batch Composition
================================================================
Mirror of test/mixtral/exp2/stress_test.py exp_a(), but for Llama3-8B (dense).

Shows that PrefillExecutionEstimator handles heterogeneous batches accurately
for a dense model because Σn² correctly captures attention cost regardless of
length distribution.  Contrast with Mixtral (MoE) where the same test yields
+244–544% over-prediction errors.

Usage:
    cd S-LoRA
    python test/llama3/exp_a/hetero_test.py
"""

import os, sys, csv, time
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/meta-llama/Meta-Llama-3-8B-HF"
MAX_POOL  = 35_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT  = "29503"

N_WARMUP = 20
N_TRAIN  = 80
N_TEST   = 80

# Same settings as Mixtral Exp A for direct comparison
SETTINGS = [
    {
        "bs": 4, "total": 1024,
        "families": {
            "uniform": [256, 256, 256, 256],
            "bimodal": [64, 64, 448, 448],
            "skewed":  [32, 32, 32, 928],
        },
    },
    {
        "bs": 8, "total": 2048,
        "families": {
            "uniform": [256] * 8,
            "bimodal": [128] * 4 + [384] * 4,
            "skewed":  [32] * 7 + [1824],
        },
    },
]


# ---------------------------------------------------------------------------
# Forward-pass helper
# ---------------------------------------------------------------------------

def run_prefill(model, token_ids_list):
    """Prefill forward pass. Returns (time_s, b_loc, b_seq_len, b_start_loc)."""
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
    return time.perf_counter() - t0, b_loc, b_seq_len, b_start_loc


def warmup(model, vocab_size, n=20):
    print(f"Warming up ({n} batches)...")
    rng = np.random.default_rng(0)
    for _ in range(n):
        batch = [rng.integers(0, vocab_size, size=64) for _ in range(4)]
        run_prefill(model, batch)
    print("Warmup done.\n")


# ---------------------------------------------------------------------------
# Experiment A
# ---------------------------------------------------------------------------

def run_exp_a(model, vocab_size):
    from slora.server.router.tracker import PrefillExecutionEstimator

    results = []
    rng = np.random.default_rng(100)

    for setting in SETTINGS:
        bs, total = setting["bs"], setting["total"]
        fams      = setting["families"]

        print(f"\n[Exp A] bs={bs} total_tokens={total}")

        # Collect timing for all families
        family_times = {name: [] for name in fams}
        for name, lens in fams.items():
            for i in range(N_TRAIN + N_TEST):
                batch = [rng.integers(0, vocab_size, size=l) for l in lens]
                pt, _, _, _ = run_prefill(model, batch)
                family_times[name].append(pt)
                if (i + 1) % 40 == 0:
                    print(f"  [{name}] {i+1}/{N_TRAIN + N_TEST}")

        # Train predictor on uniform family only (same as Mixtral Exp A)
        p_est = PrefillExecutionEstimator()
        p_est.fit(
            inference_only_tokens=[fams["uniform"]] * N_TRAIN,
            inference_only_times=family_times["uniform"][:N_TRAIN],
            coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
        )
        print(f"  Predictor fit_rmse={p_est.fit_rmse*1000:.3f}ms  "
              f"alpha={p_est._params.alpha:.3e}  beta={p_est._params.beta:.3e}")

        for name, lens in fams.items():
            test_times  = family_times[name][N_TRAIN:]
            p_pred      = p_est.predict_inference(lens)
            signed_errs = [(p_pred - a) / a * 100 for a in test_times]
            abs_errs    = [abs(e) for e in signed_errs]
            cv          = float(np.std(test_times) / np.mean(test_times) * 100)
            row = {
                "bs":                  bs,
                "total_tokens":        total,
                "family":              name,
                "sum_n2":              int(sum(l * l for l in lens)),
                "T_in":                int(sum(lens)),
                "mean_ms":             float(np.mean(test_times) * 1000),
                "std_ms":              float(np.std(test_times) * 1000),
                "cv_pct":              cv,
                "pred_ms":             float(p_pred * 1000),
                "mean_signed_err_pct": float(np.mean(signed_errs)),
                "mean_abs_err_pct":    float(np.mean(abs_errs)),
                "max_abs_err_pct":     float(np.max(abs_errs)),
            }
            results.append(row)
            print(f"  [{name:>8}] mean={row['mean_ms']:.2f}ms  cv={cv:.1f}%  "
                  f"signed_err={row['mean_signed_err_pct']:+.1f}%  "
                  f"abs_err={row['mean_abs_err_pct']:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    path   = os.path.join(OUT_DIR, "hetero.csv")
    fields = ["bs", "total_tokens", "family", "sum_n2", "T_in", "mean_ms", "std_ms",
              "cv_pct", "pred_ms", "mean_signed_err_pct", "mean_abs_err_pct", "max_abs_err_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {path}")


def print_summary(results):
    print("\n" + "=" * 65)
    print("EXP A (Llama3 control) — Heterogeneous Batch Predictor Bias")
    print("Compare to Mixtral Exp A: uniform→+5%, bimodal→+63%, skewed→+244%")
    print("=" * 65)
    for r in results:
        print(f"  bs={r['bs']}  {r['family']:>8}  "
              f"mean={r['mean_ms']:7.2f}ms  cv={r['cv_pct']:.1f}%  "
              f"signed_err={r['mean_signed_err_pct']:+6.1f}%  "
              f"abs_err={r['mean_abs_err_pct']:.1f}%")


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

    print(f"Loading model from {MODEL_DIR} ...")
    model = Llama3TpPartModel(tp_rank=0, world_size=1, weight_dir=MODEL_DIR,
                               max_total_token_num=MAX_POOL, mem_adapter_size=0, dummy=False)
    vocab_size = model.config["vocab_size"]
    print(f"Model loaded. vocab_size={vocab_size}")

    warmup(model, vocab_size, N_WARMUP)
    results = run_exp_a(model, vocab_size)
    save_results(results)
    print_summary(results)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
