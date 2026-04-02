#!/usr/bin/env python3
"""
Experiment 3 — Irreducible Variance Profiler (Llama3-8B)
=========================================================
Measures per-batch prediction error distribution and gate decision accuracy
on a log-normal heterogeneous trace matching real LLM serving workloads.

Complements test/mixtral/exp3/variance_profiler.py (same design, 2-GPU EP).

Design:
  Step 0: Generate log-normal trace (N=2048 requests, groups of bs=4 → 512 batches)
  Phase 1: Calibration — train predictor on first 256 batches (heterogeneous)
  Phase 2: Profiling — run remaining 236 batches (first 20 discarded as warmup)
           Record per-batch: actual_ms, pred_ms, signed_err_pct
  Phase 3: Gate simulation — for each profiling batch, simulate check_will_starve()
           with SFT budget M=256 tokens at various SLO thresholds

Output:
  results/trace_hetero.csv   — the generated request trace
  results/per_batch.csv      — per-batch timing and error
  results/error_distribution.csv — aggregate stats, percentiles, fractions
  results/gate_decisions.csv — FP/FN rates at various SLO thresholds

Usage:
    cd S-LoRA
    python test/llama3/exp3/variance_profiler.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR  = "/mnt/nfs/home/ramya/models/meta-llama/Meta-Llama-3-8B-HF"
MAX_POOL   = 35_000
OUT_DIR    = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT   = "29503"

BS             = 4          # requests per batch
N_REQUESTS     = 2048       # total trace length
N_CALIB        = 256        # calibration batches (first half)
N_WARMUP_PROF  = 20         # per-family warmup discarded during profiling phase
SFT_BUDGET     = 256        # SFT tokens added for gate simulation

# Log-normal params matching real LLM serving workloads
# (Azure GPT trace: median ~100, mean ~200, p90 ~450 tokens)
LN_MU    = 4.5
LN_SIGMA = 1.2
LEN_MIN  = 32
LEN_MAX  = 1024

# SLO thresholds (ms) — range from tight to generous for Llama3 (~85ms batches)
SLO_THRESHOLDS_MS = [80, 90, 100, 110, 120, 150, 200]

N_WARMUP_MODEL = 20


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------

def gen_trace(seed=42):
    """Generate log-normal heterogeneous request trace."""
    rng = np.random.default_rng(seed)
    raw = rng.lognormal(mean=LN_MU, sigma=LN_SIGMA, size=N_REQUESTS)
    lengths = np.clip(raw.astype(int), LEN_MIN, LEN_MAX).tolist()
    rows = []
    t = 0.0
    for i, l in enumerate(lengths):
        rows.append({
            "timestamp_s":   round(t, 3),
            "prompt_length": l,
            "max_new_tokens": min(l, 128),
            "request_id":    i,
        })
        t += 0.125  # 8 req/s arrival rate
    return rows, lengths


def save_trace(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "trace_hetero.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_s", "prompt_length", "max_new_tokens", "request_id"])
        w.writeheader()
        w.writerows(rows)
    return path


def describe_trace(lengths):
    a = np.array(lengths)
    print(f"Trace: n={len(a)}, min={a.min()}, max={a.max()}, "
          f"mean={a.mean():.1f}, median={np.median(a):.1f}, "
          f"p90={np.percentile(a,90):.0f}, p99={np.percentile(a,99):.0f}")


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
    print(f"Warming up ({n} batches)...")
    rng = np.random.default_rng(0)
    for _ in range(n):
        batch = [rng.integers(0, vocab_size, size=64) for _ in range(4)]
        run_prefill(model, batch)
    print("Warmup done.\n")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(model, vocab_size, batches):
    """
    batches: list of lists-of-lengths, one entry per batch.
    Returns calibration estimator + profiling results.
    """
    from slora.server.router.tracker import PrefillExecutionEstimator

    rng = np.random.default_rng(999)

    calib_batches = batches[:N_CALIB]
    prof_batches  = batches[N_CALIB:]

    # --- Phase 1: Calibration ---
    print(f"\n[Phase 1] Calibration: running {len(calib_batches)} batches...")
    calib_tokens = []
    calib_times  = []
    for i, lens in enumerate(calib_batches):
        batch = [rng.integers(0, vocab_size, size=l) for l in lens]
        t = run_prefill(model, batch)
        calib_tokens.append(lens)
        calib_times.append(t)
        if (i + 1) % 64 == 0:
            print(f"  calib {i+1}/{len(calib_batches)}")

    est = PrefillExecutionEstimator()
    est.fit(
        inference_only_tokens=calib_tokens,
        inference_only_times=calib_times,
        coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
    )
    print(f"  Calibration fit: fit_rmse={est.fit_rmse*1000:.3f}ms  "
          f"alpha={est._params.alpha:.3e}  beta={est._params.beta:.3e}  "
          f"wasted_slack={2*est.fit_rmse*1000:.1f}ms")

    # --- Phase 2: Profiling ---
    print(f"\n[Phase 2] Profiling: running {len(prof_batches)} batches "
          f"(first {N_WARMUP_PROF} discarded)...")
    per_batch = []
    for i, lens in enumerate(prof_batches):
        batch = [rng.integers(0, vocab_size, size=l) for l in lens]
        t = run_prefill(model, batch)
        sft_tokens = rng.integers(0, vocab_size, size=SFT_BUDGET)
        t_coserve = run_prefill(model, batch + [sft_tokens])
        pred = est.predict_inference(lens)
        if i >= N_WARMUP_PROF:
            signed_err = (pred - t) / t * 100
            per_batch.append({
                "batch_id":         i,
                "lengths_json":     json.dumps(lens),
                "sum_n2":           int(sum(l*l for l in lens)),
                "T_in":             int(sum(lens)),
                "actual_ms":        float(t * 1000),
                "actual_coserve_ms": float(t_coserve * 1000),
                "pred_ms":          float(pred * 1000),
                "signed_err_pct":   float(signed_err),
                "abs_err_pct":      float(abs(signed_err)),
            })
        if (i + 1) % 64 == 0:
            print(f"  prof {i+1}/{len(prof_batches)}")

    return est, per_batch


def compute_distribution(per_batch, est):
    """Compute aggregate error distribution metrics."""
    errs    = np.array([r["abs_err_pct"] for r in per_batch])
    signed  = np.array([r["signed_err_pct"] for r in per_batch])
    times   = np.array([r["actual_ms"] for r in per_batch])

    # Lag-1 autocorrelation of actual times
    if len(times) > 2:
        lag1 = float(np.corrcoef(times[:-1], times[1:])[0, 1])
    else:
        lag1 = float("nan")

    wasted_slack_ms = 2 * est.fit_rmse * 1000

    return {
        "n_batches":         len(per_batch),
        "mean_ms":           float(np.mean(times)),
        "std_ms":            float(np.std(times)),
        "cv_pct":            float(np.std(times) / np.mean(times) * 100),
        "fit_rmse_ms":       float(est.fit_rmse * 1000),
        "rmse_inflation_pct": float(2 * est.fit_rmse / (np.mean(times) / 1000) * 100),
        "wasted_slack_ms":   float(wasted_slack_ms),
        "mean_signed_err":   float(np.mean(signed)),
        "p50_err":           float(np.percentile(errs, 50)),
        "p75_err":           float(np.percentile(errs, 75)),
        "p90_err":           float(np.percentile(errs, 90)),
        "p95_err":           float(np.percentile(errs, 95)),
        "p99_err":           float(np.percentile(errs, 99)),
        "frac_gt_5pct":      float(np.mean(errs > 5)),
        "frac_gt_10pct":     float(np.mean(errs > 10)),
        "frac_gt_20pct":     float(np.mean(errs > 20)),
        "frac_gt_50pct":     float(np.mean(errs > 50)),
        "lag1_autocorr":     lag1,
    }


def simulate_gate(per_batch, est, slo_thresholds_ms):
    """Simulate check_will_starve() gate: does adding SFT_BUDGET tokens push over SLO?"""
    rows = []
    for slo_ms in slo_thresholds_ms:
        fp = fn = tp = tn = 0
        admitted = 0
        for r in per_batch:
            lens = json.loads(r["lengths_json"])
            # Predicted coserving time (inference + SFT_BUDGET tokens)
            pred_coserve_ms = est.predict_coserving(lens, [SFT_BUDGET]) * 1000
            # Gate admits SFT if predicted coserving ≤ SLO
            gate_admits = pred_coserve_ms <= slo_ms
            # Ground truth: does actual coserving time (inference + SFT tokens) fit within SLO?
            would_fit = r["actual_coserve_ms"] <= slo_ms
            if gate_admits:
                admitted += 1
                if not would_fit:
                    fp += 1   # admitted but shouldn't have
                else:
                    tp += 1
            else:
                if would_fit:
                    fn += 1   # blocked but would have fit
                else:
                    tn += 1
        n = len(per_batch)
        rows.append({
            "slo_threshold_ms": slo_ms,
            "n_batches":        n,
            "fp_count":         fp,
            "fn_count":         fn,
            "tp_count":         tp,
            "tn_count":         tn,
            "fp_rate":          fp / max(fp + tn, 1),
            "fn_rate":          fn / max(fn + tp, 1),
            "frac_admitted":    admitted / n,
        })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(per_batch, dist_row, gate_rows):
    os.makedirs(OUT_DIR, exist_ok=True)

    # per_batch.csv
    path = os.path.join(OUT_DIR, "per_batch.csv")
    fields = ["batch_id", "lengths_json", "sum_n2", "T_in",
              "actual_ms", "actual_coserve_ms", "pred_ms", "signed_err_pct", "abs_err_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(per_batch)
    print(f"Saved: {path}")

    # error_distribution.csv
    path = os.path.join(OUT_DIR, "error_distribution.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_row.keys()))
        w.writeheader(); w.writerow(dist_row)
    print(f"Saved: {path}")

    # gate_decisions.csv
    path = os.path.join(OUT_DIR, "gate_decisions.csv")
    fields = ["slo_threshold_ms", "n_batches", "fp_count", "fn_count",
              "tp_count", "tn_count", "fp_rate", "fn_rate", "frac_admitted"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(gate_rows)
    print(f"Saved: {path}")


def print_summary(dist_row, gate_rows):
    print("\n" + "=" * 65)
    print("VARIANCE PROFILER — Llama3-8B (log-normal real trace)")
    print("=" * 65)
    print(f"  n_batches={dist_row['n_batches']}  "
          f"mean={dist_row['mean_ms']:.2f}ms  "
          f"cv={dist_row['cv_pct']:.2f}%")
    print(f"  fit_rmse={dist_row['fit_rmse_ms']:.3f}ms  "
          f"rmse_inflation={dist_row['rmse_inflation_pct']:.1f}%  "
          f"wasted_slack={dist_row['wasted_slack_ms']:.1f}ms")
    print(f"  mean_signed_err={dist_row['mean_signed_err']:+.2f}%  "
          f"lag1_autocorr={dist_row['lag1_autocorr']:.3f}")
    print(f"  |err| percentiles: "
          f"p50={dist_row['p50_err']:.1f}%  p90={dist_row['p90_err']:.1f}%  "
          f"p95={dist_row['p95_err']:.1f}%  p99={dist_row['p99_err']:.1f}%")
    print(f"  frac>5%={dist_row['frac_gt_5pct']:.1%}  "
          f"frac>10%={dist_row['frac_gt_10pct']:.1%}  "
          f"frac>20%={dist_row['frac_gt_20pct']:.1%}")
    print(f"\n  Gate decisions (SFT_BUDGET={SFT_BUDGET} tokens):")
    print(f"  {'SLO(ms)':>8}  {'FP':>6}  {'FN':>6}  {'admitted':>8}")
    for r in gate_rows:
        print(f"  {r['slo_threshold_ms']:>8}  "
              f"{r['fp_rate']:>6.1%}  {r['fn_rate']:>6.1%}  "
              f"{r['frac_admitted']:>8.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    # Generate trace
    print("Generating log-normal heterogeneous trace...")
    rows, lengths = gen_trace(seed=42)
    save_trace(rows)
    describe_trace(lengths)

    # Group into batches of BS=4
    batches = [lengths[i:i+BS] for i in range(0, len(lengths)-BS+1, BS)]
    print(f"Batches: {len(batches)} total  "
          f"({N_CALIB} calibration, {len(batches)-N_CALIB} profiling)\n")

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

    warmup(model, vocab_size, N_WARMUP_MODEL)

    est, per_batch = run_experiment(model, vocab_size, batches)
    dist_row  = compute_distribution(per_batch, est)
    gate_rows = simulate_gate(per_batch, est, SLO_THRESHOLDS_MS)

    save_results(per_batch, dist_row, gate_rows)
    print_summary(dist_row, gate_rows)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
