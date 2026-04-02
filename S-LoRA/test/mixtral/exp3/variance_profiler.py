#!/usr/bin/env python3
"""
Experiment 3 — Irreducible Variance Profiler (Mixtral-8x7B EP)
===============================================================
Same design as test/llama3/exp3/variance_profiler.py, but for Mixtral EP
on 2 GPUs.

Key question: after calibrating the predictor on the same heterogeneous trace
it will be tested on (best-case scenario), what is the residual per-batch
prediction error? Is this variance irreducible — i.e., can no amount of
training data eliminate it?

Log-normal trace: μ=4.5, σ=1.2, clipped to [32, 512] (tighter for MAX_POOL=15000).
SLO thresholds tested: [200, 250, 300, 350, 400ms] (Mixtral batch ~215ms).
SFT budget: 256 tokens added to each batch for gate simulation.

Requires 2 GPUs and Mixtral-8x7B-v0.1 weights.

Usage:
    cd S-LoRA
    python test/mixtral/exp3/variance_profiler.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR  = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
MAX_POOL   = 15_000
OUT_DIR    = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT   = "29505"

BS             = 4
N_REQUESTS     = 2048
N_CALIB        = 256
N_WARMUP_PROF  = 20
SFT_BUDGETS    = [16, 32, 64, 128, 256]

# Same log-normal params as Llama3 but tighter max for memory safety
LN_MU    = 4.5
LN_SIGMA = 1.2
LEN_MIN  = 32
LEN_MAX  = 512   # tighter than Llama3 (1024) due to MAX_POOL=15000

SLO_THRESHOLDS_MS = [200, 250, 300, 350, 400]

N_WARMUP_MODEL = 20


# ---------------------------------------------------------------------------
# Trace generation (same as Llama3 but different LEN_MAX)
# ---------------------------------------------------------------------------

def gen_trace(seed=42):
    rng = np.random.default_rng(seed)
    raw = rng.lognormal(mean=LN_MU, sigma=LN_SIGMA, size=N_REQUESTS)
    lengths = np.clip(raw.astype(int), LEN_MIN, LEN_MAX).tolist()
    rows = []
    t = 0.0
    for i, l in enumerate(lengths):
        rows.append({"timestamp_s": round(t, 3), "prompt_length": l,
                     "max_new_tokens": min(l, 128), "request_id": i})
        t += 0.125
    return rows, lengths


def save_trace(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "trace_hetero.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_s", "prompt_length", "max_new_tokens", "request_id"])
        w.writeheader(); w.writerows(rows)
    return path


def describe_trace(lengths):
    a = np.array(lengths)
    print(f"Trace: n={len(a)}, min={a.min()}, max={a.max()}, "
          f"mean={a.mean():.1f}, median={np.median(a):.1f}, "
          f"p90={np.percentile(a,90):.0f}, p99={np.percentile(a,99):.0f}")


# ---------------------------------------------------------------------------
# Forward-pass helpers
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
        print(f"Warming up ({n} batches)...")
    rng = np.random.default_rng(0)
    for _ in range(n):
        batch = [rng.integers(0, vocab_size, size=64) for _ in range(4)]
        run_prefill(model, batch)
    if rank == 0:
        print("Warmup done.\n")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(model, vocab_size, rank, batches):
    from slora.server.router.tracker import PrefillExecutionEstimator

    rng = np.random.default_rng(999)  # same seed both ranks → same batches

    calib_batches = batches[:N_CALIB]
    prof_batches  = batches[N_CALIB:]

    # --- Phase 1: Calibration ---
    if rank == 0:
        print(f"\n[Phase 1] Calibration: running {len(calib_batches)} batches...")
    calib_tokens = []
    calib_times  = []
    for i, lens in enumerate(calib_batches):
        batch = [rng.integers(0, vocab_size, size=l) for l in lens]
        t = run_prefill(model, batch)
        if rank == 0:
            calib_tokens.append(lens)
            calib_times.append(t)
            if (i + 1) % 64 == 0:
                print(f"  calib {i+1}/{len(calib_batches)}")

    est = None
    if rank == 0:
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
    if rank == 0:
        print(f"\n[Phase 2] Profiling: running {len(prof_batches)} batches "
              f"(first {N_WARMUP_PROF} discarded)...")
    per_batch = []
    for i, lens in enumerate(prof_batches):
        batch = [rng.integers(0, vocab_size, size=l) for l in lens]
        t = run_prefill(model, batch)
        coserve_times = {}
        for budget in SFT_BUDGETS:
            sft_tokens = rng.integers(0, vocab_size, size=budget)
            t_cs = run_prefill(model, batch + [sft_tokens])
            coserve_times[budget] = t_cs
        if rank == 0 and i >= N_WARMUP_PROF:
            pred       = est.predict_inference(lens)
            signed_err = (pred - t) / t * 100
            row = {
                "batch_id":       i,
                "lengths_json":   json.dumps(lens),
                "sum_n2":         int(sum(l*l for l in lens)),
                "T_in":           int(sum(lens)),
                "actual_ms":      float(t * 1000),
                "pred_ms":        float(pred * 1000),
                "signed_err_pct": float(signed_err),
                "abs_err_pct":    float(abs(signed_err)),
            }
            for budget, t_cs in coserve_times.items():
                row[f"actual_coserve_ms_{budget}"] = float(t_cs * 1000)
            per_batch.append(row)
        if rank == 0 and (i + 1) % 64 == 0:
            print(f"  prof {i+1}/{len(prof_batches)}")

    return est, per_batch


def compute_distribution(per_batch, est):
    errs   = np.array([r["abs_err_pct"] for r in per_batch])
    signed = np.array([r["signed_err_pct"] for r in per_batch])
    times  = np.array([r["actual_ms"] for r in per_batch])
    lag1   = float(np.corrcoef(times[:-1], times[1:])[0, 1]) if len(times) > 2 else float("nan")

    return {
        "n_batches":          len(per_batch),
        "mean_ms":            float(np.mean(times)),
        "std_ms":             float(np.std(times)),
        "cv_pct":             float(np.std(times) / np.mean(times) * 100),
        "fit_rmse_ms":        float(est.fit_rmse * 1000),
        "rmse_inflation_pct": float(2 * est.fit_rmse / (np.mean(times) / 1000) * 100),
        "wasted_slack_ms":    float(2 * est.fit_rmse * 1000),
        "mean_signed_err":    float(np.mean(signed)),
        "p50_err":            float(np.percentile(errs, 50)),
        "p75_err":            float(np.percentile(errs, 75)),
        "p90_err":            float(np.percentile(errs, 90)),
        "p95_err":            float(np.percentile(errs, 95)),
        "p99_err":            float(np.percentile(errs, 99)),
        "frac_gt_5pct":       float(np.mean(errs > 5)),
        "frac_gt_10pct":      float(np.mean(errs > 10)),
        "frac_gt_20pct":      float(np.mean(errs > 20)),
        "frac_gt_50pct":      float(np.mean(errs > 50)),
        "lag1_autocorr":      lag1,
    }


def simulate_gate(per_batch, est, slo_thresholds_ms, sft_budget):
    rows = []
    coserve_key = f"actual_coserve_ms_{sft_budget}"
    for slo_ms in slo_thresholds_ms:
        fp = fn = tp = tn = 0
        admitted = 0
        for r in per_batch:
            lens = json.loads(r["lengths_json"])
            pred_coserve_ms = est.predict_coserving(lens, [sft_budget]) * 1000
            gate_admits = pred_coserve_ms <= slo_ms
            would_fit   = r[coserve_key] <= slo_ms
            if gate_admits:
                admitted += 1
                if not would_fit: fp += 1
                else:             tp += 1
            else:
                if would_fit: fn += 1
                else:         tn += 1
        n = len(per_batch)
        rows.append({
            "sft_budget":       sft_budget,
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
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, batches, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{TCP_PORT}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.mixtral.model import MixtralEPTpPartModel

    if rank == 0:
        print(f"Loading model from {MODEL_DIR} ...")
    model = MixtralEPTpPartModel(tp_rank=rank, world_size=world_size,
                                  weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                  mem_adapter_size=0, dummy=False)
    vocab_size = model.config["vocab_size"]
    if rank == 0:
        print(f"Model loaded. vocab_size={vocab_size}")

    warmup(model, vocab_size, rank, N_WARMUP_MODEL)
    est, per_batch = run_experiment(model, vocab_size, rank, batches)

    if rank == 0:
        mp_results["est_fit_rmse"] = est.fit_rmse
        mp_results["est_alpha"]    = est._params.alpha
        mp_results["est_beta"]     = est._params.beta
        mp_results["est_gamma"]    = est._params.gamma if est._params.gamma is not None else 0.0
        mp_results["est_c"]        = est._params.c if est._params.c is not None else 0.0
        mp_results["per_batch"]    = per_batch

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(per_batch, dist_row, gate_rows_by_budget):
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(OUT_DIR, "per_batch.csv")
    fields = ["batch_id", "lengths_json", "sum_n2", "T_in", "actual_ms", "pred_ms",
              "signed_err_pct", "abs_err_pct"] + [f"actual_coserve_ms_{b}" for b in SFT_BUDGETS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(per_batch)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "error_distribution.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_row.keys())); w.writeheader(); w.writerow(dist_row)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "gate_decisions.csv")
    all_rows = [row for rows in gate_rows_by_budget.values() for row in rows]
    fields = ["sft_budget", "slo_threshold_ms", "n_batches", "fp_count", "fn_count",
              "tp_count", "tn_count", "fp_rate", "fn_rate", "frac_admitted"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    print(f"Saved: {path}")


def print_summary(dist_row, gate_rows_by_budget):
    print("\n" + "=" * 65)
    print("VARIANCE PROFILER — Mixtral-8x7B EP (log-normal real trace)")
    print("=" * 65)
    print(f"  n_batches={dist_row['n_batches']}  "
          f"mean={dist_row['mean_ms']:.2f}ms  cv={dist_row['cv_pct']:.2f}%")
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
    for budget, gate_rows in gate_rows_by_budget.items():
        print(f"\n  Gate decisions (SFT_BUDGET={budget} tokens):")
        print(f"  {'SLO(ms)':>8}  {'FP':>6}  {'FN':>6}  {'fp_count':>8}  {'admitted':>8}")
        for r in gate_rows:
            print(f"  {r['slo_threshold_ms']:>8}  "
                  f"{r['fp_rate']:>6.1%}  {r['fn_rate']:>6.1%}  "
                  f"{r['fp_count']:>8}  {r['frac_admitted']:>8.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    print("Generating log-normal heterogeneous trace...")
    rows, lengths = gen_trace(seed=42)
    save_trace(rows)
    describe_trace(lengths)

    batches = [lengths[i:i+BS] for i in range(0, len(lengths)-BS+1, BS)]
    print(f"Batches: {len(batches)} total  "
          f"({N_CALIB} calibration, {len(batches)-N_CALIB} profiling)\n")

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, batches, mp_results), nprocs=2, join=True)

    per_batch = mp_results["per_batch"]

    # Reconstruct estimator stats for distribution computation
    class _FakeEst:
        fit_rmse = mp_results["est_fit_rmse"]
        class _params:
            pass
    _FakeEst._params.alpha = mp_results["est_alpha"]
    _FakeEst._params.beta  = mp_results["est_beta"]

    # Re-instantiate real estimator for gate simulation (predict_coserving needs it)
    from slora.server.router.tracker import PrefillExecutionEstimator, PrefillParams
    est = PrefillExecutionEstimator()
    est.fit_rmse = mp_results["est_fit_rmse"]
    est._params  = PrefillParams(
        alpha=mp_results["est_alpha"],
        beta=mp_results["est_beta"],
        gamma=mp_results["est_gamma"],
        c=mp_results["est_c"],
    )

    # Save estimator params for offline plotting
    import json as _json
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "estimator_params.json"), "w") as _f:
        _json.dump({
            "alpha":    mp_results["est_alpha"],
            "beta":     mp_results["est_beta"],
            "gamma":    mp_results["est_gamma"],
            "c":        mp_results["est_c"],
            "fit_rmse": mp_results["est_fit_rmse"],
        }, _f, indent=2)

    dist_row = compute_distribution(per_batch, est)
    gate_rows_by_budget = {
        budget: simulate_gate(per_batch, est, SLO_THRESHOLDS_MS, budget)
        for budget in SFT_BUDGETS
    }

    save_results(per_batch, dist_row, gate_rows_by_budget)
    print_summary(dist_row, gate_rows_by_budget)


if __name__ == "__main__":
    main()
