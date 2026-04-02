#!/usr/bin/env python3
"""
Experiment 3-CoServe — Co-serving Gate Validation (Llama3-8B)
==============================================================
Dense-model control companion to:
  test/mixtral/exp3_coserve/coserve_gate_validation.py

Addresses the core limitation of exp3/variance_profiler.py: gate FP/FN
used inference-only time as ground truth, not actual co-serving time.

For each profiling batch, runs TWO forward passes:
  1. Inference-only:  run_prefill(inf_tokens)           → actual_inf_ms
  2. Co-serving:      run_prefill(inf_tokens + sft_seq)  → actual_coserve_ms

Gate decision:   pred_coserve(lens, [SFT_BUDGET]) ≤ SLO  (same as exp3)
Ground truth:    actual_coserve_ms ≤ SLO                  (true co-serving)

Key output:
  - FP_true/FN_true: gate errors against actual co-serving time
  - FP_sim/FN_sim:   gate errors against inference-only proxy (exp3 method)
  - Co-serving overhead distribution: how much does adding 256 SFT tokens cost?

For Llama3 (dense, no all_to_all), overhead should be deterministic and
well-predicted by the linear model. Contrast with Mixtral EP where
all_to_all variance inflates both the overhead distribution and prediction error.

Requires 1 GPU and Meta-Llama-3-8B-HF weights.

Usage:
    cd S-LoRA
    python test/llama3/exp3_coserve/coserve_gate_validation.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR  = "/mnt/nfs/home/ramya/models/meta-llama/Meta-Llama-3-8B-HF"
MAX_POOL   = 35_000
OUT_DIR    = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT   = "29521"

BS             = 4
N_REQUESTS     = 2048
N_CALIB        = 256
N_WARMUP_PROF  = 20
SFT_BUDGET     = 256   # SFT tokens added to each batch for co-serving run

# Same log-normal params as exp3/variance_profiler.py (Llama3 version)
LN_MU    = 4.5
LN_SIGMA = 1.2
LEN_MIN  = 32
LEN_MAX  = 1024

# SLO thresholds for Llama3 (~85ms batches)
SLO_THRESHOLDS_MS = [80, 90, 100, 110, 120, 150, 200]

N_WARMUP_MODEL = 20

SFT_SEED = 7777  # fixed seed for SFT token IDs (same as Mixtral version)


# ---------------------------------------------------------------------------
# Trace generation (identical to exp3)
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
        w = csv.DictWriter(f, fieldnames=["timestamp_s", "prompt_length",
                                          "max_new_tokens", "request_id"])
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
    """Run a prefill forward pass. token_ids_list is a list of 1-D integer arrays."""
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
    from slora.server.router.tracker import PrefillExecutionEstimator

    rng     = np.random.default_rng(999)
    sft_rng = np.random.default_rng(SFT_SEED)

    calib_batches = batches[:N_CALIB]
    prof_batches  = batches[N_CALIB:]

    # -----------------------------------------------------------------------
    # Phase 1: Calibration (inference-only)
    # -----------------------------------------------------------------------
    print(f"\n[Phase 1] Calibration: running {len(calib_batches)} batches "
          "(inference-only, to fit the predictor)...")
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
          f"c={est._params.c*1000:.2f}ms")

    # -----------------------------------------------------------------------
    # Phase 2: Profiling — TWO forward passes per batch
    # -----------------------------------------------------------------------
    print(f"\n[Phase 2] Profiling: {len(prof_batches)} batches "
          f"(first {N_WARMUP_PROF} discarded).")
    print(f"  Each batch: inference-only pass + co-serving pass "
          f"(+{SFT_BUDGET} SFT tokens).")

    per_batch = []
    for i, lens in enumerate(prof_batches):
        inf_tokens = [rng.integers(0, vocab_size, size=l) for l in lens]
        sft_tokens = sft_rng.integers(0, vocab_size, size=SFT_BUDGET)

        # Pass 1: inference-only
        t_inf = run_prefill(model, inf_tokens)

        # Pass 2: co-serving (inference seqs + one SFT seq of SFT_BUDGET tokens)
        t_coserve = run_prefill(model, inf_tokens + [sft_tokens])

        if i >= N_WARMUP_PROF:
            pred_inf     = est.predict_inference(lens)
            pred_coserve = est.predict_coserving(lens, [SFT_BUDGET])
            coserve_err  = (pred_coserve - t_coserve) / t_coserve * 100
            per_batch.append({
                "batch_id":               i,
                "lengths_json":           json.dumps(lens),
                "sum_n2":                 int(sum(l*l for l in lens)),
                "T_in":                   int(sum(lens)),
                "actual_inf_ms":          float(t_inf * 1000),
                "actual_coserve_ms":      float(t_coserve * 1000),
                "coserve_overhead_ms":    float((t_coserve - t_inf) * 1000),
                "pred_inf_ms":            float(pred_inf * 1000),
                "pred_coserve_ms":        float(pred_coserve * 1000),
                "coserve_pred_err_pct":   float(coserve_err),
                "coserve_pred_abserr_pct": float(abs(coserve_err)),
            })

        if (i + 1) % 32 == 0:
            print(f"  prof {i+1}/{len(prof_batches)}")

    return est, per_batch


# ---------------------------------------------------------------------------
# Gate analysis
# ---------------------------------------------------------------------------

def analyze_gate(per_batch, slo_thresholds_ms):
    """True co-serving gate analysis vs simulated (inf-only) proxy."""
    rows = []
    for slo_ms in slo_thresholds_ms:
        fp_co = fn_co = tp_co = tn_co = 0
        fp_sim = fn_sim = tp_sim = tn_sim = 0
        admitted = 0

        for r in per_batch:
            gate_admits  = r["pred_coserve_ms"] <= slo_ms
            fits_coserve = r["actual_coserve_ms"] <= slo_ms
            fits_inf     = r["actual_inf_ms"] <= slo_ms

            if gate_admits:
                admitted += 1
                if not fits_coserve: fp_co  += 1
                else:                tp_co  += 1
                if not fits_inf:     fp_sim += 1
                else:                tp_sim += 1
            else:
                if fits_coserve: fn_co  += 1
                else:            tn_co  += 1
                if fits_inf:     fn_sim += 1
                else:            tn_sim += 1

        n = len(per_batch)
        rows.append({
            "slo_threshold_ms":  slo_ms,
            "n_batches":         n,
            "fp_coserve":        fp_co,
            "fn_coserve":        fn_co,
            "tp_coserve":        tp_co,
            "tn_coserve":        tn_co,
            "fp_rate_coserve":   fp_co / max(fp_co + tn_co, 1),
            "fn_rate_coserve":   fn_co / max(fn_co + tp_co, 1),
            "fp_sim":            fp_sim,
            "fn_sim":            fn_sim,
            "tp_sim":            tp_sim,
            "tn_sim":            tn_sim,
            "fp_rate_sim":       fp_sim / max(fp_sim + tn_sim, 1),
            "fn_rate_sim":       fn_sim / max(fn_sim + tp_sim, 1),
            "frac_admitted":     admitted / n,
        })
    return rows


def compute_overhead_stats(per_batch):
    overhead = np.array([r["coserve_overhead_ms"] for r in per_batch])
    errs     = np.array([r["coserve_pred_abserr_pct"] for r in per_batch])
    signed   = np.array([r["coserve_pred_err_pct"] for r in per_batch])
    return {
        "overhead_mean_ms":     float(np.mean(overhead)),
        "overhead_std_ms":      float(np.std(overhead)),
        "overhead_p50_ms":      float(np.percentile(overhead, 50)),
        "overhead_p90_ms":      float(np.percentile(overhead, 90)),
        "overhead_p99_ms":      float(np.percentile(overhead, 99)),
        "coserve_pred_rmse_ms": float(np.sqrt(np.mean(
            [(r["actual_coserve_ms"] - r["pred_coserve_ms"])**2 for r in per_batch]))),
        "coserve_mean_signed_err": float(np.mean(signed)),
        "coserve_p50_abserr":   float(np.percentile(errs, 50)),
        "coserve_p90_abserr":   float(np.percentile(errs, 90)),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(per_batch, overhead_stats, gate_rows):
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(OUT_DIR, "per_batch.csv")
    fields = ["batch_id", "lengths_json", "sum_n2", "T_in",
              "actual_inf_ms", "actual_coserve_ms", "coserve_overhead_ms",
              "pred_inf_ms", "pred_coserve_ms",
              "coserve_pred_err_pct", "coserve_pred_abserr_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(per_batch)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "overhead_stats.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overhead_stats.keys()))
        w.writeheader(); w.writerow(overhead_stats)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "gate_decisions.csv")
    fields = ["slo_threshold_ms", "n_batches",
              "fp_coserve", "fn_coserve", "tp_coserve", "tn_coserve",
              "fp_rate_coserve", "fn_rate_coserve",
              "fp_sim", "fn_sim", "tp_sim", "tn_sim",
              "fp_rate_sim", "fn_rate_sim",
              "frac_admitted"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(gate_rows)
    print(f"Saved: {path}")


def print_summary(per_batch, overhead_stats, gate_rows, fit_rmse_ms):
    inf_times = np.array([r["actual_inf_ms"] for r in per_batch])
    co_times  = np.array([r["actual_coserve_ms"] for r in per_batch])

    print("\n" + "=" * 72)
    print("CO-SERVING GATE VALIDATION — Llama3-8B (log-normal trace)")
    print("=" * 72)
    print(f"  n_batches={len(per_batch)}  SFT_BUDGET={SFT_BUDGET} tokens")
    print(f"  Inference-only: mean={np.mean(inf_times):.2f}ms  "
          f"std={np.std(inf_times):.2f}ms  "
          f"cv={np.std(inf_times)/np.mean(inf_times)*100:.1f}%")
    print(f"  Co-serving:     mean={np.mean(co_times):.2f}ms  "
          f"std={np.std(co_times):.2f}ms  "
          f"cv={np.std(co_times)/np.mean(co_times)*100:.1f}%")
    print(f"  SFT overhead:   mean={overhead_stats['overhead_mean_ms']:.2f}ms  "
          f"p90={overhead_stats['overhead_p90_ms']:.2f}ms  "
          f"p99={overhead_stats['overhead_p99_ms']:.2f}ms")
    print(f"  Predictor fit_rmse={fit_rmse_ms:.2f}ms  "
          f"Co-serving pred RMSE={overhead_stats['coserve_pred_rmse_ms']:.2f}ms")
    print(f"  Co-serving pred signed err={overhead_stats['coserve_mean_signed_err']:+.1f}%  "
          f"p50_abserr={overhead_stats['coserve_p50_abserr']:.1f}%  "
          f"p90_abserr={overhead_stats['coserve_p90_abserr']:.1f}%")

    print(f"\n  Gate decisions — TRUE (co-serving) vs SIMULATED (inf-only proxy):")
    print(f"  {'SLO(ms)':>8}  {'FP_true':>8}  {'FN_true':>8}  "
          f"{'FP_sim':>8}  {'FN_sim':>8}  {'admitted':>9}")
    for r in gate_rows:
        print(f"  {r['slo_threshold_ms']:>8}  "
              f"{r['fp_rate_coserve']:>8.1%}  {r['fn_rate_coserve']:>8.1%}  "
              f"{r['fp_rate_sim']:>8.1%}  {r['fn_rate_sim']:>8.1%}  "
              f"{r['frac_admitted']:>9.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    print("Generating log-normal heterogeneous trace...")
    rows, lengths = gen_trace(seed=42)
    save_trace(rows)
    describe_trace(lengths)

    batches = [lengths[i:i+BS] for i in range(0, len(lengths)-BS+1, BS)]
    print(f"Batches: {len(batches)} total  "
          f"({N_CALIB} calibration, {len(batches)-N_CALIB} profiling)\n")
    print("Note: profiling phase runs 2 forward passes per batch "
          "(inference-only + co-serving). Expect ~2× runtime vs exp3.\n")

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
    overhead_stats = compute_overhead_stats(per_batch)
    gate_rows      = analyze_gate(per_batch, SLO_THRESHOLDS_MS)

    save_results(per_batch, overhead_stats, gate_rows)
    print_summary(per_batch, overhead_stats, gate_rows,
                  fit_rmse_ms=est.fit_rmse * 1000)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
