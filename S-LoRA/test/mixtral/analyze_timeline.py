#!/usr/bin/env python3
"""
Full statistical breakdown of predictor performance on a realistic request timeline.

Runs the timeline experiment (Poisson batch sizes, log-normal seq lengths) and
collects signed errors, actual vs predicted times, and batch features so we can
answer: in practice, how bad is the predictor, and in which direction does it fail?

Usage:
    cd S-LoRA
    python test/mixtral/analyze_timeline.py
"""

import os, sys, json, csv, time, tempfile
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)

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
    "max_position_embeddings": 4096,
}

N_WARMUP      = 30
N_TRAIN       = 300   # more training for a stable fit
N_TEST        = 300   # more test for solid statistics
MAX_POOL      = 8192
VOCAB_SIZE    = 4096

TIMELINE_MEAN_BS  = 4
TIMELINE_MEAN_LEN = 64
TIMELINE_STD_LEN  = 48


# ---------------------------------------------------------------------------
# Helpers (same as main experiment)
# ---------------------------------------------------------------------------

def make_timeline_batch(seed):
    rng = np.random.default_rng(seed)
    bs  = max(1, rng.poisson(TIMELINE_MEAN_BS))
    mu  = np.log(TIMELINE_MEAN_LEN**2 / np.sqrt(TIMELINE_MEAN_LEN**2 + TIMELINE_STD_LEN**2))
    sig = np.sqrt(np.log(1 + (TIMELINE_STD_LEN / TIMELINE_MEAN_LEN)**2))
    lens = np.clip(rng.lognormal(mu, sig, bs).astype(int), 4, 256)
    return [rng.integers(0, VOCAB_SIZE, size=l) for l in lens]


def timed_prefill(model, token_ids_list):
    bs       = len(token_ids_list)
    seq_lens = [len(x) for x in token_ids_list]
    total    = sum(seq_lens)
    max_len  = max(seq_lens)
    flat     = np.concatenate([np.asarray(x, np.int64) for x in token_ids_list])

    input_ids   = torch.from_numpy(flat).cuda()
    b_seq_len   = torch.tensor(seq_lens, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(bs, dtype=torch.long, device="cuda")
    for i in range(1, bs):
        b_start_loc[i] = b_start_loc[i-1] + seq_lens[i-1]
    b_loc = torch.zeros(bs, max_len, dtype=torch.long, device="cuda")

    model.mem_manager.reset_all_pool()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total, max_len_in_batch=max_len,
                      input_ids=input_ids, b_loc=b_loc, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len, is_prefill=True)
    torch.cuda.synchronize(); dist.barrier()
    return time.perf_counter() - t0


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
    if rank == 0:
        print(f"Model ready. Warming up ({N_WARMUP} batches)...")

    for i in range(N_WARMUP):
        timed_prefill(model, make_timeline_batch(i))

    if rank == 0:
        print(f"Collecting {N_TRAIN} train + {N_TEST} test batches...")

    records = []   # (actual_s, batch_ids_list)
    for i in range(N_TRAIN + N_TEST):
        batch = make_timeline_batch(N_WARMUP + i)
        t     = timed_prefill(model, batch)
        records.append((t, batch))
        if rank == 0 and (i+1) % 100 == 0:
            print(f"  {i+1}/{N_TRAIN+N_TEST}")

    if rank == 0:
        from slora.server.router.tracker import PrefillExecutionEstimator

        def feats(batch):
            lens   = [len(x) for x in batch]
            sum_n2 = sum(l**2 for l in lens)
            T_in   = sum(lens)
            return lens, sum_n2, T_in

        train_records = records[:N_TRAIN]
        test_records  = records[N_TRAIN:]

        # --- Fit predictor ---
        est = PrefillExecutionEstimator()
        est.fit(
            inference_only_tokens=[feats(r[1])[0] for r in train_records],
            inference_only_times=[r[0] for r in train_records],
            coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
        )

        # --- Evaluate on test set, collect full per-batch data ---
        rows = []
        for actual_s, batch in test_records:
            lens, sum_n2, T_in = feats(batch)
            bs   = len(batch)
            pred_s = est.predict_inference(lens)
            signed_err_pct = (pred_s - actual_s) / actual_s * 100   # +ve = over-predict
            abs_err_pct    = abs(signed_err_pct)
            rows.append({
                "actual_ms":       actual_s * 1000,
                "pred_ms":         pred_s * 1000,
                "signed_err_pct":  signed_err_pct,
                "abs_err_pct":     abs_err_pct,
                "batch_size":      bs,
                "T_in":            T_in,
                "sum_n2":          sum_n2,
                "mean_seq_len":    T_in / bs,
            })

        results["rows"] = rows
        results["fit_rmse_ms"] = (est.fit_rmse or 0) * 1000
        results["params"] = {k: getattr(est._params, k) for k in ("alpha","beta","gamma","c")}

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(rows, fit_rmse_ms):
    actual  = np.array([r["actual_ms"]      for r in rows])
    pred    = np.array([r["pred_ms"]         for r in rows])
    signed  = np.array([r["signed_err_pct"]  for r in rows])
    abserr  = np.array([r["abs_err_pct"]     for r in rows])
    bs_arr  = np.array([r["batch_size"]      for r in rows])
    T_arr   = np.array([r["T_in"]            for r in rows])

    sep = "=" * 60

    # -----------------------------------------------------------------------
    print(sep)
    print("PREDICTOR STATISTICAL BREAKDOWN — REALISTIC TIMELINE")
    print(sep)
    print(f"  n_train={N_TRAIN}  n_test={N_TEST}  fit_RMSE={fit_rmse_ms:.3f}ms")
    print()

    # -----------------------------------------------------------------------
    print("1. ACTUAL LATENCY DISTRIBUTION")
    print("-" * 40)
    print(f"  mean   {np.mean(actual):.3f} ms")
    print(f"  median {np.median(actual):.3f} ms")
    print(f"  std    {np.std(actual):.3f} ms")
    print(f"  min    {np.min(actual):.3f} ms")
    print(f"  p10    {np.percentile(actual,10):.3f} ms")
    print(f"  p25    {np.percentile(actual,25):.3f} ms")
    print(f"  p75    {np.percentile(actual,75):.3f} ms")
    print(f"  p90    {np.percentile(actual,90):.3f} ms")
    print(f"  p99    {np.percentile(actual,99):.3f} ms")
    print(f"  max    {np.max(actual):.3f} ms")
    print()

    # -----------------------------------------------------------------------
    print("2. SIGNED ERROR DISTRIBUTION  (+ve = over-predict, -ve = under-predict)")
    print("-" * 40)
    over  = signed[signed > 0]
    under = signed[signed < 0]
    print(f"  mean signed error  {np.mean(signed):+.2f}%  "
          f"({'over' if np.mean(signed)>0 else 'under'}-predicting on average)")
    print(f"  median signed err  {np.median(signed):+.2f}%")
    print(f"  std of signed err  {np.std(signed):.2f}%")
    print()
    print(f"  over-predictions   {len(over):3d}/{len(rows)} ({len(over)/len(rows)*100:.1f}%)")
    print(f"    mean over-err    {np.mean(over):+.2f}%  max {np.max(over):+.2f}%")
    print(f"  under-predictions  {len(under):3d}/{len(rows)} ({len(under)/len(rows)*100:.1f}%)")
    print(f"    mean under-err   {np.mean(under):+.2f}%  max {np.min(under):+.2f}%")
    print()

    # -----------------------------------------------------------------------
    print("3. ABSOLUTE ERROR PERCENTILES")
    print("-" * 40)
    thresholds = [5, 10, 15, 20, 30, 50]
    pctiles    = [25, 50, 75, 90, 95, 99]
    for p in pctiles:
        print(f"  p{p:<2}  {np.percentile(abserr,p):.2f}%")
    print()
    print("  Fraction of batches EXCEEDING error threshold:")
    for t in thresholds:
        frac = (abserr > t).mean() * 100
        bar  = "█" * int(frac / 2)
        print(f"    > {t:2d}%  {frac:5.1f}%  {bar}")
    print()

    # -----------------------------------------------------------------------
    print("4. ERROR BREAKDOWN BY BATCH SIZE")
    print("-" * 40)
    print(f"  {'bs':>4}  {'n':>4}  {'mean actual':>12}  {'mean |err|':>10}  "
          f"{'bias':>8}  {'p90 |err|':>10}")
    for bs_val in sorted(set(bs_arr)):
        mask = bs_arr == bs_val
        if mask.sum() < 3:
            continue
        print(f"  {bs_val:>4}  {mask.sum():>4}  "
              f"{np.mean(actual[mask]):>10.2f}ms  "
              f"{np.mean(abserr[mask]):>9.2f}%  "
              f"{np.mean(signed[mask]):>+7.2f}%  "
              f"{np.percentile(abserr[mask],90):>9.2f}%")
    print()

    # -----------------------------------------------------------------------
    print("5. ERROR BREAKDOWN BY T_in BINS (total tokens in batch)")
    print("-" * 40)
    bins = [0, 64, 128, 256, 512, 1024, 9999]
    labels = ["[1,64]", "[65,128]", "[129,256]", "[257,512]", "[513,1024]", "[1025+]"]
    print(f"  {'T_in range':>12}  {'n':>4}  {'mean actual':>12}  {'mean |err|':>10}  "
          f"{'bias':>8}  {'p90 |err|':>10}")
    for lo, hi, lab in zip(bins, bins[1:], labels):
        mask = (T_arr > lo) & (T_arr <= hi)
        if mask.sum() < 2:
            continue
        print(f"  {lab:>12}  {mask.sum():>4}  "
              f"{np.mean(actual[mask]):>10.2f}ms  "
              f"{np.mean(abserr[mask]):>9.2f}%  "
              f"{np.mean(signed[mask]):>+7.2f}%  "
              f"{np.percentile(abserr[mask],90):>9.2f}%")
    print()

    # -----------------------------------------------------------------------
    print("6. PRACTICAL SLO IMPACT")
    print("-" * 40)
    print("  The predictor gates FT token admission via check_will_starve().")
    print("  Over-prediction → FT throttled unnecessarily (lost throughput).")
    print("  Under-prediction → FT admitted → prefill runs long → TTFT SLO miss.")
    print()
    # Simulate: suppose SLO budget leaves 2ms of headroom above predicted time.
    # Under-prediction by > 2ms = SLO violation. Over-prediction = wasted FT budget.
    for headroom_ms in [1.0, 2.0, 5.0]:
        excess   = pred - actual         # ms; +ve = over, -ve = under
        violated = (excess < -headroom_ms).mean() * 100   # under-predict by > headroom
        wasted   = (excess > headroom_ms).mean() * 100    # over-predict by > headroom
        print(f"  headroom = {headroom_ms:.0f}ms:")
        print(f"    SLO violations (under-predict > {headroom_ms:.0f}ms): {violated:.1f}% of batches")
        print(f"    FT over-throttled (over-predict > {headroom_ms:.0f}ms): {wasted:.1f}% of batches")
    print()

    # -----------------------------------------------------------------------
    print("7. WORST MISPREDICTIONS (top 10 by absolute error)")
    print("-" * 40)
    top_idx = np.argsort(abserr)[-10:][::-1]
    print(f"  {'actual ms':>10}  {'pred ms':>9}  {'err%':>7}  {'bias':>8}  "
          f"{'bs':>4}  {'T_in':>6}")
    for i in top_idx:
        r = rows[i]
        print(f"  {r['actual_ms']:>10.3f}  {r['pred_ms']:>9.3f}  "
              f"{r['abs_err_pct']:>7.1f}  {r['signed_err_pct']:>+7.1f}%  "
              f"{r['batch_size']:>4}  {r['T_in']:>6}")
    print()

    # -----------------------------------------------------------------------
    print("8. SUMMARY VERDICT")
    print("-" * 40)
    bias = np.mean(signed)
    direction = "OVER-predicts" if bias > 0 else "UNDER-predicts"
    print(f"  Mean error:     {np.mean(abserr):.1f}%")
    print(f"  Median error:   {np.median(abserr):.1f}%")
    print(f"  p90 error:      {np.percentile(abserr,90):.1f}%")
    print(f"  Systematic bias:{bias:+.1f}% ({direction} on average)")
    print()
    if abs(bias) > 5:
        print(f"  Predictor is systematically biased ({bias:+.1f}%) — "
              f"this is consistent across batches, not just noise.")
    else:
        print(f"  Predictor has low systematic bias ({bias:+.1f}%) — "
              f"errors are roughly zero-mean but with high variance.")
    print(f"  Even the median batch gets {np.median(abserr):.1f}% wrong.")
    print(f"  1 in 10 batches (p90) gets {np.percentile(abserr,90):.1f}% wrong.")
    print()


def save_csv(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "timeline_analysis.csv")
    fields = ["actual_ms","pred_ms","signed_err_pct","abs_err_pct",
              "batch_size","T_in","sum_n2","mean_seq_len"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Per-batch CSV saved to: {path}")


def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)

    config_dir = tempfile.mkdtemp(prefix="mixtral_dummy_")
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f)

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(worker, args=(2, config_dir, results), nprocs=2, join=True)

    rows = results["rows"]
    fit_rmse_ms = results["fit_rmse_ms"]

    analyze(rows, fit_rmse_ms)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    save_csv(rows, out_dir)


if __name__ == "__main__":
    main()
