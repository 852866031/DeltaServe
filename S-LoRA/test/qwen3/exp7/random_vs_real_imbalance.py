#!/usr/bin/env python3
"""
Experiment 7 — Random Tokens vs Real Content: Does Higher Imbalance → Slower Batches?
=======================================================================================
Ramya's hypothesis: random tokens cause more routing imbalance, leading to larger
per-layer bubbles, which should show up as (a) higher mean time and (b) the predictor
systematically under-predicting random-token batches.

Design:
  For each T_in in [256, 512, 1024, 2048]:
    - Run N_BATCHES batches of real MMLU content (mixed domains, BS=4, seq_len=T_in//BS)
    - Run N_BATCHES batches of random token IDs (same BS, same seq_lens)
    - Calibrate predictor on first N_CALIB batches of each type separately
    - Report: mean_ms, cv%, mean_imb, max_imb, mean_pred_err for each type

  If bubbles explain timing variance:
    random should have higher mean_imb AND higher mean_ms than real content.
    Predictor should systematically under-predict random batches.

  If communication latency is the driver:
    mean_ms should be similar despite different imbalance.
    Predictor should be equally (in)accurate for both.

Usage:
    cd S-LoRA
    python test/qwen3/exp7/random_vs_real_imbalance.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
MAX_POOL  = 20_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
NCCL_PORT = 29525

BS         = 4
N_BATCHES  = 60    # per T_in per regime (real vs random)
N_CALIB    = 30    # first N_CALIB batches used to calibrate predictor
N_WARMUP   = 20
K_EXPERTS  = 8

# Fixed uniform seq length per T_in (all seqs in a batch are the same length)
T_IN_VALUES = [256, 512, 1024, 2048]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_real_tokens():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading cais/mmlu (all subjects, test split)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"Loading tokenizer from {MODEL_DIR}...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Build a flat pool of tokenized sequences per length bucket
    # Bucket: floor(len / 64) * 64 → 64, 128, 192, 256, 320, 384, 448, 512
    pool_by_len = {}   # target_len -> list of token_id lists (trimmed/padded)
    for target in T_IN_VALUES:
        seq_len = target // BS   # per-sequence length for BS=4 batches
        pool_by_len[target] = []

    for ex in ds:
        text = (f"Question: {ex['question']}\n"
                f"A: {ex['choices'][0]}\nB: {ex['choices'][1]}\n"
                f"C: {ex['choices'][2]}\nD: {ex['choices'][3]}")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for target in T_IN_VALUES:
            seq_len = target // BS
            if len(ids) >= seq_len:
                pool_by_len[target].append(ids[:seq_len])

    for target in T_IN_VALUES:
        print(f"  T_in={target}: {len(pool_by_len[target])} real sequences "
              f"(seq_len={target//BS} each)")

    return pool_by_len


def build_batches(pool_by_len, vocab_size):
    rng = np.random.default_rng(42)
    batches = {}   # T_in -> {"real": list_of_batches, "random": list_of_batches}

    for target in T_IN_VALUES:
        seq_len = target // BS
        real_pool = list(pool_by_len[target])
        rng.shuffle(real_pool)

        # Real content batches (fixed seq_len, uniform composition)
        real_batches = []
        for i in range(0, len(real_pool) - BS + 1, BS):
            real_batches.append(real_pool[i:i + BS])
            if len(real_batches) == N_BATCHES:
                break

        # Random token batches (same seq_lens, uniform random IDs)
        random_batches = []
        for _ in range(N_BATCHES):
            batch = [rng.integers(0, vocab_size, size=seq_len).tolist()
                     for _ in range(BS)]
            random_batches.append(batch)

        batches[target] = {"real": real_batches, "random": random_batches}
        print(f"  T_in={target}: {len(real_batches)} real batches, "
              f"{len(random_batches)} random batches")

    return batches


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


def compute_layer_imbalances(log_entries, T_in):
    """
    Corrected formula: balanced = T_in * K_EXPERTS.
    Returns (per_layer_imbs, mean_imb, max_imb). 1.0=balanced, 2.0=all-on-one-rank.
    """
    if not log_entries:
        return [], 1.0, 1.0
    TK = T_in * K_EXPERTS
    balanced = TK
    if balanced == 0:
        return [], 1.0, 1.0
    layer_imbs = []
    for recv in log_entries:
        r0 = sum(recv)
        r1 = TK - r0
        layer_imbs.append(max(r0, r1) / balanced)
    return layer_imbs, float(np.mean(layer_imbs)), float(np.max(layer_imbs))


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
# Experiment
# ---------------------------------------------------------------------------

def run_regime(model, rank, ep_module, estimator_class,
               batches_list, T_in, regime_label):
    """
    Run N_BATCHES batches of one regime (real or random).
    Calibrate predictor on first N_CALIB, evaluate on remainder.
    Returns list of per-batch dicts.
    """
    from slora.server.router.tracker import PrefillExecutionEstimator

    calib_tokens = []
    calib_times  = []
    results      = []

    for i, batch in enumerate(batches_list):
        if rank == 0:
            ep_module.routing_imbalance_log.clear()
        t = run_prefill(model, batch)

        if rank == 0:
            lens = [len(x) for x in batch]
            T_in_actual = sum(lens)
            log_snap = list(ep_module.routing_imbalance_log)
            per_layer_imbs, mean_imb, max_imb = compute_layer_imbalances(log_snap, T_in_actual)

            if i < N_CALIB:
                calib_tokens.append(lens)
                calib_times.append(t)
                results.append({
                    "phase": "calib", "batch_idx": i,
                    "T_in": T_in_actual, "actual_ms": float(t * 1000),
                    "mean_imb": mean_imb, "max_imb": max_imb,
                    "pred_ms": float("nan"), "signed_err_pct": float("nan"),
                    "per_layer_imbs": per_layer_imbs,
                })
            else:
                # Fit predictor on calibration data at first eval batch
                if len(calib_tokens) == N_CALIB and not hasattr(run_regime, "_est_fitted"):
                    pass  # fitted lazily below

                # Fit on demand
                if not hasattr(run_regime, f"_est_{regime_label}_{T_in}"):
                    est = PrefillExecutionEstimator()
                    est.fit(
                        inference_only_tokens=calib_tokens,
                        inference_only_times=calib_times,
                        coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
                    )
                    setattr(run_regime, f"_est_{regime_label}_{T_in}", est)
                    if rank == 0:
                        print(f"    [{regime_label}/T_in={T_in}] predictor fit: "
                              f"fit_rmse={est.fit_rmse*1000:.2f}ms  "
                              f"alpha={est._params.alpha:.3e}  beta={est._params.beta:.3e}")

                est = getattr(run_regime, f"_est_{regime_label}_{T_in}")
                pred = est.predict_inference(lens)
                signed_err = (pred - t) / t * 100

                results.append({
                    "phase": "eval", "batch_idx": i,
                    "T_in": T_in_actual, "actual_ms": float(t * 1000),
                    "mean_imb": mean_imb, "max_imb": max_imb,
                    "pred_ms": float(pred * 1000),
                    "signed_err_pct": float(signed_err),
                    "per_layer_imbs": per_layer_imbs,
                })

    # Clean up the lazy state attributes for next call
    attr = f"_est_{regime_label}_{T_in}"
    if hasattr(run_regime, attr):
        delattr(run_regime, attr)

    return results


def run_experiment(model, vocab_size, rank, batches):
    import slora.models.qwen3_moe.layer_infer.transformer_layer_infer_ep as ep_module
    from slora.server.router.tracker import PrefillExecutionEstimator

    all_results = []   # list of dicts with T_in, regime, phase, metrics

    for T_in in T_IN_VALUES:
        for regime in ["real", "random"]:
            batches_list = batches[T_in][regime]
            if not batches_list:
                continue
            if rank == 0:
                print(f"\n  [T_in={T_in}, regime={regime}]: {len(batches_list)} batches")

            results = run_regime(model, rank, ep_module, PrefillExecutionEstimator,
                                 batches_list, T_in, regime)
            if rank == 0:
                for r in results:
                    r["T_in_target"] = T_in
                    r["regime"] = regime
                    all_results.append(r)

    return all_results


# ---------------------------------------------------------------------------
# Analysis + output
# ---------------------------------------------------------------------------

def summarize(all_results):
    rows = []
    for T_in in T_IN_VALUES:
        for regime in ["real", "random"]:
            sub = [r for r in all_results
                   if r["T_in_target"] == T_in and r["regime"] == regime
                   and r["phase"] == "eval"]
            if not sub:
                continue
            times = np.array([r["actual_ms"]      for r in sub])
            imbs  = np.array([r["mean_imb"]        for r in sub])
            maxes = np.array([r["max_imb"]         for r in sub])
            errs  = np.array([r["signed_err_pct"]  for r in sub
                              if not np.isnan(r["signed_err_pct"])])

            # Is higher imbalance within this group correlated with slower time?
            corr_imb_time = (float(np.corrcoef(imbs, times)[0, 1])
                             if len(sub) >= 3 else float("nan"))

            rows.append({
                "T_in":            T_in,
                "regime":          regime,
                "n_eval":          len(sub),
                "mean_ms":         float(np.mean(times)),
                "std_ms":          float(np.std(times)),
                "cv_pct":          float(np.std(times) / np.mean(times) * 100),
                "mean_imb":        float(np.mean(imbs)),
                "mean_max_imb":    float(np.mean(maxes)),
                "abs_max_imb":     float(np.max(maxes)),
                "mean_pred_err":   float(np.mean(errs)) if len(errs) else float("nan"),
                "std_pred_err":    float(np.std(errs))  if len(errs) else float("nan"),
                "corr_imb_time":   corr_imb_time,
            })
    return rows


def save_results(all_results, summary_rows):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Full per-batch CSV
    path = os.path.join(OUT_DIR, "per_batch.csv")
    fields = ["T_in_target", "regime", "phase", "batch_idx", "T_in",
              "actual_ms", "mean_imb", "max_imb", "pred_ms", "signed_err_pct"]
    flat = [{k: v for k, v in r.items() if k != "per_layer_imbs"}
            for r in all_results]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(flat)
    print(f"Saved: {path}")

    # Layer timeseries CSV
    path = os.path.join(OUT_DIR, "layer_timeseries.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_in_target", "regime", "batch_idx", "layer_idx",
                    "corrected_imb", "actual_ms"])
        for r in all_results:
            for li, imb in enumerate(r.get("per_layer_imbs", [])):
                w.writerow([r["T_in_target"], r["regime"], r["batch_idx"],
                             li, imb, r["actual_ms"]])
    print(f"Saved: {path}")

    # Summary CSV
    path = os.path.join(OUT_DIR, "summary.csv")
    fields = ["T_in", "regime", "n_eval", "mean_ms", "std_ms", "cv_pct",
              "mean_imb", "mean_max_imb", "abs_max_imb",
              "mean_pred_err", "std_pred_err", "corr_imb_time"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)
    print(f"Saved: {path}")


def print_summary(summary_rows):
    print("\n" + "=" * 115)
    print("EXP 7 — RANDOM TOKENS VS REAL CONTENT: IMBALANCE AND TIMING")
    print("Hypothesis: random tokens → more imbalance → more per-layer bubbles → slower/more variable")
    print("=" * 115)
    print(f"\n  {'T_in':>5}  {'regime':>7}  {'n':>3}  {'mean_ms':>8}  {'cv%':>5}  "
          f"{'mean_imb':>8}  {'max_imb_mean':>12}  {'abs_max_imb':>11}  "
          f"{'pred_err%':>9}  {'corr_imb_t':>10}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*3}  {'-'*8}  {'-'*5}  "
          f"{'-'*8}  {'-'*12}  {'-'*11}  {'-'*9}  {'-'*10}")

    for T_in in T_IN_VALUES:
        sub = [r for r in summary_rows if r["T_in"] == T_in]
        for r in sub:
            pe_str = (f"{r['mean_pred_err']:>+9.2f}" if not np.isnan(r["mean_pred_err"])
                      else "      nan")
            corr_str = (f"{r['corr_imb_time']:>+10.3f}" if not np.isnan(r["corr_imb_time"])
                        else "       nan")
            print(f"  {r['T_in']:>5}  {r['regime']:>7}  {r['n_eval']:>3}  "
                  f"{r['mean_ms']:>8.2f}  {r['cv_pct']:>5.2f}  "
                  f"{r['mean_imb']:>8.4f}  {r['mean_max_imb']:>12.4f}  "
                  f"{r['abs_max_imb']:>11.4f}  {pe_str}  {corr_str}")
        print()

    print("\n  Key questions:")
    print("  1. Does random > real for mean_imb?  (More load imbalance per layer)")
    print("  2. Does random > real for mean_ms?   (Bubbles slow down execution)")
    print("  3. Is pred_err more negative for random?  (Predictor under-predicts imbalanced batches)")
    print("  4. Is corr_imb_time > 0?  (Within each regime, more-imbalanced batches are slower)")

    print("\n  Interpretation:")
    for T_in in T_IN_VALUES:
        sub = {r["regime"]: r for r in summary_rows if r["T_in"] == T_in}
        if "real" not in sub or "random" not in sub:
            continue
        r_real = sub["real"]
        r_rand = sub["random"]
        delta_imb = r_rand["mean_imb"] - r_real["mean_imb"]
        delta_ms  = r_rand["mean_ms"]  - r_real["mean_ms"]
        delta_err = (r_rand["mean_pred_err"] - r_real["mean_pred_err"]
                     if not (np.isnan(r_rand["mean_pred_err"]) or
                             np.isnan(r_real["mean_pred_err"])) else float("nan"))
        imb_str = f"Δimb={delta_imb:+.4f}"
        ms_str  = f"Δms={delta_ms:+.1f}ms"
        err_str = (f"Δpred_err={delta_err:+.1f}%" if not np.isnan(delta_err)
                   else "Δpred_err=nan")
        verdict = ""
        if delta_imb > 0.02 and delta_ms > 5:
            verdict = "→ CONSISTENT with bubble hypothesis"
        elif delta_imb > 0.02 and abs(delta_ms) <= 5:
            verdict = "→ MORE imbalance but same timing: imbalance is NOT the bottleneck"
        elif delta_imb <= 0.01:
            verdict = "→ No meaningful imbalance difference"
        print(f"  T_in={T_in:>4}: {imb_str}  {ms_str}  {err_str}  {verdict}")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, batches, vocab_size, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{NCCL_PORT}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.qwen3_moe.model import Qwen3MoeEPTpPartModel

    if rank == 0:
        print(f"Loading model from {MODEL_DIR}...")
    model = Qwen3MoeEPTpPartModel(tp_rank=rank, world_size=world_size,
                                   weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                   mem_adapter_size=0, dummy=False)
    if rank == 0:
        print(f"Model loaded.")

    warmup(model, vocab_size, rank, N_WARMUP)
    all_results = run_experiment(model, vocab_size, rank, batches)

    if rank == 0:
        mp_results["all_results"] = all_results

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    pool_by_len = load_real_tokens()
    vocab_size  = 151936   # Qwen3-30B-A3B vocab size (filled in after tokenizer load below)

    # Re-get actual vocab size from tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    vocab_size = tok.vocab_size
    print(f"vocab_size={vocab_size}")

    batches = build_batches(pool_by_len, vocab_size)

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, batches, vocab_size, mp_results), nprocs=2, join=True)

    all_results  = mp_results["all_results"]
    summary_rows = summarize(all_results)
    save_results(all_results, summary_rows)
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
