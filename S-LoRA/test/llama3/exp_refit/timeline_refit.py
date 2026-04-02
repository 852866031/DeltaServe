#!/usr/bin/env python3
"""
Timeline Refit Experiment — Llama3-8B (dense model)
====================================================
Simulates a live serving scenario: the predictor sees a stream of heterogeneous
batches and refits every 256 batches (matching DeltaServe's production behavior).

Question: does continuous refitting on heterogeneous traffic fix the prediction
errors seen in Exp A, and how quickly?

For a dense model (Llama3), the answer should be YES — and crucially, the
initial error (Refit 1, uniform-trained) should already be low, because Σn² is
the structurally correct feature for attention-dominated computation.

Compare against test/mixtral/exp_refit/timeline_refit.py which uses the same
design on Mixtral EP (2-GPU, MoE) — expected to show high initial errors and
fragile convergence.

Timeline (bs=4, total_tokens=1024):
  Phase 0 (batches   1–256): 100% uniform [256]*4          → Refit 1
  Phase 1 (batches 257–512): 50% uniform + 50% skewed      → Refit 2
  Phase 2 (batches 513–768): 50% uniform + 50% skewed      → Refit 3
  Phase 3 (batches 769–1024): 100% uniform (reversal)      → Refit 4

At each refit, evaluate on 40-batch held-out sets of uniform, bimodal, skewed.
Eval batches are NOT added to the training buffer.

Usage:
    cd S-LoRA
    python test/llama3/exp_refit/timeline_refit.py
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

N_WARMUP    = 20
N_EVAL      = 40     # eval batches per family per checkpoint
PHASE_LEN   = 256    # training batches per phase (= refit interval)

BS          = 4
FAMILIES    = {
    "uniform": [256, 256, 256, 256],
    "bimodal": [64,  64,  448, 448],
    "skewed":  [32,  32,  32,  928],
}

PHASES = [
    # (phase_label, n_batches, composition)
    # composition: list of family names drawn round-robin
    ("phase0_uniform",   PHASE_LEN, ["uniform"]),
    ("phase1_mixed",     PHASE_LEN, ["uniform", "skewed"]),
    ("phase2_mixed",     PHASE_LEN, ["uniform", "skewed"]),
    ("phase3_reversal",  PHASE_LEN, ["uniform"]),
]


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
# Refit timeline experiment
# ---------------------------------------------------------------------------

def eval_family(model, est, family_name, lens, vocab_size, rng):
    """Run N_EVAL batches for one family, return (mean_ms, std_ms, cv_pct, pred_ms, mean_err_pct, max_err_pct)."""
    times = []
    for _ in range(N_EVAL):
        batch = [rng.integers(0, vocab_size, size=l) for l in lens]
        t = run_prefill(model, batch)
        times.append(t)
    pred      = est.predict_inference(lens)
    arr       = np.array(times)
    errs      = [(pred - a) / a * 100 for a in times]
    abs_errs  = [abs(e) for e in errs]
    return {
        "family":         family_name,
        "mean_ms":        float(np.mean(arr) * 1000),
        "std_ms":         float(np.std(arr) * 1000),
        "cv_pct":         float(np.std(arr) / np.mean(arr) * 100),
        "pred_ms":        float(pred * 1000),
        "mean_err_pct":   float(np.mean(errs)),
        "max_err_pct":    float(np.max(abs_errs)),
    }


def run_timeline_refit(model, vocab_size):
    from slora.server.router.tracker import PrefillExecutionEstimator

    train_rng = np.random.default_rng(200)
    eval_rng  = np.random.default_rng(300)

    # Accumulate training data (inference_only_tokens, inference_only_times)
    train_tokens = []   # list of lens lists
    train_times  = []   # list of floats (seconds)

    est     = PrefillExecutionEstimator()
    results = []

    for phase_idx, (phase_label, n_batches, composition) in enumerate(PHASES):
        refit_num = phase_idx + 1
        print(f"\n[Phase {phase_idx}] {phase_label}  ({n_batches} training batches, "
              f"composition={composition})")

        # --- Training batches for this phase ---
        for i in range(n_batches):
            fam_name = composition[i % len(composition)]
            lens     = FAMILIES[fam_name]
            batch    = [train_rng.integers(0, vocab_size, size=l) for l in lens]
            t        = run_prefill(model, batch)
            train_tokens.append(lens)
            train_times.append(t)
            if (i + 1) % 64 == 0:
                print(f"  training batch {i+1}/{n_batches}")

        # --- Refit predictor on all accumulated training data ---
        est.fit(
            inference_only_tokens=train_tokens,
            inference_only_times=train_times,
            coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
        )
        n_train = len(train_tokens)
        print(f"  Refit {refit_num}: n_train={n_train}  fit_rmse={est.fit_rmse*1000:.3f}ms  "
              f"alpha={est._params.alpha:.3e}  beta={est._params.beta:.3e}")

        # --- Evaluate all families (eval batches NOT added to training data) ---
        for fam_name, lens in FAMILIES.items():
            row = eval_family(model, est, fam_name, lens, vocab_size, eval_rng)
            row["phase"]   = phase_label
            row["refit_num"] = refit_num
            row["n_train"] = n_train
            row["sum_n2"]  = int(sum(l * l for l in lens))
            row["T_in"]    = int(sum(lens))
            results.append(row)
            print(f"  eval [{fam_name:>8}] mean={row['mean_ms']:7.2f}ms  "
                  f"pred={row['pred_ms']:7.2f}ms  err={row['mean_err_pct']:+6.1f}%  "
                  f"max_err={row['max_err_pct']:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    path   = os.path.join(OUT_DIR, "refit_timeline.csv")
    fields = ["refit_num", "phase", "n_train", "family", "sum_n2", "T_in",
              "mean_ms", "std_ms", "cv_pct", "pred_ms", "mean_err_pct", "max_err_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {path}")


def print_summary(results):
    print("\n" + "=" * 70)
    print("TIMELINE REFIT — Llama3-8B  (expected: low errors throughout)")
    print("=" * 70)
    print(f"  {'refit':>5}  {'phase':<22}  {'family':>8}  {'mean_ms':>8}  "
          f"{'pred_ms':>8}  {'err%':>7}  {'max_err%':>8}")
    for r in results:
        print(f"  {r['refit_num']:>5}  {r['phase']:<22}  {r['family']:>8}  "
              f"{r['mean_ms']:>8.2f}  {r['pred_ms']:>8.2f}  "
              f"{r['mean_err_pct']:>+7.1f}  {r['max_err_pct']:>8.1f}")


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
    results = run_timeline_refit(model, vocab_size)
    save_results(results)
    print_summary(results)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
