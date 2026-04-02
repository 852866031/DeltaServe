#!/usr/bin/env python3
"""
Experiment 4 — Domain-Specific Routing Variance (Mixtral-8x7B EP)
==================================================================
Same design as test/llama3/exp4/domain_experiment.py but for Mixtral EP on 2 GPUs.

Key question: does task domain (coding, medical, math, general QA, etc.) affect
Mixtral's MoE routing patterns, creating domain-specific timing variance that the
predictor cannot capture?

Hypothesis: different domains may activate different expert subsets systematically,
causing per-domain CV% to differ. If coding prompts consistently route to experts
2 & 5 while medical prompts route to experts 1 & 7, then batches from different
domains will have different (unpredictable) routing distributions, creating higher
variance than Llama3's dense model (test/llama3/exp4/domain_experiment.py).

Dataset: databricks/databricks-dolly-15k (7 domains, Apache 2.0)
Tokenizer: AutoTokenizer.from_pretrained(MODEL_DIR)

MAX_LEN=512 (tighter than Llama3 due to MAX_POOL=15000 for Mixtral EP).

Requires 2 GPUs and Mixtral-8x7B-v0.1 weights.

Usage:
    cd S-LoRA
    python test/mixtral/exp4/domain_experiment.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
MAX_POOL  = 15_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT  = "29507"

BS           = 4
MAX_LEN      = 512     # tighter than Llama3 (1024) due to MAX_POOL=15000
MIN_LEN      = 16
N_PER_DOMAIN = 200
N_CALIB      = 256
N_PROF       = 25
N_MIXED      = 50
SFT_BUDGET   = 256
N_WARMUP     = 20

SLO_THRESHOLDS_MS = [200, 250, 300, 350, 400]


# ---------------------------------------------------------------------------
# Data preparation (CPU, runs in main() before mp.spawn)
# ---------------------------------------------------------------------------

def load_and_tokenize():
    """Load dolly-15k, tokenize instructions with Mixtral tokenizer, group by domain."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading databricks/databricks-dolly-15k ...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    print(f"Loading tokenizer from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    by_domain = {}
    for ex in ds:
        cat  = ex["category"]
        text = ex["instruction"]
        ids  = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < MIN_LEN:
            continue
        ids = ids[:MAX_LEN]
        if cat not in by_domain:
            by_domain[cat] = []
        by_domain[cat].append(ids)

    print(f"\nDomain summary after tokenization (MAX_LEN={MAX_LEN}):")
    for cat in sorted(by_domain.keys()):
        lens = [len(x) for x in by_domain[cat]]
        print(f"  {cat:<26}: n={len(lens)}, mean_len={np.mean(lens):.0f}, "
              f"median={np.median(lens):.0f}, max={np.max(lens)}")
    return by_domain


def build_batches(by_domain):
    """Build calibration, per-domain, and mixed batches from precomputed token IDs."""
    rng     = np.random.default_rng(42)
    domains = sorted(by_domain.keys())

    calib_pool = {}
    prof_pool  = {}
    for cat in domains:
        examples = by_domain[cat]
        if len(examples) > N_PER_DOMAIN:
            idx = rng.choice(len(examples), N_PER_DOMAIN, replace=False)
            examples = [examples[i] for i in sorted(idx)]
        mid = len(examples) // 2
        calib_pool[cat] = examples[:mid]
        prof_pool[cat]  = examples[mid:]

    # Calibration: round-robin across domains → domain-neutral
    calib_flat = []
    max_per = min(len(calib_pool[d]) for d in domains)
    for i in range(max_per):
        for d in domains:
            calib_flat.append(calib_pool[d][i])
    calib_batches = [calib_flat[i:i+BS] for i in range(0, len(calib_flat) - BS + 1, BS)]
    calib_batches = calib_batches[:N_CALIB]

    # Per-domain profiling batches
    domain_batches_list = []  # list of (domain_name, list_of_batches) for mp.spawn
    for cat in domains:
        pool    = prof_pool[cat]
        batches = [pool[i:i+BS] for i in range(0, len(pool) - BS + 1, BS)]
        if len(batches) >= 5:
            domain_batches_list.append((cat, batches[:N_PROF]))

    # Mixed-domain batches: 1 example from each of first 4 domains
    mix_domains = [d for d, _ in domain_batches_list][:4]
    mixed_batches = []
    for i in range(N_MIXED):
        batch = []
        for d in mix_domains:
            idx = -(i + 1)
            if abs(idx) <= len(prof_pool[d]):
                batch.append(prof_pool[d][idx])
        if len(batch) == BS:
            mixed_batches.append(batch)

    print(f"\nBatches built:")
    print(f"  Calibration : {len(calib_batches)} batches")
    for cat, batches in domain_batches_list:
        print(f"  {cat:<26}: {len(batches)} profiling batches")
    print(f"  Mixed       : {len(mixed_batches)} batches  (domains: {mix_domains})")

    return calib_batches, domain_batches_list, mixed_batches


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------

def run_prefill(model, token_ids_list):
    """Both ranks must call this for every batch (2-GPU EP requires both)."""
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
# Experiment (runs inside each worker; only rank 0 records metrics)
# ---------------------------------------------------------------------------

def run_experiment(model, vocab_size, rank, calib_batches, domain_batches_list, mixed_batches):
    from slora.server.router.tracker import PrefillExecutionEstimator

    # Phase 1: Calibration
    if rank == 0:
        print(f"[Phase 1] Calibration: {len(calib_batches)} batches...")
    calib_tokens = []
    calib_times  = []
    for i, batch in enumerate(calib_batches):
        t = run_prefill(model, batch)
        if rank == 0:
            lens = [len(x) for x in batch]
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
        print(f"  Fit: fit_rmse={est.fit_rmse*1000:.3f}ms  "
              f"alpha={est._params.alpha:.3e}  beta={est._params.beta:.3e}")

    per_batch = []

    def record(batch, t, domain, batch_type):
        lens = [len(x) for x in batch]
        pred = est.predict_inference(lens)
        signed_err = (pred - t) / t * 100
        per_batch.append({
            "batch_id":       len(per_batch),
            "domain":         domain,
            "batch_type":     batch_type,
            "lengths_json":   json.dumps(lens),
            "sum_n2":         int(sum(l*l for l in lens)),
            "T_in":           int(sum(lens)),
            "actual_ms":      float(t * 1000),
            "pred_ms":        float(pred * 1000),
            "signed_err_pct": float(signed_err),
            "abs_err_pct":    float(abs(signed_err)),
        })

    # Phase 2: Per-domain profiling (both ranks run all forward passes)
    if rank == 0:
        print(f"\n[Phase 2] Per-domain profiling...")
    for domain, batches in domain_batches_list:
        if rank == 0:
            print(f"  [{domain}]: {len(batches)} batches")
        for batch in batches:
            t = run_prefill(model, batch)
            if rank == 0:
                record(batch, t, domain, "same")

    # Phase 3: Mixed-domain profiling
    if rank == 0:
        print(f"\n[Phase 3] Mixed profiling: {len(mixed_batches)} batches...")
    for batch in mixed_batches:
        t = run_prefill(model, batch)
        if rank == 0:
            record(batch, t, "mixed", "mixed")

    return est, per_batch


def compute_domain_summary(per_batch):
    domains = sorted(set(r["domain"] for r in per_batch))
    rows = []
    for domain in domains:
        sub    = [r for r in per_batch if r["domain"] == domain]
        times  = np.array([r["actual_ms"]      for r in sub])
        errs   = np.array([r["abs_err_pct"]    for r in sub])
        signed = np.array([r["signed_err_pct"] for r in sub])
        rows.append({
            "domain":       domain,
            "n_batches":    len(sub),
            "mean_ms":      float(np.mean(times)),
            "std_ms":       float(np.std(times)),
            "cv_pct":       float(np.std(times) / np.mean(times) * 100),
            "mean_T_in":    float(np.mean([r["T_in"] for r in sub])),
            "mean_err_pct": float(np.mean(signed)),
            "p50_err":      float(np.percentile(errs, 50)),
            "p90_err":      float(np.percentile(errs, 90)),
            "max_err_pct":  float(np.max(errs)),
        })
    return rows


def simulate_gate(per_batch, est, slo_thresholds_ms):
    rows = []
    for slo_ms in slo_thresholds_ms:
        fp = fn = tp = tn = 0
        admitted = 0
        for r in per_batch:
            lens            = json.loads(r["lengths_json"])
            pred_coserve_ms = est.predict_coserving(lens, [SFT_BUDGET]) * 1000
            gate_admits     = pred_coserve_ms <= slo_ms
            would_fit       = r["actual_ms"] <= slo_ms
            if gate_admits:
                admitted += 1
                if not would_fit: fp += 1
                else:             tp += 1
            else:
                if would_fit: fn += 1
                else:         tn += 1
        n = len(per_batch)
        rows.append({
            "slo_threshold_ms": slo_ms,
            "n_batches":        n,
            "fp_count":         fp,
            "fn_count":         fn,
            "fp_rate":          fp / max(fp + tn, 1),
            "fn_rate":          fn / max(fn + tp, 1),
            "frac_admitted":    admitted / n,
        })
    return rows


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, calib_batches, domain_batches_list, mixed_batches, mp_results):
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

    warmup(model, vocab_size, rank, N_WARMUP)
    est, per_batch = run_experiment(model, vocab_size, rank,
                                    calib_batches, domain_batches_list, mixed_batches)

    if rank == 0:
        mp_results["est_fit_rmse"] = est.fit_rmse
        mp_results["est_alpha"]    = est._params.alpha
        mp_results["est_beta"]     = est._params.beta
        mp_results["per_batch"]    = per_batch

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(per_batch, domain_rows, gate_rows):
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(OUT_DIR, "per_batch_domain.csv")
    fields = ["batch_id", "domain", "batch_type", "lengths_json",
              "sum_n2", "T_in", "actual_ms", "pred_ms", "signed_err_pct", "abs_err_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(per_batch)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "domain_summary.csv")
    fields = ["domain", "n_batches", "mean_ms", "std_ms", "cv_pct",
              "mean_T_in", "mean_err_pct", "p50_err", "p90_err", "max_err_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(domain_rows)
    print(f"Saved: {path}")

    # Also write as routing_variance.csv (Mixtral-specific name for comparison)
    path = os.path.join(OUT_DIR, "routing_variance.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(domain_rows)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "gate_decisions.csv")
    fields = ["slo_threshold_ms", "n_batches", "fp_count", "fn_count",
              "fp_rate", "fn_rate", "frac_admitted"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(gate_rows)
    print(f"Saved: {path}")


def print_summary(domain_rows, gate_rows):
    print("\n" + "=" * 76)
    print("DOMAIN ROUTING EXPERIMENT — Mixtral-8x7B EP (MoE, 2 GPUs)")
    print("=" * 76)
    print(f"  {'domain':<26}  {'n':>4}  {'mean_ms':>8}  {'cv%':>6}  "
          f"{'mean_err%':>9}  {'p90_err%':>8}")
    for r in domain_rows:
        print(f"  {r['domain']:<26}  {r['n_batches']:>4}  "
              f"{r['mean_ms']:>8.2f}  {r['cv_pct']:>6.2f}  "
              f"{r['mean_err_pct']:>+9.2f}  {r['p90_err']:>8.2f}")

    print(f"\n  Gate decisions (SFT_BUDGET={SFT_BUDGET} tokens, all domains pooled):")
    print(f"  {'SLO(ms)':>8}  {'FP%':>6}  {'FN%':>6}  {'admitted%':>9}")
    for r in gate_rows:
        print(f"  {r['slo_threshold_ms']:>8}  "
              f"{r['fp_rate']:>6.1%}  {r['fn_rate']:>6.1%}  "
              f"{r['frac_admitted']:>9.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    # Phase 0: Data preparation (CPU — before mp.spawn)
    by_domain = load_and_tokenize()
    calib_batches, domain_batches_list, mixed_batches = build_batches(by_domain)

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, calib_batches, domain_batches_list, mixed_batches, mp_results),
             nprocs=2, join=True)

    per_batch = mp_results["per_batch"]

    # Reconstruct estimator for gate simulation
    from slora.server.router.tracker import PrefillExecutionEstimator, PrefillParams
    est          = PrefillExecutionEstimator()
    est.fit_rmse = mp_results["est_fit_rmse"]
    est._params  = PrefillParams(
        alpha=mp_results["est_alpha"],
        beta=mp_results["est_beta"],
        gamma=0.0, c=0.0,
    )

    domain_rows = compute_domain_summary(per_batch)
    gate_rows   = simulate_gate(per_batch, est, SLO_THRESHOLDS_MS)

    save_results(per_batch, domain_rows, gate_rows)
    print_summary(domain_rows, gate_rows)


if __name__ == "__main__":
    main()
