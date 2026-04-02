#!/usr/bin/env python3
"""
Experiment 6 — Routing Collision Dissection (Llama3-8B, 1 GPU — control)
==========================================================================
Dense baseline for Exp 6A: length-controlled cross-domain CV comparison.

Hypothesis: if Llama3 (no MoE) shows uniform CV across domains at matched
T_in buckets, while Mixtral EP shows elevated CV for law_ethics, then the
law_ethics routing collision is content-driven (H1), not length-driven (H2).

No routing imbalance logger (Llama3 has no expert dispatch) and no random
ablation (content ablation is only meaningful for MoE models).

Usage:
    cd S-LoRA
    python test/llama3/exp6/routing_collision.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/meta-llama/Meta-Llama-3-8B-HF"
MAX_POOL  = 35_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
TCP_PORT  = "29511"

BS           = 4
MAX_LEN      = 512    # match Mixtral MAX_LEN for a fair comparison
MIN_LEN      = 16
MAX_PER_CELL = 80     # max examples per (domain, bucket)
N_CALIB      = 200
N_PROF       = 20     # profiling batches per (domain, bucket)
N_WARMUP     = 20

# Length buckets: per-sequence token count (same as Mixtral exp6)
BUCKETS = {
    "short":  (50,  150),
    "medium": (150, 300),
    "long":   (300, 512),
}

DOMAIN_MAP = {
    "medical":    ["anatomy", "clinical_knowledge", "college_biology", "college_medicine",
                   "human_aging", "human_sexuality", "medical_genetics", "nutrition",
                   "professional_medicine", "virology", "high_school_biology"],
    "stem_math":  ["abstract_algebra", "college_mathematics", "elementary_mathematics",
                   "formal_logic", "high_school_mathematics", "high_school_statistics"],
    "stem_sci":   ["astronomy", "college_chemistry", "college_physics", "conceptual_physics",
                   "electrical_engineering", "high_school_chemistry", "high_school_physics"],
    "stem_cs":    ["college_computer_science", "computer_security",
                   "high_school_computer_science", "machine_learning"],
    "law_ethics": ["business_ethics", "international_law", "jurisprudence",
                   "logical_fallacies", "moral_disputes", "moral_scenarios",
                   "professional_law", "philosophy"],
    "social":     ["econometrics", "high_school_geography",
                   "high_school_government_and_politics", "high_school_macroeconomics",
                   "high_school_microeconomics", "high_school_psychology",
                   "management", "marketing", "professional_accounting",
                   "professional_psychology", "public_relations", "security_studies",
                   "sociology", "us_foreign_policy"],
    "humanities": ["global_facts", "high_school_european_history",
                   "high_school_us_history", "high_school_world_history",
                   "miscellaneous", "prehistory", "world_religions"],
}
SUBJECT_TO_DOMAIN = {s: d for d, subjects in DOMAIN_MAP.items() for s in subjects}
DOMAINS = list(DOMAIN_MAP.keys())


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def format_prompt(ex):
    return (f"Question: {ex['question']}\n"
            f"A: {ex['choices'][0]}\nB: {ex['choices'][1]}\n"
            f"C: {ex['choices'][2]}\nD: {ex['choices'][3]}")


def load_and_tokenize():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading cais/mmlu (all subjects, test split)...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    by_domain_bucket = {d: {b: [] for b in BUCKETS} for d in DOMAIN_MAP}
    skipped = 0
    for ex in ds:
        domain = SUBJECT_TO_DOMAIN.get(ex["subject"])
        if domain is None:
            skipped += 1
            continue
        text = format_prompt(ex)
        ids  = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < MIN_LEN:
            continue
        ids = ids[:MAX_LEN]
        seq_len = len(ids)
        for bucket, (lo, hi) in BUCKETS.items():
            if lo <= seq_len < hi:
                by_domain_bucket[domain][bucket].append(ids)
                break

    print(f"\nDomain × bucket cell sizes (MAX_LEN={MAX_LEN}, skipped {skipped} unmapped):")
    for d in DOMAIN_MAP:
        parts = [f"{b}:{len(by_domain_bucket[d][b])}" for b in BUCKETS]
        print(f"  {d:<12}: {', '.join(parts)}")

    return by_domain_bucket


def build_batches(by_domain_bucket):
    rng = np.random.default_rng(42)

    calib_pool  = []
    prof_entries = []  # list of (domain, bucket, batches)

    for d in DOMAINS:
        for b in BUCKETS:
            examples = list(by_domain_bucket[d][b])
            if len(examples) < BS:
                continue
            if len(examples) > MAX_PER_CELL:
                idx = rng.choice(len(examples), MAX_PER_CELL, replace=False)
                examples = [examples[i] for i in sorted(idx)]
            rng.shuffle(examples)
            mid = len(examples) // 2
            calib_pool.extend(examples[:mid])
            pool = examples[mid:]
            batches = [pool[i:i+BS] for i in range(0, len(pool) - BS + 1, BS)]
            if batches:
                prof_entries.append((d, b, batches[:N_PROF]))

    rng.shuffle(calib_pool)
    calib_batches = [calib_pool[i:i+BS]
                     for i in range(0, len(calib_pool) - BS + 1, BS)]
    calib_batches = calib_batches[:N_CALIB]

    print(f"\nBatches built:")
    print(f"  Calibration    : {len(calib_batches)} batches (mixed domain/bucket)")
    for d, b, batches in sorted(prof_entries):
        print(f"  {d:<12}/{b:<6}: {len(batches)} profiling batches")

    return calib_batches, prof_entries


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
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(model, vocab_size, calib_batches, prof_entries):
    from slora.server.router.tracker import PrefillExecutionEstimator

    # Phase 1: Calibration
    print(f"[Phase 1] Calibration: {len(calib_batches)} batches...")
    calib_tokens = []
    calib_times  = []
    for i, batch in enumerate(calib_batches):
        t    = run_prefill(model, batch)
        lens = [len(x) for x in batch]
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
    print(f"  Fit: fit_rmse={est.fit_rmse*1000:.3f}ms  "
          f"alpha={est._params.alpha:.3e}  beta={est._params.beta:.3e}")

    # Phase 2: Per-(domain, bucket) profiling
    print(f"\n[Phase 2] Per-domain/bucket profiling (Exp 6A control)...")
    per_batch = []

    for (domain, bucket, batches) in sorted(prof_entries):
        print(f"  [{domain}/{bucket}]: {len(batches)} batches")
        for batch in batches:
            t    = run_prefill(model, batch)
            lens = [len(x) for x in batch]
            T_in = sum(lens)
            pred = est.predict_inference(lens)
            signed_err = (pred - t) / t * 100
            per_batch.append({
                "domain":         domain,
                "bucket":         bucket,
                "batch_id":       len(per_batch),
                "T_in":           T_in,
                "actual_ms":      float(t * 1000),
                "pred_ms":        float(pred * 1000),
                "signed_err_pct": float(signed_err),
                "abs_err_pct":    float(abs(signed_err)),
            })

    return est, per_batch


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_domain_bucket_summary(per_batch):
    keys = sorted(set((r["domain"], r["bucket"]) for r in per_batch))
    rows = []
    for (domain, bucket) in keys:
        sub   = [r for r in per_batch if r["domain"] == domain and r["bucket"] == bucket]
        times = np.array([r["actual_ms"]      for r in sub])
        errs  = np.array([r["abs_err_pct"]    for r in sub])
        t_ins = np.array([r["T_in"]           for r in sub])
        rows.append({
            "domain":       domain,
            "bucket":       bucket,
            "n_batches":    len(sub),
            "mean_ms":      float(np.mean(times)),
            "std_ms":       float(np.std(times)),
            "cv_pct":       float(np.std(times) / np.mean(times) * 100),
            "mean_T_in":    float(np.mean(t_ins)),
            "mean_err_pct": float(np.mean(np.array([r["signed_err_pct"] for r in sub]))),
            "p90_err":      float(np.percentile(errs, 90)),
        })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(per_batch, summary_rows):
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(OUT_DIR, "per_batch.csv")
    fields = ["domain", "bucket", "batch_id", "T_in",
              "actual_ms", "pred_ms", "signed_err_pct", "abs_err_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(per_batch)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "domain_bucket_summary.csv")
    fields = ["domain", "bucket", "n_batches", "mean_ms", "std_ms", "cv_pct",
              "mean_T_in", "mean_err_pct", "p90_err"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)
    print(f"Saved: {path}")


def print_summary(summary_rows):
    print("\n" + "=" * 90)
    print("EXP 6 — ROUTING COLLISION DISSECTION (Llama3-8B, dense, 1 GPU — CONTROL)")
    print("=" * 90)
    for bucket in BUCKETS:
        bucket_rows = sorted(
            [r for r in summary_rows if r["bucket"] == bucket],
            key=lambda x: x["cv_pct"], reverse=True,
        )
        if not bucket_rows:
            continue
        print(f"\n  [{bucket} bucket]")
        print(f"  {'domain':<12}  {'n':>4}  {'mean_ms':>8}  {'cv%':>6}  "
              f"{'T_in_mean':>9}  {'mean_err%':>9}  {'p90_err%':>8}")
        for r in bucket_rows:
            print(f"  {r['domain']:<12}  {r['n_batches']:>4}  "
                  f"{r['mean_ms']:>8.2f}  {r['cv_pct']:>6.2f}  "
                  f"{r['mean_T_in']:>9.0f}  {r['mean_err_pct']:>+9.2f}  "
                  f"{r['p90_err']:>8.2f}")

    print(f"\n  Expected: uniform cv% across all domains within each bucket.")
    print(f"  If law_ethics shows elevated cv% here too → length is confounded.")
    print(f"  If law_ethics cv% matches STEM → content-driven (supports H1).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    by_domain_bucket = load_and_tokenize()
    calib_batches, prof_entries = build_batches(by_domain_bucket)

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{TCP_PORT}",
                            world_size=1, rank=0)

    from slora.models.llama3.model import Llama3TpPartModel

    print(f"\nLoading model from {MODEL_DIR}...")
    model = Llama3TpPartModel(tp_rank=0, world_size=1, weight_dir=MODEL_DIR,
                               max_total_token_num=MAX_POOL, mem_adapter_size=0,
                               dummy=False)
    vocab_size = model.config["vocab_size"]
    print(f"Model loaded. vocab_size={vocab_size}")

    warmup(model, vocab_size, N_WARMUP)

    est, per_batch  = run_experiment(model, vocab_size, calib_batches, prof_entries)
    summary_rows    = compute_domain_bucket_summary(per_batch)

    save_results(per_batch, summary_rows)
    print_summary(summary_rows)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
