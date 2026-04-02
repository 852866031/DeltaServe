#!/usr/bin/env python3
"""
Experiment 6 — Routing Collision Dissection (Mixtral-8x7B EP, 2 GPUs)
=======================================================================
Three-pronged attack to determine whether law_ethics routing variance from
Exp 5 is caused by vocabulary content (H1) or token length (H2).

  Exp 6A: Length-Controlled Cross-Domain Comparison
    At matched T_in buckets (short/medium/long), if law_ethics CV > STEM
    domains: content (not length) is the cause.

  Exp 6B: Routing Imbalance Logger
    `transformer_layer_infer_ep._ffn` is instrumented to capture recv_counts
    per layer per batch (rank 0 only). Measures imbalance_ratio =
    max(rank_0_load, rank_1_load) / balanced_load. Correlates with actual_ms.

  Exp 6C: Random Token Ablation
    Take law_ethics long-bucket prompts, keep exact token lengths, replace
    IDs with random integers. If real law CV >> random at same lengths:
    vocabulary patterns (not length) cause routing collision.

Requires 2 GPUs and Mixtral-8x7B-v0.1 weights.

Usage:
    cd S-LoRA
    python test/mixtral/exp6/routing_collision.py
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
TCP_PORT  = "29510"

BS           = 4
MAX_LEN      = 512
MIN_LEN      = 16
MAX_PER_CELL = 80    # max examples per (domain, bucket) → 20 batches of BS=4
N_CALIB      = 200
N_PROF       = 20    # profiling batches per (domain, bucket)
N_WARMUP     = 20
K_EXPERTS    = 2     # num_experts_per_tok for Mixtral

# Length buckets: per-sequence token count
BUCKETS = {
    "short":  (50,  150),   # batch T_in = 200–600
    "medium": (150, 300),   # batch T_in = 600–1200
    "long":   (300, 512),   # batch T_in = 1200–2048
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
# Data preparation (CPU, runs in main() before mp.spawn)
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
    vocab_size = tokenizer.vocab_size

    # by_domain_bucket[domain][bucket] = list of token ID lists
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

    return by_domain_bucket, vocab_size


def build_batches(by_domain_bucket, vocab_size):
    rng = np.random.default_rng(42)

    calib_pool = []
    # prof_entries: list of (domain, bucket, batches) for mp.spawn pickling
    prof_entries = []

    for d in DOMAINS:
        for b in BUCKETS:
            examples = list(by_domain_bucket[d][b])
            if len(examples) < BS:
                continue  # skip cells with fewer examples than one batch
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

    # Random ablation (Exp 6C): law_ethics long bucket, same lengths, random IDs
    law_long = by_domain_bucket["law_ethics"]["long"]
    random_ablation_batches = []
    if len(law_long) >= BS:
        src = law_long[:MAX_PER_CELL]
        src_batches = [src[i:i+BS] for i in range(0, len(src) - BS + 1, BS)]
        for raw_batch in src_batches[:N_PROF]:
            rand_batch = [rng.integers(0, vocab_size, size=len(seq)).tolist()
                          for seq in raw_batch]
            random_ablation_batches.append(rand_batch)

    # Calibration: mixed-domain mixed-bucket, round-robin shuffled
    rng.shuffle(calib_pool)
    calib_batches = [calib_pool[i:i+BS]
                     for i in range(0, len(calib_pool) - BS + 1, BS)]
    calib_batches = calib_batches[:N_CALIB]

    print(f"\nBatches built:")
    print(f"  Calibration    : {len(calib_batches)} batches (mixed domain/bucket)")
    for d, b, batches in sorted(prof_entries):
        print(f"  {d:<12}/{b:<6}: {len(batches)} profiling batches")
    print(f"  random_ablation: {len(random_ablation_batches)} batches "
          f"(law_ethics long, random IDs)")

    return calib_batches, prof_entries, random_ablation_batches


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


def compute_imbalance(log_entries, T_in):
    """
    Compute inter-rank load imbalance from rank 0's recv_counts log.

    log_entries : list of [recv_from_r0, recv_from_r1] per layer (32 entries).
    T_in        : total prefill tokens in the batch.

    For K=2 top-experts and R=2 ranks:
      TK = T_in * K_EXPERTS = total work items split across both ranks.
      rank_0_load (this layer) = sum(recv_counts)
      rank_1_load              = TK - rank_0_load
      imbalance_ratio          = max(rank_0_load, rank_1_load) / (TK / 2)
        = 1.0 if perfectly balanced, 2.0 if all work on one rank.

    Returns (mean_imbalance_ratio, max_imbalance_ratio) across all logged layers.
    """
    if not log_entries:
        return 1.0, 1.0
    TK = T_in * K_EXPERTS
    balanced = TK / 2
    if balanced == 0:
        return 1.0, 1.0
    layer_ratios = []
    for recv in log_entries:
        r0_load = sum(recv)
        r1_load = TK - r0_load
        layer_ratios.append(max(r0_load, r1_load) / balanced)
    return float(np.mean(layer_ratios)), float(np.max(layer_ratios))


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

def run_experiment(model, vocab_size, rank,
                   calib_batches, prof_entries, random_ablation_batches):
    from slora.server.router.tracker import PrefillExecutionEstimator
    import slora.models.mixtral.layer_infer.transformer_layer_infer_ep as ep_module

    # --- Phase 1: Calibration ---
    if rank == 0:
        print(f"[Phase 1] Calibration: {len(calib_batches)} batches...")
    calib_tokens = []
    calib_times  = []
    for i, batch in enumerate(calib_batches):
        if rank == 0:
            ep_module.routing_imbalance_log.clear()
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

    # --- Phase 2: Per-(domain, bucket) profiling ---
    if rank == 0:
        print(f"\n[Phase 2] Per-domain/bucket profiling (Exp 6A + 6B)...")
    per_batch = []

    for (domain, bucket, batches) in sorted(prof_entries):
        if rank == 0:
            print(f"  [{domain}/{bucket}]: {len(batches)} batches")
        for batch in batches:
            if rank == 0:
                ep_module.routing_imbalance_log.clear()
            t = run_prefill(model, batch)
            if rank == 0:
                lens = [len(x) for x in batch]
                T_in = sum(lens)
                pred = est.predict_inference(lens)
                signed_err = (pred - t) / t * 100
                log_snap = list(ep_module.routing_imbalance_log)
                mean_imb, max_imb = compute_imbalance(log_snap, T_in)
                per_batch.append({
                    "domain":               domain,
                    "bucket":               bucket,
                    "batch_id":             len(per_batch),
                    "T_in":                 T_in,
                    "n_layers_logged":      len(log_snap),
                    "mean_imbalance_ratio": mean_imb,
                    "max_imbalance_ratio":  max_imb,
                    "actual_ms":            float(t * 1000),
                    "pred_ms":              float(pred * 1000),
                    "signed_err_pct":       float(signed_err),
                    "abs_err_pct":          float(abs(signed_err)),
                })

    # --- Phase 3: Random ablation (Exp 6C) ---
    if rank == 0:
        print(f"\n[Phase 3] Random ablation: {len(random_ablation_batches)} batches...")
    random_ablation_results = []

    for batch in random_ablation_batches:
        if rank == 0:
            ep_module.routing_imbalance_log.clear()
        t = run_prefill(model, batch)
        if rank == 0:
            lens = [len(x) for x in batch]
            T_in = sum(lens)
            log_snap = list(ep_module.routing_imbalance_log)
            mean_imb, max_imb = compute_imbalance(log_snap, T_in)
            random_ablation_results.append({
                "bucket":               "long",
                "batch_id":             len(random_ablation_results),
                "T_in":                 T_in,
                "mean_imbalance_ratio": mean_imb,
                "max_imbalance_ratio":  max_imb,
                "actual_ms":            float(t * 1000),
            })

    return est, per_batch, random_ablation_results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_domain_bucket_summary(per_batch):
    keys = sorted(set((r["domain"], r["bucket"]) for r in per_batch))
    rows = []
    for (domain, bucket) in keys:
        sub   = [r for r in per_batch if r["domain"] == domain and r["bucket"] == bucket]
        times = np.array([r["actual_ms"]            for r in sub])
        imbs  = np.array([r["mean_imbalance_ratio"] for r in sub])
        t_ins = np.array([r["T_in"]                 for r in sub])
        corr  = (float(np.corrcoef(imbs, times)[0, 1])
                 if len(sub) >= 3 else float("nan"))
        rows.append({
            "domain":             domain,
            "bucket":             bucket,
            "n_batches":          len(sub),
            "mean_ms":            float(np.mean(times)),
            "std_ms":             float(np.std(times)),
            "cv_pct":             float(np.std(times) / np.mean(times) * 100),
            "mean_T_in":          float(np.mean(t_ins)),
            "mean_imbalance":     float(np.mean(imbs)),
            "max_imbalance":      float(np.max(imbs)),
            "corr_imbalance_ms":  corr,
        })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(per_batch, summary_rows, random_ablation_results):
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(OUT_DIR, "per_batch.csv")
    fields = ["domain", "bucket", "batch_id", "T_in", "n_layers_logged",
              "mean_imbalance_ratio", "max_imbalance_ratio",
              "actual_ms", "pred_ms", "signed_err_pct", "abs_err_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(per_batch)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "domain_bucket_summary.csv")
    fields = ["domain", "bucket", "n_batches", "mean_ms", "std_ms", "cv_pct",
              "mean_T_in", "mean_imbalance", "max_imbalance", "corr_imbalance_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "random_ablation.csv")
    fields = ["bucket", "batch_id", "T_in", "mean_imbalance_ratio",
              "max_imbalance_ratio", "actual_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(random_ablation_results)
    print(f"Saved: {path}")


def print_summary(summary_rows, random_ablation_results, per_batch):
    print("\n" + "=" * 110)
    print("EXP 6 — ROUTING COLLISION DISSECTION (Mixtral-8x7B EP, 2 GPUs)")
    print("=" * 110)

    # Exp 6A: cross-domain comparison at matched T_in bucket
    for bucket in BUCKETS:
        bucket_rows = sorted(
            [r for r in summary_rows if r["bucket"] == bucket],
            key=lambda x: x["cv_pct"], reverse=True,
        )
        if not bucket_rows:
            continue
        print(f"\n  [{bucket} bucket]")
        print(f"  {'domain':<12}  {'n':>4}  {'mean_ms':>8}  {'cv%':>6}  "
              f"{'T_in_mean':>9}  {'mean_imb':>8}  {'max_imb':>7}  {'corr_imb_ms':>11}")
        for r in bucket_rows:
            corr_str = f"{r['corr_imbalance_ms']:>+11.3f}" if not np.isnan(r["corr_imbalance_ms"]) else "        nan"
            print(f"  {r['domain']:<12}  {r['n_batches']:>4}  "
                  f"{r['mean_ms']:>8.2f}  {r['cv_pct']:>6.2f}  "
                  f"{r['mean_T_in']:>9.0f}  {r['mean_imbalance']:>8.3f}  "
                  f"{r['max_imbalance']:>7.3f}  {corr_str}")

    # Exp 6C: random ablation comparison
    if random_ablation_results:
        ra_times = np.array([r["actual_ms"]            for r in random_ablation_results])
        ra_imbs  = np.array([r["mean_imbalance_ratio"] for r in random_ablation_results])
        law_long = [r for r in per_batch
                    if r["domain"] == "law_ethics" and r["bucket"] == "long"]
        ll_times = np.array([r["actual_ms"]            for r in law_long])
        ll_imbs  = np.array([r["mean_imbalance_ratio"] for r in law_long])

        print(f"\n  [Exp 6C — Random Ablation vs Real law_ethics (long bucket)]")
        print(f"  {'group':<25}  {'n':>4}  {'mean_ms':>8}  {'cv%':>6}  {'mean_imb':>8}  {'max_imb':>7}")
        if len(ra_times) > 0:
            print(f"  {'random_ablation':<25}  {len(ra_times):>4}  "
                  f"{np.mean(ra_times):>8.2f}  "
                  f"{np.std(ra_times)/np.mean(ra_times)*100:>6.2f}  "
                  f"{np.mean(ra_imbs):>8.3f}  {np.max(ra_imbs):>7.3f}")
        if len(ll_times) > 0:
            print(f"  {'law_ethics_real':<25}  {len(ll_times):>4}  "
                  f"{np.mean(ll_times):>8.2f}  "
                  f"{np.std(ll_times)/np.mean(ll_times)*100:>6.2f}  "
                  f"{np.mean(ll_imbs):>8.3f}  {np.max(ll_imbs):>7.3f}")

    # Interpretation guidance
    print(f"\n  Interpretation:")
    print(f"    H1 (content) confirmed if:")
    print(f"      - law_ethics mean_imb > 1.2 while STEM domains mean_imb ≈ 1.0 (same bucket)")
    print(f"      - corr_imb_ms is strongly positive for law_ethics")
    print(f"      - random_ablation mean_imb ≈ 1.0 (same lengths, balanced routing)")
    print(f"    H2 (length) confirmed if:")
    print(f"      - mean_imb is uniform across domains within same bucket")
    print(f"      - cv% differences disappear when controlling for T_in")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, calib_batches, prof_entries,
           random_ablation_batches, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{TCP_PORT}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.mixtral.model import MixtralEPTpPartModel

    if rank == 0:
        print(f"Loading model from {MODEL_DIR}...")
    model = MixtralEPTpPartModel(tp_rank=rank, world_size=world_size,
                                  weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                  mem_adapter_size=0, dummy=False)
    vocab_size = model.config["vocab_size"]
    if rank == 0:
        print(f"Model loaded. vocab_size={vocab_size}")

    warmup(model, vocab_size, rank, N_WARMUP)
    est, per_batch, random_ablation_results = run_experiment(
        model, vocab_size, rank, calib_batches, prof_entries, random_ablation_batches)

    if rank == 0:
        mp_results["est_fit_rmse"]    = est.fit_rmse
        mp_results["est_alpha"]       = est._params.alpha
        mp_results["est_beta"]        = est._params.beta
        mp_results["per_batch"]       = per_batch
        mp_results["random_ablation"] = random_ablation_results

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    by_domain_bucket, vocab_size = load_and_tokenize()
    calib_batches, prof_entries, random_ablation_batches = build_batches(
        by_domain_bucket, vocab_size)

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker,
             args=(2, calib_batches, prof_entries, random_ablation_batches, mp_results),
             nprocs=2, join=True)

    per_batch       = mp_results["per_batch"]
    random_ablation = mp_results["random_ablation"]

    from slora.server.router.tracker import PrefillExecutionEstimator, PrefillParams
    est          = PrefillExecutionEstimator()
    est.fit_rmse = mp_results["est_fit_rmse"]
    est._params  = PrefillParams(
        alpha=mp_results["est_alpha"],
        beta=mp_results["est_beta"],
        gamma=0.0, c=0.0,
    )

    summary_rows = compute_domain_bucket_summary(per_batch)
    save_results(per_batch, summary_rows, random_ablation)
    print_summary(summary_rows, random_ablation, per_batch)


if __name__ == "__main__":
    main()
