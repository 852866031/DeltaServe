#!/usr/bin/env python3
"""
Experiment 5 — MMLU Knowledge-Domain Routing Variance (Qwen3-30B-A3B EP)
=========================================================================
Same design as test/mixtral/exp5/mmlu_domain_experiment.py but for Qwen3 EP on 2 GPUs.

Tests whether knowledge domain (medical vs legal vs math vs CS vs humanities)
creates domain-specific MoE routing variance in Qwen3-30B-A3B.

MAX_LEN=512. Requires 2 GPUs and Qwen3-30B-A3B weights.

Usage:
    cd S-LoRA
    python test/qwen3/exp5/mmlu_domain_experiment.py
"""

import os, sys, csv, json, time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR  = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
MAX_POOL   = 20_000
OUT_DIR    = os.path.join(os.path.dirname(__file__), "results")
NCCL_PORT  = 29520

BS           = 4
MAX_LEN      = 512
MIN_LEN      = 16
N_PER_DOMAIN = 400
N_CALIB      = 200
N_PROF       = 50
N_WARMUP     = 20
SFT_BUDGET   = 256

SLO_THRESHOLDS_MS = [200, 250, 300, 350, 400]

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

    print("Loading cais/mmlu (all subjects, test split) ...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    print(f"Loading tokenizer from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    by_domain = {d: [] for d in DOMAIN_MAP}
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
        by_domain[domain].append(ids)

    print(f"\nDomain summary after tokenization (MAX_LEN={MAX_LEN}, skipped {skipped}):")
    for d in DOMAIN_MAP:
        lens = [len(x) for x in by_domain[d]]
        if lens:
            print(f"  {d:<12}: n={len(lens):5d}, "
                  f"mean_len={np.mean(lens):.0f}, "
                  f"std={np.std(lens):.0f}, "
                  f"max={np.max(lens)}")
    return by_domain


def build_batches(by_domain):
    rng     = np.random.default_rng(42)
    domains = list(DOMAIN_MAP.keys())

    calib_pool = {}
    prof_pool  = {}
    for d in domains:
        examples = by_domain[d]
        if len(examples) > N_PER_DOMAIN:
            idx = rng.choice(len(examples), N_PER_DOMAIN, replace=False)
            examples = [examples[i] for i in sorted(idx)]
        mid = len(examples) // 2
        calib_pool[d] = examples[:mid]
        prof_pool[d]  = examples[mid:]

    # Calibration: round-robin across domains
    calib_flat = []
    max_per = min(len(calib_pool[d]) for d in domains)
    for i in range(max_per):
        for d in domains:
            calib_flat.append(calib_pool[d][i])
    calib_batches = [calib_flat[i:i+BS] for i in range(0, len(calib_flat) - BS + 1, BS)]
    calib_batches = calib_batches[:N_CALIB]

    # Per-domain profiling
    domain_batches_list = []
    for d in domains:
        pool    = prof_pool[d]
        batches = [pool[i:i+BS] for i in range(0, len(pool) - BS + 1, BS)]
        if len(batches) >= 5:
            domain_batches_list.append((d, batches[:N_PROF]))

    print(f"\nBatches built:")
    print(f"  Calibration : {len(calib_batches)} batches")
    for d, batches in domain_batches_list:
        print(f"  {d:<12}: {len(batches)} profiling batches")

    return calib_batches, domain_batches_list


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
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(model, vocab_size, rank, calib_batches, domain_batches_list):
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

    # Phase 2: Per-domain profiling
    if rank == 0:
        print(f"\n[Phase 2] Per-domain profiling...")
    per_batch = []

    for domain, batches in domain_batches_list:
        if rank == 0:
            print(f"  [{domain}]: {len(batches)} batches")
        for batch in batches:
            t = run_prefill(model, batch)
            if rank == 0:
                lens = [len(x) for x in batch]
                pred = est.predict_inference(lens)
                signed_err = (pred - t) / t * 100
                per_batch.append({
                    "batch_id":       len(per_batch),
                    "domain":         domain,
                    "lengths_json":   json.dumps(lens),
                    "sum_n2":         int(sum(l*l for l in lens)),
                    "T_in":           int(sum(lens)),
                    "actual_ms":      float(t * 1000),
                    "pred_ms":        float(pred * 1000),
                    "signed_err_pct": float(signed_err),
                    "abs_err_pct":    float(abs(signed_err)),
                })

    return est, per_batch


def compute_domain_summary(per_batch):
    domains = list(DOMAIN_MAP.keys())
    rows = []
    for domain in domains:
        sub = [r for r in per_batch if r["domain"] == domain]
        if not sub:
            continue
        times  = np.array([r["actual_ms"]      for r in sub])
        errs   = np.array([r["abs_err_pct"]    for r in sub])
        signed = np.array([r["signed_err_pct"] for r in sub])
        t_ins  = np.array([r["T_in"]           for r in sub])
        rows.append({
            "domain":       domain,
            "n_batches":    len(sub),
            "mean_ms":      float(np.mean(times)),
            "std_ms":       float(np.std(times)),
            "cv_pct":       float(np.std(times) / np.mean(times) * 100),
            "mean_T_in":    float(np.mean(t_ins)),
            "std_T_in":     float(np.std(t_ins)),
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
            "fp_rate":          fp / n,
            "fn_rate":          fn / n,
            "frac_admitted":    admitted / n,
        })
    return rows


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, calib_batches, domain_batches_list, mp_results):
    dist.init_process_group(backend="nccl",
                            init_method=f"tcp://127.0.0.1:{NCCL_PORT}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    from slora.models.qwen3_moe.model import Qwen3MoeEPTpPartModel

    if rank == 0:
        print(f"Loading model from {MODEL_DIR} ...")
    model = Qwen3MoeEPTpPartModel(tp_rank=rank, world_size=world_size,
                                   weight_dir=MODEL_DIR, max_total_token_num=MAX_POOL,
                                   mem_adapter_size=0, dummy=False)
    vocab_size = model.config["vocab_size"]
    if rank == 0:
        print(f"Model loaded. vocab_size={vocab_size}")

    warmup(model, vocab_size, rank, N_WARMUP)
    est, per_batch = run_experiment(model, vocab_size, rank,
                                    calib_batches, domain_batches_list)

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
    fields = ["batch_id", "domain", "lengths_json", "sum_n2", "T_in",
              "actual_ms", "pred_ms", "signed_err_pct", "abs_err_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(per_batch)
    print(f"Saved: {path}")

    path = os.path.join(OUT_DIR, "domain_summary.csv")
    fields = ["domain", "n_batches", "mean_ms", "std_ms", "cv_pct",
              "mean_T_in", "std_T_in", "mean_err_pct", "p50_err", "p90_err", "max_err_pct"]
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
    print("\n" + "=" * 86)
    print("MMLU KNOWLEDGE-DOMAIN EXPERIMENT — Qwen3-30B-A3B EP (MoE, 2 GPUs)")
    print("=" * 86)
    print(f"  {'domain':<12}  {'n':>4}  {'mean_ms':>8}  {'cv%':>6}  "
          f"{'T_in_mean':>9}  {'T_in_std':>8}  {'mean_err%':>9}  {'p90_err%':>8}")
    for r in domain_rows:
        print(f"  {r['domain']:<12}  {r['n_batches']:>4}  "
              f"{r['mean_ms']:>8.2f}  {r['cv_pct']:>6.2f}  "
              f"{r['mean_T_in']:>9.0f}  {r['std_T_in']:>8.0f}  "
              f"{r['mean_err_pct']:>+9.2f}  {r['p90_err']:>8.2f}")

    print(f"\n  Gate decisions (SFT_BUDGET={SFT_BUDGET} tokens, all domains pooled):")
    print(f"  {'SLO(ms)':>8}  {'FP_count':>8}  {'FP%(all)':>8}  {'FN%(all)':>8}  {'admitted%':>9}")
    for r in gate_rows:
        print(f"  {r['slo_threshold_ms']:>8}  "
              f"{r['fp_count']:>8}  "
              f"{r['fp_rate']:>8.1%}  {r['fn_rate']:>8.1%}  "
              f"{r['frac_admitted']:>9.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    by_domain = load_and_tokenize()
    calib_batches, domain_batches_list = build_batches(by_domain)

    manager    = mp.Manager()
    mp_results = manager.dict()
    mp.spawn(worker, args=(2, calib_batches, domain_batches_list, mp_results),
             nprocs=2, join=True)

    per_batch = mp_results["per_batch"]

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
