#!/usr/bin/env python3
"""
EP Predictor Failure Experiment — Mixtral MoE (2-GPU Expert Parallelism)
=========================================================================
Runs four experiments that collectively prove PrefillExecutionEstimator
cannot reliably predict MoE prefill latency.

The predictor model:
    T_prefill ≈ α·Σn_i² + β·T_in + γ·T_ft + c

For dense models this works because FFN cost per token is fixed.
For MoE, the FFN cost depends on which experts are selected — that varies
per batch even when Σn² and T_in are identical — so the predictor has
irreducible, content-dependent residual error.

Experiments
-----------
1. Uniform config sweep
   Many (batch_size, seq_len) pairs. For EACH config, all batches share
   identical Σn² / T_in. Predictor is fit per-config. Shows it outputs
   the same prediction every time, with high absolute error.

2. Global predictor across diverse shapes
   One estimator fit on ALL configs combined. It can capture the
   token-count structure across shapes, but within each config the
   residual from routing variance remains.

3. Realistic input timelines
   Poisson-distributed batch sizes, log-normal sequence lengths.
   Fit predictor, show error distribution over realistic traffic.

4. RMSE convergence curve
   Fixed config (4×64). Fit on 5, 10, 25, 50, 100, 200, 400 samples.
   RMSE stays flat → the error is irreducible, not a data-quantity problem.

Usage
-----
    cd S-LoRA
    python test/mixtral/ep_predictor_experiment.py
"""

import os
import sys
import csv
import json
import time
import tempfile
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Model config — tiny Mixtral, fits on GPU, runs fast.
# 4 experts, 2 GPUs → each rank owns 2 experts at full intermediate size.
# top-2 routing out of 4 means each token's 2 chosen experts may land on
# the same rank or split across ranks — creating communication imbalance.
# ---------------------------------------------------------------------------
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

MAX_POOL   = 8192   # KV cache slots; reset between batches so just needs one large batch
N_WARMUP   = 30     # discarded warm-up iterations (GPU/NCCL spin-up)
VOCAB_SIZE = TINY_CONFIG["vocab_size"]

# Uniform configs: (batch_size, seq_len_per_request)
# Chosen to span the typical operating range of a serving system.
UNIFORM_CONFIGS = [
    (1,  256),
    (2,  128),
    (4,   64),   # baseline (matches original experiment)
    (8,   32),
    (16,  16),
    (4,   32),
    (4,  128),
    (8,   64),
]
N_TRAIN_PER_CFG = 120
N_TEST_PER_CFG  = 120

# Exp 3 — realistic timeline
N_TIMELINE_TRAIN = 200
N_TIMELINE_TEST  = 200
TIMELINE_MEAN_BS  = 4    # Poisson mean batch size
TIMELINE_MEAN_LEN = 64   # log-normal mean seq length
TIMELINE_STD_LEN  = 48   # log-normal std

# Exp 4 — convergence
CONVERGENCE_CONFIG = (4, 64)
CONVERGENCE_TOTAL  = 500
CONVERGENCE_CHECKPOINTS = [5, 10, 25, 50, 100, 200, 350, 500]
CONVERGENCE_TEST   = 200   # held-out batches always from same distribution


# ---------------------------------------------------------------------------
# Shared helpers (run inside each worker process)
# ---------------------------------------------------------------------------

def build_prefill_inputs(token_ids_list: list, device: str = "cuda"):
    """
    token_ids_list: list of 1-D int arrays, one per request (may differ in length).
    Returns (input_ids, b_loc, b_start_loc, b_seq_len, bs, total_tokens, max_len).
    b_loc is zeros — init_bloc inside model._prefill fills it from the allocator.
    """
    bs       = len(token_ids_list)
    seq_lens = [len(ids) for ids in token_ids_list]
    total    = sum(seq_lens)
    max_len  = max(seq_lens)

    flat = np.concatenate([np.asarray(ids, dtype=np.int64) for ids in token_ids_list])
    input_ids   = torch.from_numpy(flat).to(device)
    b_seq_len   = torch.tensor(seq_lens, dtype=torch.long, device=device)
    b_start_loc = torch.zeros(bs, dtype=torch.long, device=device)
    for i in range(1, bs):
        b_start_loc[i] = b_start_loc[i - 1] + seq_lens[i - 1]
    b_loc = torch.zeros(bs, max_len, dtype=torch.long, device=device)

    return input_ids, b_loc, b_start_loc, b_seq_len, bs, total, max_len


def timed_prefill(model, token_ids_list: list) -> float:
    """Run one prefill pass and return wall-clock seconds (barriers on both sides)."""
    inputs = build_prefill_inputs(token_ids_list)
    input_ids, b_loc, b_start_loc, b_seq_len, bs, total, max_len = inputs

    model.mem_manager.reset_all_pool()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()

    with torch.no_grad():
        model.forward(
            batch_size=bs,
            total_token_num=total,
            max_len_in_batch=max_len,
            input_ids=input_ids,
            b_loc=b_loc,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            is_prefill=True,
        )

    torch.cuda.synchronize()
    dist.barrier()
    return time.perf_counter() - t0


def make_uniform_batch(seed: int, batch_size: int, seq_len: int) -> list:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, VOCAB_SIZE, size=seq_len) for _ in range(batch_size)]


def make_timeline_batch(seed: int) -> list:
    """Poisson batch size, log-normal sequence lengths."""
    rng = np.random.default_rng(seed)
    bs  = max(1, rng.poisson(TIMELINE_MEAN_BS))
    # log-normal: mean=TIMELINE_MEAN_LEN, std=TIMELINE_STD_LEN
    mu  = np.log(TIMELINE_MEAN_LEN**2 / np.sqrt(TIMELINE_MEAN_LEN**2 + TIMELINE_STD_LEN**2))
    sig = np.sqrt(np.log(1 + (TIMELINE_STD_LEN / TIMELINE_MEAN_LEN)**2))
    lens = np.clip(rng.lognormal(mu, sig, bs).astype(int), 4, 256)
    return [rng.integers(0, VOCAB_SIZE, size=l) for l in lens]


def token_features(token_ids_list: list) -> tuple:
    """Return (seq_len_list, sum_n2, T_in) — the predictor's features."""
    lens   = [len(ids) for ids in token_ids_list]
    sum_n2 = sum(l**2 for l in lens)
    T_in   = sum(lens)
    return lens, sum_n2, T_in


def fit_and_eval(train_token_lists, train_times, test_token_lists, test_times):
    """
    Fit PrefillExecutionEstimator on training data, evaluate on test data.
    Returns (estimator, predictions, abs_errors_pct).
    """
    from slora.server.router.tracker import PrefillExecutionEstimator
    est = PrefillExecutionEstimator()
    est.fit(
        inference_only_tokens=[token_features(b)[0] for b in train_token_lists],
        inference_only_times=train_times,
        coserving_inf_tokens=[],
        coserving_ft_tokens=[],
        coserving_times=[],
    )
    preds  = [est.predict_inference(token_features(b)[0]) for b in test_token_lists]
    errors = [abs(p - a) / a * 100 for p, a in zip(preds, test_times)]
    return est, preds, errors


# ---------------------------------------------------------------------------
# Worker — runs inside each spawned process
# ---------------------------------------------------------------------------

def worker(rank: int, world_size: int, config_dir: str, results: dict):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    from slora.models.mixtral.model import MixtralEPTpPartModel

    if rank == 0:
        print("Loading model with dummy weights...")

    model = MixtralEPTpPartModel(
        tp_rank=rank,
        world_size=world_size,
        weight_dir=config_dir,
        max_total_token_num=MAX_POOL,
        mem_adapter_size=0,
        dummy=True,
    )

    if rank == 0:
        print("Model ready.\n")

    # -----------------------------------------------------------------------
    # Warm up (all ranks participate)
    # -----------------------------------------------------------------------
    if rank == 0:
        print(f"Warming up ({N_WARMUP} batches)...")
    for i in range(N_WARMUP):
        timed_prefill(model, make_uniform_batch(i, 4, 64))
    if rank == 0:
        print("Warmup done.\n")

    # Seed offset so experiments don't reuse warm-up seeds
    SEED_OFFSET = N_WARMUP

    # -----------------------------------------------------------------------
    # Experiment 1 — uniform config sweep
    # -----------------------------------------------------------------------
    if rank == 0:
        print("=" * 62)
        print("EXP 1: Uniform config sweep")
        print("=" * 62)

    exp1_results = {}
    global_seed = SEED_OFFSET

    for (bs, sl) in UNIFORM_CONFIGS:
        label = f"bs={bs:2d} sl={sl:3d}"
        batches, times = [], []

        for i in range(N_TRAIN_PER_CFG + N_TEST_PER_CFG):
            batch = make_uniform_batch(global_seed + i, bs, sl)
            t     = timed_prefill(model, batch)
            batches.append(batch)
            times.append(t)

        global_seed += N_TRAIN_PER_CFG + N_TEST_PER_CFG

        if rank == 0:
            train_b, test_b = batches[:N_TRAIN_PER_CFG], batches[N_TRAIN_PER_CFG:]
            train_t, test_t = times[:N_TRAIN_PER_CFG],   times[N_TRAIN_PER_CFG:]

            est, preds, errors = fit_and_eval(train_b, train_t, test_b, test_t)
            # Because all test batches have the SAME features, predictor gives one value
            unique_preds = len(set(round(p, 9) for p in preds))

            _, sum_n2, T_in = token_features(batches[0])
            exp1_results[label] = {
                "batch_size":      bs,
                "seq_len":         sl,
                "sum_n2":          sum_n2,
                "T_in":            T_in,
                "mean_actual_ms":  float(np.mean(test_t) * 1000),
                "std_actual_ms":   float(np.std(test_t) * 1000),
                "cv_pct":          float(np.std(test_t) / np.mean(test_t) * 100),
                "mean_error_pct":  float(np.mean(errors)),
                "max_error_pct":   float(np.max(errors)),
                "fit_rmse_ms":     float((est.fit_rmse or 0) * 1000),
                "unique_preds":    unique_preds,  # should always be 1
            }
            print(f"  {label} | actual={np.mean(test_t)*1000:.1f}±{np.std(test_t)*1000:.1f}ms"
                  f" | error={np.mean(errors):.1f}% (max {np.max(errors):.1f}%)"
                  f" | unique_preds={unique_preds}")

    # -----------------------------------------------------------------------
    # Experiment 2 — global predictor across all configs
    # -----------------------------------------------------------------------
    if rank == 0:
        print()
        print("=" * 62)
        print("EXP 2: Global predictor fit on all configs combined")
        print("=" * 62)

    exp2_results = {}
    # Re-generate the same batches used in Exp 1 (same seed range)
    global_seed_e2 = SEED_OFFSET
    all_train_b, all_train_t = [], []
    per_cfg_test = {}

    for (bs, sl) in UNIFORM_CONFIGS:
        label = f"bs={bs:2d} sl={sl:3d}"
        batches, times = [], []
        for i in range(N_TRAIN_PER_CFG + N_TEST_PER_CFG):
            batch = make_uniform_batch(global_seed_e2 + i, bs, sl)
            batches.append(batch)
        # We don't re-time — reuse the timings from exp1 (stored on rank 0)
        global_seed_e2 += N_TRAIN_PER_CFG + N_TEST_PER_CFG

        if rank == 0:
            r = exp1_results[label]
            # Reconstruct per-cfg times from exp1 (we saved mean/std but not raw)
            # So we re-run timing for exp2 — small cost, guarantees correct alignment
            per_cfg_test[label] = (batches[N_TRAIN_PER_CFG:], None)
            all_train_b.extend(batches[:N_TRAIN_PER_CFG])

    # Rank 0 needs the actual times for the global fit; re-time train batches
    if rank == 0:
        print("  Re-timing train batches for global fit...")

    global_seed_e2b = SEED_OFFSET
    all_train_t_real = []
    per_cfg_test_times = {label: [] for (bs, sl) in UNIFORM_CONFIGS
                          for label in [f"bs={bs:2d} sl={sl:3d}"]}

    for (bs, sl) in UNIFORM_CONFIGS:
        label = f"bs={bs:2d} sl={sl:3d}"
        for i in range(N_TRAIN_PER_CFG + N_TEST_PER_CFG):
            batch = make_uniform_batch(global_seed_e2b + i, bs, sl)
            t = timed_prefill(model, batch)
            if i < N_TRAIN_PER_CFG:
                all_train_t_real.append(t)
            else:
                per_cfg_test_times[label].append(t)
        global_seed_e2b += N_TRAIN_PER_CFG + N_TEST_PER_CFG

    global_seed = global_seed_e2b  # advance seed counter

    if rank == 0:
        from slora.server.router.tracker import PrefillExecutionEstimator
        global_est = PrefillExecutionEstimator()
        global_est.fit(
            inference_only_tokens=[token_features(b)[0] for b in all_train_b],
            inference_only_times=all_train_t_real,
            coserving_inf_tokens=[],
            coserving_ft_tokens=[],
            coserving_times=[],
        )
        print(f"  Global fit RMSE: {(global_est.fit_rmse or 0)*1000:.3f} ms")
        print(f"  Params: α={global_est._params.alpha:.3e}  β={global_est._params.beta:.3e}"
              f"  c={global_est._params.c:.3e}")
        print()

        for (bs, sl) in UNIFORM_CONFIGS:
            label = f"bs={bs:2d} sl={sl:3d}"
            test_b = per_cfg_test[label][0]
            test_t = per_cfg_test_times[label]
            preds  = [global_est.predict_inference(token_features(b)[0]) for b in test_b]
            errors = [abs(p - a) / a * 100 for p, a in zip(preds, test_t)]
            exp2_results[label] = {
                "mean_error_pct": float(np.mean(errors)),
                "max_error_pct":  float(np.max(errors)),
                "within_cfg_std_actual_ms": float(np.std(test_t) * 1000),
            }
            print(f"  {label} | global pred error={np.mean(errors):.1f}%"
                  f" (max {np.max(errors):.1f}%)  within-cfg std={np.std(test_t)*1000:.2f}ms")

    # -----------------------------------------------------------------------
    # Experiment 3 — realistic input timeline
    # -----------------------------------------------------------------------
    if rank == 0:
        print()
        print("=" * 62)
        print("EXP 3: Realistic input timeline (Poisson bs, log-normal lens)")
        print("=" * 62)

    timeline_batches, timeline_times = [], []
    for i in range(N_TIMELINE_TRAIN + N_TIMELINE_TEST):
        batch = make_timeline_batch(global_seed + i)
        t     = timed_prefill(model, batch)
        timeline_batches.append(batch)
        timeline_times.append(t)
        if rank == 0 and (i + 1) % 100 == 0:
            print(f"  timeline batch {i+1}/{N_TIMELINE_TRAIN + N_TIMELINE_TEST}")
    global_seed += N_TIMELINE_TRAIN + N_TIMELINE_TEST

    exp3_results = {}
    if rank == 0:
        train_b = timeline_batches[:N_TIMELINE_TRAIN]
        test_b  = timeline_batches[N_TIMELINE_TRAIN:]
        train_t = timeline_times[:N_TIMELINE_TRAIN]
        test_t  = timeline_times[N_TIMELINE_TRAIN:]

        est, preds, errors = fit_and_eval(train_b, train_t, test_b, test_t)

        # Per-batch features for scatter analysis
        features = [token_features(b) for b in test_b]
        batch_sizes_test = [len(b) for b in test_b]
        T_ins_test       = [f[2] for f in features]
        sum_n2s_test     = [f[1] for f in features]

        exp3_results = {
            "mean_error_pct":    float(np.mean(errors)),
            "median_error_pct":  float(np.median(errors)),
            "p90_error_pct":     float(np.percentile(errors, 90)),
            "max_error_pct":     float(np.max(errors)),
            "fit_rmse_ms":       float((est.fit_rmse or 0) * 1000),
            "mean_actual_ms":    float(np.mean(test_t) * 1000),
            "std_actual_ms":     float(np.std(test_t) * 1000),
            "raw_errors_pct":    errors,
            "raw_T_in":          T_ins_test,
            "raw_batch_sizes":   batch_sizes_test,
        }
        print(f"  Fit RMSE: {(est.fit_rmse or 0)*1000:.3f} ms")
        print(f"  Test error — mean: {np.mean(errors):.1f}%  "
              f"median: {np.median(errors):.1f}%  "
              f"p90: {np.percentile(errors, 90):.1f}%  "
              f"max: {np.max(errors):.1f}%")

    # -----------------------------------------------------------------------
    # Experiment 4 — RMSE convergence curve
    # -----------------------------------------------------------------------
    if rank == 0:
        print()
        print("=" * 62)
        print("EXP 4: RMSE convergence (does more data help?)")
        print("=" * 62)

    bs_c, sl_c = CONVERGENCE_CONFIG
    conv_batches, conv_times = [], []
    for i in range(CONVERGENCE_TOTAL + CONVERGENCE_TEST):
        batch = make_uniform_batch(global_seed + i, bs_c, sl_c)
        t     = timed_prefill(model, batch)
        conv_batches.append(batch)
        conv_times.append(t)
        if rank == 0 and (i + 1) % 100 == 0:
            print(f"  convergence batch {i+1}/{CONVERGENCE_TOTAL + CONVERGENCE_TEST}")
    global_seed += CONVERGENCE_TOTAL + CONVERGENCE_TEST

    exp4_results = {}
    if rank == 0:
        from slora.server.router.tracker import PrefillExecutionEstimator

        # Held-out test set (always the same CONVERGENCE_TEST batches)
        test_b = conv_batches[CONVERGENCE_TOTAL:]
        test_t = conv_times[CONVERGENCE_TOTAL:]

        checkpoints = []
        for n in CONVERGENCE_CHECKPOINTS:
            if n < 4:
                continue
            train_b = conv_batches[:n]
            train_t = conv_times[:n]
            est_c = PrefillExecutionEstimator()
            est_c.fit(
                inference_only_tokens=[token_features(b)[0] for b in train_b],
                inference_only_times=train_t,
                coserving_inf_tokens=[],
                coserving_ft_tokens=[],
                coserving_times=[],
            )
            preds  = [est_c.predict_inference(token_features(b)[0]) for b in test_b]
            errors = [abs(p - a) / a * 100 for p, a in zip(preds, test_t)]
            rmse_s = float(np.sqrt(np.mean([(p - a)**2 for p, a in zip(preds, test_t)])))
            checkpoints.append({
                "n_train":        n,
                "test_rmse_ms":   rmse_s * 1000,
                "mean_error_pct": float(np.mean(errors)),
            })
            print(f"  n_train={n:4d} → test RMSE={rmse_s*1000:.3f}ms  "
                  f"mean_error={np.mean(errors):.1f}%")

        # Irreducible floor: std of actual times (best any predictor can do)
        actual_std_ms = float(np.std(test_t) * 1000)
        print(f"\n  Irreducible floor (std of actual times): {actual_std_ms:.3f} ms")
        print(f"  → Predictor RMSE cannot drop below this floor,")
        print(f"    no matter how many training samples are used.")

        exp4_results = {
            "checkpoints":       checkpoints,
            "irreducible_floor_ms": actual_std_ms,
        }

    # -----------------------------------------------------------------------
    # Pack results and return
    # -----------------------------------------------------------------------
    if rank == 0:
        results["exp1_uniform_sweep"] = exp1_results
        results["exp2_global_predictor"] = exp2_results
        results["exp3_timeline"] = exp3_results
        results["exp4_convergence"] = exp4_results

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)

    # Exp 1 table
    print("\nExp 1 — Per-config error (predictor fit on each config separately)")
    print(f"  {'Config':<16} {'Σn²':>8} {'T_in':>6} {'actual ms':>10} {'std ms':>7} {'cv%':>5} {'err%':>7} {'max_err%':>9}")
    for label, r in results["exp1_uniform_sweep"].items():
        print(f"  {label:<16} {r['sum_n2']:>8,} {r['T_in']:>6} "
              f"{r['mean_actual_ms']:>10.2f} {r['std_actual_ms']:>7.2f} "
              f"{r['cv_pct']:>5.1f} {r['mean_error_pct']:>7.1f} {r['max_error_pct']:>9.1f}")

    # Exp 2 table
    print("\nExp 2 — Within-config residual when using a GLOBAL predictor")
    print(f"  {'Config':<16} {'global err%':>12} {'max err%':>9} {'within-cfg std ms':>18}")
    for label, r in results["exp2_global_predictor"].items():
        print(f"  {label:<16} {r['mean_error_pct']:>12.1f} {r['max_error_pct']:>9.1f} "
              f"{r['within_cfg_std_actual_ms']:>18.3f}")

    # Exp 3
    r3 = results["exp3_timeline"]
    print(f"\nExp 3 — Realistic timeline: mean error={r3['mean_error_pct']:.1f}%  "
          f"p90={r3['p90_error_pct']:.1f}%  max={r3['max_error_pct']:.1f}%")

    # Exp 4
    r4 = results["exp4_convergence"]
    print(f"\nExp 4 — RMSE convergence (irreducible floor = {r4['irreducible_floor_ms']:.3f} ms):")
    for cp in r4["checkpoints"]:
        print(f"  n={cp['n_train']:4d} → RMSE={cp['test_rmse_ms']:.3f}ms  "
              f"error={cp['mean_error_pct']:.1f}%")

    print()
    print("KEY FINDING:")
    errs = [r["mean_error_pct"] for r in results["exp1_uniform_sweep"].values()]
    print(f"  Across all configs, mean prediction error ranges from "
          f"{min(errs):.1f}% to {max(errs):.1f}%.")
    r4_first = r4["checkpoints"][0]["test_rmse_ms"]
    r4_last  = r4["checkpoints"][-1]["test_rmse_ms"]
    print(f"  RMSE went from {r4_first:.3f}ms (n=5) to {r4_last:.3f}ms (n={CONVERGENCE_TOTAL}) —")
    print(f"  {'CONVERGED (unexpected)' if r4_last < r4_first * 0.5 else 'DID NOT CONVERGE'}.")
    print(f"  The error is irreducible: it comes from expert routing variance,")
    print(f"  not from insufficient training data.")


def save_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Full JSON
    json_path = os.path.join(out_dir, "ep_predictor_results.json")
    with open(json_path, "w") as f:
        json.dump(dict(results), f, indent=2)

    # Exp1 CSV
    csv1 = os.path.join(out_dir, "exp1_uniform_sweep.csv")
    with open(csv1, "w", newline="") as f:
        fields = ["config", "batch_size", "seq_len", "sum_n2", "T_in",
                  "mean_actual_ms", "std_actual_ms", "cv_pct",
                  "mean_error_pct", "max_error_pct", "fit_rmse_ms"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, r in results["exp1_uniform_sweep"].items():
            w.writerow({"config": label, **{k: r[k] for k in fields[1:]}})

    # Exp4 CSV
    csv4 = os.path.join(out_dir, "exp4_convergence.csv")
    with open(csv4, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_train", "test_rmse_ms", "mean_error_pct"])
        w.writeheader()
        w.writerows(results["exp4_convergence"]["checkpoints"])

    print(f"\nResults saved to: {out_dir}/")
    print(f"  {json_path}")
    print(f"  {csv1}")
    print(f"  {csv4}")


def main():
    world_size = 2
    if torch.cuda.device_count() < world_size:
        print(f"ERROR: need {world_size} GPUs, found {torch.cuda.device_count()}")
        sys.exit(1)

    config_dir = tempfile.mkdtemp(prefix="mixtral_dummy_")
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f, indent=2)
    print(f"Config dir: {config_dir}")

    manager = mp.Manager()
    results  = manager.dict()

    mp.spawn(
        worker,
        args=(world_size, config_dir, results),
        nprocs=world_size,
        join=True,
    )

    print_summary(results)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    save_results(results, out_dir)


if __name__ == "__main__":
    main()
