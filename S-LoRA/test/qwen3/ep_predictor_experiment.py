#!/usr/bin/env python3
"""
EP Predictor Failure Experiment — Qwen3-30B-A3B (2-GPU Expert Parallelism)
===========================================================================
Mirrors test/mixtral/ep_predictor_experiment.py for Qwen3, to compare
predictor behaviour on a second MoE architecture.

Key Qwen3 differences exercised here:
  - model_type = "qwen3_moe"
  - Config fields: num_experts, moe_intermediate_size, head_dim (explicit)
  - Stacked expert weights: gate_up_proj (num_experts, 2*moe_inter, hidden)
  - MLP key prefix "mlp" (not "block_sparse_moe")
  - Per-head q_norm / k_norm before RoPE
  - num_experts_per_tok = 2 in tiny config (8 in real model)

The tiny config uses a non-standard head_dim (128) with hidden_size=1024 and
num_attention_heads=16, so hidden_size // num_heads = 64 ≠ head_dim.
This specifically exercises the head_dim override path in Qwen3's layer infer.

Experiments
-----------
1. Uniform config sweep     — per-config predictor fit; shows flat RMSE
2. Global predictor         — one estimator across all configs
3. Realistic timeline       — Poisson bs, log-normal seq lengths
4. RMSE convergence curve   — does more data reduce RMSE?

Usage
-----
    cd S-LoRA
    python test/qwen3/ep_predictor_experiment.py
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
# Tiny Qwen3-MoE config — fits on GPU, runs fast, exercises all Qwen3 paths.
#
# Non-standard head_dim: hidden_size=1024, num_attention_heads=16
#   → hidden_size // num_heads = 64, but head_dim = 128
# This matches the real Qwen3-30B-A3B pattern (2048 // 32 = 64, head_dim=128).
#
# 8 experts, 2 GPUs → each rank owns 4 experts at full moe_intermediate_size.
# ---------------------------------------------------------------------------
TINY_CONFIG = {
    "model_type":             "qwen3_moe",
    "hidden_size":            1024,
    "num_attention_heads":    16,
    "num_key_value_heads":    4,
    "head_dim":               128,     # non-standard: != hidden_size // num_heads
    "num_experts":            8,
    "num_experts_per_tok":    2,
    "moe_intermediate_size":  512,
    "intermediate_size":      2048,    # dense FFN (not used by MoE layers, but may be read)
    "num_hidden_layers":      4,
    "vocab_size":             4096,
    "rms_norm_eps":           1e-6,
    "max_position_embeddings": 4096,
}

NCCL_PORT  = 29514    # Qwen3 dummy exp1; see notes/feedback_working_style.md for full port map
MAX_POOL   = 8192
N_WARMUP   = 30
VOCAB_SIZE = TINY_CONFIG["vocab_size"]

UNIFORM_CONFIGS = [
    (1,  256),
    (2,  128),
    (4,   64),
    (8,   32),
    (16,  16),
    (4,   32),
    (4,  128),
    (8,   64),
]
N_TRAIN_PER_CFG = 120
N_TEST_PER_CFG  = 120

N_TIMELINE_TRAIN = 200
N_TIMELINE_TEST  = 200
TIMELINE_MEAN_BS  = 4
TIMELINE_MEAN_LEN = 64
TIMELINE_STD_LEN  = 48

CONVERGENCE_CONFIG      = (4, 64)
CONVERGENCE_TOTAL       = 500
CONVERGENCE_CHECKPOINTS = [5, 10, 25, 50, 100, 200, 350, 500]
CONVERGENCE_TEST        = 200


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_prefill_inputs(token_ids_list: list, device: str = "cuda"):
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
    inputs = build_prefill_inputs(token_ids_list)
    input_ids, b_loc, b_start_loc, b_seq_len, bs, total, max_len = inputs

    model.mem_manager.free_all()
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
    rng = np.random.default_rng(seed)
    bs  = max(1, rng.poisson(TIMELINE_MEAN_BS))
    mu  = np.log(TIMELINE_MEAN_LEN**2 / np.sqrt(TIMELINE_MEAN_LEN**2 + TIMELINE_STD_LEN**2))
    sig = np.sqrt(np.log(1 + (TIMELINE_STD_LEN / TIMELINE_MEAN_LEN)**2))
    lens = np.clip(rng.lognormal(mu, sig, bs).astype(int), 4, 256)
    return [rng.integers(0, VOCAB_SIZE, size=l) for l in lens]


def token_features(token_ids_list: list) -> tuple:
    lens   = [len(ids) for ids in token_ids_list]
    sum_n2 = sum(l**2 for l in lens)
    T_in   = sum(lens)
    return lens, sum_n2, T_in


def fit_and_eval(train_token_lists, train_times, test_token_lists, test_times):
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
# Worker
# ---------------------------------------------------------------------------

def worker(rank: int, world_size: int, config_dir: str, results: dict):
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{NCCL_PORT}",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    from slora.models.qwen3_moe.model import Qwen3MoeEPTpPartModel

    if rank == 0:
        print("Loading Qwen3-MoE with dummy weights...")

    model = Qwen3MoeEPTpPartModel(
        tp_rank=rank,
        world_size=world_size,
        weight_dir=config_dir,
        max_total_token_num=MAX_POOL,
        mem_adapter_size=0,
        dummy=True,
    )

    if rank == 0:
        print("Model ready.\n")

    # Warmup
    if rank == 0:
        print(f"Warming up ({N_WARMUP} batches)...")
    for i in range(N_WARMUP):
        timed_prefill(model, make_uniform_batch(i, 4, 64))
    if rank == 0:
        print("Warmup done.\n")

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
                "unique_preds":    unique_preds,
            }
            print(f"  {label} | actual={np.mean(test_t)*1000:.1f}±{np.std(test_t)*1000:.1f}ms"
                  f" | error={np.mean(errors):.1f}% (max {np.max(errors):.1f}%)"
                  f" | unique_preds={unique_preds}")

    # -----------------------------------------------------------------------
    # Experiment 2 — global predictor
    # -----------------------------------------------------------------------
    if rank == 0:
        print()
        print("=" * 62)
        print("EXP 2: Global predictor fit on all configs combined")
        print("=" * 62)

    exp2_results = {}
    global_seed_e2 = SEED_OFFSET
    all_train_b = []
    per_cfg_test = {}

    for (bs, sl) in UNIFORM_CONFIGS:
        label = f"bs={bs:2d} sl={sl:3d}"
        batches = []
        for i in range(N_TRAIN_PER_CFG + N_TEST_PER_CFG):
            batch = make_uniform_batch(global_seed_e2 + i, bs, sl)
            batches.append(batch)
        global_seed_e2 += N_TRAIN_PER_CFG + N_TEST_PER_CFG

        if rank == 0:
            per_cfg_test[label] = (batches[N_TRAIN_PER_CFG:], None)
            all_train_b.extend(batches[:N_TRAIN_PER_CFG])

    if rank == 0:
        print("  Re-timing train batches for global fit...")

    global_seed_e2b = SEED_OFFSET
    all_train_t_real = []
    per_cfg_test_times = {f"bs={bs:2d} sl={sl:3d}": [] for (bs, sl) in UNIFORM_CONFIGS}

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

    global_seed = global_seed_e2b

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
    # Experiment 3 — realistic timeline
    # -----------------------------------------------------------------------
    if rank == 0:
        print()
        print("=" * 62)
        print("EXP 3: Realistic timeline (Poisson bs, log-normal lens)")
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
        exp3_results = {
            "mean_error_pct":   float(np.mean(errors)),
            "median_error_pct": float(np.median(errors)),
            "p90_error_pct":    float(np.percentile(errors, 90)),
            "max_error_pct":    float(np.max(errors)),
            "fit_rmse_ms":      float((est.fit_rmse or 0) * 1000),
            "mean_actual_ms":   float(np.mean(test_t) * 1000),
            "std_actual_ms":    float(np.std(test_t) * 1000),
            "raw_errors_pct":   errors,
            "raw_T_in":         [token_features(b)[2] for b in test_b],
            "raw_batch_sizes":  [len(b) for b in test_b],
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

        actual_std_ms = float(np.std(test_t) * 1000)
        print(f"\n  Irreducible floor (std of actual times): {actual_std_ms:.3f} ms")

        exp4_results = {
            "checkpoints":          checkpoints,
            "irreducible_floor_ms": actual_std_ms,
        }

    if rank == 0:
        results["exp1_uniform_sweep"]    = exp1_results
        results["exp2_global_predictor"] = exp2_results
        results["exp3_timeline"]         = exp3_results
        results["exp4_convergence"]      = exp4_results

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    print("\n" + "=" * 62)
    print("SUMMARY — Qwen3-30B-A3B (tiny dummy config)")
    print("=" * 62)

    print("\nExp 1 — Per-config error")
    print(f"  {'Config':<16} {'Σn²':>8} {'T_in':>6} {'actual ms':>10} {'std ms':>7} {'cv%':>5} {'err%':>7} {'max_err%':>9}")
    for label, r in results["exp1_uniform_sweep"].items():
        print(f"  {label:<16} {r['sum_n2']:>8,} {r['T_in']:>6} "
              f"{r['mean_actual_ms']:>10.2f} {r['std_actual_ms']:>7.2f} "
              f"{r['cv_pct']:>5.1f} {r['mean_error_pct']:>7.1f} {r['max_error_pct']:>9.1f}")

    print("\nExp 2 — Global predictor residual")
    print(f"  {'Config':<16} {'global err%':>12} {'max err%':>9} {'within-cfg std ms':>18}")
    for label, r in results["exp2_global_predictor"].items():
        print(f"  {label:<16} {r['mean_error_pct']:>12.1f} {r['max_error_pct']:>9.1f} "
              f"{r['within_cfg_std_actual_ms']:>18.3f}")

    r3 = results["exp3_timeline"]
    print(f"\nExp 3 — Realistic timeline: mean error={r3['mean_error_pct']:.1f}%  "
          f"p90={r3['p90_error_pct']:.1f}%  max={r3['max_error_pct']:.1f}%")

    r4 = results["exp4_convergence"]
    print(f"\nExp 4 — RMSE convergence (floor = {r4['irreducible_floor_ms']:.3f} ms):")
    for cp in r4["checkpoints"]:
        print(f"  n={cp['n_train']:4d} → RMSE={cp['test_rmse_ms']:.3f}ms  "
              f"error={cp['mean_error_pct']:.1f}%")

    errs = [r["mean_error_pct"] for r in results["exp1_uniform_sweep"].values()]
    r4_first = r4["checkpoints"][0]["test_rmse_ms"]
    r4_last  = r4["checkpoints"][-1]["test_rmse_ms"]
    print(f"\nKEY: prediction error {min(errs):.1f}–{max(errs):.1f}% across configs. "
          f"RMSE {r4_first:.3f}ms→{r4_last:.3f}ms "
          f"({'converged' if r4_last < r4_first * 0.5 else 'did not converge'}).")


def save_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "ep_predictor_results.json")
    with open(json_path, "w") as f:
        json.dump(dict(results), f, indent=2)

    csv1 = os.path.join(out_dir, "exp1_uniform_sweep.csv")
    with open(csv1, "w", newline="") as f:
        fields = ["config", "batch_size", "seq_len", "sum_n2", "T_in",
                  "mean_actual_ms", "std_actual_ms", "cv_pct",
                  "mean_error_pct", "max_error_pct", "fit_rmse_ms"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, r in results["exp1_uniform_sweep"].items():
            w.writerow({"config": label, **{k: r[k] for k in fields[1:]}})

    csv4 = os.path.join(out_dir, "exp4_convergence.csv")
    with open(csv4, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_train", "test_rmse_ms", "mean_error_pct"])
        w.writeheader()
        w.writerows(results["exp4_convergence"]["checkpoints"])

    print(f"\nResults saved to: {out_dir}/")


def main():
    world_size = 2
    if torch.cuda.device_count() < world_size:
        print(f"ERROR: need {world_size} GPUs, found {torch.cuda.device_count()}")
        sys.exit(1)

    # Write tiny config to a temp dir — model._init_config reads config.json from here
    config_dir = tempfile.mkdtemp(prefix="qwen3_dummy_")
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f, indent=2)
    print(f"Tiny config written to: {config_dir}")
    print(f"  hidden_size={TINY_CONFIG['hidden_size']}, "
          f"head_dim={TINY_CONFIG['head_dim']} (non-standard: hidden/heads="
          f"{TINY_CONFIG['hidden_size']//TINY_CONFIG['num_attention_heads']})")
    print(f"  num_experts={TINY_CONFIG['num_experts']}, "
          f"num_experts_per_tok={TINY_CONFIG['num_experts_per_tok']}, "
          f"moe_intermediate_size={TINY_CONFIG['moe_intermediate_size']}")
    print()

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
