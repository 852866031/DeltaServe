#!/usr/bin/env python3
"""
Experiment 2 — Mixtral Predictor Stress Test
=============================================
Five experiments targeting conditions where PrefillExecutionEstimator and
DecodeExecutionEstimator systematically fail, exposing predictor weaknesses
that cause SLO violations or SFT starvation in DeltaServe's check_will_starve().

Requires 2 GPUs (EP mode) and Mixtral-8x7B-v0.1 weights.

Usage:
    cd S-LoRA
    python test/mixtral/exp2/stress_test.py

Experiments:
    A — Heterogeneous batch composition (uniform vs bimodal vs skewed lengths)
    B — Decode predictor under progressive K growth (100 decode steps)
    C — Content-dependent routing variance (random/constant/low_ids/high_ids)
    D — RMSE convergence vs training set size (high-CV vs low-CV configs)
    E — SLO gate error rate (FP/FN vs slack level)
"""

import os, sys, json, csv, time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = "/mnt/nfs/home/ramya/models/mistralai/Mixtral-8x7B-v0.1"
MAX_POOL  = 15_000
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------

def run_prefill(model, token_ids_list):
    """Prefill forward pass. Returns (time_s, b_loc, b_seq_len, b_start_loc)."""
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
    return time.perf_counter() - t0, b_loc, b_seq_len, b_start_loc


def run_decode_step(model, b_loc, b_seq_len, b_start_loc, vocab_size):
    """One decode step. Returns (time_s, new_b_loc, new_b_seq_len)."""
    bs          = int(b_seq_len.shape[0])
    b_loc_d     = torch.cat([b_loc, torch.zeros(bs, 1, dtype=torch.long, device="cuda")], dim=1)
    b_seq_len_d = b_seq_len + 1
    total_d     = int(b_seq_len_d.sum().item())
    max_len_d   = int(b_seq_len_d.max().item())
    input_ids_d = torch.randint(0, vocab_size, (bs,), dtype=torch.long, device="cuda")

    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(batch_size=bs, total_token_num=total_d, max_len_in_batch=max_len_d,
                      input_ids=input_ids_d, b_loc=b_loc_d, b_start_loc=b_start_loc,
                      b_seq_len=b_seq_len_d, is_prefill=False)
    torch.cuda.synchronize(); dist.barrier()
    return time.perf_counter() - t0, b_loc_d, b_seq_len_d


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
# Experiment D — RMSE convergence vs. training set size
# ---------------------------------------------------------------------------

def exp_d(model, vocab_size, rank):
    """Show that prefill RMSE plateaus for high-CV configs regardless of n_train."""
    from slora.server.router.tracker import PrefillExecutionEstimator

    configs = [
        {"label": "high_cv_bs4_sl256",  "bs": 4,  "sl": 256,  "cv_class": "high"},
        {"label": "high_cv_bs1_sl1024", "bs": 1,  "sl": 1024, "cv_class": "high"},
        {"label": "low_cv_bs32_sl256",  "bs": 32, "sl": 256,  "cv_class": "low"},
        {"label": "low_cv_bs1_sl8192",  "bs": 1,  "sl": 8192, "cv_class": "low"},
    ]
    N_TOTAL        = 500
    N_TEST         = 80
    N_TRAIN_POINTS = [5, 10, 20, 40, 80, 160, 320, 500]

    results = []
    rng = np.random.default_rng(400)

    for cfg in configs:
        bs, sl = cfg["bs"], cfg["sl"]
        label  = cfg["label"]
        if rank == 0:
            print(f"\n[Exp D] Collecting {N_TOTAL + N_TEST} prefill times for {label} ...")

        times = []
        for i in range(N_TOTAL + N_TEST):
            batch = [rng.integers(0, vocab_size, size=sl) for _ in range(bs)]
            pt, _, _, _ = run_prefill(model, batch)
            times.append(pt)
            if rank == 0 and (i + 1) % 100 == 0:
                print(f"  [{label}] {i+1}/{N_TOTAL + N_TEST}")

        if rank == 0:
            train_all  = times[:N_TOTAL]
            test_times = times[N_TOTAL:]
            lens       = [sl] * bs
            actual_std = float(np.std(test_times) * 1000)
            actual_cv  = float(np.std(test_times) / np.mean(test_times) * 100)
            print(f"  [{label}] actual_cv={actual_cv:.1f}% actual_std={actual_std:.3f}ms")

            for n_train in N_TRAIN_POINTS:
                p_est = PrefillExecutionEstimator()
                p_est.fit(
                    inference_only_tokens=[lens] * n_train,
                    inference_only_times=train_all[:n_train],
                    coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
                )
                p_pred  = p_est.predict_inference(lens)
                rmse_ms = float(np.sqrt(np.mean([(p_pred - a)**2 for a in test_times])) * 1000)
                mae_pct = float(np.mean([abs(p_pred - a) / a * 100 for a in test_times]))
                results.append({
                    "label":          label,
                    "cv_class":       cfg["cv_class"],
                    "n_train":        n_train,
                    "rmse_ms":        rmse_ms,
                    "fit_rmse_ms":    float((p_est.fit_rmse or 0) * 1000),
                    "mae_pct":        mae_pct,
                    "actual_std_ms":  actual_std,
                    "actual_cv_pct":  actual_cv,
                    "actual_mean_ms": float(np.mean(test_times) * 1000),
                })
                print(f"  [{label}] n_train={n_train:3d}  rmse={rmse_ms:.3f}ms  mae={mae_pct:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Experiment B — Decode predictor under progressive K growth
# ---------------------------------------------------------------------------

def exp_b(model, vocab_size, rank):
    """Run 100 consecutive decode steps; fit on first W steps, eval on rest."""
    from slora.server.router.tracker import DecodeExecutionEstimator

    BS, SL        = 16, 256
    N_STEPS       = 100
    TRAIN_WINDOWS = [10, 30, 50]
    N_TRIALS      = 5

    if rank == 0:
        print(f"\n[Exp B] bs={BS} sl={SL}, {N_STEPS} decode steps x {N_TRIALS} trials")

    rng = np.random.default_rng(200)
    all_decode_times = []   # (N_TRIALS, N_STEPS)

    for trial in range(N_TRIALS):
        batch = [rng.integers(0, vocab_size, size=SL) for _ in range(BS)]
        _, b_loc, b_seq_len, b_start_loc = run_prefill(model, batch)

        decode_times = []
        for step in range(N_STEPS):
            dt, b_loc, b_seq_len = run_decode_step(model, b_loc, b_seq_len, b_start_loc, vocab_size)
            decode_times.append(dt)
            if rank == 0 and (step + 1) % 25 == 0:
                print(f"  trial={trial} step={step+1}/{N_STEPS} dt={dt*1000:.2f}ms")
        all_decode_times.append(decode_times)

    results = []
    if rank == 0:
        arr = np.array(all_decode_times)          # (N_TRIALS, N_STEPS)
        mean_times = arr.mean(axis=0)             # (N_STEPS,)
        k_values   = [BS * (SL + step + 1) for step in range(N_STEPS)]

        cv_early = arr[:, :10].std() / arr[:, :10].mean() * 100
        cv_late  = arr[:, 90:].std() / arr[:, 90:].mean() * 100
        print(f"  CV steps 0-9: {cv_early:.1f}%   CV steps 90-99: {cv_late:.1f}%")
        print(f"  step=0  actual={mean_times[0]*1000:.2f}ms  K={k_values[0]}")
        print(f"  step=49 actual={mean_times[49]*1000:.2f}ms  K={k_values[49]}")
        print(f"  step=99 actual={mean_times[99]*1000:.2f}ms  K={k_values[99]}")

        for train_window in TRAIN_WINDOWS:
            train_idx = list(range(train_window))
            test_idx  = list(range(train_window, N_STEPS))

            d_est = DecodeExecutionEstimator()
            d_est.fit(
                total_tokens=[k_values[i] for i in train_idx],
                batch_sizes=[BS] * train_window,
                times=[mean_times[i] for i in train_idx],
            )

            for step in test_idx:
                K        = k_values[step]
                pred_t   = d_est.predict(total_tokens=K, batch_size=BS)
                actual_t = mean_times[step]
                err_pct  = (pred_t - actual_t) / actual_t * 100
                results.append({
                    "train_window":    train_window,
                    "step":            step,
                    "K":               K,
                    "actual_ms":       float(actual_t * 1000),
                    "pred_ms":         float(pred_t * 1000),
                    "signed_err_pct":  float(err_pct),
                })

    return results


# ---------------------------------------------------------------------------
# Experiment E — SLO gate error rate
# ---------------------------------------------------------------------------

def exp_e(model, vocab_size, rank):
    """FP (SLO violations) and FN (SFT starvation) rates vs slack level."""
    from slora.server.router.tracker import PrefillExecutionEstimator

    N_TRAIN       = 200
    N_TEST        = 200
    SLO_SLACKS_MS = [50, 100, 150, 200, 300, 500]

    # Log-normal seq length: mean=64, std=48
    log_mean = np.log(64**2 / np.sqrt(64**2 + 48**2))
    log_std  = np.sqrt(np.log(1 + 48**2 / 64**2))

    rng = np.random.default_rng(500)

    def sample_batch():
        bs  = max(1, min(16, rng.poisson(4)))
        sls = np.clip(
            np.round(rng.lognormal(log_mean, log_std, size=bs)).astype(int),
            16, 4096
        )
        # Clip total tokens to fit in KV pool
        while int(sls.sum()) > MAX_POOL - 1000:
            sls = np.clip(sls // 2, 16, 4096)
        return [rng.integers(0, vocab_size, size=int(sl)) for sl in sls]

    if rank == 0:
        print(f"\n[Exp E] Realistic batches: {N_TRAIN} train + {N_TEST} test")

    train_batches, train_times = [], []
    test_batches,  test_times  = [], []

    for i in range(N_TRAIN + N_TEST):
        batch = sample_batch()
        pt, _, _, _ = run_prefill(model, batch)
        if i < N_TRAIN:
            train_batches.append(batch)
            train_times.append(pt)
        else:
            test_batches.append(batch)
            test_times.append(pt)
        if rank == 0 and (i + 1) % 50 == 0:
            bs_i = len(batch)
            sl_i = len(batch[0])
            print(f"  [{i+1}/{N_TRAIN + N_TEST}] bs={bs_i} sl={sl_i} pt={pt*1000:.2f}ms")

    results = []
    if rank == 0:
        p_est = PrefillExecutionEstimator()
        p_est.fit(
            inference_only_tokens=[[len(s) for s in b] for b in train_batches],
            inference_only_times=train_times,
            coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
        )

        preds = [p_est.predict_inference([len(s) for s in b]) for b in test_batches]

        for slack_ms in SLO_SLACKS_MS:
            slack = slack_ms / 1000.0
            tp = fp = tn = fn = 0
            for pred_t, actual_t in zip(preds, test_times):
                gate_admit = pred_t < slack
                true_admit = actual_t < slack
                if     gate_admit and     true_admit: tp += 1
                elif   gate_admit and not true_admit: fp += 1   # SLO violation
                elif not gate_admit and   true_admit: fn += 1   # SFT starvation
                else:                                 tn += 1

            total_pos = tp + fn   # batches that would actually fit
            total_neg = fp + tn   # batches that would actually violate
            fp_rate = fp / total_neg if total_neg > 0 else 0.0
            fn_rate = fn / total_pos if total_pos > 0 else 0.0
            results.append({
                "slack_ms":        slack_ms,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "fp_rate":         float(fp_rate),
                "fn_rate":         float(fn_rate),
                "admit_rate_pred": float((tp + fp) / N_TEST),
                "admit_rate_true": float((tp + fn) / N_TEST),
            })
            print(f"  slack={slack_ms:3d}ms: FP={fp_rate:.1%}  FN={fn_rate:.1%}  "
                  f"admit_pred={results[-1]['admit_rate_pred']:.1%}  "
                  f"admit_true={results[-1]['admit_rate_true']:.1%}")

    return results


# ---------------------------------------------------------------------------
# Experiment A — Heterogeneous batch composition
# ---------------------------------------------------------------------------

def exp_a(model, vocab_size, rank):
    """Train on uniform-length batches; evaluate on uniform/bimodal/skewed."""
    from slora.server.router.tracker import PrefillExecutionEstimator

    N_TRAIN = 80
    N_TEST  = 80

    settings = [
        {
            "bs": 4, "total": 1024,
            "families": {
                "uniform": [256, 256, 256, 256],
                "bimodal": [64, 64, 448, 448],
                "skewed":  [32, 32, 32, 928],
            },
        },
        {
            "bs": 8, "total": 2048,
            "families": {
                "uniform": [256] * 8,
                "bimodal": [128] * 4 + [384] * 4,
                "skewed":  [32] * 7 + [1824],
            },
        },
    ]

    results = []
    rng = np.random.default_rng(100)

    for setting in settings:
        bs, total = setting["bs"], setting["total"]
        fams      = setting["families"]

        if rank == 0:
            print(f"\n[Exp A] bs={bs} total_tokens={total}")

        family_times = {name: [] for name in fams}
        for name, lens in fams.items():
            for i in range(N_TRAIN + N_TEST):
                batch = [rng.integers(0, vocab_size, size=l) for l in lens]
                pt, _, _, _ = run_prefill(model, batch)
                family_times[name].append(pt)
                if rank == 0 and (i + 1) % 40 == 0:
                    print(f"  [{name}] {i+1}/{N_TRAIN + N_TEST}")

        if rank == 0:
            # Train predictor on uniform family only
            p_est = PrefillExecutionEstimator()
            p_est.fit(
                inference_only_tokens=[fams["uniform"]] * N_TRAIN,
                inference_only_times=family_times["uniform"][:N_TRAIN],
                coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
            )

            for name, lens in fams.items():
                test_times = family_times[name][N_TRAIN:]
                p_pred     = p_est.predict_inference(lens)
                signed_errs = [(p_pred - a) / a * 100 for a in test_times]
                abs_errs    = [abs(e) for e in signed_errs]
                cv = float(np.std(test_times) / np.mean(test_times) * 100)
                row = {
                    "bs":                  bs,
                    "total_tokens":        total,
                    "family":              name,
                    "sum_n2":              int(sum(l * l for l in lens)),
                    "T_in":                int(sum(lens)),
                    "mean_ms":             float(np.mean(test_times) * 1000),
                    "std_ms":              float(np.std(test_times) * 1000),
                    "cv_pct":              cv,
                    "pred_ms":             float(p_pred * 1000),
                    "mean_signed_err_pct": float(np.mean(signed_errs)),
                    "mean_abs_err_pct":    float(np.mean(abs_errs)),
                    "max_abs_err_pct":     float(np.max(abs_errs)),
                }
                results.append(row)
                print(f"  [{name:>8}] mean={row['mean_ms']:.2f}ms  cv={cv:.1f}%  "
                      f"signed_err={row['mean_signed_err_pct']:+.1f}%  "
                      f"abs_err={row['mean_abs_err_pct']:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Experiment C — Content-dependent routing variance
# ---------------------------------------------------------------------------

def exp_c(model, vocab_size, rank):
    """Compare routing variance for different token content regimes."""
    from slora.server.router.tracker import PrefillExecutionEstimator

    BS, SL  = 4, 256
    N_TRAIN = 80
    N_TEST  = 80

    regimes = {
        "random":   lambda rng: rng.integers(0, vocab_size, size=SL),
        "constant": lambda rng: np.ones(SL, dtype=np.int64),
        "low_ids":  lambda rng: rng.integers(0, 100, size=SL),
        "high_ids": lambda rng: rng.integers(vocab_size - 100, vocab_size, size=SL),
    }

    results = []
    rng = np.random.default_rng(300)

    if rank == 0:
        print(f"\n[Exp C] bs={BS} sl={SL}, content regimes: {list(regimes.keys())}")

    regime_times = {name: [] for name in regimes}
    for name, token_fn in regimes.items():
        for i in range(N_TRAIN + N_TEST):
            batch = [token_fn(rng) for _ in range(BS)]
            pt, _, _, _ = run_prefill(model, batch)
            regime_times[name].append(pt)
            if rank == 0 and (i + 1) % 40 == 0:
                print(f"  [{name}] {i+1}/{N_TRAIN + N_TEST}")

    if rank == 0:
        p_est = PrefillExecutionEstimator()
        p_est.fit(
            inference_only_tokens=[[SL] * BS] * N_TRAIN,
            inference_only_times=regime_times["random"][:N_TRAIN],
            coserving_inf_tokens=[], coserving_ft_tokens=[], coserving_times=[],
        )

        for name in regimes:
            test_times   = regime_times[name][N_TRAIN:]
            p_pred       = p_est.predict_inference([SL] * BS)
            signed_errs  = [(p_pred - a) / a * 100 for a in test_times]
            cv = float(np.std(test_times) / np.mean(test_times) * 100)
            row = {
                "regime":              name,
                "mean_ms":             float(np.mean(test_times) * 1000),
                "std_ms":              float(np.std(test_times) * 1000),
                "cv_pct":              cv,
                "pred_ms":             float(p_pred * 1000),
                "mean_signed_err_pct": float(np.mean(signed_errs)),
                "mean_abs_err_pct":    float(np.mean([abs(e) for e in signed_errs])),
            }
            results.append(row)
            print(f"  [{name:>10}] mean={row['mean_ms']:.2f}ms  cv={cv:.1f}%  "
                  f"signed_err={row['mean_signed_err_pct']:+.1f}%")

    return results


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(rank, world_size, results):
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29502",
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

    warmup(model, vocab_size, rank)

    # Run in order D → B → E → A → C (cheapest first, most expensive last)
    d_results = exp_d(model, vocab_size, rank)
    b_results = exp_b(model, vocab_size, rank)
    e_results = exp_e(model, vocab_size, rank)
    a_results = exp_a(model, vocab_size, rank)
    c_results = exp_c(model, vocab_size, rank)

    if rank == 0:
        results["d"] = d_results
        results["b"] = b_results
        results["e"] = e_results
        results["a"] = a_results
        results["c"] = c_results

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_csv(rows, fields, fname):
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


def save_results(all_results):
    os.makedirs(OUT_DIR, exist_ok=True)

    _save_csv(all_results["d"],
              ["label", "cv_class", "n_train", "rmse_ms", "fit_rmse_ms", "mae_pct",
               "actual_std_ms", "actual_cv_pct", "actual_mean_ms"],
              "convergence.csv")

    _save_csv(all_results["b"],
              ["train_window", "step", "K", "actual_ms", "pred_ms", "signed_err_pct"],
              "k_growth.csv")

    _save_csv(all_results["e"],
              ["slack_ms", "tp", "fp", "tn", "fn", "fp_rate", "fn_rate",
               "admit_rate_pred", "admit_rate_true"],
              "gate_errors.csv")

    _save_csv(all_results["a"],
              ["bs", "total_tokens", "family", "sum_n2", "T_in", "mean_ms", "std_ms",
               "cv_pct", "pred_ms", "mean_signed_err_pct", "mean_abs_err_pct", "max_abs_err_pct"],
              "hetero.csv")

    _save_csv(all_results["c"],
              ["regime", "mean_ms", "std_ms", "cv_pct", "pred_ms",
               "mean_signed_err_pct", "mean_abs_err_pct"],
              "content.csv")

    jpath = os.path.join(OUT_DIR, "stress_test.json")
    with open(jpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {jpath}")


def print_summary(all_results):
    print("\n" + "=" * 70)
    print("EXP D — RMSE Convergence (should plateau for high-CV configs)")
    print("=" * 70)
    labels = list(dict.fromkeys(r["label"] for r in all_results["d"]))
    for label in labels:
        rows = [r for r in all_results["d"] if r["label"] == label]
        cv = rows[0]["actual_cv_pct"]
        std = rows[0]["actual_std_ms"]
        print(f"\n  {label}  (CV={cv:.1f}%  actual_std={std:.3f}ms)")
        print(f"  {'n_train':>8}  {'rmse_ms':>10}  {'mae_pct':>8}")
        for r in rows:
            print(f"  {r['n_train']:>8}  {r['rmse_ms']:>10.3f}  {r['mae_pct']:>7.1f}%")

    print("\n" + "=" * 70)
    print("EXP B — Decode Signed Error vs K Growth")
    print("=" * 70)
    windows = sorted(set(r["train_window"] for r in all_results["b"]))
    for w in windows:
        rows = sorted([r for r in all_results["b"] if r["train_window"] == w],
                      key=lambda x: x["step"])
        # Show 5 representative steps
        indices = [0, len(rows)//4, len(rows)//2, 3*len(rows)//4, -1]
        print(f"\n  train_window={w}:")
        for idx in indices:
            r = rows[idx]
            print(f"    step={r['step']:3d}  K={r['K']:6d}  "
                  f"actual={r['actual_ms']:.2f}ms  pred={r['pred_ms']:.2f}ms  "
                  f"err={r['signed_err_pct']:+.1f}%")

    print("\n" + "=" * 70)
    print("EXP E — SLO Gate Error Rates")
    print("=" * 70)
    print(f"  {'slack_ms':>10}  {'FP_rate':>10}  {'FN_rate':>10}  "
          f"{'admit_pred':>12}  {'admit_true':>12}")
    for r in sorted(all_results["e"], key=lambda x: x["slack_ms"]):
        print(f"  {r['slack_ms']:>10}  {r['fp_rate']:>9.1%}  {r['fn_rate']:>9.1%}  "
              f"  {r['admit_rate_pred']:>10.1%}  {r['admit_rate_true']:>10.1%}")

    print("\n" + "=" * 70)
    print("EXP A — Heterogeneous Batch Predictor Bias")
    print("=" * 70)
    for r in all_results["a"]:
        print(f"  bs={r['bs']}  {r['family']:>8}  cv={r['cv_pct']:.1f}%  "
              f"signed_err={r['mean_signed_err_pct']:+.1f}%  "
              f"abs_err={r['mean_abs_err_pct']:.1f}%")

    print("\n" + "=" * 70)
    print("EXP C — Content-Dependent Routing Variance (bs=4 sl=256)")
    print("=" * 70)
    for r in all_results["c"]:
        print(f"  {r['regime']:>10}  cv={r['cv_pct']:.1f}%  "
              f"signed_err={r['mean_signed_err_pct']:+.1f}%  "
              f"abs_err={r['mean_abs_err_pct']:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs"); sys.exit(1)
    if not os.path.isdir(MODEL_DIR):
        print(f"Model not found at {MODEL_DIR}"); sys.exit(1)

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(worker, args=(2, results), nprocs=2, join=True)

    all_results = dict(results)
    save_results(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()
