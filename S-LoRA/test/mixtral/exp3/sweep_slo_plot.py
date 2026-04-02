#!/usr/bin/env python3
"""
Oracle vs Actual Predictor CDF Plot
====================================
CPU-only. No GPU or model loading required.

For each SFT budget, plots two CDF curves against SLO threshold:
  - Oracle:  CDF of actual_coserve_ms  (admits exactly the batches that truly fit)
  - Actual:  CDF of pred_coserve_ms    (what the live predictor admits)

The horizontal gap between the curves is the SFT throughput lost to prediction error.
For Mixtral EP, this gap ≈ 2×fit_rmse (the wasted slack from EP routing variance).

Usage:
    cd S-LoRA
    python test/mixtral/exp3/sweep_slo_plot.py
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SFT_BUDGETS = [16, 32, 64, 128, 256]
SLO_VALUES  = np.arange(150, 405, 5)   # ms


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    per_batch = pd.read_csv(os.path.join(RESULTS_DIR, "per_batch.csv"))

    # Load estimator params — from JSON if available, else recover from per_batch
    json_path = os.path.join(RESULTS_DIR, "estimator_params.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            params = json.load(f)
        # Estimator stores params in seconds; convert to ms to match pred_ms units
        alpha    = params["alpha"] * 1000
        beta     = params["beta"]  * 1000
        c        = params["c"]     * 1000
        fit_rmse = params["fit_rmse"]
        print(f"Loaded estimator params from JSON: "
              f"alpha={alpha:.3e}  beta={beta:.3e}  c={c:.3e}  fit_rmse={fit_rmse*1000:.2f}ms")
    else:
        # Recover from per_batch: pred_ms = (alpha*S + beta*T + c) * (1 + 2*fit_rmse)
        err_dist = pd.read_csv(os.path.join(RESULTS_DIR, "error_distribution.csv"))
        fit_rmse = float(err_dist["fit_rmse_ms"].iloc[0]) / 1000
        inflation = 1 + 2 * fit_rmse
        raw_pred  = per_batch["pred_ms"].values / inflation  # ms, un-inflated
        S   = per_batch["sum_n2"].values
        Tin = per_batch["T_in"].values
        X   = np.column_stack([S, Tin, np.ones(len(S))])
        alpha, beta, c = np.linalg.lstsq(X, raw_pred, rcond=None)[0]
        print(f"Recovered estimator params from per_batch.csv: "
              f"alpha={alpha:.3e}  beta={beta:.3e}  c={c:.3e}  fit_rmse={fit_rmse*1000:.2f}ms")

    return per_batch, alpha, beta, c, fit_rmse


# ---------------------------------------------------------------------------
# Compute CDFs
# ---------------------------------------------------------------------------

def compute_cdfs(per_batch, alpha, beta, c, fit_rmse):
    inflation = 1 + 2 * fit_rmse
    S   = per_batch["sum_n2"].values
    Tin = per_batch["T_in"].values

    results = {}
    for budget in SFT_BUDGETS:
        actual_coserve = per_batch[f"actual_coserve_ms_{budget}"].values

        # Actual predictor: inflated prediction including SFT tokens
        pred_coserve = (
            alpha * (S + budget**2) + beta * (Tin + budget) + c
        ) * inflation  # ms

        oracle_admitted = np.array([np.mean(actual_coserve <= slo) for slo in SLO_VALUES])
        actual_admitted = np.array([np.mean(pred_coserve   <= slo) for slo in SLO_VALUES])

        results[budget] = {
            "oracle":        oracle_admitted,
            "actual":        actual_admitted,
            "mean_coserve":  float(np.mean(actual_coserve)),
            "pred_coserve":  pred_coserve,
            "actual_coserve": actual_coserve,
        }

    return results


# ---------------------------------------------------------------------------
# Measure horizontal gap at 50% admission
# ---------------------------------------------------------------------------

def gap_at_50pct(slo_values, oracle, actual):
    """Return SLO at which oracle/actual reach 50% admission, and their gap."""
    def slo_at_frac(curve, frac=0.5):
        idx = np.searchsorted(curve, frac)
        if idx >= len(curve):
            return slo_values[-1]
        return float(slo_values[idx])

    slo_oracle = slo_at_frac(oracle)
    slo_actual = slo_at_frac(actual)
    return slo_oracle, slo_actual, slo_actual - slo_oracle


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(results, fit_rmse, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    fig.suptitle(
        "Oracle vs Actual Predictor: SFT Admission Rate\n"
        f"Mixtral-8x7B EP  |  fit_rmse = {fit_rmse*1000:.1f}ms  |  "
        f"wasted slack ≈ {2*fit_rmse*1000:.0f}ms",
        fontsize=13,
    )

    blue = "#2166ac"
    red  = "#d6604d"

    for ax, budget in zip(axes.flat, SFT_BUDGETS):
        r = results[budget]
        oracle = r["oracle"]
        actual = r["actual"]
        mean_cs = r["mean_coserve"]

        ax.fill_between(SLO_VALUES, actual, oracle,
                        alpha=0.18, color=red, label="_gap")
        ax.plot(SLO_VALUES, oracle, color=blue, lw=2,
                label="Oracle (actual coserving time)")
        ax.plot(SLO_VALUES, actual, color=red,  lw=2, ls="--",
                label="Actual predictor")

        # Vertical line at mean coserving time
        ax.axvline(mean_cs, color="gray", lw=1, ls=":", alpha=0.7)
        ax.text(mean_cs + 2, 0.04, f"mean\n{mean_cs:.0f}ms",
                color="gray", fontsize=7.5, va="bottom")

        ax.set_title(f"SFT budget = {budget} tokens", fontsize=11)
        ax.set_xlim(SLO_VALUES[0], SLO_VALUES[-1])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("SLO threshold (ms)")
        ax.set_ylabel("Fraction admitted")
        ax.grid(True, alpha=0.3)

    # Hide unused subplot (2×3 grid has 6 cells, we use 5)
    axes.flat[-1].set_visible(False)

    # Shared legend
    handles = [
        mpatches.Patch(color=blue, label="Oracle (actual coserving time)"),
        mpatches.Patch(color=red,  label="Actual predictor"),
        mpatches.Patch(color=red,  alpha=0.3, label="Throughput gap (FN)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"oracle_vs_actual_cdf.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    per_batch, alpha, beta, c, fit_rmse = load_data()
    results = compute_cdfs(per_batch, alpha, beta, c, fit_rmse)

    print(f"\nWasted slack: {2*fit_rmse*1000:.1f}ms  (2 × fit_rmse = 2 × {fit_rmse*1000:.1f}ms)")
    for budget in SFT_BUDGETS:
        r = results[budget]
        slo_o, slo_a, gap = gap_at_50pct(SLO_VALUES, r["oracle"], r["actual"])
        print(f"  budget={budget:3d}: oracle 50% @ {slo_o:.0f}ms, "
              f"actual 50% @ {slo_a:.0f}ms, gap={gap:.0f}ms")

    plot(results, fit_rmse, RESULTS_DIR)


if __name__ == "__main__":
    main()
