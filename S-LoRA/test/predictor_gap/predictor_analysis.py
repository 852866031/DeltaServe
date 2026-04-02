#!/usr/bin/env python3
"""
Predictor Analysis — Two complementary views of where the predictor fails
=========================================================================

Figure 1 — Error Autocorrelation (Exp 3 trace):
  Residuals (actual_ms - pred_ms) in batch-arrival order.
  If errors are caused by a missing feature, consecutive batches with similar
  features would have correlated residuals → autocorrelation > 0 at short lags.
  If errors are pure hardware noise, residuals are white noise → autocorr ≈ 0.
  Llama3 and Mixtral EP side-by-side.

Figure 2 — CDF of Actual Execution Time at Fixed Composition (Predictor Gap exp):
  For each T_in, the empirical CDF of actual_ms across 50 test trials where the
  batch composition is IDENTICAL. The predictor's single estimate is marked as a
  vertical dashed line.
  Shows: predictor is right about the mean (line sits at ~50th percentile) but the
  distribution is wide (Mixtral) or narrow (Llama3). Llama3 and Mixtral side-by-side.

Usage:
    cd S-LoRA
    python test/predictor_gap/predictor_analysis.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXP3_MIXTRAL  = "test/mixtral/exp3/results/per_batch.csv"
EXP3_LLAMA3   = "test/llama3/exp3/results/per_batch.csv"
GAP_MIXTRAL   = "test/mixtral/predictor_gap/results/fixed_composition.csv"
GAP_LLAMA3    = "test/llama3/predictor_gap/results/fixed_composition.csv"
OUT_DIR       = "test/predictor_gap/results/error_analysis"

T_IN_VALUES   = [256, 512, 1024, 2048]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def autocorr(x, max_lag=40):
    x = np.array(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    lags, acf = [], []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            c = np.dot(x[:n-lag], x[lag:]) / (np.var(x) * (n - lag))
            acf.append(c)
        lags.append(lag)
    return np.array(lags), np.array(acf)


def confidence_band(n, max_lag):
    """95% confidence band for white-noise autocorrelation."""
    return 1.96 / np.sqrt(n - np.arange(max_lag + 1))


# ---------------------------------------------------------------------------
# Figure 1 — Error Autocorrelation
# ---------------------------------------------------------------------------

def plot_autocorrelation(ax, rows, label, color):
    residuals = [float(r["actual_ms"]) - float(r["pred_ms"]) for r in rows]
    n = len(residuals)
    lags, acf = autocorr(residuals, max_lag=40)
    cb = confidence_band(n, 40)

    ax.bar(lags, acf, color=color, alpha=0.6, width=0.8, label=label)
    ax.fill_between(lags, -cb, cb, color="gray", alpha=0.2, label="95% CI (white noise)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (batches)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{label}\nError Autocorrelation (n={n} batches)")
    ax.set_xlim(-0.5, 40.5)
    ax.set_ylim(-0.4, 0.4)
    ax.legend(fontsize=8)

    # Annotate fraction of lags outside CI
    outside = np.sum(np.abs(acf[1:]) > cb[1:])
    ax.text(0.97, 0.95, f"{outside}/40 lags outside 95% CI",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


# ---------------------------------------------------------------------------
# Figure 2 — CDF at Fixed Composition
# ---------------------------------------------------------------------------

def plot_cdfs(axes_row, gap_rows, model_label, color, T_in_values):
    for ax, T_in in zip(axes_row, T_in_values):
        test_rows = [r for r in gap_rows
                     if int(r["T_in"]) == T_in and int(r["is_train"]) == 0]
        if not test_rows:
            ax.set_visible(False)
            continue

        times = np.sort([float(r["actual_ms"]) for r in test_rows])
        pred  = float(test_rows[0]["pred_ms"])
        n     = len(times)

        # Empirical CDF
        cdf = np.arange(1, n + 1) / n
        ax.step(times, cdf, where="post", color=color, linewidth=1.8, label="Actual CDF")

        # Predictor estimate
        ax.axvline(pred, color="black", linestyle="--", linewidth=1.5, label=f"Predictor ({pred:.0f}ms)")

        # Mark predictor percentile
        pct = np.searchsorted(times, pred) / n * 100
        ax.axhline(pct / 100, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(pred + (times[-1] - times[0]) * 0.02, pct / 100 + 0.03,
                f"  {pct:.0f}th pct", fontsize=7, color="black")

        # Annotate std
        std = float(np.std(times))
        ax.text(0.97, 0.05, f"std = {std:.1f}ms\ncv = {std/np.mean(times)*100:.1f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Actual execution time (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{model_label}  T_in={T_in}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    exp3_mixtral = load_csv(EXP3_MIXTRAL)
    exp3_llama3  = load_csv(EXP3_LLAMA3)
    gap_mixtral  = load_csv(GAP_MIXTRAL)
    gap_llama3   = load_csv(GAP_LLAMA3)

    # Sort exp3 by batch_id to preserve arrival order
    exp3_mixtral.sort(key=lambda r: int(r["batch_id"]))
    exp3_llama3.sort(key=lambda r: int(r["batch_id"]))

    # -------------------------------------------------------------------
    # Figure 1 — Autocorrelation (2 panels side by side)
    # -------------------------------------------------------------------
    fig1, (ax_l3, ax_mx) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig1.suptitle(
        "Predictor Error Autocorrelation\n"
        "White noise → no missing feature can explain the residuals",
        fontsize=11, fontweight="bold"
    )
    plot_autocorrelation(ax_l3, exp3_llama3,  "Llama3-8B (dense)", "#2196F3")
    plot_autocorrelation(ax_mx, exp3_mixtral, "Mixtral-8x7B EP",   "#F44336")
    plt.tight_layout()
    p1_pdf = os.path.join(OUT_DIR, "autocorrelation.pdf")
    p1_png = os.path.join(OUT_DIR, "autocorrelation.png")
    fig1.savefig(p1_pdf, bbox_inches="tight")
    fig1.savefig(p1_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {p1_pdf}")
    print(f"Saved: {p1_png}")

    # -------------------------------------------------------------------
    # Figure 2 — CDF at fixed composition (2 rows × 4 T_in panels)
    # -------------------------------------------------------------------
    fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle(
        "Execution Time CDF at Fixed Batch Composition\n"
        "Predictor (dashed) correctly estimates the mean — but the distribution width is irreducible",
        fontsize=11, fontweight="bold"
    )
    plot_cdfs(axes[0], gap_llama3,  "Llama3-8B (dense, 1 GPU)", "#2196F3", T_IN_VALUES)
    plot_cdfs(axes[1], gap_mixtral, "Mixtral-8x7B EP (2 GPU)",  "#F44336", T_IN_VALUES)
    plt.tight_layout()
    p2_pdf = os.path.join(OUT_DIR, "cdf_fixed_composition.pdf")
    p2_png = os.path.join(OUT_DIR, "cdf_fixed_composition.png")
    fig2.savefig(p2_pdf, bbox_inches="tight")
    fig2.savefig(p2_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {p2_pdf}")
    print(f"Saved: {p2_png}")

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------
    print("\n--- Autocorrelation summary (exp3 residuals) ---")
    for label, rows in [("Llama3", exp3_llama3), ("Mixtral", exp3_mixtral)]:
        res = np.array([float(r["actual_ms"]) - float(r["pred_ms"]) for r in rows])
        _, acf = autocorr(res, max_lag=40)
        cb = confidence_band(len(res), 40)
        outside = np.sum(np.abs(acf[1:]) > cb[1:])
        print(f"  {label}: mean_residual={res.mean():.2f}ms  std={res.std():.2f}ms  "
              f"lags_outside_95CI={outside}/40  "
              f"max_|acf|={np.max(np.abs(acf[1:])):.3f}")

    print("\n--- CDF summary (fixed composition, test set) ---")
    for label, rows in [("Llama3", gap_llama3), ("Mixtral", gap_mixtral)]:
        print(f"  {label}:")
        for T_in in T_IN_VALUES:
            sub = [r for r in rows if int(r["T_in"]) == T_in and int(r["is_train"]) == 0]
            if not sub:
                continue
            times = np.array([float(r["actual_ms"]) for r in sub])
            pred  = float(sub[0]["pred_ms"])
            pct   = np.searchsorted(np.sort(times), pred) / len(times) * 100
            print(f"    T_in={T_in:4d}: mean={np.mean(times):.1f}ms  std={np.std(times):.2f}ms  "
                  f"pred={pred:.1f}ms ({pct:.0f}th pct)")


if __name__ == "__main__":
    main()
