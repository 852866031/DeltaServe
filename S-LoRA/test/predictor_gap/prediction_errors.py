#!/usr/bin/env python3
"""
Predictor Error Analysis — operational view of where the predictor fails
========================================================================

Uses exp3 per_batch.csv (realistic log-normal trace, 236 batches each).

Three panels per model (Llama3 left column, Mixtral right column):

Panel 1 — Error distribution: histogram of (actual_ms - pred_ms).
  Negative = predictor over-estimated (conservative, SFT blocked unnecessarily).
  Positive = predictor under-estimated (FP risk: admitted SFT that violated SLO).
  Shows how often and by how much the predictor is wrong in each direction.

Panel 2 — FP and FN rate vs SLO threshold:
  FP = predictor said batch fits under SLO, actual_ms > SLO (admitted bad batch).
  FN = predictor said batch violates SLO, actual_ms ≤ SLO (blocked good batch).
  X-axis sweeps SLO from min to max of actual_ms range.
  Shows the operating region where FPs actually happen.

Panel 3 — Scatter: actual_ms vs pred_ms, coloured by error magnitude.
  Diagonal = perfect prediction. Shows structure of errors (do large batches
  get worse predictions? do certain T_in values cluster?).

Usage:
    cd S-LoRA
    python test/predictor_gap/prediction_errors.py
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP3_MIXTRAL = "test/mixtral/exp3/results/per_batch.csv"
EXP3_LLAMA3  = "test/llama3/exp3/results/per_batch.csv"
OUT_DIR      = "test/predictor_gap/results/error_analysis"


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def compute_gate(actual, pred, slo_values):
    """Return FP rate and FN rate at each SLO threshold."""
    actual = np.array(actual)
    pred   = np.array(pred)
    fp_rates, fn_rates = [], []
    for slo in slo_values:
        admitted  = pred <= slo          # predictor says fits
        violates  = actual > slo         # actually violates
        fp = np.sum(admitted & violates)
        tn = np.sum(~admitted & violates)
        tp = np.sum(admitted & ~violates)
        fn = np.sum(~admitted & ~violates)
        # FP rate = FP / (FP + TN) — fraction of violating batches wrongly admitted
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # FN rate = FN / (FN + TP) — fraction of safe batches wrongly blocked
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fp_rates.append(fp_rate * 100)
        fn_rates.append(fn_rate * 100)
    return np.array(fp_rates), np.array(fn_rates)


def plot_model(axes, rows, model_label, color):
    actual  = np.array([float(r["actual_ms"]) for r in rows])
    pred    = np.array([float(r["pred_ms"])   for r in rows])
    T_in    = np.array([float(r["T_in"])      for r in rows])
    error   = actual - pred   # positive = under-estimate = FP risk

    # --- Panel 1: Error distribution ---
    ax = axes[0]
    bins = np.linspace(error.min(), error.max(), 40)
    neg  = error[error <= 0]
    pos  = error[error > 0]
    ax.hist(neg, bins=bins, color="steelblue", alpha=0.75,
            label=f"Over-estimate (SFT blocked)\nn={len(neg)}, mean={neg.mean():.1f}ms")
    ax.hist(pos, bins=bins, color="tomato", alpha=0.75,
            label=f"Under-estimate (FP risk)\nn={len(pos)}, mean={pos.mean():.1f}ms")
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("actual_ms − pred_ms  (ms)")
    ax.set_ylabel("Number of batches")
    ax.set_title(f"{model_label}\nPrediction Error Distribution\n"
                 f"RMSE={np.sqrt(np.mean(error**2)):.1f}ms  "
                 f"mean={error.mean():.1f}ms  std={error.std():.1f}ms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate fraction positive (FP risk)
    frac_pos = len(pos) / len(error) * 100
    ax.text(0.97, 0.95, f"{frac_pos:.0f}% of batches under-estimated",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Panel 2: FP/FN rate vs SLO ---
    ax = axes[1]
    slo_grid = np.linspace(actual.min() * 0.9, actual.max() * 1.05, 300)
    fp_rates, fn_rates = compute_gate(actual, pred, slo_grid)

    ax.plot(slo_grid, fp_rates, color="tomato",    linewidth=2, label="FP rate (admitted SLO violators)")
    ax.plot(slo_grid, fn_rates, color="steelblue", linewidth=2, label="FN rate (blocked safe batches)")
    ax.set_xlabel("SLO threshold (ms)")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"{model_label}\nGate Error Rate vs SLO")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 105)

    # Mark a few round SLO values
    for slo_mark in [100, 150, 200, 250, 300]:
        if actual.min() < slo_mark < actual.max() * 1.05:
            fp_at = np.interp(slo_mark, slo_grid, fp_rates)
            fn_at = np.interp(slo_mark, slo_grid, fn_rates)
            ax.annotate(f"SLO={slo_mark}ms\nFP={fp_at:.0f}%\nFN={fn_at:.0f}%",
                        xy=(slo_mark, fp_at), xytext=(slo_mark + 5, fp_at + 10),
                        fontsize=7, arrowprops=dict(arrowstyle="-", color="gray"))

    # --- Panel 3: Scatter actual vs pred, coloured by |error_pct| ---
    ax = axes[2]
    abs_err_pct = np.abs(error / actual * 100)
    sc = ax.scatter(actual, pred, c=abs_err_pct, cmap="RdYlGn_r",
                    vmin=0, vmax=30, s=18, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="|error| %")
    lo = min(actual.min(), pred.min()) * 0.95
    hi = max(actual.max(), pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual execution time (ms)")
    ax.set_ylabel("Predicted execution time (ms)")
    ax.set_title(f"{model_label}\nActual vs Predicted\n(colour = |error| %)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Print per-SLO summary
    print(f"\n  {model_label} gate summary:")
    print(f"  {'SLO':>6}  {'FP%':>6}  {'FN%':>6}  {'n_FP':>6}  {'n_FN':>6}")
    for slo_mark in [150, 200, 250, 300, 350]:
        admitted = pred <= slo_mark
        violates = actual > slo_mark
        fp = np.sum(admitted & violates)
        fn_count = np.sum(~admitted & ~violates)
        fp_r = fp / max(np.sum(violates), 1) * 100
        fn_r = fn_count / max(np.sum(~violates), 1) * 100
        print(f"  {slo_mark:>6}  {fp_r:>6.1f}  {fn_r:>6.1f}  {fp:>6}  {fn_count:>6}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows_llama3  = load_csv(EXP3_LLAMA3)
    rows_mixtral = load_csv(EXP3_MIXTRAL)

    # Sort by batch_id
    rows_llama3.sort(key=lambda r: int(r["batch_id"]))
    rows_mixtral.sort(key=lambda r: int(r["batch_id"]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Predictor Error Analysis — Exp 3 log-normal trace\n"
        "(realistic heterogeneous batches, predictor calibrated on first 256 batches)",
        fontsize=12, fontweight="bold"
    )

    print("\n=== Predictor Error Analysis ===")
    plot_model(axes[0], rows_llama3,  "Llama3-8B (dense, 1 GPU)",  "#2196F3")
    plot_model(axes[1], rows_mixtral, "Mixtral-8x7B EP (2 GPU)",   "#F44336")

    plt.tight_layout()
    pdf = os.path.join(OUT_DIR, "prediction_errors.pdf")
    png = os.path.join(OUT_DIR, "prediction_errors.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {pdf}")
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()
