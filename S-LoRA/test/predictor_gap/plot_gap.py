#!/usr/bin/env python3
"""
Predictor Gap — Oracle vs Current Predictor Figure
===================================================
Generates a 3-panel figure showing:

  Panel A (key): Execution time distribution at fixed batch composition.
    At identical T_in, Llama3 execution time varies by ~0.7ms (std) while
    Mixtral EP varies by ~9ms. The predictor correctly estimates the MEAN
    but cannot predict where in the distribution each execution falls.
    The violin width = irreducible variance = oracle gap.

  Panel B: Scatter — current predictor vs oracle (actual execution time).
    Oracle = 45° diagonal. Mixtral scatter is wide; Llama3 is tight.

  Panel C: SLO gate false-positive rate as a function of SLO threshold.
    Oracle gate = 0% FP by construction. Mixtral current predictor > 0% at
    tight SLOs. Llama3 current predictor ≈ 0%. The gap = research gap.

Usage:
    python test/predictor_gap/plot_gap.py

Reads:
    test/mixtral/predictor_gap/results/fixed_composition.csv
    test/llama3/predictor_gap/results/fixed_composition.csv
    test/mixtral/exp3/results/per_batch.csv
    test/llama3/exp3/results/per_batch.csv
"""

import os, sys, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

MIXTRAL_FIXED = os.path.join(BASE, "mixtral", "predictor_gap", "results", "fixed_composition.csv")
LLAMA3_FIXED  = os.path.join(BASE, "llama3",  "predictor_gap", "results", "fixed_composition.csv")
MIXTRAL_EXP3  = os.path.join(BASE, "mixtral", "exp3", "results", "per_batch.csv")
LLAMA3_EXP3   = os.path.join(BASE, "llama3",  "exp3", "results", "per_batch.csv")

T_IN_VALUES = [256, 512, 1024, 2048]

# Colors
C_LLAMA  = "#2196F3"   # blue
C_MIX    = "#F44336"   # red
C_ORACLE = "#4CAF50"   # green
C_GRAY   = "#9E9E9E"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get_fixed_comp(rows, model_label):
    """Returns dict: T_in -> array of actual_ms (test set only)."""
    result = {}
    pred_by_tin = {}
    for T_in in T_IN_VALUES:
        sub = [r for r in rows if int(r["T_in"]) == T_in and int(r["is_train"]) == 0]
        if sub:
            result[T_in] = np.array([float(r["actual_ms"]) for r in sub])
            pred_by_tin[T_in] = float(sub[0]["pred_ms"])
    return result, pred_by_tin


def get_exp3(rows):
    """Returns arrays: actual_ms, pred_ms, T_in."""
    actual = np.array([float(r["actual_ms"]) for r in rows])
    pred   = np.array([float(r["pred_ms"])   for r in rows])
    t_in   = np.array([float(r["T_in"])      for r in rows])
    return actual, pred, t_in


def compute_fp_curve(rows, slo_range_ms):
    """
    Compute FP rate (FP / (FP+TN)) for current predictor and oracle
    across a range of SLO thresholds.

    Current predictor admits if pred_ms <= slo_ms.
    Oracle gate admits if actual_ms <= slo_ms.

    FP rate = fraction of "would-violate-SLO" batches that are incorrectly admitted.
    = FP / (FP + TN)
    """
    actual = np.array([float(r["actual_ms"]) for r in rows])
    pred   = np.array([float(r["pred_ms"])   for r in rows])

    pred_fp_rates   = []
    oracle_fp_rates = []

    for slo in slo_range_ms:
        would_violate = actual > slo   # "actual negatives" — batch would hurt SLO

        # Current predictor
        pred_admits = pred <= slo
        fp = np.sum(pred_admits & would_violate)
        tn = np.sum(~pred_admits & would_violate)
        denom = fp + tn
        pred_fp_rates.append(fp / denom if denom > 0 else 0.0)

        # Oracle gate (admits iff actual <= slo → FP always 0)
        oracle_fp_rates.append(0.0)

    return np.array(pred_fp_rates), np.array(oracle_fp_rates)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_A(ax, llama_data, mix_data, llama_pred, mix_pred):
    """Violin plots of execution time at fixed features."""

    x_positions = np.arange(len(T_IN_VALUES))
    width = 0.35
    offset = 0.18

    # For each T_in, plot two violins: Llama3 and Mixtral
    for i, T_in in enumerate(T_IN_VALUES):
        ll_times = llama_data.get(T_in)
        mx_times = mix_data.get(T_in)

        if ll_times is not None and len(ll_times) > 2:
            vp = ax.violinplot(ll_times, positions=[i - offset], widths=width,
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(C_LLAMA)
                body.set_alpha(0.6)
            vp["cmedians"].set_color(C_LLAMA)
            vp["cmedians"].set_linewidth(2)
            # Predictor estimate
            pred = llama_pred.get(T_in)
            if pred:
                ax.plot([i - offset - width/2 + 0.02, i - offset + width/2 - 0.02],
                        [pred, pred], color=C_LLAMA, linewidth=2.5,
                        linestyle="--", zorder=5)

        if mx_times is not None and len(mx_times) > 2:
            vp = ax.violinplot(mx_times, positions=[i + offset], widths=width,
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(C_MIX)
                body.set_alpha(0.6)
            vp["cmedians"].set_color(C_MIX)
            vp["cmedians"].set_linewidth(2)
            # Predictor estimate
            pred = mix_pred.get(T_in)
            if pred:
                ax.plot([i + offset - width/2 + 0.02, i + offset + width/2 - 0.02],
                        [pred, pred], color=C_MIX, linewidth=2.5,
                        linestyle="--", zorder=5)

            # Annotate std
            std = np.std(mx_times)
            mean = np.mean(mx_times)
            ax.annotate(f"σ={std:.1f}ms", xy=(i + offset, mean + std + 2),
                        ha="center", va="bottom", fontsize=7.5,
                        color=C_MIX, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{t}" for t in T_IN_VALUES])
    ax.set_xlabel("Total Input Tokens (T_in)", fontsize=11)
    ax.set_ylabel("Execution Time (ms)", fontsize=11)
    ax.set_title("(A) Execution Time at Fixed Batch Composition\n"
                 "(predictor = dashed line; spread = irreducible oracle gap)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor=C_LLAMA, alpha=0.7, label="Llama3-8B (dense)"),
        mpatches.Patch(facecolor=C_MIX,   alpha=0.7, label="Mixtral-8x7B EP (MoE)"),
        Line2D([0], [0], color=C_GRAY, linewidth=2, linestyle="--",
               label="Predictor estimate (mean)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc="upper left")


def panel_B(ax_l, ax_r, llama_exp3, mix_exp3):
    """Scatter plots: actual vs pred for both models."""

    def _scatter(ax, actual, pred, color, title, rmse_label):
        err = np.abs(pred - actual) / actual * 100
        large = err > 10

        ax.scatter(actual[~large], pred[~large], s=12, alpha=0.5,
                   color=color, label="≤10% error")
        ax.scatter(actual[large],  pred[large],  s=18, alpha=0.7,
                   color="#FF5722", marker="x", linewidths=1.2, label=">10% error")

        lo = min(actual.min(), pred.min()) * 0.95
        hi = max(actual.max(), pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color=C_ORACLE, linewidth=1.8,
                linestyle="--", label="Oracle (perfect prediction)", zorder=5)
        ax.fill_between([lo, hi], [lo*0.9, hi*0.9], [lo*1.1, hi*1.1],
                        alpha=0.07, color=C_ORACLE, label="±10% band")

        rmse = np.sqrt(np.mean((pred - actual)**2))
        ax.text(0.97, 0.04, f"RMSE = {rmse:.1f} ms", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Actual (ms)", fontsize=10)
        ax.set_ylabel("Predicted (ms)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

    ll_actual, ll_pred, _ = llama_exp3
    mx_actual, mx_pred, _ = mix_exp3

    _scatter(ax_l, ll_actual, ll_pred, C_LLAMA,
             "(B1) Llama3-8B\nCurrent Predictor vs Oracle", "")
    _scatter(ax_r, mx_actual, mx_pred, C_MIX,
             "(B2) Mixtral-8x7B EP\nCurrent Predictor vs Oracle", "")

    # Add arrow + annotation on Mixtral plot to show research gap
    y_mid = (mx_actual.max() + mx_actual.min()) / 2
    ax_r.annotate("", xy=(y_mid * 0.7, y_mid * 1.35),
                  xytext=(y_mid * 0.7, y_mid * 0.65),
                  arrowprops=dict(arrowstyle="<->", color="#FF5722", lw=1.5))
    ax_r.text(y_mid * 0.68, y_mid, "Research gap\n(all_to_all\njitter)",
              ha="right", va="center", fontsize=7.5, color="#FF5722")


def panel_C(ax, llama_rows, mix_rows):
    """FP rate vs SLO threshold for both models + oracle."""

    llama_actual = np.array([float(r["actual_ms"]) for r in llama_rows])
    mix_actual   = np.array([float(r["actual_ms"]) for r in mix_rows])

    # Build SLO range around each model's mean
    mix_mean = np.mean(mix_actual)
    mix_std  = np.std(mix_actual)
    slo_range = np.linspace(mix_mean - mix_std, mix_mean + 3 * mix_std, 120)

    mix_fp, oracle_fp  = compute_fp_curve(mix_rows,   slo_range)
    ll_fp,  _          = compute_fp_curve(llama_rows, slo_range)

    ax.plot(slo_range, mix_fp * 100,    color=C_MIX,   linewidth=2.2,
            label="Mixtral-8x7B EP (current predictor)")
    ax.plot(slo_range, ll_fp * 100,     color=C_LLAMA, linewidth=2.2,
            label="Llama3-8B (current predictor)")
    ax.plot(slo_range, oracle_fp * 100, color=C_ORACLE, linewidth=1.8,
            linestyle="--", label="Oracle gate (0% FP by construction)")

    # Shade research gap
    ax.fill_between(slo_range, oracle_fp * 100, mix_fp * 100,
                    alpha=0.15, color=C_MIX, label="Research gap")

    ax.set_xlabel("SLO Threshold (ms)", fontsize=11)
    ax.set_ylabel("False Positive Rate (%)\n[SLO violations / all SLO-violating batches]",
                  fontsize=9.5)
    ax.set_title("(C) SLO Gate: Current Predictor vs Oracle\n"
                 "(FP = SLO-violating batch incorrectly admitted)", fontsize=10)
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # Annotate worst-case Mixtral FP
    worst_idx = np.argmax(mix_fp)
    worst_slo = slo_range[worst_idx]
    worst_fp  = mix_fp[worst_idx] * 100
    ax.annotate(f"{worst_fp:.0f}% FP\n@ SLO={worst_slo:.0f}ms",
                xy=(worst_slo, worst_fp),
                xytext=(worst_slo + mix_std * 0.3, worst_fp - 15),
                arrowprops=dict(arrowstyle="->", color=C_MIX),
                fontsize=8.5, color=C_MIX)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check which files exist
    missing = []
    for path in [MIXTRAL_EXP3, LLAMA3_EXP3]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("Missing required files:")
        for m in missing: print(f"  {m}")
        sys.exit(1)

    fixed_available = (os.path.exists(MIXTRAL_FIXED) and os.path.exists(LLAMA3_FIXED))
    if not fixed_available:
        print("Fixed-composition data not found; Panel A will be skipped.")
        print(f"  Expected: {MIXTRAL_FIXED}")
        print(f"  Expected: {LLAMA3_FIXED}")

    # Load data
    mix_exp3_rows   = load_csv(MIXTRAL_EXP3)
    ll_exp3_rows    = load_csv(LLAMA3_EXP3)
    mix_exp3        = get_exp3(mix_exp3_rows)
    ll_exp3         = get_exp3(ll_exp3_rows)

    if fixed_available:
        mix_fixed_rows = load_csv(MIXTRAL_FIXED)
        ll_fixed_rows  = load_csv(LLAMA3_FIXED)
        mix_fixed, mix_pred_by_tin = get_fixed_comp(mix_fixed_rows, "Mixtral")
        ll_fixed,  ll_pred_by_tin  = get_fixed_comp(ll_fixed_rows,  "Llama3")

    # Build figure
    if fixed_available:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
        ax_A           = axes[0]
        ax_B_llama     = axes[1]
        ax_B_mix       = axes[2]
        ax_C           = axes[3]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        ax_A           = None
        ax_B_llama     = axes[0]
        ax_B_mix       = axes[1]
        ax_C           = axes[2]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    if fixed_available:
        panel_A(ax_A, ll_fixed, mix_fixed, ll_pred_by_tin, mix_pred_by_tin)

    panel_B(ax_B_llama, ax_B_mix, ll_exp3, mix_exp3)
    panel_C(ax_C, ll_exp3_rows, mix_exp3_rows)

    plt.suptitle(
        "DeltaServe Predictor Gap: Current Predictor vs Oracle\n"
        "Mixtral-8x7B EP all_to_all jitter creates an irreducible 9ms RMSE floor "
        "that no static predictor can overcome",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, "predictor_gap.pdf")
    out_png = os.path.join(OUT_DIR, "predictor_gap.png")
    plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    # Print summary stats
    print("\n--- Summary ---")
    if fixed_available:
        print("Panel A (fixed composition, test set):")
        print(f"  {'T_in':>6}  {'Llama3 std':>12}  {'Mixtral std':>12}  {'ratio':>6}")
        for T_in in T_IN_VALUES:
            ll = ll_fixed.get(T_in)
            mx = mix_fixed.get(T_in)
            if ll is not None and mx is not None:
                ratio = np.std(mx) / np.std(ll) if np.std(ll) > 0 else float("inf")
                print(f"  {T_in:>6}  {np.std(ll):>10.3f}ms  {np.std(mx):>10.3f}ms  {ratio:>6.1f}×")

    ll_actual, ll_pred, _ = ll_exp3
    mx_actual, mx_pred, _ = mix_exp3
    ll_rmse = np.sqrt(np.mean((ll_pred - ll_actual)**2))
    mx_rmse = np.sqrt(np.mean((mx_pred - mx_actual)**2))
    print(f"\nPanel B (exp3 scatter):")
    print(f"  Llama3  RMSE: {ll_rmse:.1f}ms")
    print(f"  Mixtral RMSE: {mx_rmse:.1f}ms  ({mx_rmse/ll_rmse:.1f}× higher)")

    print(f"\nPanel C (SLO gate):")
    slo_vals = np.linspace(np.mean(mx_actual) - np.std(mx_actual),
                           np.mean(mx_actual) + 3*np.std(mx_actual), 120)
    mx_fp, _ = compute_fp_curve(mix_exp3_rows, slo_vals)
    peak_slo = slo_vals[np.argmax(mx_fp)]
    peak_fp  = np.max(mx_fp) * 100
    print(f"  Mixtral peak FP rate: {peak_fp:.0f}% at SLO={peak_slo:.0f}ms")
    print(f"  Oracle FP rate: 0% (by construction)")


if __name__ == "__main__":
    main()
