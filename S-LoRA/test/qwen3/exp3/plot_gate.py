#!/usr/bin/env python3
"""
Experiment 3 - Oracle vs Simulated Gate Plot (Qwen3-30B-A3B EP)
================================================================
Two panels:

  Panel A: Scatter — predicted coserve time vs actual coserve time.
    Each point is one batch. The diagonal = oracle (perfect predictor).
    SLO threshold shown as horizontal + vertical lines. Points are coloured
    by gate outcome: TP (green), TN (grey), FP (red), FN (orange).

  Panel B: FP rate and FN rate vs SLO threshold, for each SFT budget.
    Shows how the gate degrades at tight SLOs (700ms) and clears up at
    looser SLOs (≥900ms).

Usage:
    cd S-LoRA
    python test/qwen3/exp3/plot_gate.py [--budget 64] [--slo 900]

Reads:
    test/qwen3/exp3/results/per_batch.csv
    test/qwen3/exp3/results/estimator_params.json
"""

import os, sys, csv, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PER_BATCH   = os.path.join(RESULTS_DIR, "per_batch.csv")
PARAMS      = os.path.join(RESULTS_DIR, "estimator_params.json")
OUT_DIR     = RESULTS_DIR

SFT_BUDGETS       = [16, 32, 64, 128, 256]
SLO_THRESHOLDS_MS = [700, 800, 900, 1000, 1100, 1200]

C_TP = "#4CAF50"   # green
C_TN = "#BDBDBD"   # grey
C_FP = "#F44336"   # red
C_FN = "#FF9800"   # orange


def load_data():
    with open(PARAMS) as f:
        p = json.load(f)

    per_batch = []
    with open(PER_BATCH) as f:
        for row in csv.DictReader(f):
            per_batch.append(row)
    return p, per_batch


def get_pred_coserve(lens, p, budget):
    """Replicate predict_coserving: alpha*sum_n2 + beta*T_in + beta*T_ft + c, with RMSE inflation."""
    alpha, beta, c, fit_rmse = p["alpha"], p["beta"], p["c"], p["fit_rmse"]
    sum_n2 = sum(l * l for l in lens)
    T_in   = sum(lens)
    T_ft   = budget
    pred_s = alpha * sum_n2 + beta * (T_in + T_ft) + c
    # inflation used in predict_coserving
    inflated = pred_s * (1 + 2 * fit_rmse / max(pred_s, 1e-9))
    return inflated * 1000  # ms


def classify(gate_admits, would_fit):
    if gate_admits and would_fit:     return "TP"
    if gate_admits and not would_fit: return "FP"
    if not gate_admits and would_fit: return "FN"
    return "TN"


def plot(per_batch, params, scatter_budget, scatter_slo, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Qwen3-30B-A3B EP — Exp 3: Oracle vs Simulated Gate", fontsize=13, fontweight="bold")

    # ── Panel A: scatter ──────────────────────────────────────────────────
    ax = axes[0]
    coserve_key = f"actual_coserve_ms_{scatter_budget}"

    actuals, preds, colours = [], [], []
    for r in per_batch:
        lens   = json.loads(r["lengths_json"])
        actual = float(r[coserve_key])
        pred   = get_pred_coserve(lens, params, scatter_budget)
        ga     = pred <= scatter_slo
        wf     = actual <= scatter_slo
        label  = classify(ga, wf)
        actuals.append(actual)
        preds.append(pred)
        colours.append({"TP": C_TP, "TN": C_TN, "FP": C_FP, "FN": C_FN}[label])

    actuals = np.array(actuals)
    preds   = np.array(preds)

    ax.scatter(actuals, preds, c=colours, s=18, alpha=0.7, linewidths=0)

    # diagonal = oracle
    lo = min(actuals.min(), preds.min()) * 0.97
    hi = max(actuals.max(), preds.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Oracle (y=x)")

    # SLO lines
    ax.axhline(scatter_slo, color="#9C27B0", lw=1.2, ls="--", label=f"SLO={scatter_slo}ms (pred)")
    ax.axvline(scatter_slo, color="#9C27B0", lw=1.2, ls=":",  label=f"SLO={scatter_slo}ms (actual)")

    legend_patches = [
        mpatches.Patch(color=C_TP, label="TP (admitted, fits)"),
        mpatches.Patch(color=C_FP, label="FP (admitted, violates SLO)"),
        mpatches.Patch(color=C_FN, label="FN (blocked, would have fit)"),
        mpatches.Patch(color=C_TN, label="TN (blocked, would violate)"),
        plt.Line2D([0],[0], color="k",      ls="--", lw=1.2, label="Oracle (y=x)"),
        plt.Line2D([0],[0], color="#9C27B0",ls="--", lw=1.2, label=f"SLO={scatter_slo}ms"),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5, loc="upper left")
    ax.set_xlabel("Actual coserve time (ms)", fontsize=10)
    ax.set_ylabel("Predicted coserve time (ms)", fontsize=10)
    ax.set_title(f"Panel A: Oracle vs Gate  (SFT budget={scatter_budget} tok, SLO={scatter_slo}ms)", fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # ── Panel B: FP/FN rate vs SLO threshold ─────────────────────────────
    ax2 = axes[1]

    cmap   = plt.get_cmap("tab10")
    styles = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]

    for idx, budget in enumerate(SFT_BUDGETS):
        coserve_key = f"actual_coserve_ms_{budget}"
        fp_rates, fn_rates = [], []
        for slo_ms in SLO_THRESHOLDS_MS:
            fp = fn = tp = tn = 0
            for r in per_batch:
                lens   = json.loads(r["lengths_json"])
                pred   = get_pred_coserve(lens, params, budget)
                ga     = pred <= slo_ms
                wf     = float(r[coserve_key]) <= slo_ms
                if ga and not wf:     fp += 1
                elif not ga and wf:   fn += 1
                elif ga and wf:       tp += 1
                else:                 tn += 1
            n = len(per_batch)
            fp_rates.append(fp / max(fp + tn, 1) * 100)
            fn_rates.append(fn / max(fn + tp, 1) * 100)

        color = cmap(idx)
        ls    = styles[idx % len(styles)]
        ax2.plot(SLO_THRESHOLDS_MS, fp_rates, color=color, ls=ls,  lw=1.8,
                 marker="o", ms=5, label=f"FP  budget={budget}")
        ax2.plot(SLO_THRESHOLDS_MS, fn_rates, color=color, ls="--", lw=1.2,
                 marker="s", ms=4, alpha=0.7, label=f"FN  budget={budget}")

    ax2.set_xlabel("SLO threshold (ms)", fontsize=10)
    ax2.set_ylabel("Rate (%)", fontsize=10)
    ax2.set_title("Panel B: FP / FN rate vs SLO threshold (all SFT budgets)", fontsize=10)
    ax2.set_xticks(SLO_THRESHOLDS_MS)
    ax2.set_ylim(-2, 110)
    ax2.axhline(0, color="k", lw=0.6, ls="--")
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_path.rsplit("/", 1)[0], exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=64,
                        help="SFT budget (tokens) for scatter panel (default 64)")
    parser.add_argument("--slo",    type=int, default=900,
                        help="SLO threshold (ms) for scatter panel (default 900)")
    args = parser.parse_args()

    params, per_batch = load_data()

    out_pdf = os.path.join(OUT_DIR, "gate_oracle_vs_simulated.pdf")
    out_png = os.path.join(OUT_DIR, "gate_oracle_vs_simulated.png")
    plot(per_batch, params, args.budget, args.slo, out_pdf)
    plot(per_batch, params, args.budget, args.slo, out_png)


if __name__ == "__main__":
    main()
