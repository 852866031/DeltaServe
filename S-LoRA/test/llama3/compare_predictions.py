"""
compare_predictions.py
----------------------
Compare predictor accuracy between Llama3 (dense) and Mixtral (MoE).

Usage:
    python compare_predictions.py \
        --llama3  prediction_stats_meta-llama_Meta-Llama-3-8B.csv \
        --mixtral prediction_stats_<mixtral_path>.csv

Outputs per model:
  - RMSE (prefill / decode)
  - Mean absolute error %
  - Per-bucket timing variance (same total_tokens → should be tight for Llama3, wide for Mixtral)
  - Scatter plot: predicted vs actual colored by model (saved to compare_predictions.png)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_csv(path: str):
    """Load prediction stats CSV. Returns list of row dicts."""
    import csv
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["execution_duration"] = float(row["execution_duration"])
            row["predicted_duration"] = float(row["predicted_duration"])
            row["batch_index"] = int(row["batch_index"])
            try:
                row["inference_tokens"] = json.loads(row["inference_tokens"])
            except Exception:
                row["inference_tokens"] = []
            try:
                row["finetuning_tokens"] = json.loads(row["finetuning_tokens"])
            except Exception:
                row["finetuning_tokens"] = []
            rows.append(row)
    return rows


def total_tokens(row):
    inf = row["inference_tokens"]
    ft = row["finetuning_tokens"]
    try:
        t_inf = sum(sum(r) if isinstance(r, list) else r for r in inf) if inf else 0
    except Exception:
        t_inf = sum(inf) if inf else 0
    try:
        t_ft = sum(sum(r) if isinstance(r, list) else r for r in ft) if ft else 0
    except Exception:
        t_ft = sum(ft) if ft else 0
    return t_inf + t_ft


def rmse(actual, predicted):
    a = np.array(actual)
    p = np.array(predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae_pct(actual, predicted):
    a = np.array(actual)
    p = np.array(predicted)
    return float(np.mean(np.abs(a - p) / np.maximum(a, 1e-9))) * 100


def analyze(rows, label):
    prefill = [r for r in rows if r["batch_type"] == "prefill"]
    decode  = [r for r in rows if r["batch_type"] == "decode"]

    print(f"\n{'='*60}")
    print(f"  {label}  ({len(rows)} total rows: {len(prefill)} prefill, {len(decode)} decode)")
    print(f"{'='*60}")

    for tag, subset in [("PREFILL", prefill), ("DECODE", decode)]:
        if not subset:
            print(f"  {tag}: no data")
            continue
        actual    = [r["execution_duration"] for r in subset]
        predicted = [r["predicted_duration"]  for r in subset]
        r = rmse(actual, predicted)
        m = mae_pct(actual, predicted)
        print(f"  {tag}: RMSE={r*1000:.2f}ms  MAE%={m:.1f}%  n={len(subset)}")

        # Per-bucket variance
        tokens = [total_tokens(row) for row in subset]
        if any(t > 0 for t in tokens):
            bucket_residuals = {}
            for row, tok in zip(subset, tokens):
                if tok == 0:
                    continue
                # log2-bucket: 0-63 / 64-127 / 128-255 / etc.
                b = 2 ** int(math.log2(max(tok, 1)))
                bucket_residuals.setdefault(b, []).append(
                    abs(row["execution_duration"] - row["predicted_duration"])
                )
            print(f"  {tag} per-bucket residual std-dev (ms):")
            for bkt in sorted(bucket_residuals):
                errs = np.array(bucket_residuals[bkt]) * 1000
                print(f"    [{bkt:5d}-{bkt*2-1:5d} tokens] "
                      f"n={len(errs):3d}  mean={errs.mean():.2f}ms  std={errs.std():.2f}ms")

    return prefill, decode


def plot(llama3_prefill, mixtral_prefill, llama3_decode, mixtral_decode, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[compare_predictions] matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, l3, mx, title in [
        (axes[0], llama3_prefill, mixtral_prefill, "Prefill"),
        (axes[1], llama3_decode,  mixtral_decode,  "Decode"),
    ]:
        for rows, color, label in [(l3, "steelblue", "Llama3"), (mx, "crimson", "Mixtral")]:
            if not rows:
                continue
            actual    = [r["execution_duration"] * 1000 for r in rows]
            predicted = [r["predicted_duration"]  * 1000 for r in rows]
            ax.scatter(actual, predicted, alpha=0.5, s=20, color=color, label=label)

        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="perfect")
        ax.set_xlabel("Actual (ms)")
        ax.set_ylabel("Predicted (ms)")
        ax.set_title(f"{title}: predicted vs actual")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n[compare_predictions] Plot saved → {out_path}")


def variance_comparison(l3_rows, mx_rows, tag):
    """Print side-by-side per-bucket std-dev summary for the key comparison."""
    def bucket_std(rows):
        bkt_errs = {}
        for row in rows:
            tok = total_tokens(row)
            if tok == 0:
                continue
            b = 2 ** int(math.log2(max(tok, 1)))
            bkt_errs.setdefault(b, []).append(
                abs(row["execution_duration"] - row["predicted_duration"])
            )
        return {b: np.std(v) for b, v in bkt_errs.items()}

    l3_std  = bucket_std(l3_rows)
    mx_std  = bucket_std(mx_rows)
    all_bkts = sorted(set(l3_std) | set(mx_std))

    print(f"\n  {tag} same-bucket residual std-dev comparison (ms):")
    print(f"  {'bucket':>14}  {'Llama3':>10}  {'Mixtral':>10}  {'ratio':>8}")
    for b in all_bkts:
        l3 = l3_std.get(b, float("nan")) * 1000
        mx = mx_std.get(b, float("nan")) * 1000
        ratio = mx / l3 if l3 > 0 else float("nan")
        print(f"  {b:5d}-{b*2-1:<5d}       {l3:8.2f}ms    {mx:8.2f}ms    {ratio:6.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama3",  required=True, help="Llama3 prediction stats CSV")
    parser.add_argument("--mixtral", required=True, help="Mixtral prediction stats CSV")
    parser.add_argument("--plot",    default="compare_predictions.png")
    args = parser.parse_args()

    if not Path(args.llama3).exists():
        print(f"ERROR: {args.llama3} not found", file=sys.stderr)
        sys.exit(1)
    if not Path(args.mixtral).exists():
        print(f"ERROR: {args.mixtral} not found", file=sys.stderr)
        sys.exit(1)

    l3_rows  = load_csv(args.llama3)
    mx_rows  = load_csv(args.mixtral)

    l3_prefill, l3_decode   = analyze(l3_rows,  "Llama3 (dense)")
    mx_prefill, mx_decode   = analyze(mx_rows,  "Mixtral MoE")

    print(f"\n{'='*60}")
    print("  KEY COMPARISON: same-bucket residual variance")
    print(f"{'='*60}")
    variance_comparison(l3_prefill, mx_prefill, "PREFILL")
    variance_comparison(l3_decode,  mx_decode,  "DECODE")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for tag, l3_sub, mx_sub in [("Prefill", l3_prefill, mx_prefill),
                                  ("Decode",  l3_decode,  mx_decode)]:
        if not l3_sub or not mx_sub:
            continue
        l3_r = rmse([r["execution_duration"] for r in l3_sub],
                    [r["predicted_duration"]  for r in l3_sub])
        mx_r = rmse([r["execution_duration"] for r in mx_sub],
                    [r["predicted_duration"]  for r in mx_sub])
        ratio = mx_r / l3_r if l3_r > 0 else float("nan")
        verdict = "WORSE (expected)" if ratio > 1.2 else ("SIMILAR" if ratio > 0.8 else "BETTER")
        print(f"  {tag}: Llama3 RMSE={l3_r*1000:.2f}ms  Mixtral RMSE={mx_r*1000:.2f}ms  "
              f"ratio={ratio:.2f}x  [{verdict}]")

    plot(l3_prefill, mx_prefill, l3_decode, mx_decode, args.plot)


if __name__ == "__main__":
    main()
