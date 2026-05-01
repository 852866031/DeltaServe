"""plot_loose_tight.py — single-trace 4-panel plots for loose & tight runs.

For a given base configuration suffix, this looks up the two timeline-shape
variants (`_loose` and `_tight`) and emits one 4-subplot PNG per variant.
Unlike compare_kv_plot.py / compare_graphs_plot.py, there is no comparison
between two configurations — each plot just shows the single trace for that
workload shape.

Each PNG has the same 4 subplots used by the comparison scripts:

  1. Scheduled request timeline (from timeline_<mode>.csv)
  2. TTFT percentile curve
  3. Per-request E2E latency vs time
  4. Cumulative finetune tokens

Inputs (under eval/llama3/output/, suffix scheme from auto_benchmark.py):
  timeline_results<base><mode>.csv        per-request results
  bwd_log<base><mode>.csv                 finetune backward log
  timeline_<bare-mode>.csv                input timeline (eval/llama3/)

where <mode> is `_loose` or `_tight` and <bare-mode> is `loose` or `tight`.

Run the upstream benchmarks first, e.g.:
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph --packed_kv --loose
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph --packed_kv --tight

Then:
  python plot_loose_tight.py --suffix _decode_prefill_bwd_kv

Default suffix is `_decode_prefill_bwd_kv` (the all-graphs + packed_kv
configuration). Pass `--suffix ""` for a bare run with no graphs/allocator
tags, where the inputs are e.g. `timeline_results_loose.csv`.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from bwd_graph_plot import (
    ensure_exists,
    load_results,
    parse_bwd_log_csv,
)
from compare_graphs_plot import load_timeline, plot_request_timeline


_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "output")
PLOTS_DIR = os.path.join(_HERE, "plots")

MODE_COLORS = {
    "loose": "tab:blue",
    "tight": "tab:red",
}


# ---------------- Single-trace plot helpers ----------------
def plot_ttft_percentile_single(ax, results_df, label: str, color: str):
    percentiles = np.array([0, 20, 40, 60, 80, 100])
    ttft = results_df.loc[results_df["ok"], "ttft_s"].dropna().astype(float).to_numpy()
    if len(ttft) > 0:
        values = np.percentile(ttft, percentiles)
        ax.plot(percentiles, values, marker="o", color=color, label=label)
    ax.set_title("TTFT Percentile Curve")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("TTFT (s)")
    ax.legend()


def plot_latency_vs_time_single(ax, results_df, label: str, color: str):
    ok = results_df[results_df["ok"]]
    latencies = ok["latency_s"].astype(float).to_numpy()
    avg_latency = float(np.mean(latencies)) if len(latencies) > 0 else float("nan")
    series_label = (
        f"{label} (avg {avg_latency:.3f}s)" if np.isfinite(avg_latency) else label
    )
    ax.scatter(
        ok["t_rel_s"],
        ok["latency_s"],
        s=10,
        color=color,
        alpha=0.7,
        label=series_label,
    )
    if np.isfinite(avg_latency):
        ax.axhline(avg_latency, color=color, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Request E2E Latency vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (s)")
    ax.legend()


def plot_bwd_cumulative_single(ax, log_tuple, label: str, color: str):
    t, cum, avg = log_tuple
    series_label = f"{label} ({avg:.1f} tok/s)" if np.isfinite(avg) else label
    ax.plot(t, cum, color=color, label=series_label)
    ax.set_title("Finetuning Cumulative Tokens")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Tokens")
    ax.legend()


# ---------------- Per-mode figure ----------------
def make_figure_for_mode(
    mode: str,
    base_suffix: str,
    output_dir: str,
    plots_dir: str,
    timeline_csv_dir: str,
    out_path: str,
) -> None:
    """Build one 4-panel PNG for `mode` ∈ {'loose', 'tight'}."""
    full_suffix = f"{base_suffix}_{mode}"
    results_csv = os.path.join(output_dir, f"timeline_results{full_suffix}.csv")
    bwd_log_csv = os.path.join(output_dir, f"bwd_log{full_suffix}.csv")
    timeline_csv = os.path.join(timeline_csv_dir, f"timeline_{mode}.csv")

    for p in (timeline_csv, results_csv, bwd_log_csv):
        ensure_exists(p)

    timeline_df = load_timeline(timeline_csv)
    results_df = load_results(results_csv)
    bwd_log = parse_bwd_log_csv(bwd_log_csv)

    color = MODE_COLORS.get(mode, "tab:gray")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    plot_request_timeline(axes[0], timeline_df)
    plot_ttft_percentile_single(axes[1], results_df, label=mode, color=color)
    plot_latency_vs_time_single(axes[2], results_df, label=mode, color=color)
    plot_bwd_cumulative_single(axes[3], bwd_log, label=mode, color=color)

    config_tag = base_suffix.lstrip("_") or "baseline"
    fig.suptitle(f"{mode}  ({config_tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[plot_loose_tight] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--suffix",
        default="_decode_prefill_bwd_kv",
        help="Base configuration suffix from auto_benchmark, WITHOUT the "
             "trailing _loose/_tight tag. Examples: '' (no graphs/allocator), "
             "'_decode_prefill_bwd', '_decode_prefill_bwd_kv' (default). "
             "The script appends _loose / _tight itself to find the per-mode files.",
    )
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="Where auto_benchmark writes its CSVs.")
    ap.add_argument("--plots-dir", default=PLOTS_DIR,
                    help="Where to write the PNGs.")
    ap.add_argument("--timeline-csv-dir", default=_HERE,
                    help="Directory containing timeline_loose.csv and timeline_tight.csv.")
    ap.add_argument("--mode", choices=("loose", "tight", "both"), default="both",
                    help="Which mode(s) to plot. Default: both.")
    args = ap.parse_args()

    modes = ("loose", "tight") if args.mode == "both" else (args.mode,)

    failures = []
    for mode in modes:
        config_tag = args.suffix.lstrip("_") or "baseline"
        out_path = os.path.join(args.plots_dir, f"{mode}_{config_tag}.png")
        try:
            make_figure_for_mode(
                mode=mode,
                base_suffix=args.suffix,
                output_dir=args.output_dir,
                plots_dir=args.plots_dir,
                timeline_csv_dir=args.timeline_csv_dir,
                out_path=out_path,
            )
        except FileNotFoundError as e:
            failures.append((mode, str(e)))

    if failures:
        for mode, msg in failures:
            print(f"[plot_loose_tight] {mode}: missing input — {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
