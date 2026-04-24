"""compare_graphs_plot.py — ablation plots for the CUDA graph features.

Produces three side-by-side comparison PNGs, each pitting the baseline
(no CUDA graph) against a progressively more-enabled variant:

  plots/compare_prefill.png              baseline vs prefill graph
  plots/compare_prefill_decode.png       baseline vs prefill+decode graph
  plots/compare_prefill_decode_bwd.png   baseline vs prefill+decode+bwd graph

Each figure has the same 3-panel layout as bwd_graph_plot.py:
  [TTFT percentile] [E2E latency vs time] [cumulative finetune tokens].

All four CSVs must live under eval/llama3/output/ (produced by
auto_benchmark.py with the matching flags). Missing variants are skipped
with a warning so partial ablations still produce what they can.

Run the upstream benchmarks first, e.g.:
  python auto_benchmark.py --co                                           # baseline
  python auto_benchmark.py --co --prefill_graph                           # prefill
  python auto_benchmark.py --co --decode_graph --prefill_graph            # prefill+decode
  python auto_benchmark.py --co --decode_graph --prefill_graph --bwd_graph
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt

from bwd_graph_plot import (
    ensure_exists,
    load_results,
    parse_bwd_log_csv,
    plot_ttft_percentile,
    plot_latency_vs_time,
    plot_bwd_cumulative,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "output")
PLOTS_DIR = os.path.join(_HERE, "plots")

BASELINE_LABEL = "no graph"
BASELINE_COLOR = "tab:blue"
VARIANT_COLOR = "tab:orange"

# (png_stem, variant_label, tag_suffix). Tag order is decode < prefill < bwd
# (matches auto_benchmark's filename composer), so the progressive variants
# stamp as _prefill, _decode_prefill, _decode_prefill_bwd.
COMPARISONS = [
    ("compare_prefill",            "prefill",              "_prefill"),
    ("compare_prefill_decode",     "prefill+decode",       "_decode_prefill"),
    ("compare_prefill_decode_bwd", "prefill+decode+bwd",   "_decode_prefill_bwd"),
]


def _make_figure(base_results, base_log, var_results, var_log,
                 variant_label: str, out_path: str, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = (BASELINE_LABEL, variant_label)
    colors = (BASELINE_COLOR, VARIANT_COLOR)
    plot_ttft_percentile(axes[0], base_results, var_results, labels=labels, colors=colors)
    plot_latency_vs_time(axes[1], base_results, var_results, labels=labels, colors=colors)
    plot_bwd_cumulative(axes[2], base_log, var_log, labels=labels, colors=colors)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="Where auto_benchmark writes its CSVs.")
    ap.add_argument("--plots-dir", default=PLOTS_DIR,
                    help="Where to write the comparison PNGs.")
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    base_results_csv = os.path.join(args.output_dir, "timeline_results.csv")
    base_log_csv = os.path.join(args.output_dir, "bwd_log.csv")
    try:
        ensure_exists(base_results_csv)
        ensure_exists(base_log_csv)
    except FileNotFoundError as e:
        sys.exit(f"[compare_graphs_plot] baseline CSV missing: {e}")

    base_results = load_results(base_results_csv)
    base_log = parse_bwd_log_csv(base_log_csv)

    wrote = 0
    for png_stem, variant_label, suffix in COMPARISONS:
        var_results_csv = os.path.join(args.output_dir, f"timeline_results{suffix}.csv")
        var_log_csv = os.path.join(args.output_dir, f"bwd_log{suffix}.csv")
        missing = [p for p in (var_results_csv, var_log_csv) if not os.path.exists(p)]
        if missing:
            print(f"[compare_graphs_plot] skip {png_stem}: missing {missing}",
                  file=sys.stderr)
            continue

        var_results = load_results(var_results_csv)
        var_log = parse_bwd_log_csv(var_log_csv)
        out_path = os.path.join(args.plots_dir, f"{png_stem}.png")
        title = f"{BASELINE_LABEL}  vs  {variant_label}"
        _make_figure(base_results, base_log, var_results, var_log,
                     variant_label, out_path, title)
        print(f"[compare_graphs_plot] Wrote {out_path}")
        wrote += 1

    if wrote == 0:
        sys.exit("[compare_graphs_plot] no variant CSVs found — nothing to compare")


if __name__ == "__main__":
    main()
