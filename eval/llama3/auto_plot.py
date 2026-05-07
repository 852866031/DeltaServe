"""auto_plot.py — single-trace 4-panel plots for loose, tight, and nutanix runs.

For each workload-shape variant the auto_benchmark.py emits, this script
builds one 4-subplot PNG showing the trace for that run.

Subplots (left → right):

  1. Scheduled request timeline (from timeline_<mode>.csv)
  2. Per-request E2E latency vs time
  3. Throughput (tokens/s) over time — two curves: inference-only and
     total (inference + finetune); the gap between them is shaded and
     equals the FT contribution at each instant.
  4. TTFT SLO satisfaction rate over time — rolling-window % of requests
     whose TTFT met the SLO threshold, with a 95% reference line.

Defaults assume the alpaca + all-optimizations-on configuration:
  --co --decode_graph --prefill_graph --bwd_graph --packed_kv --alpaca

which produces output suffix `_decode_prefill_bwd_kv_alpaca`. Override
via `--suffix` if you ran a different combination.

The TTFT SLO threshold (subplot 4) is read from the YAML config file that
matches the suffix (alpaca → serving_config_finetuning_alpaca.yaml,
otherwise → serving_config_finetuning.yaml). Override with `--slo`.

Inputs (under eval/llama3/output/, suffix scheme from auto_benchmark.py):
  timeline_results<base>_<mode>.csv       per-request results
  bwd_log<base>_<mode>.csv                finetune backward log
  timeline_<mode>.csv                     input timeline (eval/llama3/)

Run upstream first, e.g.:
  python auto_benchmark.py --co --all_graph --packed_kv --alpaca --loose
  python auto_benchmark.py --co --all_graph --packed_kv --alpaca --tight
  python auto_benchmark.py --co --all_graph --packed_kv --alpaca --nutanix

Then:
  python auto_plot.py
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
CONFIG_DIR = os.path.join(_HERE, "config")
DEFAULT_SLO_FALLBACK = 0.35


def _config_yaml_for_suffix(suffix: str) -> str:
    """Pick the server YAML whose tags match the run suffix. Used to read
    the TTFT SLO so the satisfaction-rate subplot's reference matches the
    actual SLO the server enforced."""
    if "_alpaca" in suffix:
        return os.path.join(CONFIG_DIR, "serving_config_finetuning_alpaca.yaml")
    if "_kv" in suffix:
        return os.path.join(CONFIG_DIR, "serving_config_finetuning_packed.yaml")
    return os.path.join(CONFIG_DIR, "serving_config_finetuning.yaml")


def _read_ttft_slo_from_yaml(yaml_path: str):
    """Return float(slo.ttft_slo) from `yaml_path`, or None on any failure."""
    try:
        import yaml  # PyYAML — already a project dep (used by dserve/server/config.py)
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        v = data.get("slo", {}).get("ttft_slo")
        return float(v) if v is not None else None
    except Exception:
        return None

ALL_MODES = ("loose", "tight", "nutanix")

MODE_COLORS = {
    "loose":   "tab:blue",
    "tight":   "tab:red",
    "nutanix": "tab:green",
}

# Color used to shade the FT contribution in the throughput stack —
# kept stable across modes so the eye reads the FT band consistently.
FT_SHADE_COLOR = "tab:orange"


# ---------------- Token-distribution helpers ----------------
def _distribute_to_bins(t_starts, t_ends, weights, t_max, bin_s):
    """For each (t_start, t_end, weight), spread `weight` uniformly across
    1 s bins covering [t_start, t_end]. Returns a per-bin array."""
    n_bins = int(np.ceil(t_max / bin_s)) + 1
    bins = np.zeros(n_bins, dtype=float)
    for ts, te, w in zip(t_starts, t_ends, weights):
        if te <= ts:
            # Degenerate (no decode duration recorded) — drop weight in start bin.
            b = int(ts // bin_s)
            if 0 <= b < n_bins:
                bins[b] += float(w)
            continue
        rate = float(w) / (te - ts)
        b_start = max(0, int(np.floor(ts / bin_s)))
        b_end = min(n_bins, int(np.ceil(te / bin_s)))
        for b in range(b_start, b_end):
            lo = max(ts, b * bin_s)
            hi = min(te, (b + 1) * bin_s)
            if hi > lo:
                bins[b] += rate * (hi - lo)
    return bins


def _ft_per_bin(rel_t, cum_tok, t_max, bin_s):
    """Convert (timestamp, cumulative-tokens) sequence into per-bin token
    deltas via linear interpolation at bin edges."""
    n_bins = int(np.ceil(t_max / bin_s)) + 1
    if len(rel_t) == 0 or len(cum_tok) == 0:
        return np.zeros(n_bins)
    edges = np.arange(n_bins + 1) * bin_s
    cum_at_edges = np.interp(
        edges, rel_t, cum_tok, left=0.0, right=float(cum_tok[-1])
    )
    deltas = np.diff(cum_at_edges)
    deltas[deltas < 0] = 0.0  # cumulative shouldn't decrease; clip noise
    return deltas


# ---------------- Single-trace plot helpers ----------------
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


def plot_throughput_curves(
    ax, results_df, timeline_df, bwd_log,
    color_inf: str, bin_s: float = 1.0,
):
    """Two-curve throughput view: inference-only tokens/s and total
    (inference + finetune) tokens/s. The area BETWEEN the curves — which
    is the FT contribution at each instant — is shaded.

    Both series are on the request-timeline time origin; the FT origin is
    the bwd-log's first event, so very early seconds may be slightly
    misaligned by the FT-trigger latency (a few s).
    """
    # Join max_new_tokens onto results via row_id (timeline df.index == row_id).
    tl = timeline_df.copy()
    tl["row_id"] = tl.index
    merged = results_df.merge(
        tl[["row_id", "max_new_tokens"]],
        left_on="idx", right_on="row_id", how="left",
    )
    ok = merged[
        merged["ok"]
        & merged["ttft_s"].notna()
        & merged["latency_s"].notna()
        & merged["max_new_tokens"].notna()
    ]

    if len(ok) > 0:
        inf_t_start = (ok["t_rel_s"].astype(float) + ok["ttft_s"].astype(float)).to_numpy()
        inf_t_end   = (ok["t_rel_s"].astype(float) + ok["latency_s"].astype(float)).to_numpy()
        inf_tokens  = ok["max_new_tokens"].astype(float).to_numpy()
    else:
        inf_t_start = inf_t_end = inf_tokens = np.array([], dtype=float)

    rel_t, cum_tok, _ = bwd_log

    t_max = max(
        float(inf_t_end.max()) if len(inf_t_end) else 0.0,
        float(rel_t[-1]) if len(rel_t) else 0.0,
    )
    if t_max <= 0:
        ax.set_title("Throughput (no data)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tokens / s")
        return

    inf_per_s = _distribute_to_bins(inf_t_start, inf_t_end, inf_tokens, t_max, bin_s) / bin_s
    ft_per_s  = _ft_per_bin(rel_t, cum_tok, t_max, bin_s) / bin_s

    # Pad both to a common length so the fill_between aligns.
    n = max(len(inf_per_s), len(ft_per_s))
    inf_per_s = np.pad(inf_per_s, (0, n - len(inf_per_s)))
    ft_per_s  = np.pad(ft_per_s,  (0, n - len(ft_per_s)))
    centers = (np.arange(n) + 0.5) * bin_s
    total = inf_per_s + ft_per_s

    inf_avg, ft_avg, tot_avg = inf_per_s.mean(), ft_per_s.mean(), total.mean()

    # Shade the gap between inference and total — that gap IS the FT contribution.
    ax.fill_between(
        centers, inf_per_s, total,
        color=FT_SHADE_COLOR, alpha=0.30, hatch="//", linewidth=0,
        label=f"Finetune contribution (avg {ft_avg:.0f} tok/s)",
    )
    # Two curves on top.
    ax.plot(
        centers, inf_per_s, color=color_inf, linewidth=1.4,
        label=f"Inference (avg {inf_avg:.0f} tok/s)",
    )
    ax.plot(
        centers, total, color="black", linewidth=1.4,
        label=f"Total (avg {tot_avg:.0f} tok/s)",
    )
    ax.set_title("Throughput (tokens/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tokens / s")
    ax.legend(loc="best", fontsize=8)


def plot_ttft_satisfaction_rate(
    ax, results_df, slo_s: float, window_s: float = 5.0,
):
    """Rolling % of requests in a `window_s` window whose TTFT ≤ `slo_s`.
    Attribution is by request arrival time (t_rel_s)."""
    ok = results_df[results_df["ok"] & results_df["ttft_s"].notna()]
    if len(ok) == 0:
        ax.set_title(f"TTFT Satisfaction Rate (no data)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("% of requests")
        return

    t = ok["t_rel_s"].astype(float).to_numpy()
    sat = (ok["ttft_s"].astype(float) <= slo_s).to_numpy().astype(int)

    t_max = float(t.max())
    grid = np.arange(0.0, t_max + 1.0, 1.0)
    rates = np.full(len(grid), np.nan)
    half = window_s / 2.0
    for i, t0 in enumerate(grid):
        mask = (t >= t0 - half) & (t <= t0 + half)
        n = int(mask.sum())
        if n > 0:
            rates[i] = sat[mask].mean() * 100.0

    overall = sat.mean() * 100.0

    ax.plot(
        grid, rates, color="tab:purple", linewidth=1.5,
        label=f"Satisfaction ({int(window_s)}s window) — overall {overall:.1f}%",
    )
    ax.axhline(95.0, color="tab:red", linestyle="--", linewidth=1, alpha=0.6,
               label="95% target")
    ax.set_ylim(0, 105)
    ax.set_title(f"TTFT Satisfaction Rate (SLO ≤ {slo_s:.2f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% of requests")
    ax.legend(loc="best", fontsize=8)


# ---------------- Per-mode figure ----------------
def make_figure_for_mode(
    mode: str,
    base_suffix: str,
    output_dir: str,
    plots_dir: str,
    timeline_csv_dir: str,
    out_path: str,
    slo_s: float,
    window_s: float,
) -> None:
    """Build one 4-panel PNG for `mode` ∈ {'loose', 'tight', 'nutanix'}."""
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
    plot_latency_vs_time_single(axes[1], results_df, label=mode, color=color)
    plot_throughput_curves(axes[2], results_df, timeline_df, bwd_log, color_inf=color)
    plot_ttft_satisfaction_rate(axes[3], results_df, slo_s=slo_s, window_s=window_s)

    config_tag = base_suffix.lstrip("_") or "baseline"
    fig.suptitle(f"{mode}  ({config_tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[auto_plot] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--suffix",
        default="_decode_prefill_bwd_kv_alpaca",
        help="Base configuration suffix from auto_benchmark, WITHOUT the "
             "trailing _<mode> tag. Default '_decode_prefill_bwd_kv_alpaca' "
             "matches '--co --all_graph --packed_kv --alpaca'.",
    )
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="Where auto_benchmark writes its CSVs.")
    ap.add_argument("--plots-dir", default=PLOTS_DIR,
                    help="Where to write the PNGs.")
    ap.add_argument("--timeline-csv-dir", default=_HERE,
                    help="Directory containing timeline_<mode>.csv inputs.")
    ap.add_argument(
        "--mode",
        choices=ALL_MODES + ("all",),
        default="all",
        help="Which mode(s) to plot. 'all' (default) emits one PNG per mode "
             "in {loose, tight, nutanix}; modes whose input CSVs are missing "
             "are skipped with a warning.",
    )
    ap.add_argument(
        "--slo", type=float, default=None,
        help="TTFT SLO threshold (seconds) for the satisfaction-rate subplot. "
             "Default: read slo.ttft_slo from the YAML matching --suffix "
             "(alpaca → serving_config_finetuning_alpaca.yaml, otherwise the "
             "appropriate non-alpaca finetuning YAML). Falls back to "
             f"{DEFAULT_SLO_FALLBACK}s if the YAML can't be read.",
    )
    ap.add_argument(
        "--config-yaml",
        default=None,
        help="Override the YAML the SLO is read from. Default: auto-pick by "
             "suffix tags (see --slo for the resolution rule).",
    )
    ap.add_argument(
        "--window", type=float, default=5.0,
        help="Rolling-window size (seconds) for the satisfaction-rate "
             "subplot. Default 5s.",
    )
    args = ap.parse_args()

    # Resolve SLO: explicit --slo wins; else read from YAML; else fallback.
    slo_s = args.slo
    if slo_s is None:
        yaml_path = args.config_yaml or _config_yaml_for_suffix(args.suffix)
        slo_s = _read_ttft_slo_from_yaml(yaml_path)
        if slo_s is not None:
            print(f"[auto_plot] SLO = {slo_s:.3f}s (read from {os.path.basename(yaml_path)})")
        else:
            slo_s = DEFAULT_SLO_FALLBACK
            print(
                f"[auto_plot] SLO = {slo_s:.3f}s (fallback — couldn't read "
                f"slo.ttft_slo from {yaml_path})",
                file=sys.stderr,
            )
    else:
        print(f"[auto_plot] SLO = {slo_s:.3f}s (from --slo)")

    modes = ALL_MODES if args.mode == "all" else (args.mode,)

    skipped = []
    wrote = 0
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
                slo_s=slo_s,
                window_s=args.window,
            )
            wrote += 1
        except FileNotFoundError as e:
            skipped.append((mode, str(e)))

    for mode, msg in skipped:
        print(f"[auto_plot] {mode}: skipped (missing input — {msg})", file=sys.stderr)

    if wrote == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
