#!/usr/bin/env python3
"""5-panel auto-plot ported from DeltaServe-vLLM/eval/auto_plot.py.

For each workload-shape mode + config produces one PNG with:
  1. Scheduled request timeline (req/s bars + output tok/s line)
  2. Per-request E2E latency vs time (scatter + avg line)
  3. Throughput tok/s bands (inference + finetune contributions)
  4. TTFT SLO satisfaction rate vs target
  5. E2E latency percentile curve with p99 highlighted

Compatible with the apples-to-apples DeltaServe-vLLM plots in
/tmp/dsv-recon/DeltaServe-vLLM/eval/plots/.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_TIMELINE_DIR = _REPO / "eval" / "llama3" / "timelines"
_OUT = _HERE / "output"
_PLOTS = _HERE / "plots"
_PLOTS.mkdir(exist_ok=True)


def detect_gpu_subdir() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0,
        )
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name: return "A100"
        if "5090" in name: return "5090"
        if "H200" in name or "H100" in name: return "A100"
    except Exception:
        pass
    return "A100"


def load_results(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    out = {
        "rid": [r["rid"] for r in rows],
        "sent_t":  np.array([float(r["sent_t"] or "nan") for r in rows]),
        "ttft_s":  np.array([float(r["ttft_s"] or "nan") for r in rows]),
        "latency_s": np.array([float(r["latency_s"] or "nan") for r in rows]),
        "is_ft":   np.array([int(r["is_ft"] or "0") for r in rows], dtype=bool),
        "tokens":  np.array([int(r["completion_tokens"] or "0") for r in rows]),
        "n": n,
    }
    out["ok"] = ~np.isnan(out["latency_s"])
    out["t_rel_s"] = out["sent_t"]
    return out


def load_timeline(path: Path) -> dict:
    rows = list(csv.DictReader(open(path)))
    return {
        "t_rel_s": np.array([float(r["timestamp_s"]) - 5.0 for r in rows]),  # subtract anchor offset
        "max_new_tokens": np.array([int(r["max_new_tokens"]) for r in rows]),
    }


def plot_request_timeline(ax, tl):
    t = tl["t_rel_s"]
    tok = tl["max_new_tokens"]
    if len(t) == 0:
        ax.set_title("Request Timeline (empty)"); return
    bucket = np.floor(t).astype(int)
    bucket = np.clip(bucket, 0, None)
    n = int(bucket.max()) + 1
    req = np.bincount(bucket, minlength=n).astype(float)
    tps = np.bincount(bucket, weights=tok, minlength=n).astype(float)
    centers = np.arange(n, dtype=float)
    ax.bar(centers, req, width=1.0, color="tab:gray", alpha=0.55,
           align="edge", label="req/s", edgecolor="none")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Requests / s", color="tab:gray")
    ax.tick_params(axis="y", labelcolor="tab:gray")
    ax.set_xlim(0, n); ax.set_ylim(bottom=0)
    ax.set_title("Scheduled Request Timeline")
    ax2 = ax.twinx()
    ax2.plot(centers + 0.5, tps, color="tab:green", marker="o",
             markersize=3, linewidth=1.5, label="tokens/s")
    ax2.set_ylabel("Output tokens / s", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green"); ax2.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def plot_latency_vs_time(ax, res, label, color):
    ok = res["ok"]; t = res["t_rel_s"][ok]; lat = res["latency_s"][ok]
    avg = float(np.nanmean(lat)) if lat.size else float("nan")
    series_label = f"{label} (avg {avg:.3f}s)" if np.isfinite(avg) else label
    ax.scatter(t, lat, s=10, color=color, alpha=0.7, label=series_label)
    if np.isfinite(avg):
        ax.axhline(avg, color=color, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Request E2E Latency vs Time")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Latency (s)")
    ax.legend()


def plot_throughput(ax, res, faux_stats: Optional[dict], label, color):
    """Plot inference tokens/s bars; overlay FT contribution band."""
    ok = res["ok"]; t = res["t_rel_s"][ok]; tok = res["tokens"][ok]
    is_ft = res["is_ft"][ok]
    if t.size == 0:
        ax.set_title("Throughput (empty)"); return
    bins = np.floor(t).astype(int); bins = np.clip(bins, 0, None)
    n = int(bins.max()) + 1
    inf_t = np.bincount(bins[~is_ft[:bins.size]], weights=tok[~is_ft[:bins.size]], minlength=n).astype(float)
    ft_t  = np.bincount(bins[is_ft[:bins.size]], weights=tok[is_ft[:bins.size]], minlength=n).astype(float)
    centers = np.arange(n, dtype=float) + 0.5
    ax.bar(centers, inf_t, width=1.0, color=color, alpha=0.55, label="inference tok/s")
    ax.bar(centers, ft_t, width=1.0, bottom=inf_t, color="tab:red", alpha=0.6, label="FT tok/s")
    if faux_stats:
        ax.text(0.02, 0.95,
                f"backward: {faux_stats.get('backward_calls', 0)} fires, "
                f"{faux_stats.get('backward_total_ms', 0):.0f}ms total\n"
                f"cuda_graph={faux_stats.get('cuda_graph', False)}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Output tokens / s")
    ax.set_title("Throughput (inference + finetune)")
    ax.legend(loc="upper right", fontsize=8)


def plot_ttft_slo(ax, res, label, color, slo_s=0.05):
    """Rolling-window TTFT SLO satisfaction rate."""
    ok = res["ok"]; t = res["t_rel_s"][ok]; ttft = res["ttft_s"][ok]
    sat = (ttft <= slo_s).astype(float)
    if sat.size == 0:
        ax.set_title("TTFT SLO satisfaction (empty)"); return
    # Sort by time, rolling mean.
    order = np.argsort(t); ts = t[order]; ss = sat[order]
    win = max(10, len(ss) // 10)
    rolling = np.array([np.mean(ss[max(0, i-win):i+1]) for i in range(len(ss))]) * 100
    ax.plot(ts, rolling, color=color, lw=1.8, label=f"{label} (overall {sat.mean()*100:.1f}%)")
    ax.axhline(95, color="0.5", linestyle="--", linewidth=1, label="95% target")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(f"Rolling sat-rate vs TTFT≤{slo_s*1000:.0f}ms (%)")
    ax.set_ylim(0, 105); ax.set_title("TTFT SLO Satisfaction")
    ax.legend(fontsize=8)


def plot_latency_percentile(ax, res, label, color):
    ok = res["ok"]; lat = res["latency_s"][ok]; lat = lat[np.isfinite(lat)]
    if lat.size == 0:
        ax.set_title("Latency percentile (empty)"); return
    sl = np.sort(lat)
    pct = np.arange(1, sl.size+1, dtype=float) / sl.size * 100
    p50 = float(np.percentile(lat, 50))
    p95 = float(np.percentile(lat, 95))
    p99 = float(np.percentile(lat, 99))
    ax.plot(pct, sl, color=color, linewidth=1.8,
            label=f"{label} (p50 {p50:.2f}s · p95 {p95:.2f}s · p99 {p99:.2f}s)")
    if not getattr(ax, "_guides", False):
        for p in (50, 95): ax.axvline(p, color="0.78", linestyle=":", linewidth=0.7)
        ax.axvline(99, color="0.35", linestyle="--", linewidth=1.4, label="p99")
        ax._guides = True
    ax.hlines(p99, 0, 99, colors=color, linestyles="--", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Percentile"); ax.set_ylabel("Latency (s)")
    ax.set_title("E2E Latency Percentile")
    ax.legend(fontsize=8)


def render_5panel(co_path: Path, inf_path: Path, timeline_path: Path,
                  faux_stats: Optional[dict], out_path: Path, title: str):
    co_res = load_results(co_path)
    inf_res = load_results(inf_path)
    tl = load_timeline(timeline_path)

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    # Row 1: 4 panels
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    # Row 2: 1 wide panel
    ax5 = fig.add_subplot(gs[1, :])

    plot_request_timeline(ax1, tl)
    plot_latency_vs_time(ax2, inf_res, "inf-only", "tab:blue")
    plot_latency_vs_time(ax2, co_res, "co-serving", "tab:red")
    plot_throughput(ax3, co_res, faux_stats, "co-serving", "tab:orange")
    plot_ttft_slo(ax4, inf_res, "inf-only", "tab:blue", slo_s=0.05)
    plot_ttft_slo(ax4, co_res, "co-serving", "tab:red", slo_s=0.05)
    plot_latency_percentile(ax5, inf_res, "inf-only", "tab:blue")
    plot_latency_percentile(ax5, co_res, "co-serving", "tab:red")

    fig.suptitle(title, fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="tight", choices=["tight", "loose"])
    ap.add_argument("--co-csv", help="Path to co-serving timeline_results CSV (default: auto)")
    ap.add_argument("--inf-csv", help="Path to inf-only timeline_results CSV (default: auto)")
    ap.add_argument("--label", default="sglang DeltaServe", help="Plot title prefix")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    gpu = args.gpu or detect_gpu_subdir()
    tl_path = _TIMELINE_DIR / gpu / f"timeline_{args.shape}.csv"

    co_csv = Path(args.co_csv) if args.co_csv else (_OUT / f"timeline_results_{args.shape}_co25_precap.csv")
    if not co_csv.exists():
        co_csv = _OUT / f"timeline_results_{args.shape}_co10.csv"  # fallback
    inf_csv = Path(args.inf_csv) if args.inf_csv else (_OUT / f"timeline_results_{args.shape}_inf.csv")

    # Read faux stats (last run's, if present)
    faux_stats = None
    stats_p = Path("/tmp/sglang_ds_gates/faux_stats.json")
    if stats_p.exists():
        import json
        try: faux_stats = json.load(open(stats_p))
        except Exception: pass

    out_path = _PLOTS / f"{args.shape}_co_5panel.png"
    title = f"{args.label} — {args.shape} timeline  (co-serve vs inf-only, Llama-3.2-1B / H200)"
    render_5panel(co_csv, inf_csv, tl_path, faux_stats, out_path, title)


if __name__ == "__main__":
    main()
