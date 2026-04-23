"""Compare an eager backward run against an --enable-bwd-cuda-graph run.

Reads two timeline_results CSVs and two bwd_log CSVs produced by
orchestrate_run_timeline.py and replays the comparable plots from
auto_plot.py side-by-side.

Inputs (fixed paths, all under this directory):
  - timeline_results.csv           (baseline / eager backward)
  - timeline_results_bwd_graph.csv (--enable-bwd-cuda-graph)
  - bwd_log.csv                    (baseline)
  - bwd_log_bwd_graph.csv          (--enable-bwd-cuda-graph)

Output:
  - bwd_graph_comparison.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------- Fixed paths ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS_BASELINE = os.path.join(_HERE, "timeline_results.csv")
RESULTS_BWD_GRAPH = os.path.join(_HERE, "timeline_results_bwd_graph.csv")
BWD_LOG_BASELINE = os.path.join(_HERE, "bwd_log.csv")
BWD_LOG_BWD_GRAPH = os.path.join(_HERE, "bwd_log_bwd_graph.csv")
OUT_PATH = os.path.join(_HERE, "bwd_graph_comparison.png")

BASELINE_LABEL = "eager"
BWD_GRAPH_LABEL = "bwd graph"
BASELINE_COLOR = "tab:blue"
BWD_GRAPH_COLOR = "tab:orange"


# ---------------- Utilities ----------------
def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_results(csv_path: str) -> pd.DataFrame:
    """
    Expected format (from orchestrate_run_timeline.py):
      idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s
    """
    df = pd.read_csv(csv_path)
    required = ["idx", "t_rel_s", "latency_s", "status", "ttft_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    df["ok"] = df["status"].astype(str).str.strip().eq("ok")
    return df


def parse_bwd_log_csv(csv_path: str):
    """
    Parse backward log CSV:
      timestamp,epoch,batch_idx,batch_tokens,batch_loss,total_processed_tokens

    Returns:
      rel_time_s: np.ndarray
      cum_tokens: np.ndarray
      avg_tok_s: float
    """
    df = pd.read_csv(csv_path)
    required = ["timestamp", "batch_tokens"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"{csv_path} has no valid timestamp rows.")

    # Collapse rows that share a timestamp (ts resolution is only 1s in the
    # logger) down to the last row, which holds the latest cumulative value
    # for that moment. Without this the line can double back mid-second.
    df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    t0 = df["timestamp"].iloc[0]
    df["rel_time_s"] = (df["timestamp"] - t0).dt.total_seconds()

    if "total_processed_tokens" in df.columns:
        cum_tokens = df["total_processed_tokens"].astype(float).to_numpy()
    else:
        cum_tokens = df["batch_tokens"].astype(float).cumsum().to_numpy()

    rel_time_s = df["rel_time_s"].to_numpy()
    elapsed = float(rel_time_s[-1]) if len(rel_time_s) > 0 else 0.0
    total = float(cum_tokens[-1]) if len(cum_tokens) > 0 else 0.0
    avg_tok_s = total / elapsed if elapsed > 0 else float("nan")
    return rel_time_s, cum_tokens, avg_tok_s


# ---------------- Plotting ----------------
def plot_ttft_percentile(ax, base_results: pd.DataFrame, graph_results: pd.DataFrame):
    percentiles = np.array([0, 20, 40, 60, 80, 100])
    for df, label, color in (
        (base_results, BASELINE_LABEL, BASELINE_COLOR),
        (graph_results, BWD_GRAPH_LABEL, BWD_GRAPH_COLOR),
    ):
        ttft = df.loc[df["ok"], "ttft_s"].dropna().astype(float).to_numpy()
        if len(ttft) == 0:
            continue
        values = np.percentile(ttft, percentiles)
        ax.plot(percentiles, values, marker="o", color=color, label=label)
    ax.set_title("TTFT Percentile Curve")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("TTFT (s)")
    ax.legend()


def plot_latency_vs_time(ax, base_results: pd.DataFrame, graph_results: pd.DataFrame):
    for df, label, color in (
        (base_results, BASELINE_LABEL, BASELINE_COLOR),
        (graph_results, BWD_GRAPH_LABEL, BWD_GRAPH_COLOR),
    ):
        ok = df[df["ok"]]
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


def plot_bwd_cumulative(ax, base_log, graph_log):
    base_t, base_cum, base_avg = base_log
    graph_t, graph_cum, graph_avg = graph_log

    base_label = f"{BASELINE_LABEL} ({base_avg:.1f} tok/s)" if np.isfinite(base_avg) else BASELINE_LABEL
    graph_label = f"{BWD_GRAPH_LABEL} ({graph_avg:.1f} tok/s)" if np.isfinite(graph_avg) else BWD_GRAPH_LABEL

    ax.plot(base_t, base_cum, color=BASELINE_COLOR, label=base_label)
    ax.plot(graph_t, graph_cum, color=BWD_GRAPH_COLOR, label=graph_label)
    ax.set_title("Finetuning Cumulative Tokens")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Tokens")
    ax.legend()


# ---------------- Main ----------------
def main():
    for p in (RESULTS_BASELINE, RESULTS_BWD_GRAPH, BWD_LOG_BASELINE, BWD_LOG_BWD_GRAPH):
        ensure_exists(p)

    base_results = load_results(RESULTS_BASELINE)
    graph_results = load_results(RESULTS_BWD_GRAPH)
    base_log = parse_bwd_log_csv(BWD_LOG_BASELINE)
    graph_log = parse_bwd_log_csv(BWD_LOG_BWD_GRAPH)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_ttft_percentile(axes[0], base_results, graph_results)
    plot_latency_vs_time(axes[1], base_results, graph_results)
    plot_bwd_cumulative(axes[2], base_log, graph_log)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160)
    print(f"[bwd_graph_plot] Saved comparison figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
