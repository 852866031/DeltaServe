"""
auto_benchmark.py — Drive inference traffic to the Mixtral server and
observe PrefillExecutionEstimator behavior under different length distributions.

Traffic phases (total ~600 requests, enough to trigger ~2 refits at 256 each):
    Phase 1 (200 req): UNIFORM  — all requests 256 tokens
    Phase 2 (200 req): SKEWED   — mostly short (32 tok) + one long (800+ tok) per batch
    Phase 3 (200 req): BIMODAL  — mix of 64-tok and 448-tok requests

After each batch, the server prints predictor fit_rmse to its log.
This script collects per-request TTFT and latency so you can see whether
the predictor's over-conservative behavior causes request queuing.

Usage:
    cd S-LoRA
    # Server must already be running:
    #   python test/mixtral/launch_mixtral.py
    python test/mixtral/auto_benchmark.py [--port 8000] [--bs 4]

Results written to test/mixtral/benchmark_results.csv
"""

import argparse
import csv
import os
import random
import string
import time
import requests
import threading

BASE_URL   = "http://127.0.0.1:{port}/generate"
OUT_CSV    = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
VOCAB_SIZE = 32000  # Mixtral


def random_ids(n: int) -> list:
    return [random.randint(1, VOCAB_SIZE - 1) for _ in range(n)]


def random_text(n_tokens_approx: int) -> str:
    # Rough approximation: ~1.3 tokens per word
    n_words = max(1, int(n_tokens_approx / 1.3))
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
             for _ in range(n_words)]
    return " ".join(words)


def send_request(url, prompt, max_new_tokens=1):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=120)
        ttft = time.perf_counter() - t0
        ok = r.status_code == 200
    except Exception as e:
        ttft = time.perf_counter() - t0
        ok = False
    return ttft, ok


def run_phase(url, phase_name, lengths, n_requests, results, bs=4, max_new=1):
    """Send n_requests, each with prompt length drawn from `lengths` list (round-robin)."""
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}  ({n_requests} requests, bs={bs})")
    print(f"Lengths: {lengths}")
    print(f"{'='*60}")

    futures = []
    lock = threading.Lock()

    def worker(idx, length):
        prompt = random_text(length)
        ttft, ok = send_request(url, prompt, max_new_tokens=max_new)
        with lock:
            results.append({
                "phase": phase_name,
                "req_idx": idx,
                "prompt_len_approx": length,
                "ttft_s": round(ttft, 4),
                "ok": ok,
            })
            if (idx + 1) % 50 == 0:
                recent = [r for r in results if r["phase"] == phase_name]
                mean_ttft = sum(r["ttft_s"] for r in recent) / len(recent)
                print(f"  [{idx+1}/{n_requests}] mean_ttft={mean_ttft*1000:.0f}ms")

    threads = []
    for i in range(n_requests):
        length = lengths[i % len(lengths)]
        t = threading.Thread(target=worker, args=(i, length))
        threads.append(t)

    # Send in batches of `bs` concurrently, with a small gap between batches
    for i in range(0, len(threads), bs):
        batch = threads[i:i+bs]
        for t in batch:
            t.start()
        for t in batch:
            t.join()
        time.sleep(0.05)   # small gap so server can batch naturally


def wait_for_server(url, timeout=120):
    print(f"Waiting for server at {url} ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url.replace("/generate", "/health"), timeout=2)
            if r.status_code == 200:
                print("Server is up.")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("Server did not start in time.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",        type=int, default=8000)
    parser.add_argument("--bs",          type=int, default=4,   help="Concurrent requests per batch")
    parser.add_argument("--n_per_phase", type=int, default=200, help="Requests per phase")
    args = parser.parse_args()

    url = BASE_URL.format(port=args.port)
    if not wait_for_server(url):
        import sys; sys.exit(1)

    random.seed(42)
    results = []

    # Phase 1: UNIFORM — all 256 tokens
    # Calibration baseline; predictor is already fitted on ~50-tok profiling batches.
    # After 256 batches here, first refit happens on mostly uniform 256-tok data.
    run_phase(url, "uniform_256", lengths=[256], n_requests=args.n_per_phase,
              results=results, bs=args.bs)

    # Phase 2: SKEWED — one dominant long seq per batch
    # [32, 32, 32, 800] for bs=4: Σn²≫uniform, T_in≈896≈uniform.
    # This is the Exp A failure scenario. Predictor should over-predict → excessive queuing.
    # After another 256 batches, second refit fires with BOTH uniform and skewed data
    # → predictor learns α≈0 → error should drop.
    skewed = [32, 32, 32, 800]   # 4 requests, total ≈ 896 tokens
    run_phase(url, "skewed_32x3_800", lengths=skewed, n_requests=args.n_per_phase,
              results=results, bs=args.bs)

    # Phase 3: BIMODAL — mix of short and long
    bimodal = [64, 64, 448, 448]   # 4 requests, total = 1024 tokens
    run_phase(url, "bimodal_64_448", lengths=bimodal, n_requests=args.n_per_phase,
              results=results, bs=args.bs)

    # Save results
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "req_idx", "prompt_len_approx",
                                           "ttft_s", "ok"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {len(results)} rows → {OUT_CSV}")

    # Summary per phase
    print("\nSummary:")
    print(f"{'Phase':>25}  {'N':>5}  {'mean_ttft_ms':>14}  {'p90_ttft_ms':>12}  {'ok%':>6}")
    import statistics
    for phase in ["uniform_256", "skewed_32x3_800", "bimodal_64_448"]:
        rows = [r for r in results if r["phase"] == phase and r["ok"]]
        if not rows:
            continue
        ttfts = sorted(r["ttft_s"] for r in rows)
        p90  = ttfts[int(0.9 * len(ttfts))]
        mean = statistics.mean(ttfts)
        ok_pct = len(rows) / args.n_per_phase * 100
        print(f"  {phase:>25}  {len(rows):>5}  {mean*1000:>13.0f}ms  {p90*1000:>11.0f}ms  {ok_pct:>5.0f}%")

    print("\nNow watch the server log for predictor refit lines:")
    print("  [Router]: Error for prefill estimator: <rmse>")
    print("Compare RMSE before/after phase transitions to see the online adaptation.")
    print("\nAlso check batch_prediction_stats_*.csv after stopping the server (Ctrl+C).")
