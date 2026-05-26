#!/usr/bin/env python3
"""auto_benchmark_sglang.py — DeltaServe-on-sglang timeline benchmark.

Adapted from DeltaServe-vLLM's eval/auto_benchmark.py:
- Launches `python -m sglang.launch_server` with --enable-finetuning when --co
  is set; otherwise plain inference.
- Drives requests off the same timeline CSV format as the vLLM harness
  (timestamp_s, prompt_length, max_new_tokens).
- Each request is a streaming POST /generate (sglang's native SSE) — TTFT is
  the time to the first `data:` chunk; latency is the time to the chunk
  with finish_reason != null.
- A configurable fraction of timeline rows are sent with is_finetuning=True
  to exercise the FT dispatch path.
- Writes per-request metrics to output/timeline_results<suffix>.csv with the
  same columns as the vLLM harness.

Usage:
    python auto_benchmark_sglang.py --co --tight --ft-fraction 0.1
    python auto_benchmark_sglang.py --tight                  # inference-only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiohttp

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_TIMELINE_DIR = _REPO / "eval" / "llama3" / "timelines"
OUTPUT_DIR = _HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def detect_gpu_subdir() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0,
        )
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name:
            return "A100"
        if "5090" in name:
            return "5090"
        if "H200" in name or "H100" in name:
            return "A100"  # closest baseline; treat H-series like A100
    except Exception:
        pass
    return "A100"


@dataclass
class TimelineRow:
    timestamp_s: float
    prompt_length: int
    max_new_tokens: int


def load_timeline(path: Path) -> List[TimelineRow]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(TimelineRow(
                timestamp_s=float(r["timestamp_s"]),
                prompt_length=int(r["prompt_length"]),
                max_new_tokens=int(r["max_new_tokens"]),
            ))
    rows.sort(key=lambda r: r.timestamp_s)
    return rows


def build_server_cmd(model_path: str, port: int, co: bool, mps_pct: int) -> List[str]:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp-size", "1",
        "--mem-fraction-static", "0.5",
        "--disable-cuda-graph",
    ]
    if co:
        cmd += ["--enable-finetuning", "--backward-mps-percentage", str(mps_pct)]
    return cmd


def wait_for_health(port: int, timeout_s: float = 180) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionResetError, OSError):
            pass
        time.sleep(1)
    return False


# Reusable prompt token used to hit a given target prompt_length (close enough
# for benchmarking — sglang's tokenizer turns " hello" into ~1 token).
_PROMPT_FILLER = " hello" * 2048


def make_prompt(prompt_length: int) -> str:
    # Each " hello" is ~1 token after BPE; chain enough to exceed target.
    return _PROMPT_FILLER[:prompt_length * 6][: prompt_length * 6]


@dataclass
class RequestResult:
    rid: str
    sent_t: float
    ttft_s: Optional[float]
    latency_s: Optional[float]
    chunks: int
    avg_tbt_s: Optional[float]
    worst_tbt_s: Optional[float]
    is_ft: bool
    prompt_length: int
    max_new_tokens: int
    completion_tokens: int
    error: Optional[str]


async def stream_one(session: aiohttp.ClientSession, port: int,
                     row: TimelineRow, is_ft: bool, sent_t: float) -> RequestResult:
    prompt = make_prompt(row.prompt_length)
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": row.max_new_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        "stream": True,
        "is_finetuning": is_ft,
    }
    t0 = time.monotonic()
    ttft = None
    tbts: List[float] = []
    last_chunk_t = None
    chunks = 0
    completion_tokens = 0
    rid = ""
    err = None
    try:
        async with session.post(
            f"http://127.0.0.1:{port}/generate", json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            async for line in resp.content:
                if not line:
                    continue
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]" or not data:
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                if last_chunk_t is not None:
                    tbts.append(now - last_chunk_t)
                last_chunk_t = now
                chunks += 1
                meta = obj.get("meta_info") or {}
                rid = meta.get("id") or rid
                completion_tokens = meta.get("completion_tokens", completion_tokens)
                if meta.get("finish_reason") is not None:
                    break
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    latency = (time.monotonic() - t0) if ttft is not None else None
    return RequestResult(
        rid=rid, sent_t=sent_t, ttft_s=ttft, latency_s=latency, chunks=chunks,
        avg_tbt_s=(sum(tbts)/len(tbts) if tbts else None),
        worst_tbt_s=(max(tbts) if tbts else None),
        is_ft=is_ft, prompt_length=row.prompt_length,
        max_new_tokens=row.max_new_tokens, completion_tokens=completion_tokens,
        error=err,
    )


async def drive(port: int, timeline: List[TimelineRow], ft_fraction: float,
                t_anchor: float) -> List[RequestResult]:
    results: List[RequestResult] = []
    pending: List[asyncio.Task] = []
    sent_count = 0
    async with aiohttp.ClientSession() as session:
        for i, row in enumerate(timeline):
            target_t = t_anchor + row.timestamp_s
            now = time.monotonic()
            if target_t > now:
                await asyncio.sleep(target_t - now)
            is_ft = (i % max(1, int(round(1/ft_fraction))) == 0) if ft_fraction > 0 else False
            sent_count += 1
            task = asyncio.create_task(
                stream_one(session, port, row, is_ft, time.monotonic() - t_anchor)
            )
            pending.append(task)
        results = await asyncio.gather(*pending)
    return results


def write_csv(path: Path, results: List[RequestResult]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rid", "sent_t", "ttft_s", "latency_s", "chunks",
            "avg_tbt_s", "worst_tbt_s", "is_ft", "prompt_length",
            "max_new_tokens", "completion_tokens", "error",
        ])
        for r in results:
            w.writerow([
                r.rid, f"{r.sent_t:.4f}",
                f"{r.ttft_s:.4f}" if r.ttft_s is not None else "",
                f"{r.latency_s:.4f}" if r.latency_s is not None else "",
                r.chunks,
                f"{r.avg_tbt_s:.4f}" if r.avg_tbt_s is not None else "",
                f"{r.worst_tbt_s:.4f}" if r.worst_tbt_s is not None else "",
                int(r.is_ft), r.prompt_length, r.max_new_tokens,
                r.completion_tokens, r.error or "",
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/weka/home/jianshu.she/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
                    help="HF model path or repo id")
    ap.add_argument("--port", type=int, default=30200)
    ap.add_argument("--co", action="store_true",
                    help="Enable --enable-finetuning + send is_finetuning=true on a fraction of reqs.")
    ap.add_argument("--ft-fraction", type=float, default=0.1,
                    help="Fraction of timeline rows to mark as FT (only meaningful with --co).")
    ap.add_argument("--mps-pct", type=int, default=20)
    sg = ap.add_mutually_exclusive_group()
    sg.add_argument("--tight", action="store_true")
    sg.add_argument("--loose", action="store_true")
    ap.add_argument("--timeline-gpu", default=None, help="Override timelines/<gpu>/ subdir (default: auto-detect).")
    ap.add_argument("--launch-server", action="store_true", default=True,
                    help="Launch the server (default). Pass --no-launch-server to use an already-running server.")
    ap.add_argument("--no-launch-server", dest="launch_server", action="store_false")
    args = ap.parse_args()

    gpu = args.timeline_gpu or detect_gpu_subdir()
    shape = "tight" if args.tight else ("loose" if args.loose else "tight")
    tl_path = _TIMELINE_DIR / gpu / f"timeline_{shape}.csv"
    if not tl_path.exists():
        print(f"timeline missing: {tl_path}", file=sys.stderr)
        sys.exit(2)
    timeline = load_timeline(tl_path)
    print(f"[bench] gpu={gpu} shape={shape} rows={len(timeline)} co={args.co} ft_frac={args.ft_fraction}")

    server_proc = None
    log_path = None
    if args.launch_server:
        log_path = OUTPUT_DIR / f"server_{shape}_{'co' if args.co else 'inf'}.log"
        cmd = build_server_cmd(args.model, args.port, args.co, args.mps_pct)
        print(f"[bench] launching: {' '.join(cmd)}")
        server_proc = subprocess.Popen(
            cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
            preexec_fn=os.setsid,
        )
        print(f"[bench] server pid={server_proc.pid} log={log_path}")
        if not wait_for_health(args.port):
            print("[bench] server failed to come up; tail log:", file=sys.stderr)
            try:
                print("".join(open(log_path).readlines()[-40:]), file=sys.stderr)
            except Exception:
                pass
            os.killpg(server_proc.pid, signal.SIGTERM)
            sys.exit(3)
        print(f"[bench] server healthy")

    # Drop a couple of warmup requests so the first timeline row doesn't see cold start.
    print(f"[bench] warmup...")
    asyncio.run(drive(args.port, timeline[:3], ft_fraction=0.0, t_anchor=time.monotonic()))

    print(f"[bench] running timeline...")
    t_anchor = time.monotonic()
    results = asyncio.run(drive(args.port, timeline, ft_fraction=(args.ft_fraction if args.co else 0.0), t_anchor=t_anchor))

    suffix = f"_{shape}{'_co' if args.co else '_inf'}"
    out_csv = OUTPUT_DIR / f"timeline_results{suffix}.csv"
    write_csv(out_csv, results)
    print(f"[bench] wrote {out_csv}")

    n_ok = sum(1 for r in results if r.error is None)
    n_err = len(results) - n_ok
    ft_count = sum(1 for r in results if r.is_ft)
    ttfts = [r.ttft_s for r in results if r.ttft_s is not None]
    lats = [r.latency_s for r in results if r.latency_s is not None]
    if ttfts:
        ttfts.sort()
        print(f"[bench] ttft_s: mean={sum(ttfts)/len(ttfts):.3f}  p50={ttfts[len(ttfts)//2]:.3f}  p95={ttfts[int(len(ttfts)*0.95)]:.3f}")
    if lats:
        lats.sort()
        print(f"[bench] latency_s: mean={sum(lats)/len(lats):.3f}  p50={lats[len(lats)//2]:.3f}  p95={lats[int(len(lats)*0.95)]:.3f}")
    print(f"[bench] reqs ok={n_ok} err={n_err}; ft_tagged={ft_count}")

    if server_proc is not None:
        os.killpg(server_proc.pid, signal.SIGTERM)
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(server_proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
