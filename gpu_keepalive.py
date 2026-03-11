#!/usr/bin/env python3
"""
gpu_keepalive.py — Keeps GPU utilization high with continuous matrix multiplications.

Run while the inference server is not active to prevent the job from being killed.
Stop with Ctrl+C.

Usage:
  python gpu_keepalive.py
  python gpu_keepalive.py --size 4096 --device 0
"""

import argparse
import signal
import sys
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Keep GPU busy with matmuls")
    ap.add_argument("--size", type=int, default=8192, help="Matrix dimension NxN (default: 8192)")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--report_interval", type=int, default=50, help="Print stats every N iters")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.device}")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"[keepalive] Device : {torch.cuda.get_device_name(device)}")
    print(f"[keepalive] Matrix : {args.size}x{args.size} {args.dtype}")
    print(f"[keepalive] Press Ctrl+C to stop.\n")

    A = torch.randn(args.size, args.size, dtype=dtype, device=device)
    B = torch.randn(args.size, args.size, dtype=dtype, device=device)

    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False
        print("\n[keepalive] Stopping...", flush=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    i = 0
    t0 = time.monotonic()
    while running:
        _ = torch.mm(A, B)
        torch.cuda.synchronize(device)
        i += 1
        if i % args.report_interval == 0:
            elapsed = time.monotonic() - t0
            tflops = (2 * args.size ** 3 * args.report_interval) / elapsed / 1e12
            mem_gb = torch.cuda.memory_allocated(device) / 1e9
            print(
                f"[keepalive] iter={i:>8}  {tflops:5.1f} TFLOPS  mem={mem_gb:.2f} GB",
                flush=True,
            )
            t0 = time.monotonic()

    print("[keepalive] Done.", flush=True)


if __name__ == "__main__":
    main()
