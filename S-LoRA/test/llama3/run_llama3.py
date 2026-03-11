#!/usr/bin/env python3
"""
run_llama3.py — All-in-one: launch server, warm up, co-serve, infer, shut down.

Usage:
  python run_llama3.py                        # co-serving (default)
  python run_llama3.py --no-finetuning        # inference only
  python run_llama3.py --num_requests 20 --max_new_tokens 100 --port 9001
"""

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional, Tuple

import aiohttp

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
LORA_DIR = os.path.join(SCRIPT_DIR, "adapters", "llama3-toy-lora")
FT_LORA_DIR = os.path.join(SCRIPT_DIR, "adapters", "llama3-toy-lora-ft")
FINETUNING_CONFIG = os.path.join(SCRIPT_DIR, "config", "finetuning_config.json")
NO_FINETUNING_CONFIG = os.path.join(SCRIPT_DIR, "config", "no_finetuning_config.json")
FT_LOG_PATH = os.path.join(SCRIPT_DIR, "bwd_log.csv")

WARMUP_PROMPTS = [
    "Instruction:\nWhat is 2 + 2?\n### Response: ",
    "Instruction:\nName the capital of France.\n### Response: ",
    "Instruction:\nDescribe gravity in one sentence.\n### Response: ",
]

INFERENCE_PROMPTS = [
    "Instruction:\nExplain the difference between supervised and unsupervised learning.\n### Response: ",
    "Instruction:\nWrite a haiku about neural networks.\n### Response: ",
    "Instruction:\nWhat is the capital of Japan?\n### Response: ",
    "Instruction:\nDescribe the water cycle briefly.\n### Response: ",
    "Instruction:\nWhat are the primary colors?\n### Response: ",
    "Instruction:\nExplain what a transformer model is.\n### Response: ",
    "Instruction:\nWhat is the Pythagorean theorem?\n### Response: ",
    "Instruction:\nName three famous scientists.\n### Response: ",
    "Instruction:\nDescribe how photosynthesis works.\n### Response: ",
    "Instruction:\nWhat does CPU stand for?\n### Response: ",
]


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def check_adapters() -> None:
    missing = []
    if not os.path.isdir(LORA_DIR):
        missing.append(LORA_DIR)
    if not os.path.isdir(FT_LORA_DIR):
        missing.append(FT_LORA_DIR)
    if missing:
        print("[run_llama3] ERROR: Missing adapter directories:")
        for p in missing:
            print(f"  {p}")
        print("[run_llama3] Run: python adapter_train.py && cp -r adapters/llama3-toy-lora adapters/llama3-toy-lora-ft")
        sys.exit(1)


def check_mps() -> None:
    exe = shutil.which("nvidia-cuda-mps-control")
    if not exe:
        print("[run_llama3] WARNING: nvidia-cuda-mps-control not found; MPS may not be available.")
        return
    try:
        p = subprocess.Popen(
            [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, _ = p.communicate("get_server_list\nquit\n", timeout=2.0)
        if p.returncode != 0:
            print("[run_llama3] WARNING: MPS daemon does not appear to be running.")
            print("[run_llama3]   Start it with: sudo nvidia-cuda-mps-control -d")
    except Exception:
        print("[run_llama3] WARNING: Could not query MPS daemon status.")


# ---------------------------------------------------------------------------
# Server process helpers
# ---------------------------------------------------------------------------
def build_server_cmd(port: int, enable_finetuning: bool) -> list:
    config_path = FINETUNING_CONFIG if enable_finetuning else NO_FINETUNING_CONFIG
    cmd = [
        sys.executable, "-m", "slora.server.api_server",
        "--max_total_token_num", "25000",
        "--model", BASE_MODEL,
        "--tokenizer_mode", "auto",
        "--pool-size-lora", "0",
        "--port", str(port),
        "--ft_log_path", FT_LOG_PATH,
        "--finetuning_config_path", config_path,
        "--lora", LORA_DIR,
        "--swap",
        "--enable_unified_mem_manager",
        "--unified_mem_manager_max_size", "6",
        "--scheduler", "slora_plus",
    ]
    return cmd


def terminate_process_tree(p: subprocess.Popen, grace_s: float = 3.0) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass

    t0 = time.monotonic()
    while time.monotonic() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.05)

    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Server readiness
# ---------------------------------------------------------------------------
async def wait_for_server(server: str, max_wait_s: float = 240.0, poll_s: float = 0.5) -> None:
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(f"{server}/health", timeout=aiohttp.ClientTimeout(total=1.0)) as r:
                    if r.status == 200:
                        print(f"[run_llama3] Server is up at {server}", flush=True)
                        return
            except Exception:
                pass
            # Also probe /generate
            try:
                payload = {
                    "model_dir": BASE_MODEL,
                    "lora_dir": LORA_DIR,
                    "inputs": "ping",
                    "parameters": {"do_sample": False, "ignore_eos": True, "max_new_tokens": 2},
                }
                async with session.post(
                    f"{server}/generate", json=payload, timeout=aiohttp.ClientTimeout(total=2.0)
                ) as r:
                    if r.status == 200:
                        print(f"[run_llama3] Server is up at {server}", flush=True)
                        return
            except Exception:
                pass

            if time.monotonic() - t0 > max_wait_s:
                raise TimeoutError(f"Server did not become ready within {max_wait_s:.0f}s")
            await asyncio.sleep(poll_s)


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------
async def send_request(
    session: aiohttp.ClientSession,
    server: str,
    idx: int,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[int, str, float, Optional[float], Optional[float], str]:
    """Returns (idx, status, latency_s, ttft_s, avg_tbt_s, generated_text)."""
    url = f"{server}/generate"
    payload = {
        "model_dir": BASE_MODEL,
        "lora_dir": LORA_DIR,
        "inputs": prompt,
        "parameters": {"do_sample": False, "ignore_eos": False, "max_new_tokens": max_new_tokens},
    }
    t0 = time.monotonic()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300.0)) as resp:
            body = await resp.read()
            latency = time.monotonic() - t0
        data = json.loads(body)
        ttft = data.get("ttft")
        avg_tbt = data.get("avg_tbt")
        text = data.get("generated_text", ["<no text>"])
        text = text[0] if isinstance(text, list) else str(text)
        return (idx, "ok", latency, ttft, avg_tbt, text)
    except Exception as e:
        latency = time.monotonic() - t0
        return (idx, f"error:{type(e).__name__}", latency, None, None, str(e))


# ---------------------------------------------------------------------------
# Finetuning control
# ---------------------------------------------------------------------------
async def start_finetuning(server: str) -> bool:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{server}/start_finetuning", timeout=aiohttp.ClientTimeout(total=5.0)) as r:
                ok = r.status == 200
                print(f"[run_llama3] start_finetuning → HTTP {r.status}", flush=True)
                return ok
        except Exception as e:
            print(f"[run_llama3] start_finetuning error: {e}", flush=True)
            return False


async def exit_finetuning(server: str) -> bool:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{server}/exit_finetuning", timeout=aiohttp.ClientTimeout(total=5.0)) as r:
                ok = r.status == 200
                print(f"[run_llama3] exit_finetuning → HTTP {r.status}", flush=True)
                return ok
        except Exception as e:
            print(f"[run_llama3] exit_finetuning error: {e}", flush=True)
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> None:
    server = f"http://127.0.0.1:{args.port}"
    enable_ft = not args.no_finetuning

    # --- Preflight ---
    check_adapters()
    check_mps()

    # --- Launch server ---
    cmd = build_server_cmd(args.port, enable_finetuning=enable_ft)
    print(f"[run_llama3] Launching server: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    p = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=SCRIPT_DIR,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    async def pump_logs() -> None:
        assert p.stdout is not None
        while True:
            line = await loop.run_in_executor(None, p.stdout.readline)
            if not line:
                break
            print("[server]", line.rstrip(), flush=True)

    log_task = asyncio.create_task(pump_logs())

    def _on_sigint() -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    try:
        # --- Wait for server ---
        print("[run_llama3] Waiting for server to be ready...", flush=True)
        wait_task = asyncio.create_task(wait_for_server(server, max_wait_s=240.0))
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait({wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if stop_event.is_set():
            return
        # Re-raise any exception from wait_for_server
        if wait_task in done:
            wait_task.result()

        # --- Warmup ---
        print(f"[run_llama3] Warming up ({len(WARMUP_PROMPTS)} requests)...", flush=True)
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i, prompt in enumerate(WARMUP_PROMPTS):
                if stop_event.is_set():
                    return
                idx, status, latency, _, _, _ = await send_request(session, server, i, prompt, max_new_tokens=10)
                print(f"[warmup]  req={idx} status={status} latency={latency:.3f}s", flush=True)
        print("[run_llama3] Warmup complete.", flush=True)

        # --- Start finetuning ---
        if enable_ft:
            print("[run_llama3] Starting co-serving (finetuning)...", flush=True)
            ok = await start_finetuning(server)
            if not ok:
                print("[run_llama3] WARNING: Failed to start finetuning; continuing inference only.", flush=True)
            else:
                print("[run_llama3] Co-serving active.", flush=True)

        # --- Concurrent inference ---
        prompts = [INFERENCE_PROMPTS[i % len(INFERENCE_PROMPTS)] for i in range(args.num_requests)]
        print(f"[run_llama3] Sending {args.num_requests} inference requests concurrently...", flush=True)

        connector2 = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector2) as session:
            tasks = [
                asyncio.create_task(send_request(session, server, i, prompt, args.max_new_tokens))
                for i, prompt in enumerate(prompts)
            ]
            results = await asyncio.gather(*tasks)

        # Print results sorted by idx
        print("\n" + "=" * 70)
        print(f"{'IDX':>4}  {'STATUS':<12}  {'LATENCY':>9}  {'TTFT':>8}  {'AVG_TBT':>9}")
        print("-" * 70)
        for idx, status, latency, ttft, avg_tbt, text in sorted(results, key=lambda x: x[0]):
            ttft_s = f"{ttft:.4f}s" if ttft is not None else "   n/a  "
            avg_tbt_s = f"{avg_tbt:.4f}s" if avg_tbt is not None else "   n/a  "
            print(f"{idx:>4}  {status:<12}  {latency:>8.3f}s  {ttft_s:>8}  {avg_tbt_s:>9}")
            short_prompt = prompts[idx].split("\n")[1][:50]
            print(f"      prompt : {short_prompt}")
            print(f"      output : {text[:120]}")
            print()
        print("=" * 70 + "\n")

        # --- Stop finetuning ---
        if enable_ft:
            print("[run_llama3] Stopping finetuning...", flush=True)
            ok = await exit_finetuning(server)
            if ok:
                print("[run_llama3] Finetuning stopped.", flush=True)

        await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        print("[run_llama3] Shutting down server...", flush=True)
        terminate_process_tree(p, grace_s=5.0)
        log_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(log_task), timeout=1.0)
        except (Exception, asyncio.CancelledError):
            pass
        print("[run_llama3] Done.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch Llama 3 8B co-serving demo")
    ap.add_argument("--port", type=int, default=9000, help="Server port (default: 9000)")
    ap.add_argument("--max_new_tokens", type=int, default=50, help="Tokens to generate per request (default: 50)")
    ap.add_argument("--num_requests", type=int, default=10, help="Number of concurrent inference requests (default: 10)")
    ap.add_argument("--no-finetuning", action="store_true", help="Skip co-serving; run inference only")
    args = ap.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
