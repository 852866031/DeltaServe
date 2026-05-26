"""Phase 6 acceptance: BackwardProcess subprocess + IPC round-trip.

CPU-only smoke test. Spawns the backward subprocess, sends a pickled
``{"op": "step", "acts": {...}}`` over a ZMQ PAIR socket, verifies the
synthetic grads come back (~ act * 0.01), and that the child inherits
``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=15`` via its env.
"""

import os
import pickle
import socket
import sys
import time

# Make the in-tree sglang importable when running from a checkout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch  # noqa: E402
import zmq  # noqa: E402

from sglang.srt.deltaserve.backward_process import spawn_backward_process  # noqa: E402


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _read_child_env(pid: int) -> dict:
    with open(f"/proc/{pid}/environ", "rb") as f:
        raw = f.read()
    env = {}
    for entry in raw.split(b"\x00"):
        if not entry or b"=" not in entry:
            continue
        k, _, v = entry.partition(b"=")
        env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return env


def main():
    port = _pick_free_port()
    channel_addr = f"tcp://127.0.0.1:{port}"

    proc = spawn_backward_process(
        channel_addr=channel_addr,
        model_name="llama3-stub",
        mps_pct=15,
    )

    # Wait for the child to bind the TCP port. The child binds the ZMQ
    # socket *before* importing the (slow) sglang stack, so this becomes
    # available within a second; we still give it 60s for safety on cold
    # imports. This wait is subprocess boot time, not the 5s round-trip
    # budget for the step message.
    bind_deadline = time.time() + 60.0
    while time.time() < bind_deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(("127.0.0.1", port))
            probe.close()
            break
        except OSError:
            probe.close()
            time.sleep(0.1)
    else:
        raise RuntimeError("child never bound the IPC port")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PAIR)
    sock.setsockopt(zmq.LINGER, 0)
    # 30s receive window: the round-trip itself is sub-second once the
    # child has finished importing sglang (the slow part), so the first
    # recv mostly waits for that import to complete.
    sock.setsockopt(zmq.RCVTIMEO, 30000)
    try:
        sock.connect(channel_addr)

        x = torch.randn(4)
        msg = {"op": "step", "acts": {"x": x}}

        send_t = time.time()
        sock.send(pickle.dumps(msg))
        raw = sock.recv()
        reply = pickle.loads(raw)
        roundtrip_ms = (time.time() - send_t) * 1000
        print(f"step round-trip: {roundtrip_ms:.1f} ms")

        assert set(reply.keys()) == set(msg["acts"].keys()), (
            f"reply keys {set(reply.keys())} != acts keys {set(msg['acts'].keys())}"
        )
        expected = x * 0.01
        assert torch.allclose(reply["x"], expected, atol=1e-6), (
            f"grad mismatch: got {reply['x']}, expected {expected}"
        )

        # Verify the child process inherited the MPS env var.
        env = _read_child_env(proc.pid)
        assert env.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") == "15", (
            f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE in child env = "
            f"{env.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!r}, expected '15'"
        )

        # Ask the child to shut down cleanly.
        sock.send(pickle.dumps({"op": "shutdown"}))
        try:
            sock.recv()
        except zmq.Again:
            pass
    finally:
        sock.close(linger=0)
        ctx.term()
        try:
            proc.wait(timeout=2.0)
        except Exception:
            proc.kill()
            proc.wait(timeout=2.0)

    print("phase6 ok")


if __name__ == "__main__":
    main()
