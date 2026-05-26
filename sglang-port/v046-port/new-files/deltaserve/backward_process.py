"""DeltaServe backward subprocess — child entry point and parent-side spawner.

Phase 6 scope: subprocess lifecycle + IPC round-trip only. The child binds a
ZMQ PAIR socket; the parent (engine / test) connects to it. Messages are
plain pickled dicts: ``{"op": "step", "acts": {...}}`` → ``{"key": tensor, ...}``.

The parent-side helper ``spawn_backward_process`` sets
``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` in the child's env dict only — the
parent's own environment is untouched so the inference scheduler keeps the
full GPU partition.
"""

import os
import pickle
import subprocess
import sys
from typing import Optional

import zmq

_MPS_PERCENTAGE_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def _build_service(model_name: str, mps_pct: int):
    # Phase 6: only the Llama3 stub is wired up.
    from sglang.srt.deltaserve.bwd_services.llama3 import Llama3BackwardService

    return Llama3BackwardService(
        model_name=model_name, device="cpu", mps_pct=mps_pct
    )


def main(channel_addr: str, model_name: str, mps_pct):
    mps_pct = int(mps_pct)

    # Bind the IPC socket *before* the heavy sglang import so the parent's
    # connect succeeds immediately. The first recv blocks until the import
    # finishes (which can take >10s); after that round-trip latency is
    # dominated by the actual service.step() call.
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PAIR)
    sock.bind(channel_addr)

    service = _build_service(model_name, mps_pct)
    service.start()

    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.ContextTerminated:
                break
            msg = pickle.loads(raw)
            op = msg.get("op")
            if op == "shutdown":
                sock.send(pickle.dumps({"ok": True}))
                break
            if op == "step":
                grads = service.step(msg.get("acts", {}))
                sock.send(pickle.dumps(grads))
                continue
            if op == "apply_grads":
                service.apply_grads(msg.get("grads", {}))
                sock.send(pickle.dumps({"ok": True}))
                continue
            sock.send(pickle.dumps({"error": f"unknown op: {op!r}"}))
    finally:
        service.stop()
        sock.close(linger=0)
        ctx.term()


def spawn_backward_process(
    channel_addr: str,
    model_name: str,
    mps_pct: int,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """Spawn the backward subprocess with the child-only MPS env applied.

    The parent's ``os.environ`` is *not* mutated — only the child's env dict.
    """
    child_env = dict(os.environ if env is None else env)
    child_env[_MPS_PERCENTAGE_ENV] = str(int(mps_pct))
    return subprocess.Popen(
        [sys.executable, __file__, channel_addr, model_name, str(int(mps_pct))],
        env=child_env,
    )


if __name__ == "__main__":
    # When invoked as a script (subprocess.Popen([python, __file__, ...]))
    # the in-tree sglang package is not on sys.path (a system-installed
    # sglang may shadow it). Walk up three levels from this file
    # (deltaserve/ -> srt/ -> sglang/) to land on the package root that
    # *contains* sglang/, and prepend it so the bwd_services import below
    # resolves to the in-tree code.
    _pkg_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    # argv: [script, channel_addr, model_name, mps_pct]
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
