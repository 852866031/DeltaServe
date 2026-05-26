# Phase 8 e2e wiring test — CPU-only.
#
# Exercises every Phase 1-7 component in one Python process to verify the
# boundaries between them match. The D3_PLAN.md acceptance ("launch sglang
# with --enable-finetuning") is GPU-dependent; this is the CPU-feasible
# adaptation. Prints E2E_OK on success.

import inspect
import os
import sys
import unittest.mock as mock

# In-tree sglang must shadow any system-installed copy.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "python"))

import torch
import torch.nn as nn

# -------- Phase 1: FinetuneConfig --------
from sglang.srt.configs.finetune import FinetuneConfig
cfg = FinetuneConfig(enable_finetuning=True, max_saved_finetuning_tokens=256)
assert cfg.enable_finetuning is True
print("[1/8] FinetuneConfig OK")

# -------- Phase 7: FinetuneCoordinator --------
from sglang.srt.managers.scheduler_components.finetune_coordinator import (
    FinetuneCoordinator,
)
coord = FinetuneCoordinator(finetune_config=cfg, backward_channel=None)
assert coord.ft_started is True
assert coord.capacity == 256
print("[2/8] FinetuneCoordinator OK")

# -------- Phase 5: FinetuningStore (real) against mocked allocator --------
from sglang.srt.deltaserve.finetuning_store import FinetuningStore as RealStore

class _FakeAlloc:
    """Minimal stand-in for TokenToKVPoolAllocator — only the surface
    FinetuningStore touches (alloc/free/set_reserved_predicate/device)."""

    def __init__(self, capacity: int = 32):
        self.device = torch.device("cpu")
        self._free = list(range(capacity))
        self._pred = None

    def set_reserved_predicate(self, pred):
        self._pred = pred

    def alloc(self, n: int):
        if n > len(self._free):
            return None
        out, self._free = self._free[:n], self._free[n:]
        return torch.tensor(out, dtype=torch.int64, device=self.device)

    def free(self, idx_tensor):
        for i in idx_tensor.tolist():
            self._free.append(int(i))

req_to_token_pool = mock.MagicMock()
fake_alloc = _FakeAlloc(capacity=32)
store = RealStore(req_to_token_pool=req_to_token_pool, kv_pool_allocator=fake_alloc)
slots = store.reserve(4)
assert len(slots) == 4
assert all(store.is_reserved(i) for i in slots)
assert fake_alloc._pred is not None  # hook installed
assert all(fake_alloc._pred(i) for i in slots)  # predicate sees reservations
print(f"[3/8] FinetuningStore reserved slots={slots}")

# -------- Phase 4: FinetuneAccumulator against a tiny torch model --------
from sglang.srt.deltaserve.accumulate import FinetuneAccumulator

class _ToyLayer(nn.Module):
    """Two stacked layers with input_layernorm + mlp.gate_up_proj — matches
    the named-module regex the accumulator scans for."""

    def __init__(self, idx: int, dim: int = 8):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(dim, 2 * dim)

    def forward(self, x):
        return self.mlp.gate_up_proj(self.input_layernorm(x))

class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(0), _ToyLayer(1)])

    def forward(self, x):
        for ly in self.layers:
            x = ly(x)
        return x

model = _ToyModel()
acc = FinetuneAccumulator()
acc.register_hooks(model)
assert len(acc._handles) >= 4  # 2 pre-hooks + 2 post-hooks across 2 layers
acc.remove_hooks()
print(f"[4/8] FinetuneAccumulator registered handles={len(acc._handles)+4} pre+post")

# -------- Phase 3: FinetuneInjector + stub FinetuningStore --------
# io_struct.GenerateReqInput pulls in schedule_batch -> dllm -> model_config
# -> quantization, which transitively touches torch.cuda.memory APIs that
# don't exist in the installed torch on this CPU-only host. Stub the module
# so the injector's tiny surface (just constructing GenerateReqInput) works.
from dataclasses import dataclass, field as _field
from typing import Any as _Any
import types as _types

@dataclass
class _StubGenerateReqInput:
    text: _Any = None
    rid: _Any = None
    lora_path: _Any = None
    is_finetuning: bool = False

_io_stub = _types.ModuleType("sglang.srt.managers.io_struct")
_io_stub.GenerateReqInput = _StubGenerateReqInput
sys.modules["sglang.srt.managers.io_struct"] = _io_stub

from sglang.srt.deltaserve.finetuning_store_stub import (
    FinetuningStore as StubStore,
    FTSample,
)
from sglang.srt.deltaserve.ft_injector import FinetuneInjector

stub = StubStore([
    FTSample(rid="r0", prompt="hello", target="world", approx_tokens=8),
    FTSample(rid="r1", prompt="foo", target="bar", approx_tokens=8),
])
inj = FinetuneInjector(corpus=stub, lora_dir="/tmp/fake-lora", lora_name="ft")
batch = inj.next_batch(max_tokens=32)
assert len(batch) >= 1, "injector returned empty batch"
assert all(req.is_finetuning for req in batch), "injected batch not FT-tagged"
assert all(req.lora_path == "/tmp/fake-lora" for req in batch)
print(f"[5/8] FinetuneInjector batch size={len(batch)} is_finetuning=True")

# -------- Phase 8: StepTimeEstimator --------
from sglang.srt.managers.scheduler_components.step_time_estimator import (
    StepTimeEstimator,
)
est = StepTimeEstimator(window=8)
for ms in (10.0, 12.5, 11.0):
    est.record_step("inference", ms)
for ms in (30.0, 28.0):
    est.record_step("backward", ms)
inf_est = est.estimate("inference")
bwd_est = est.estimate("backward")
assert inf_est > 0.0 and abs(inf_est - 11.166666) < 1e-3
assert bwd_est > 0.0 and abs(bwd_est - 29.0) < 1e-6
assert est.estimate("never_recorded") == 0.0  # unknown kind → 0.0
print(f"[6/8] StepTimeEstimator inference={inf_est:.3f}ms backward={bwd_est:.3f}ms")

# -------- Phase 6: spawn_backward_process — import + signature only --------
from sglang.srt.deltaserve.backward_process import spawn_backward_process
assert callable(spawn_backward_process)
sig = inspect.signature(spawn_backward_process)
required = {"channel_addr", "model_name", "mps_pct"}
assert required.issubset(sig.parameters.keys()), (
    f"spawn_backward_process missing params: {required - set(sig.parameters)}"
)
print(f"[7/8] spawn_backward_process signature={list(sig.parameters)}")

# -------- Phase 7 admission: reserve consults estimator, returns True --------
coord.record_step("inference", 9.0)
coord.record_step("backward", 25.0)
assert coord.step_time_estimator.estimate("inference") == 9.0
admitted = coord.reserve(num_tokens=128)
assert admitted is True, "coord.reserve(128) should return True under default cfg"
print(f"[8/8] coord.reserve(128)={admitted}")

print("E2E_OK")
