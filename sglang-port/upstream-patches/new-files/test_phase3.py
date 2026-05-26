"""Phase 3 acceptance test.

This dev box has torch 2.6.0 and missing optional kernel deps; importing
sglang normally fails on transitive CUDA/quantization imports. None of those
matter for the Phase 3 dataclass change. We bypass sglang's package __init__
chain by pre-registering empty packages (with __path__) and stubbing the
transitive modules io_struct depends on. The real io_struct.py,
finetuning_store_stub.py, and ft_injector.py are loaded as-is from disk.
"""

import os
import sys
import types
from enum import Enum

os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")

_PKG_ROOT = "/tmp/sglang_work/sglang/python/sglang"


def _make_pkg(name: str, path: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


# Register the package roots as bare modules so their real __init__.py
# files never run (they pull in broken optional deps on this box).
for pkg, sub in [
    ("sglang", ""),
    ("sglang.srt", "/srt"),
    ("sglang.srt.managers", "/srt/managers"),
    ("sglang.srt.deltaserve", "/srt/deltaserve"),
]:
    _make_pkg(pkg, _PKG_ROOT + sub)


def _stub(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# Stubs for everything io_struct.py imports — none of these are exercised
# by the dataclass construction path.
class _BaseFinishReason:
    pass


class _Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


_stub(
    "sglang.srt.managers.schedule_batch",
    BaseFinishReason=_BaseFinishReason,
    Modality=_Modality,
)
_stub("sglang.srt.managers.embed_types", PositionalEmbeds=object)
_stub("sglang.srt.lora", )
_stub("sglang.srt.lora.lora_registry", LoRARef=object)
_stub("sglang.srt.multimodal", )
_stub("sglang.srt.multimodal.mm_utils", has_valid_data=lambda *a, **kw: False)
_stub("sglang.srt.observability", )
_stub(
    "sglang.srt.observability.req_time_stats",
    APIServerReqTimeStats=object,
    DPControllerReqTimeStats=object,
    SchedulerReqTimeStats=object,
)
_stub("sglang.srt.sampling", )
_stub("sglang.srt.sampling.sampling_params", SamplingParams=object)
_stub("sglang.srt.utils", ImageData=object, VideoData=object)


# Now load io_struct, finetuning_store_stub, ft_injector via importlib so we
# get the real source — Python will use the registered parent packages' __path__.
import importlib  # noqa: E402

io_struct = importlib.import_module("sglang.srt.managers.io_struct")
GenerateReqInput = io_struct.GenerateReqInput

finetuning_store_stub = importlib.import_module(
    "sglang.srt.deltaserve.finetuning_store_stub"
)
FinetuningStore = finetuning_store_stub.FinetuningStore
FTSample = finetuning_store_stub.FTSample

ft_injector_mod = importlib.import_module("sglang.srt.deltaserve.ft_injector")
FinetuneInjector = ft_injector_mod.FinetuneInjector


# ---- Acceptance assertions ----
samples = [
    FTSample(rid=f"ft-{i}", prompt=f"p{i}", target=f"t{i}", approx_tokens=64)
    for i in range(5)
]
injector = FinetuneInjector(FinetuningStore(samples), lora_dir="/tmp/fake-lora")

batch = injector.next_batch(max_tokens=1000)
assert len(batch) >= 1, f"expected >=1 req, got {len(batch)}"
for r in batch:
    assert r.is_finetuning is True, "is_finetuning not propagated"
    assert r.lora_path == "/tmp/fake-lora", "lora_path not set"

injector.commit([r.rid for r in batch])

direct = GenerateReqInput(text="hello", is_finetuning=True)
assert direct.is_finetuning is True

print("phase3 ok")
