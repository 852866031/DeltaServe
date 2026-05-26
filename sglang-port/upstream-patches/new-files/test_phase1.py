"""Phase 1 acceptance test for DeltaServe sglang port."""
import ast
import inspect
from pathlib import Path

from sglang.srt.configs.finetune import FinetuneConfig
from sglang.srt.server_args import ServerArgs

# D2: ServerArgs has the 3 new fields and accepts them via constructor.
args = ServerArgs(
    model_path="dummy",
    enable_finetuning=True,
    backward_mps_percentage=20,
)
assert args.enable_finetuning is True, args.enable_finetuning
assert args.backward_mps_percentage == 20, args.backward_mps_percentage
assert args.finetune_config is None, args.finetune_config

# D1: FinetuneConfig default-disabled.
fc = FinetuneConfig()
assert fc.enable_finetuning is False, fc.enable_finetuning

# D3: Scheduler.__init__ accepts finetune_config: Optional[FinetuneConfig] = None.
# Verified via AST parse to sidestep an env-level torch/sglang version skew that
# blocks importing the scheduler module (torch 2.6 vs sglang's pinned 2.11).
sched_src = Path("python/sglang/srt/managers/scheduler.py").read_text()
tree = ast.parse(sched_src)
sched_cls = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
)
init_fn = next(
    n for n in sched_cls.body
    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
)
arg_names = [a.arg for a in init_fn.args.args] + [a.arg for a in init_fn.args.kwonlyargs]
assert "finetune_config" in arg_names, arg_names
# default of None — kwarg has a default (so it's in args.defaults or kw_defaults).
defaults_by_name = dict(zip(
    [a.arg for a in init_fn.args.args[-len(init_fn.args.defaults):]],
    init_fn.args.defaults,
)) if init_fn.args.defaults else {}
ft_default = defaults_by_name.get("finetune_config")
assert isinstance(ft_default, ast.Constant) and ft_default.value is None, ast.dump(ft_default)

print("phase1 ok")
