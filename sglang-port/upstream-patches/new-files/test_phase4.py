import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/tmp/sglang_work/sglang/python")
from sglang.srt.deltaserve.accumulate import FinetuneAccumulator

H = 8
lin0 = nn.Linear(H, H)
lin1 = nn.Linear(H, H)
model = nn.Sequential(lin0, lin1)

acc = FinetuneAccumulator()
acc.register_hooks(model, module_map={lin0: 0, lin1: 1})
acc.begin_step(
    req_ids=["ft0", "inf1"],
    finetune_mask=torch.tensor([True, False]),
)

_ = model(torch.randn(2, H))

assert acc.layer_in[0].shape[0] == 1, (
    f"expected 1 FT row in layer_in[0], got {acc.layer_in[0].shape[0]}"
)
assert acc.mlp_gate_up[0].shape[0] == 1, (
    f"expected 1 FT row in mlp_gate_up[0], got {acc.mlp_gate_up[0].shape[0]}"
)

layer_in, mlp_gate_up = acc.pop_step()
assert acc.layer_in == {}, f"pop_step did not clear layer_in: {acc.layer_in}"
assert acc.mlp_gate_up == {}, f"pop_step did not clear mlp_gate_up: {acc.mlp_gate_up}"
assert 0 in layer_in and 1 in layer_in, f"snapshot missing layer keys: {layer_in.keys()}"

print("phase4 ok")
