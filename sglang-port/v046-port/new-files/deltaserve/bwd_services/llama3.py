"""Phase 6 Llama3 backward service stub.

# Phase 6 stub: real GQA backward arrives in Phase 7.
"""

from typing import Dict

import torch

from sglang.srt.deltaserve.bwd_services.base import BackwardService


class Llama3BackwardService(BackwardService):
    def step(self, activations: Dict) -> Dict[str, torch.Tensor]:
        grads: Dict[str, torch.Tensor] = {}
        for key, value in activations.items():
            if isinstance(value, torch.Tensor):
                grads[key] = value * 0.01
            else:
                grads[key] = torch.as_tensor(value) * 0.01
        return grads

    def apply_grads(self, grad_dict: Dict) -> None:
        return None
