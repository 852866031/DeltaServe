# SPDX-License-Identifier: Apache-2.0
"""Per-sample activation capture for finetuning-tagged requests.

When a forward batch contains FT-tagged samples mixed with inference samples,
this accumulator captures the FT rows of the residual stream entering each
decoder layer (``layer_in``) and the fused ``gate_up_proj`` output inside each
MLP (``mlp_gate_up``). Capture is gated on a per-row boolean ``finetune_mask``
set via ``begin_step``; ``pop_step`` snapshots and resets state for the next
forward.

Registration uses PyTorch's canonical ``module.register_forward_pre_hook`` /
``register_forward_hook`` directly — additive, not monkey-patching of
``forward``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_LAYER_IN_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.input_layernorm$")
_GATE_UP_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate_up_proj$")


class FinetuneAccumulator:
    def __init__(self) -> None:
        self.layer_in: Dict[int, torch.Tensor] = {}
        self.mlp_gate_up: Dict[int, torch.Tensor] = {}
        self._finetune_mask: Optional[torch.Tensor] = None
        self._req_ids: List[str] = []
        self._handles: List = []

    def register_hooks(
        self,
        model: nn.Module,
        module_map: Optional[Dict[nn.Module, int]] = None,
    ) -> None:
        if module_map is not None:
            for mod, idx in module_map.items():
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_pre_hook(idx))
                )
                self._handles.append(
                    mod.register_forward_hook(self._make_post_hook(idx))
                )
            return

        for name, mod in model.named_modules():
            m_in = _LAYER_IN_RE.search(name)
            if m_in is not None:
                idx = int(m_in.group(1))
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_pre_hook(idx))
                )
                continue
            m_gu = _GATE_UP_RE.search(name)
            if m_gu is not None:
                idx = int(m_gu.group(1))
                self._handles.append(
                    mod.register_forward_hook(self._make_post_hook(idx))
                )

    def _make_pre_hook(self, idx: int):
        def pre_hook(module, args):
            if self._finetune_mask is None:
                return
            if len(args) >= 2 and args[1] is not None:
                val = args[0] + args[1]
            else:
                val = args[0]
            self.layer_in[idx] = val[self._finetune_mask].detach().clone()

        return pre_hook

    def _make_post_hook(self, idx: int):
        def post_hook(module, inputs, output):
            if self._finetune_mask is None:
                return
            out = output[0] if isinstance(output, tuple) else output
            self.mlp_gate_up[idx] = out[self._finetune_mask].detach().clone()

        return post_hook

    def begin_step(
        self,
        req_ids: List[str],
        finetune_mask: torch.Tensor,
    ) -> None:
        self._req_ids = list(req_ids)
        self._finetune_mask = finetune_mask
        self.layer_in = {}
        self.mlp_gate_up = {}

    def pop_step(
        self,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        snapshot = (self.layer_in, self.mlp_gate_up)
        self.layer_in = {}
        self.mlp_gate_up = {}
        self._finetune_mask = None
        self._req_ids = []
        return snapshot

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
