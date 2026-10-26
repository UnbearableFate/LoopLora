"""Backbone-independent frozen linear wrapper and explicit per-call state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from .adapters import AdapterBase


@dataclass
class MemoryContext:
    """Mutable call context; masks describe this call, state is replaced on return.

    online reads before writing; read_only disables writes; write_only returns
    the base projection and updates memory without reading. A supplied state
    retains its dtype. Fork states explicitly for independent questions/depths.
    """

    state: Tensor | None = None
    loop_steps: int = 1
    valid: Tensor | None = None
    write: Tensor | None = None
    reset: Tensor | None = None
    mode: Literal["online", "write_only", "read_only"] = "online"
    stats: dict[str, Tensor] = field(default_factory=dict, init=False)


class AdaptedLinear(nn.Module):
    def __init__(self, base: nn.Linear, adapter: AdapterBase, context: MemoryContext | None = None) -> None:
        super().__init__()
        if (base.in_features, base.out_features) != (adapter.cfg.d_in, adapter.cfg.d_out):
            raise ValueError("base linear and adapter dimensions must match")
        self.base = base
        self.base.requires_grad_(False)
        self.adapter = adapter
        self.context = context
        self.base_call_count = 0
        self.base_token_count = 0

    def set_context(self, context: MemoryContext | None) -> None:
        self.context = context

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in {2, 3}:
            raise ValueError("AdaptedLinear expects [B,d_in] or [B,T,d_in]")
        context = self.context if self.context is not None else MemoryContext()
        if context.mode not in {"online", "write_only", "read_only"}:
            raise ValueError("mode must be online, write_only, or read_only")
        single_token = x.ndim == 2
        adapter_input = x.to(dtype=self.adapter.A.dtype)
        sequence = adapter_input.unsqueeze(1) if single_token else adapter_input

        def sequence_mask(mask: Tensor | None) -> Tensor | None:
            if single_token and mask is not None and mask.ndim == 1:
                return mask.unsqueeze(1)
            return mask

        valid = sequence_mask(context.valid)
        write = sequence_mask(context.write)
        reset = sequence_mask(context.reset)
        if context.mode == "read_only":
            write = torch.zeros(sequence.shape[:2], dtype=torch.bool, device=x.device)
        if context.mode == "write_only":
            u, state, valid, write, reset = self.adapter._prepare(
                sequence, context.state, context.loop_steps, valid, write, reset,
            )
            if self.adapter.has_memory:
                _, state = self.adapter.scan_prewrite(state, u, valid, write, reset)
            elif state is not None:
                state = torch.where(reset.any(dim=1)[:, None, None], torch.zeros_like(state), state)
            stats = self.adapter._stats(valid, write, context.loop_steps)
            stats["reads"] = torch.zeros_like(stats["reads"])
            branch = None
        else:
            branch, state, stats = self.adapter.process_sequence(
                sequence, context.state, context.loop_steps, valid, write, reset,
            )
        # Frozen weights still participate in the graph for upstream gradients.
        base = self.base(x)
        self.base_call_count += 1
        self.base_token_count += sequence.shape[0] * sequence.shape[1]
        context.state = state
        context.stats = stats
        if branch is None:
            return base
        branch = branch[:, 0] if single_token else branch
        return base + branch.to(dtype=base.dtype)
