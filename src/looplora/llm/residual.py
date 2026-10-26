"""Adapter input from a decoder's residual stream, output at its o_proj."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from looplora.core.wrapper import MemoryContext


class ResidualInputLinear(nn.Module):
    """Qwen's attention projection with an independently captured adapter input.

    The decoder pre-hook supplies the pre-layer residual. The base projection
    always receives the actual attention output; only A's input changes.
    Contexts, masks and memory follow the same contract as AdaptedLinear.
    """

    def __init__(self,base: nn.Linear,adapter):
        super().__init__()
        if base.out_features!=adapter.cfg.d_out:
            raise ValueError('base and adapter output dimensions must match')
        self.base,self.adapter = base,adapter
        self.base.requires_grad_(False)
        self.context = None
        self.base_call_count = self.base_token_count = 0
        self._residual = None

    def capture(self,residual: Tensor) -> None:
        self._residual = residual

    def forward(self,x: Tensor) -> Tensor:
        residual,self._residual = self._residual,None
        if residual is None:
            raise RuntimeError('residual input requires the decoder pre-hook before o_proj')
        if (x.ndim!=3 or residual.ndim!=3 or x.shape[:2]!=residual.shape[:2]
                or residual.shape[-1]!=self.adapter.cfg.d_in):
            raise ValueError('residual must match the projection batch/tokens and adapter width')
        context = self.context if self.context is not None else MemoryContext()
        if context.mode not in {'online','write_only','read_only'}:
            raise ValueError('unknown memory context mode')
        sequence = residual.to(dtype=self.adapter.A.dtype)
        write = (torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device)
                 if context.mode=='read_only' else context.write)
        if context.mode=='write_only':
            u,state,valid,write,reset = self.adapter._prepare(
                sequence,context.state,context.loop_steps,context.valid,write,context.reset)
            if self.adapter.has_memory:
                state = self.adapter.scan_final(state,u,valid,write,reset)
            elif state is not None:
                state = torch.where(reset.any(1)[:,None,None],torch.zeros_like(state),state)
            stats = self.adapter._stats(valid,write,context.loop_steps)
            stats['reads'] = torch.zeros_like(stats['reads'])
            branch = None
        else:
            branch,state,stats = self.adapter.process_sequence(
                sequence,context.state,context.loop_steps,context.valid,write,context.reset)
        base = self.base(x)
        self.base_call_count += 1
        self.base_token_count += x.shape[0]*x.shape[1]
        context.state,context.stats = state,stats
        return base if branch is None else base+branch.to(base.dtype)
