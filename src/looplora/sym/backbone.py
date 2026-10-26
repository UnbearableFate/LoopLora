"""Frozen random symbolic features; there is no task-aware base projection."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from looplora.infra.seeds import derive_seed


class SymbolicBackbone(nn.Module):
    def __init__(self, V: int = 1024, n_rel: int = 4, d_e: int = 64,
                 max_hop: int = 16, backbone_seed: int = 0) -> None:
        super().__init__()
        if min(V, n_rel, d_e, max_hop) < 1:
            raise ValueError("backbone dimensions must be positive")
        self.V, self.d_e, self.max_hop = V, d_e, max_hop
        for name, count in (("entities", V + 1), ("relations", n_rel), ("hops", max_hop + 1)):
            generator = torch.Generator().manual_seed(derive_seed(backbone_seed, "backbone", name))
            self.register_buffer(name, torch.randn(count, d_e, generator=generator) / d_e**0.5)

    def facts(self, facts: Tensor) -> Tensor:
        return torch.cat((self.entities[facts[..., 0]], self.relations[facts[..., 1]],
                          self.entities[facts[..., 2]]), dim=-1)

    def queries(self, start: Tensor, rel: Tensor, hop_given: Tensor) -> Tensor:
        if torch.any((hop_given < -1) | (hop_given > self.max_hop)):
            raise ValueError("stated hop is outside the backbone hop table")
        head = self.entities[start]
        hop_index = hop_given.clamp_min(0)
        return torch.cat((head, self.relations[rel] + self.hops[hop_index],
                          torch.zeros_like(head)), dim=-1)

    def logits(self, output: Tensor) -> Tensor:
        return output @ self.entities.T
