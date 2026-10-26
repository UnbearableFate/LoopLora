"""Fact-only random-code writer for the memory-only LLM reader control."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from looplora.infra.seeds import derive_seed
from looplora.sym.writers import OracleWriter


class OracleRandomWriter(OracleWriter):
    """Global (entity, relation) keys and global entity values, all fixed.

    Entity values are exactly OracleWriter's global codes at the same seed.
    Lift head IDs into a disjoint pair-code namespace, then use its unchanged
    decay-first scan (lambda = beta = 1), without re-normalizing the keys.
    """

    def __init__(self, V: int, n_rel: int, seed: int = 17, *, r_mem: int = 32) -> None:
        if r_mem not in {32, 128}:
            raise ValueError("P1-L oracle memory rank must be32 or128")
        super().__init__(V, r_mem=r_mem, seed=seed)
        self.n_rel = n_rel
        generator = torch.Generator().manual_seed(derive_seed(seed, "oracle", "pair_codes"))
        self.register_buffer("pair_codes", F.normalize(
            torch.randn(V, n_rel, self.r_mem, generator=generator), dim=-1))

    def forward(self, facts: Tensor, valid: Tensor) -> Tensor:
        # Invalid slots may hold sentinels; they never address either table.
        facts = facts.masked_fill(~valid[..., None], 0)
        offset = self.V * self.n_rel
        lifted = torch.stack((facts[..., 0] * self.n_rel + facts[..., 1],
                              facts[..., 1], offset + facts[..., 2]), dim=-1)
        table = torch.cat((self.pair_codes.flatten(0, 1), self.global_codes), dim=0)
        codes = table.unsqueeze(0).expand(len(facts), -1, -1)
        return super().forward(lifted, valid, codes)
