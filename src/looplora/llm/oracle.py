"""Fact-only oracle writers for the memory-only LLM reader control: random keys (P1-L) and rotated keys (WP-L6)."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.core.memory import delta_update, zeros_state
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


def relation_rotations(seed: int, n_rel: int, r: int) -> Tensor:
    """Fixed orthogonal Pi_rel for each relation: the Q factor of a seeded Gaussian with the QR sign fixed.

    LAPACK may return either sign for every column of Q. Scaling the columns by sign(diag R) selects the unique
    factorization with diag R > 0, so the matrices depend on the seed only. float32 Gaussian, float64 QR.
    """
    rotations = []
    for rel in range(n_rel):
        generator = torch.Generator().manual_seed(derive_seed(seed, "oracle", "rotation", rel))
        q, upper = torch.linalg.qr(torch.randn(r, r, generator=generator).double())
        rotations.append((q * torch.sign(torch.diagonal(upper))).float())
    return torch.stack(rotations)


class RotatedOracleWriter(OracleWriter):
    """Fact-only oracle with rotated keys: key(e, rel) = normalize(Pi_rel @ code(e)); value(e) is the entity code.

    Addresses and values come from one entity code table, so after a read the reader maps the bridge's value code
    to its next key by the fixed linear map Pi_rel instead of an unrelated random key. Facts are written in list
    order with the unchanged decay-first rule (lambda = beta = 1). Invalid slots may hold sentinels: they are
    zeroed before any table is indexed and are never written, as in OracleRandomWriter.
    """

    def __init__(self, V: int, n_rel: int, seed: int = 17, *, r_mem: int = 32) -> None:
        if r_mem not in {32, 128}:
            raise ValueError("oracle memory rank must be 32 or 128")
        super().__init__(V, r_mem=r_mem, seed=seed)
        self.n_rel = n_rel
        self.register_buffer("rotations", relation_rotations(seed, n_rel, r_mem))

    def entity_codes(self) -> Tensor:
        """[V + 1, r_mem] entity code table that keys rotate and values copy."""
        return self.global_codes

    def value_of(self, codes: Tensor) -> Tensor:
        return codes

    def fact_codes(self, facts: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Keys and values [batch, facts, r_mem] of every fact slot (invalid slots hold entity 0, relation 0)."""
        facts = facts.masked_fill(~valid[..., None], 0)
        table = self.entity_codes()
        rotated = torch.matmul(self.rotations[facts[..., 1]], table[facts[..., 0]].unsqueeze(-1)).squeeze(-1)
        return F.normalize(rotated, dim=-1), self.value_of(table[facts[..., 2]])

    def forward(self, facts: Tensor, valid: Tensor) -> Tensor:
        keys, values = self.fact_codes(facts, valid)
        state = zeros_state(len(facts), self.r_mem, keys.dtype, keys.device)
        for t in range(facts.shape[1]):
            state = delta_update(state, keys[:, t], values[:, t], valid[:, t], lam=1.0, beta=1.0)
        return state


class OracleRotWriter(RotatedOracleWriter):
    """oracle_rot: fixed codes. Values are OracleWriter.global_codes, as in oracle_random; keys rotate them."""


class OracleSharedWriter(RotatedOracleWriter):
    """oracle_shared: a trainable table C (initialized to the global codes) supplies keys and values.

    key(e, rel) = normalize(Pi_rel @ C_e) and value(e) = normalize(C_e); C is a writer parameter trained with
    the reader, and the fixed global codes stay in the state only as its initialization.
    """

    def __init__(self, V: int, n_rel: int, seed: int = 17, *, r_mem: int = 32) -> None:
        super().__init__(V, n_rel, seed, r_mem=r_mem)
        self.C = nn.Parameter(self.global_codes.clone())

    def entity_codes(self) -> Tensor:
        return self.C

    def value_of(self, codes: Tensor) -> Tensor:
        return F.normalize(codes, dim=-1)
