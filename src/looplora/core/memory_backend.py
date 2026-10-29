"""Memory states and read backends for the read-only memory-QA path (docs/prereg/SLOT-LOOP-v1.md, plan 11.2).

A matrix state is the original [B, dv, dk] delta or ridge state. A slot state keeps the written key/value pairs themselves:
the actual unit DPFP keys and raw values of the support facts, in write order. Reads are owner-indexed: ``owner[n]`` names the
episode row whose memory the n-th flattened query reads, so a question batch is expanded by an index map and the state is
never copied per question. Everything here is read-only and functional (no function writes into a state it was given).
Online writing is undefined and refuses. Capacity is a hard limit: more facts than slots raises, nothing is truncated.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor


class CapacityError(ValueError):
    """More facts were written than the slot state has slots."""


@dataclass(frozen=True)
class MatrixState:
    matrix: Tensor  # [B, dv, dk]

    def __post_init__(self) -> None:
        if self.matrix.ndim != 3 or not self.matrix.is_floating_point():
            raise ValueError("a matrix state must be a floating tensor [B, dv, dk]")

    @property
    def rows(self) -> int:
        return self.matrix.shape[0]

    def tensors(self) -> dict[str, Tensor]:
        return {"matrix": self.matrix}


@dataclass(frozen=True)
class SlotState:
    keys: Tensor    # [B, S, dk] the written unit features
    values: Tensor  # [B, S, dv] the written raw values
    valid: Tensor   # [B, S] bool, a prefix of True (slots are filled in write order)
    count: Tensor   # [B] int64, the number of valid slots

    def __post_init__(self) -> None:
        if self.keys.ndim != 3 or self.values.ndim != 3 or self.keys.shape[:2] != self.values.shape[:2]:
            raise ValueError("slot keys and values must be [B, S, dk] and [B, S, dv]")
        if not (self.keys.is_floating_point() and self.values.is_floating_point()):
            raise ValueError("slot keys and values must be floating tensors")
        if self.valid.dtype != torch.bool or self.valid.shape != self.keys.shape[:2]:
            raise ValueError("slot valid must be a bool tensor [B, S]")
        if self.count.dtype != torch.int64 or self.count.shape != self.keys.shape[:1]:
            raise ValueError("slot count must be an int64 tensor [B]")

    @property
    def rows(self) -> int:
        return self.keys.shape[0]

    @property
    def capacity(self) -> int:
        return self.keys.shape[1]

    def tensors(self) -> dict[str, Tensor]:
        return {"keys": self.keys, "values": self.values, "valid": self.valid, "count": self.count}


MemoryState = MatrixState | SlotState


def build_slots(keys: Tensor, values: Tensor, capacity: int, valid: Tensor | None = None) -> SlotState:
    """Slot state of the written facts ``keys`` [B, E, dk] and ``values`` [B, E, dv], zero padded to ``capacity`` slots.

    ``valid`` [B, E] marks the written facts (default: all). The facts must fill a prefix: writes happen in order and leave
    no hole. A row with more facts than slots raises CapacityError, whatever the other rows hold.
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if keys.ndim != 3 or values.ndim != 3 or keys.shape[:2] != values.shape[:2]:
        raise ValueError("keys and values must be [B, E, dk] and [B, E, dv]")
    batch, facts = keys.shape[:2]
    if valid is None:
        valid = torch.ones(batch, facts, dtype=torch.bool, device=keys.device)
    if valid.dtype != torch.bool or valid.shape != (batch, facts):
        raise ValueError("valid must be a bool tensor [B, E]")
    count = valid.sum(dim=1)
    if not bool((valid == (torch.arange(facts, device=keys.device)[None] < count[:, None])).all()):
        raise ValueError("facts must fill a prefix of the slots, in write order")
    if batch and int(count.max()) > capacity:
        raise CapacityError(f"{int(count.max())} facts exceed the {capacity} slots; slots are never truncated")
    if facts > capacity:
        keys, values, valid = keys[:, :capacity], values[:, :capacity], valid[:, :capacity]
    elif facts < capacity:
        pad = capacity - facts
        keys = torch.cat((keys, keys.new_zeros(batch, pad, keys.shape[2])), dim=1)
        values = torch.cat((values, values.new_zeros(batch, pad, values.shape[2])), dim=1)
        valid = torch.cat((valid, valid.new_zeros(batch, pad)), dim=1)
    return SlotState(keys, values, valid, count.to(torch.int64))


def state_tensors(state: MemoryState) -> dict[str, Tensor]:
    return state.tensors()


def persistent_bytes(state: MemoryState) -> int:
    """Bytes of everything the state keeps: ``numel * element_size`` over all its tensors, masks and counts included."""
    return sum(tensor.numel() * tensor.element_size() for tensor in state.tensors().values())


def state_digest(state: MemoryState) -> str:
    """blake2b-128 of the state's names, dtypes, shapes and raw bytes (read-only isolation checks)."""
    digest = hashlib.blake2b(digest_size=16)
    for name, tensor in state.tensors().items():
        host = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{host.dtype}|{tuple(host.shape)}|".encode())
        digest.update(host.reshape(-1).view(torch.uint8).numpy().tobytes() if host.numel() else b"")
    return digest.hexdigest()


def clone_state(state: MemoryState) -> MemoryState:
    return type(state)(**{name: tensor.clone() for name, tensor in state.tensors().items()})


def select_rows(state: MemoryState, rows: Tensor) -> MemoryState:
    return type(state)(**{name: tensor.index_select(0, rows) for name, tensor in state.tensors().items()})


def reset_rows(state: MemoryState, rows: Tensor) -> MemoryState:
    """A copy of the state with the given episode rows emptied (zeros, no valid slot, count zero)."""
    return type(state)(**{name: tensor.index_fill(0, rows, 0) for name, tensor in state.tensors().items()})


def _check_read(state: MemoryState, query: Tensor, owner: Tensor) -> None:
    if query.ndim != 2 or owner.ndim != 1 or owner.shape[0] != query.shape[0]:
        raise ValueError("query must be [N, dk] with one owner entry per row")
    if owner.dtype != torch.int64:
        raise ValueError("owner must be an int64 tensor")


class MemoryBackend:
    """Read semantics of one memory kind. Backends hold settings (gain, temperature), never memory."""

    name = "backend"
    state_type: type = MatrixState

    def __init__(self, gain: float = 1.0) -> None:
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be a finite positive number")
        self.gain = float(gain)

    def check(self, state: MemoryState) -> None:
        if not isinstance(state, self.state_type):
            raise TypeError(f"{self.name} backend reads a {self.state_type.__name__}, not a {type(state).__name__}")

    def read(self, state: MemoryState, query: Tensor, owner: Tensor) -> Tensor:
        """query [N, dk], owner [N] -> readout [N, dv]; row n reads only the memory of episode row owner[n]."""
        raise NotImplementedError

    def select_rows(self, state: MemoryState, rows: Tensor) -> MemoryState:
        self.check(state)
        return select_rows(state, rows)

    def clone_state(self, state: MemoryState) -> MemoryState:
        self.check(state)
        return clone_state(state)

    def reset_rows(self, state: MemoryState, rows: Tensor) -> MemoryState:
        self.check(state)
        return reset_rows(state, rows)

    def persistent_bytes(self, state: MemoryState) -> int:
        self.check(state)
        return persistent_bytes(state)

    def online_write(self, *args, **kwargs):
        raise NotImplementedError(
            "online_write is not supported: only build-final, read-only memory-QA is defined (no silent skip, no overwrite "
            "of the oldest fact, no accumulation across questions)")

    def settings(self) -> dict:
        return {"backend": self.name, "gain": self.gain}

    def _scale(self, readout: Tensor) -> Tensor:
        return readout if self.gain == 1.0 else readout * self.gain


class MatrixBackend(MemoryBackend):
    """m = g M q with the stored matrix of the owner: the original read, operation for operation."""

    name = "matrix"
    state_type = MatrixState

    def read(self, state: MatrixState, query: Tensor, owner: Tensor) -> Tensor:
        self.check(state)
        _check_read(state, query, owner)
        rows = state.matrix.index_select(0, owner)
        return self._scale(torch.matmul(rows, query.to(rows.dtype).unsqueeze(-1)).squeeze(-1))


class _SlotBackend(MemoryBackend):
    state_type = SlotState

    def _scores(self, state: SlotState, query: Tensor, owner: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.check(state)
        _check_read(state, query, owner)
        keys = state.keys.index_select(0, owner)
        scores = torch.bmm(keys, query.to(keys.dtype).unsqueeze(-1)).squeeze(-1)
        return scores, state.valid.index_select(0, owner), state.values.index_select(0, owner)


class SoftSlotBackend(_SlotBackend):
    """m = g sum_i softmax_i(k_i . q / tau [+ log w_i]) v_i over the valid slots; an empty row reads zero.

    ``decay`` (lam in (0, 1]) adds the recency log-weight log(lam^(count - 1 - i)) to the logit of slot i: the writer's own
    recency weighting, as a descriptive control. A row without a valid slot has no softmax: its weights and readout are zero,
    and no NaN enters the graph.
    """

    name = "soft"

    def __init__(self, tau: float, gain: float = 1.0, decay: float | None = None) -> None:
        super().__init__(gain)
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be a finite positive number")
        if decay is not None and not (0 < decay <= 1):
            raise ValueError("decay must lie in (0, 1]")
        self.tau, self.decay = float(tau), decay

    def settings(self) -> dict:
        return {**super().settings(), "tau": self.tau, "recency_decay": self.decay}

    def _weights(self, state: SlotState, query: Tensor, owner: Tensor) -> tuple[Tensor, Tensor]:
        scores, valid, values = self._scores(state, query, owner)
        logits = scores / self.tau
        if self.decay is not None:
            age = (state.count.index_select(0, owner)[:, None] - 1
                   - torch.arange(logits.shape[1], device=logits.device)[None]).clamp_min(0)
            logits = logits + age.to(logits.dtype) * math.log(self.decay)
        empty = ~valid.any(dim=-1)
        # Empty rows are softmaxed over zeros (finite, differentiable) and then zeroed.
        logits = logits.masked_fill(~valid, -torch.inf).masked_fill(empty[:, None], 0.0)
        weights = torch.softmax(logits, dim=-1).masked_fill(empty[:, None], 0.0)
        return weights, values

    def weights(self, state: SlotState, query: Tensor, owner: Tensor) -> Tensor:
        """The attention weights [N, S] a read would use (diagnostics)."""
        return self._weights(state, query, owner)[0]

    def read(self, state: SlotState, query: Tensor, owner: Tensor) -> Tensor:
        weights, values = self._weights(state, query, owner)
        return self._scale(torch.bmm(weights.unsqueeze(1), values).squeeze(1))


class HardSlotBackend(_SlotBackend):
    """m = g v_j with j the valid slot of the largest k . q (ties: the lowest slot index); an empty row reads zero."""

    name = "hard"

    def index(self, state: SlotState, query: Tensor, owner: Tensor) -> tuple[Tensor, Tensor]:
        """The selected slot [N] and whether the row has any valid slot [N]."""
        scores, valid, _ = self._scores(state, query, owner)
        scores = scores.masked_fill(~valid, -torch.inf)
        best = scores.amax(dim=-1, keepdim=True)
        slots = torch.arange(scores.shape[1], device=scores.device)[None]
        chosen = torch.where((scores == best) & valid, slots, scores.shape[1]).amin(dim=-1)
        return chosen.clamp_max(scores.shape[1] - 1), valid.any(dim=-1)

    def read(self, state: SlotState, query: Tensor, owner: Tensor) -> Tensor:
        chosen, nonempty = self.index(state, query, owner)
        values = state.values.index_select(0, owner)
        picked = values.gather(1, chosen[:, None, None].expand(-1, 1, values.shape[2])).squeeze(1)
        return self._scale(picked.masked_fill(~nonempty[:, None], 0.0))


@dataclass(frozen=True)
class MemoryRead:
    """What a read sees in place of a tensor state: a backend, the memory of the B episodes and each query row's owner.

    ``owner`` maps the rows of the flattened question batch to rows of ``state``; expanding questions only repeats owners.
    Instances are immutable; select/expand/clone return new handles over the same (or cloned) state tensors.
    """

    backend: MemoryBackend
    state: MemoryState
    owner: Tensor

    @classmethod
    def bind(cls, backend: MemoryBackend, state: MemoryState, owner: Tensor | None = None) -> "MemoryRead":
        backend.check(state)
        if owner is None:
            owner = torch.arange(state.rows, device=next(iter(state.tensors().values())).device)
        if owner.ndim != 1 or owner.dtype != torch.int64:
            raise ValueError("owner must be an int64 vector")
        if owner.numel() and (int(owner.min()) < 0 or int(owner.max()) >= state.rows):
            raise ValueError("owner names an episode row outside the state")
        return cls(backend, state, owner)

    @property
    def rows(self) -> int:
        return self.owner.shape[0]

    def check_fp32(self) -> None:
        for tensor in self.state.tensors().values():
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise ValueError("LLM adapter memory must be fp32")

    def expand(self, queries: int) -> "MemoryRead":
        """Episode-major flattening of B x Q questions: each owner is repeated Q times, the state stays at B rows."""
        if queries < 1:
            raise ValueError("queries must be positive")
        return replace(self, owner=self.owner.repeat_interleave(queries))

    def select(self, rows: Tensor, final: Tensor | None = None) -> "MemoryRead":
        return replace(self, owner=self.owner.index_select(0, rows))

    def read(self, query: Tensor) -> Tensor:
        return self.backend.read(self.state, query, self.owner)

    def clone(self) -> "MemoryRead":
        return replace(self, state=self.backend.clone_state(self.state), owner=self.owner.clone())
