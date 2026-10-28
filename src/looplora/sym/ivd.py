"""IVD (F8): separate key interference from memory load on frozen symbolic checkpoints.

Off-path facts are added to the memory of an unchanged, frozen reader. The
target chains, queries and answers of every episode never change; the extras
have heads outside the episode and tails inside it, so the candidate answer set
(the entities of the original twelve facts) is identical in every cell. Two
factors are crossed:

* load: the first ``n`` of at most 24 extras (nested loads, fixed slots);
* key overlap: ``low`` (constructed or selected to be near orthogonal to the
  relevant keys), ``nat`` (unmodified random entities), ``high``.

The oracle writer's overlap is constructed directly in per-episode code tables;
the learned (factorized) writer's overlap is selected by measured key cosine
from a pool of candidate heads. Relevant facts keep their original relative
order and the extras occupy the same slots in every overlap arm, so the age
and order of the relevant facts are matched within a load. For the learned
writer a decay-only control replaces the extras by writes with beta = 0.

Nothing here trains, edits a checkpoint, or touches existing code paths.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from looplora.core.memory import delta_update, zeros_state
from looplora.data.relations import EpisodeArrays
from looplora.infra.seeds import derive_seed
from . import followup_stats as fs
from .evaluate import _require_finite, _slice
from .model import PreparedBatch, SymbolicModel
from .writers import FactorizedWriter, OracleWriter

LOADS: Mapping[str, int] = {"L6": 6, "L24": 24}
OVERLAPS = ("low", "nat", "high")
HIGH_COSINE = 0.5
POOL_CANDIDATES = 128
WRITERS = ("oracle_random", "joint_fact")
DECISION_LOAD = "L24"
DECISION_HOPS = (2, 3)
GAIN_MINIMUM = 0.05
SPLIT = "test"
HOP_GROUPS = fs.HOP_GROUPS


def cell_names(writer: str, loads: Mapping[str, int] = LOADS) -> list[str]:
    """Evaluation cells: baseline, load x overlap, and (learned writer) decay-only controls."""
    if writer not in WRITERS:
        raise ValueError(f"IVD supports {WRITERS}, not {writer!r}")
    names = ["base"]
    for load in loads:
        names.extend(f"{load}-{kind}" for kind in OVERLAPS)
    if writer == "joint_fact":
        names.extend(f"{load}-decay" for load in loads)
    return names


def episode_entities(arrays: EpisodeArrays) -> np.ndarray:
    """Sorted entity ids [B, m] of the original facts; every episode must have the same m."""
    rows = [np.unique(facts[valid][:, (0, 2)]) for facts, valid in zip(arrays.facts, arrays.fact_valid)]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("IVD requires the same number of entities in every episode")
    return np.stack(rows).astype(np.int64)


def draw_extras(arrays: EpisodeArrays, entities: np.ndarray, *, seed: int, n_extra: int,
                pool: tuple[int, int], candidates: int = POOL_CANDIDATES) -> tuple[np.ndarray, np.ndarray]:
    """Candidate extra heads [B, P] outside the episode and slot tails [B, n_extra] inside it."""
    heads, tails = [], []
    for b in range(len(arrays.episode_seed)):
        rng = np.random.default_rng(derive_seed(seed, "ivd", "extras", int(arrays.episode_seed[b])))
        outside = np.setdiff1d(np.arange(pool[0], pool[1]), entities[b])
        size = min(candidates, len(outside))
        if size < n_extra:
            raise ValueError("entity pool is too small for the requested number of extras")
        heads.append(rng.choice(outside, size, replace=False))
        tails.append(rng.choice(entities[b], n_extra, replace=True))
    return np.stack(heads).astype(np.int64), np.stack(tails).astype(np.int64)


def shuffle_order(episode_seed: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Per-episode random permutations [B, n]: slot order is independent of overlap rank."""
    return np.stack([np.random.default_rng(derive_seed(seed, "ivd", "order", int(episode))).permutation(n)
                     for episode in episode_seed]).astype(np.int64)


def slot_layout(episode_seed: np.ndarray, n_facts: int, n_extra: int, seed: int) -> np.ndarray:
    """Boolean [B, n_facts + n_extra]; True marks an extra slot. Relevant slots keep fact order."""
    total = n_facts + n_extra
    is_extra = np.ones((len(episode_seed), total), dtype=bool)
    for b, episode in enumerate(episode_seed):
        rng = np.random.default_rng(derive_seed(seed, "ivd", "slots", int(episode)))
        is_extra[b, rng.choice(total, n_facts, replace=False)] = False
    return is_extra


def assemble(arrays: EpisodeArrays, heads: np.ndarray, tails: np.ndarray, is_extra: np.ndarray,
             load: int) -> tuple[EpisodeArrays, np.ndarray]:
    """Interleave extras 0..load-1 at their fixed slots; returns (arrays, extra flags [B, E+load]).

    The sequence of a smaller load is the subsequence of a larger load that drops
    the later extras, so relevant facts never move relative to each other.
    """
    count, n_facts = arrays.facts.shape[:2]
    if not arrays.fact_valid.all():
        raise ValueError("IVD requires episodes without padded facts")
    if load < 0 or load > is_extra.shape[1] - n_facts or heads.shape[1] < load or tails.shape[1] < load:
        raise ValueError("load exceeds the drawn extras")
    facts = np.zeros((count, n_facts + load, 3), dtype=arrays.facts.dtype)
    flags = np.zeros((count, n_facts + load), dtype=bool)
    for b in range(count):
        rank = np.cumsum(is_extra[b]) - 1
        positions = np.flatnonzero(~is_extra[b] | (rank < load))
        relevant = iter(range(n_facts))
        for k, slot in enumerate(positions):
            if is_extra[b, slot]:
                j = int(rank[slot])
                facts[b, k] = (heads[b, j], 0, tails[b, j])
                flags[b, k] = True
            else:
                facts[b, k] = arrays.facts[b, next(relevant)]
    extended = replace(arrays, facts=facts, fact_valid=np.ones((count, n_facts + load), dtype=bool))
    return extended, flags


@torch.no_grad()
def write_keys(model: SymbolicModel, facts: Tensor) -> Tensor:
    """Deployed unit write keys [..., r_mem] of arbitrary fact triples (factorized writer only)."""
    writer = model.writer
    if not isinstance(writer, FactorizedWriter):
        raise ValueError("write_keys requires the factorized writer")
    raw = F.linear(writer.u(model.backbone.facts(facts)), writer.K)
    return raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-6)


def overlap(extra_keys: Tensor, relevant_keys: Tensor) -> Tensor:
    """Largest absolute cosine of each extra key with any relevant key: [B, n]."""
    cosine = torch.einsum("bnr,ber->bne", F.normalize(extra_keys, dim=-1), F.normalize(relevant_keys, dim=-1))
    return cosine.abs().amax(dim=-1)


@torch.no_grad()
def select_by_overlap(model: SymbolicModel, facts: Tensor, candidate_heads: Tensor, kind: str, n: int,
                      order: Tensor) -> tuple[Tensor, Tensor]:
    """Choose ``n`` of the candidate heads by measured key overlap (learned writer).

    ``nat`` keeps the first ``n`` random candidates, ``low`` takes the ``n``
    smallest and ``high`` the ``n`` largest overlap. ``order`` ([B, n], a
    random permutation per episode) fixes which selected head sits in which
    extra slot, so slot order and age do not depend on overlap rank and the
    smaller load is a random subset of the larger one. Returns
    (heads [B, n], overlap [B, n]).
    """
    if kind not in OVERLAPS:
        raise ValueError(f"unknown overlap arm {kind!r}")
    zeros = torch.zeros_like(candidate_heads)
    candidate_keys = write_keys(model, torch.stack((candidate_heads, zeros, zeros), dim=-1))
    rho = overlap(candidate_keys, write_keys(model, facts))
    if kind == "nat":
        ranking = torch.arange(candidate_heads.shape[1], device=candidate_heads.device).expand_as(candidate_heads)
    else:
        ranking = torch.argsort(rho, dim=1, descending=kind == "high", stable=True)
    chosen = ranking[:, :n].gather(1, order)
    return candidate_heads.gather(1, chosen), rho.gather(1, chosen)


@torch.no_grad()
def constructed_codes(global_codes: Tensor, facts: Tensor, entities: Tensor, heads: Tensor, kind: str, *,
                      episode_seed: np.ndarray, seed: int, cosine: float = HIGH_COSINE) -> Tensor:
    """Per-episode oracle code tables [B, V+1, r] with controlled extra-head overlap.

    ``low`` heads are unit vectors orthogonal to the codes of every episode
    entity; ``high`` heads have cosine ``cosine`` with the code of one relevant
    head (slot j targets relevant fact j mod E) and are orthogonal to every
    other direction of the episode span; ``nat`` leaves the global codes. The
    random directions are drawn per episode, independent of batching.
    """
    if kind not in OVERLAPS:
        raise ValueError(f"unknown overlap arm {kind!r}")
    count, n = heads.shape
    codes = global_codes.unsqueeze(0).repeat(count, 1, 1)
    if kind == "nat":
        return codes
    rows = torch.arange(count, device=codes.device)[:, None]
    span = codes[rows, entities].double()
    if span.shape[1] >= codes.shape[-1]:
        raise ValueError("episode entities must be fewer than the code dimension")
    basis, _ = torch.linalg.qr(span.transpose(1, 2))
    noise = torch.as_tensor(np.stack([
        np.random.default_rng(derive_seed(seed, "ivd", "oracle-noise", kind, int(episode))).standard_normal(
            (n, codes.shape[-1])) for episode in episode_seed]), dtype=torch.float64, device=codes.device)
    unit = F.normalize(noise - (noise @ basis) @ basis.transpose(1, 2), dim=-1)
    if kind == "low":
        new = unit
    else:
        slots = torch.arange(n, device=codes.device) % facts.shape[1]
        target = codes[rows, facts[:, slots, 0]].double()
        new = cosine * target + (1 - cosine ** 2) ** 0.5 * unit
    codes[rows, heads] = new.to(codes.dtype)
    return codes


@torch.no_grad()
def decay_only_memory(model: SymbolicModel, arrays: EpisodeArrays, decay_slots: np.ndarray) -> Tensor:
    """Factorized-writer memory in which the flagged slots only decay (beta = 0)."""
    writer = model.writer
    if not isinstance(writer, FactorizedWriter):
        raise ValueError("decay-only controls require the factorized writer")
    device = model.device
    valid = torch.as_tensor(arrays.fact_valid, dtype=torch.bool, device=device)
    facts = torch.as_tensor(arrays.facts, dtype=torch.long, device=device)
    facts = torch.where(valid[..., None], facts, torch.zeros_like(facts))
    x = model.backbone.facts(facts)
    x = torch.where(valid[..., None], x, torch.zeros_like(x))
    raw = F.linear(writer.u(x), writer.K)
    keys = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-6)
    values = F.linear(F.linear(x[..., 2 * writer.d_e:], writer.A_v), writer.V)
    decay = torch.as_tensor(decay_slots, dtype=torch.bool, device=device)
    state = zeros_state(len(x), writer.r_mem, x.dtype, x.device)
    lam, beta = writer.a_lambda.sigmoid(), writer.a_beta.sigmoid()
    for t in range(x.shape[1]):
        scale = (~decay[:, t]).to(x.dtype)
        state = delta_update(state, keys[:, t], values[:, t], valid[:, t], lam, beta * scale)
    return state


@torch.no_grad()
def prepare_cell(model: SymbolicModel, extended: EpisodeArrays, *, candidate_mask: Tensor,
                 codes: Tensor | None = None, decay_slots: np.ndarray | None = None,
                 decode_codes: Tensor | None = None) -> PreparedBatch:
    """Prepared batch for one cell; the candidate set is always the original episode's.

    ``codes`` are the constructed oracle codes (they set the write keys and values of the
    extras); ``decode_codes``, when given, replace the table the reader decodes against, so that
    a full-vocabulary decode does not see the constructed codes of the extras' heads.
    """
    options = {} if codes is None else {"oracle_codes": codes}
    prepared = model.prepare(extended, **options)
    if decay_slots is not None:
        prepared = replace(prepared, memory=decay_only_memory(model, extended, decay_slots))
    if decode_codes is not None:
        prepared = replace(prepared, codes=decode_codes)
    return replace(prepared, candidate_mask=candidate_mask)


@torch.no_grad()
def start_keys(model: SymbolicModel, prepared: PreparedBatch, extended: EpisodeArrays) -> Tensor:
    """Write key of the fact whose head is each query's start: [B, Q, r_mem]."""
    if isinstance(model.writer, OracleWriter):
        return prepared.u
    device = model.device
    facts = torch.as_tensor(extended.facts, dtype=torch.long, device=device)
    valid = torch.as_tensor(extended.fact_valid, dtype=torch.bool, device=device)
    start = torch.as_tensor(extended.q_start, dtype=torch.long, device=device)
    rel = torch.as_tensor(extended.q_rel, dtype=torch.long, device=device)
    matches = ((facts[:, None, :, 0] == start[..., None]) & (facts[:, None, :, 1] == rel[..., None])
               & valid[:, None, :])
    positions = torch.arange(facts.shape[1], device=device)
    last = torch.where(matches, positions, -1).amax(dim=-1).clamp_min(0)
    keys = write_keys(model, facts)
    rows = torch.arange(len(facts), device=device)
    return keys[rows[:, None], last]


@torch.no_grad()
def read_residual(model: SymbolicModel, prepared: PreparedBatch, extended: EpisodeArrays) -> Tensor:
    """Stored-pair residual ||M k - v|| / ||v|| at each query's start fact: [B, Q].

    ``k`` is the write key of the start fact, so this measures what the memory
    holds for that fact, independent of the reader's learned query projection.
    """
    key = start_keys(model, prepared, extended)
    read = torch.einsum("bij,bqj->bqi", prepared.memory, key)
    value = prepared.true_value
    residual = torch.linalg.vector_norm(read - value, dim=-1) / torch.linalg.vector_norm(value, dim=-1).clamp_min(1e-12)
    return torch.where(prepared.retrieval_valid, residual, torch.full_like(residual, float("nan")))


def _cell_parts(name: str) -> tuple[str, str]:
    load, _, kind = name.partition("-")
    return load, kind


@torch.no_grad()
def evaluate_cells(model: SymbolicModel, arrays: EpisodeArrays, *, depths: list[int], seed: int,
                   pool: tuple[int, int], batch_size: int = 64, split: str = "test",
                   loads: Mapping[str, int] = LOADS,
                   candidates: int = POOL_CANDIDATES) -> dict[str, np.ndarray]:
    """Evaluate one frozen checkpoint on every IVD cell; flat arrays like ``evaluate`` samples.

    Rows carry split, condition (cell name), episode_id, query_id, hop, L, prediction,
    answer, correct, full_correct, candidate_size, read_residual and the mean/max
    measured overlap of the cell's extras (NaN for the baseline). ``full_correct``
    decodes the full vocabulary against the unmodified entity codes in every cell.
    """
    writer = model.config["writer"]
    names = cell_names(writer, loads)
    max_extra = max(loads.values())
    entities = episode_entities(arrays)
    n_facts = arrays.facts.shape[1]
    candidate_heads, tails = draw_extras(arrays, entities, seed=seed, n_extra=max_extra, pool=pool,
                                         candidates=candidates)
    is_extra = slot_layout(arrays.episode_seed, n_facts, max_extra, seed)
    slot_order = shuffle_order(arrays.episode_seed, max_extra, seed)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    previous_mode = model.training
    model.eval()
    device = model.device
    try:
        for begin in range(0, len(arrays.episode_seed), batch_size):
            window = slice(begin, min(begin + batch_size, len(arrays.episode_seed)))
            part = _slice(arrays, window)
            base = model.prepare(part)
            facts = torch.as_tensor(part.facts, dtype=torch.long, device=device)
            ents = torch.as_tensor(entities[window], dtype=torch.long, device=device)
            heads_pool = torch.as_tensor(candidate_heads[window], dtype=torch.long, device=device)
            order = torch.as_tensor(slot_order[window], dtype=torch.long, device=device)
            rows = torch.arange(len(facts), device=device)[:, None]
            arms: dict[str, tuple[np.ndarray, Tensor, Tensor | None]] = {}
            for kind in OVERLAPS:
                if writer == "oracle_random":
                    heads = heads_pool[:, :max_extra]
                    codes = constructed_codes(model.writer.global_codes, facts, ents, heads, kind,
                                              episode_seed=part.episode_seed, seed=seed)
                    rho = overlap(codes[rows, heads], codes[rows, facts[:, :, 0]])
                else:
                    heads, rho = select_by_overlap(model, facts, heads_pool, kind, max_extra, order)
                    codes = None
                arms[kind] = (heads.cpu().numpy(), rho, codes)
            for name in names:
                if name == "base":
                    prepared, extended = base, part
                    mean_rho = max_rho = torch.full((len(facts),), float("nan"), device=device)
                else:
                    load_name, kind = _cell_parts(name)
                    load = loads[load_name]
                    decay = kind == "decay"
                    heads_np, rho, codes = arms["nat" if decay else kind]
                    extended, flags = assemble(part, heads_np, tails[window], is_extra[window], load)
                    prepared = prepare_cell(model, extended, candidate_mask=base.candidate_mask, codes=codes,
                                            decay_slots=flags if decay else None, decode_codes=base.codes)
                    mean_rho, max_rho = rho[:, :load].mean(dim=1), rho[:, :load].amax(dim=1)
                    if decay:  # nothing is written in a decay-only cell, so there is no key overlap to report
                        mean_rho = max_rho = torch.full_like(mean_rho, float("nan"))
                shape = part.answer.shape
                _require_finite(prepared.memory, "memory", f"cell={name}, batch_start={begin}")
                residual = read_residual(model, prepared, extended).cpu().numpy()
                candidate_size = prepared.candidate_mask.sum(dim=-1).cpu().numpy()
                for depth in depths:
                    logits = model.read(prepared, depth)["logits"]
                    _require_finite(logits, "logits", f"cell={name}, L={depth}, batch_start={begin}")
                    restricted = logits.masked_fill(~prepared.candidate_mask[:, None, :], -torch.inf)
                    prediction = restricted.argmax(-1).cpu().numpy()
                    full = logits.argmax(-1).cpu().numpy()
                    values = dict(
                        split=np.full(shape, split), condition=np.full(shape, name),
                        episode_id=np.broadcast_to(part.episode_seed[:, None], shape),
                        query_id=np.broadcast_to(np.arange(shape[1]), shape), hop=part.q_hop_true,
                        L=np.full(shape, depth, dtype=np.int64), prediction=prediction, answer=part.answer,
                        correct=prediction == part.answer, full_correct=full == part.answer,
                        candidate_size=np.broadcast_to(candidate_size[:, None], shape),
                        read_residual=residual,
                        overlap_mean=np.broadcast_to(mean_rho.cpu().numpy()[:, None], shape),
                        overlap_max=np.broadcast_to(max_rho.cpu().numpy()[:, None], shape))
                    for field, value in values.items():
                        chunks[field].append(np.asarray(value).ravel().copy())
    finally:
        model.train(previous_mode)
    return {field: np.concatenate(parts) for field, parts in chunks.items()}


def overlap_summary(samples: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Mean measured overlap per cell (episode-level values, first depth only)."""
    rows = []
    depth = int(samples["L"].min())
    for name in sorted(set(samples["condition"].tolist())):
        mask = (samples["condition"] == name) & (samples["L"] == depth) & (samples["query_id"] == 0)
        mean = samples["overlap_mean"][mask]
        rows.append(dict(condition=name, episodes=int(mask.sum()),
                         overlap_mean=float(np.nanmean(mean)) if np.isfinite(mean).any() else None,
                         overlap_max=float(np.nanmax(samples["overlap_max"][mask])) if np.isfinite(mean).any() else None))
    return rows


def depth_gain(samples_by_seed: Mapping[int, Mapping[str, np.ndarray]], condition: str, *,
               hops: tuple[int, ...] = DECISION_HOPS, split: str = SPLIT, resamples: int = 2000,
               confidence: float = 0.95, label: tuple = (), field: str = "correct") -> dict[str, Any]:
    """Pooled (L4 - L1) accuracy gain of one cell on ``hops`` (joint episode bootstrap) with both accuracies.

    ``field`` is ``correct`` (candidate-restricted) or ``full_correct`` (full-vocabulary exact match).
    """
    aligned = fs.aligned_terms(samples_by_seed, [(split, condition, 4), (split, condition, 1)], field=field)
    keep = np.isin(aligned["hop"], hops)
    values = {seed: (term[0] - term[1])[keep] for seed, term in aligned["values"].items()}
    result = fs.pooled_mean_contrast(values, aligned["episode_id"][keep], resamples=resamples,
                                     confidence=confidence, label=("ivd", "gain", condition, *label))
    result["accuracy_l4"] = float(np.mean([term[0][keep].mean() for term in aligned["values"].values()]))
    result["accuracy_l1"] = float(np.mean([term[1][keep].mean() for term in aligned["values"].values()]))
    return result


def interference_contrast(samples_by_seed: Mapping[int, Mapping[str, np.ndarray]], *, load: str = DECISION_LOAD,
                          hops: tuple[int, ...] = DECISION_HOPS, split: str = SPLIT, resamples: int = 2000,
                          confidence: float = 0.975, label: tuple = ()) -> dict[str, Any]:
    """Registered F8 contrast: (L4 - L1)_low minus (L4 - L1)_high at one load, paired by query.

    A positive value means that the high key-overlap extras lower the depth gain
    more than the low-overlap extras do. ``passes`` is the registered pattern: at
    least 5 pp with a confidence interval lower bound above zero.
    """
    terms = [(split, f"{load}-{kind}", depth) for kind in ("low", "high") for depth in (4, 1)]
    aligned = fs.aligned_terms(samples_by_seed, terms)
    keep = np.isin(aligned["hop"], hops)
    values = {seed: ((term[0] - term[1]) - (term[2] - term[3]))[keep] for seed, term in aligned["values"].items()}
    result = fs.pooled_mean_contrast(values, aligned["episode_id"][keep], resamples=resamples,
                                     confidence=confidence, label=("ivd", "interference", load, *label))
    result["passes"] = fs.passes_gain(result, GAIN_MINIMUM)
    return result


def cell_table(samples_by_seed: Mapping[int, Mapping[str, np.ndarray]], writer: str, *,
               hops: tuple[int, ...] = DECISION_HOPS, split: str = SPLIT, resamples: int = 2000,
               confidence: float = 0.95, field: str = "correct") -> list[dict[str, Any]]:
    """Descriptive rows per cell: depth gain with its interval, accuracies, residual and achieved overlap."""
    rows = []
    for name in cell_names(writer):
        gain = depth_gain(samples_by_seed, name, hops=hops, split=split, resamples=resamples,
                          confidence=confidence, label=(writer,) if field == "correct" else (writer, field),
                          field=field)
        residual, mean_overlap = [], []
        for samples in samples_by_seed.values():
            cell = (np.asarray(samples["split"]) == split) & (np.asarray(samples["condition"]) == name) & (
                np.asarray(samples["L"]) == 1) & np.isin(np.asarray(samples["hop"]), hops)
            residual.append(float(np.nanmean(np.asarray(samples["read_residual"])[cell])))
            overlap_values = np.asarray(samples["overlap_mean"])[cell]
            mean_overlap.append(float(np.nanmean(overlap_values)) if np.isfinite(overlap_values).any() else None)
        rows.append(dict(writer=writer, condition=name, field=field, hops=",".join(str(h) for h in hops),
                         gain=gain["difference"],
                         gain_ci_low=gain["ci_low"], gain_ci_high=gain["ci_high"], confidence=confidence,
                         accuracy_l1=gain["accuracy_l1"], accuracy_l4=gain["accuracy_l4"],
                         read_residual_l1=float(np.mean(residual)),
                         overlap_mean=None if mean_overlap[0] is None else float(np.mean(mean_overlap)),
                         **{f"gain_seed_{entry['seed']}": entry["difference"] for entry in gain["per_seed"]}))
    return rows


def accuracy_table(samples_by_seed: Mapping[int, Mapping[str, np.ndarray]], writer: str, *, depth: int,
                   hops: tuple[int, ...] = DECISION_HOPS, split: str = SPLIT, resamples: int = 2000,
                   confidence: float = 0.95, field: str = "correct") -> list[dict[str, Any]]:
    """Descriptive pooled accuracy of every cell at one depth, with a joint episode bootstrap interval."""
    rows = []
    for name in cell_names(writer):
        aligned = fs.aligned_terms(samples_by_seed, [(split, name, depth)], field=field)
        keep = np.isin(aligned["hop"], hops)
        values = {seed: term[0][keep] for seed, term in aligned["values"].items()}
        label = ("ivd", "accuracy", writer, name, depth, *hops) + (() if field == "correct" else (field,))
        result = fs.pooled_mean_contrast(values, aligned["episode_id"][keep], resamples=resamples,
                                         confidence=confidence, label=label)
        rows.append(dict(writer=writer, condition=name, field=field, L=depth, hops=",".join(str(h) for h in hops),
                         accuracy=result["difference"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                         confidence=confidence))
    return rows
