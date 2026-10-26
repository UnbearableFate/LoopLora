"""NumPy relation episodes with independent, reproducible episode seeds.

See docs/tasks/WP-B_contract.md for rate and counterfactual batch conventions.
All public entity ids belong to the chosen split pool; V denotes insufficient
information and is used only as an answer. Presentation order defines updates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from looplora.infra.seeds import derive_seed, stable_hash

GENERATOR_VERSION = "relations-v1"
INSUFFICIENT = 1024
_TASKS = {"t1", "t2_chain_end", "t2_walk", "t2_typed", "t3_update", "t3_missing"}
_SPLITS = {"train", "val", "test", "test_unseen"}


@dataclass(frozen=True)
class TaskSpec:
    """Fixed episode shapes, disjoint entity pools, and task construction knobs.

    ``chain_lengths`` counts final base edges, excluding obsolete update facts
    and typed distractors. ``hops`` is sampled uniformly. Missingness is an
    episode event anchored on query zero; other overlapping queries may fail too.
    """

    name: str
    E: int = 24
    Q: int = 4
    V: int = 1024
    n_rel: int = 4
    seen: tuple[int, int] = (0, 768)
    unseen: tuple[int, int] = (768, 1024)
    head_sharing: float = 0.0
    chain_lengths: tuple[int, ...] | None = None
    hops: tuple[int, ...] = (1, 2, 3)
    base: str | None = None
    n_distract: int = 0
    n_updates: int = 2
    update_query_rate: float = 0.5
    missing_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.name not in _TASKS:
            raise ValueError(f"unknown task: {self.name!r}")
        for name in ("E", "Q", "V", "n_rel"):
            _integer(name, getattr(self, name), minimum=1)
        if max(self.V, self.n_rel) > np.iinfo(np.int32).max:
            raise ValueError("entity and relation ids must fit int32")
        for name in ("n_distract", "n_updates"):
            _integer(name, getattr(self, name), minimum=0)
        for name in ("head_sharing", "update_query_rate", "missing_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("seen", "unseen"):
            pool = tuple(getattr(self, name))
            if len(pool) != 2:
                raise ValueError(f"{name} must be half-open (start, stop) bounds")
            for bound in pool:
                _integer(name, bound, minimum=0)
            if not 0 <= pool[0] < pool[1] <= self.V:
                raise ValueError(f"invalid {name} entity pool")
            object.__setattr__(self, name, pool)
        if max(self.seen[0], self.unseen[0]) < min(self.seen[1], self.unseen[1]):
            raise ValueError("seen and unseen entity pools must be disjoint")
        hops = tuple(self.hops)
        if not hops or len(set(hops)) != len(hops):
            raise ValueError("hops must be nonempty and distinct")
        for hop in hops:
            _integer("hop", hop, minimum=1)
        object.__setattr__(self, "hops", hops)
        allowed_bases = {
            "t3_update": ("t1", "t2_chain_end"),
            "t3_missing": ("t2_walk", "t1"),
        }
        if self.name in allowed_bases:
            base = self.base or allowed_bases[self.name][0]
            if base not in allowed_bases[self.name]:
                raise ValueError(f"invalid base {base!r} for {self.name}")
            object.__setattr__(self, "base", base)
        elif self.base is not None:
            raise ValueError("base is only defined for t3 tasks")
        if self.n_distract and self.name != "t2_typed":
            raise ValueError("n_distract is only defined for t2_typed")
        if self.name == "t2_typed" and self.n_distract and self.n_rel < 2:
            raise ValueError("typed distractors require n_rel >= 2")
        if self.head_sharing and self.n_rel < 2:
            raise ValueError("head sharing requires n_rel >= 2")
        base_edges = self.E
        if self.name == "t3_update":
            base_edges -= self.n_updates
            if self.n_updates > base_edges:
                raise ValueError("each updated key has exactly two occurrences")
            if self.update_query_rate and not self.n_updates:
                raise ValueError("updated queries require at least one update")
            if self.base == "t1" and self.update_query_rate < 1 and base_edges == self.n_updates:
                raise ValueError("untouched queries require an untouched key")
        if self.name == "t2_typed":
            base_edges -= self.n_distract
        if base_edges < 1:
            raise ValueError("at least one base edge is required")
        if self.kind != "t1":
            lengths = (base_edges,) if self.chain_lengths is None else tuple(self.chain_lengths)
            if not lengths:
                raise ValueError("at least one chain is required")
            for length in lengths:
                _integer("chain length", length, minimum=1)
            if sum(lengths) != base_edges:
                raise ValueError("chain_lengths must sum to the number of final base edges")
            if max(hops) > max(lengths):
                raise ValueError("a requested hop exceeds every chain length")
            if self.name == "t2_typed":
                if self.n_distract > (base_edges + len(lengths)) * (self.n_rel - 1):
                    raise ValueError("too many distinct typed distractor keys")
            if self.name == "t3_update" and self.n_updates:
                eligible = [i for i, length in enumerate(lengths) if length >= max(hops)]
                if self.update_query_rate < 1:
                    clean = min(eligible, key=lambda i: lengths[i])
                    if self.n_updates > base_edges - lengths[clean]:
                        raise ValueError("not enough update keys outside an untouched eligible chain")
                    if self.update_query_rate > 0 and len(eligible) < 2:
                        raise ValueError("mixed update queries need two chains covering every hop")
            object.__setattr__(self, "chain_lengths", lengths)
        elif self.chain_lengths is not None:
            raise ValueError("t1 bases do not use chain_lengths")

    @property
    def kind(self) -> str:
        """Question semantics, independent of interference or history."""
        if self.name in ("t3_update", "t3_missing"):
            return self.base or ("t1" if self.name == "t3_update" else "t2_walk")
        return "t2_chain_end" if self.name == "t2_typed" else self.name

    @property
    def insufficient(self) -> int:
        return self.V


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class EpisodeArrays:
    facts: NDArray[np.int32]
    fact_valid: NDArray[np.bool_]
    q_start: NDArray[np.int32]
    q_rel: NDArray[np.int32]
    q_hop_given: NDArray[np.int32]
    q_hop_true: NDArray[np.int32]
    answer: NDArray[np.int32]
    episode_seed: NDArray[np.int64]
    meta: dict[str, Any]


@dataclass(frozen=True)
class _Layout:
    facts: NDArray[np.int32]
    node_count: int
    chain_nodes: int
    offsets: NDArray[np.int32]
    edge_offsets: NDArray[np.int32]
    lengths: NDArray[np.int32]
    eligible: NDArray[np.int32]
    n_eligible: NDArray[np.int32]
    digest: str


@lru_cache(maxsize=128)
def _layout(spec: TaskSpec) -> _Layout:
    edges = spec.E - (spec.n_updates if spec.name == "t3_update" else 0)
    edges -= spec.n_distract if spec.name == "t2_typed" else 0
    if spec.kind == "t1":
        facts = np.zeros((edges, 3), dtype=np.int32)
        facts[:, 0] = np.arange(edges)
        facts[:, 2] = np.arange(edges, 2 * edges)
        empty = np.empty(0, dtype=np.int32)
        lengths = offsets = edge_offsets = n_eligible = empty
        eligible = np.empty((0, 0), dtype=np.int32)
        node_count = chain_nodes = 2 * edges
    else:
        lengths = np.asarray(spec.chain_lengths, dtype=np.int32)
        offsets = np.cumsum(np.r_[0, lengths[:-1] + 1], dtype=np.int32)
        edge_offsets = np.cumsum(np.r_[0, lengths[:-1]], dtype=np.int32)
        heads = np.concatenate([np.arange(o, o + n) for o, n in zip(offsets, lengths)])
        facts = np.zeros((edges, 3), dtype=np.int32)
        facts[:, 0], facts[:, 2] = heads, heads + 1
        chain_nodes = edges + len(lengths)
        node_count = chain_nodes + (spec.n_distract if spec.name == "t2_typed" else 0)
        eligible = np.zeros((len(spec.hops), len(lengths)), dtype=np.int32)
        n_eligible = np.zeros(len(spec.hops), dtype=np.int32)
        for i, hop in enumerate(spec.hops):
            choices = np.flatnonzero(lengths >= hop)
            eligible[i, :len(choices)] = choices
            n_eligible[i] = len(choices)
    if spec.name == "t3_update":
        node_count += spec.n_updates
    if spec.name == "t3_missing":
        node_count += 2
    return _Layout(facts, node_count, chain_nodes, offsets, edge_offsets, lengths,
                   eligible, n_eligible, stable_hash(asdict(spec)))


def _empty(spec: TaskSpec, count: int, meta: dict[str, Any]) -> EpisodeArrays:
    def questions(fill: int = 0) -> NDArray[np.int32]:
        return np.full((count, spec.Q), fill, dtype=np.int32)

    return EpisodeArrays(
        facts=np.empty((count, spec.E, 3), dtype=np.int32),
        fact_valid=np.ones((count, spec.E), dtype=np.bool_),
        q_start=questions(), q_rel=questions(), q_hop_given=questions(-1),
        q_hop_true=questions(), answer=questions(),
        episode_seed=np.empty(count, dtype=np.int64), meta=meta,
    )


def _t1_facts(spec: TaskSpec, layout: _Layout, rng: np.random.Generator) -> NDArray[np.int32]:
    facts = layout.facts.copy()
    edges = len(facts)
    if spec.head_sharing == 0:
        facts[:, 1] = rng.integers(spec.n_rel, size=edges)
        return facts
    available: dict[int, list[int]] = {}
    next_head = 0
    for i in range(edges):
        if available and rng.random() < spec.head_sharing:
            heads = tuple(available)
            head = heads[int(rng.integers(len(heads)))]
        else:
            head = next_head
            next_head += 1
            available[head] = list(range(spec.n_rel))
        relations = available[head]
        rel = relations.pop(int(rng.integers(len(relations))))
        if not relations:
            del available[head]
        facts[i, :2] = head, rel
    return facts


def _chain_queries(
    spec: TaskSpec, layout: _Layout, rng: np.random.Generator,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    draws = rng.random((3, spec.Q))
    hop_index = (draws[0] * len(spec.hops)).astype(np.int32)
    hops = np.asarray(spec.hops, dtype=np.int32)[hop_index]
    chain_index = (draws[1] * layout.n_eligible[hop_index]).astype(np.int32)
    chains = layout.eligible[hop_index, chain_index]
    lengths = layout.lengths[chains]
    positions = lengths - hops
    if spec.kind == "t2_walk":
        positions = (draws[2] * (positions + 1)).astype(np.int32)
    starts = layout.offsets[chains] + positions
    return starts, starts + hops, hops


def _update_keys(spec: TaskSpec, layout: _Layout, rng: np.random.Generator) -> NDArray[np.int32]:
    edges = len(layout.facts)
    if not spec.n_updates:
        return np.empty(0, dtype=np.int32)
    if spec.kind == "t1":
        return rng.choice(edges, spec.n_updates, replace=False).astype(np.int32)
    eligible = np.flatnonzero(layout.lengths >= max(spec.hops)).tolist()
    candidates = np.ones(edges, dtype=np.bool_)
    if spec.update_query_rate < 1:
        clean = min(eligible, key=lambda i: layout.lengths[i])
        begin = layout.edge_offsets[clean]
        candidates[begin:begin + layout.lengths[clean]] = False
        eligible.remove(clean)
    forced: list[int] = []
    if spec.update_query_rate > 0:
        chain = int(rng.choice(eligible))
        edge = int(layout.edge_offsets[chain] + layout.lengths[chain] - 1)
        forced = [edge]
        candidates[edge] = False
    rest = rng.choice(np.flatnonzero(candidates), spec.n_updates - len(forced), replace=False)
    return np.asarray([*forced, *rest], dtype=np.int32)


def _updated_queries(
    spec: TaskSpec, layout: _Layout, facts: NDArray[np.int32],
    updated: NDArray[np.int32], rng: np.random.Generator,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    wanted = rng.random(spec.Q) < spec.update_query_rate
    flags = np.zeros(len(facts), dtype=np.bool_)
    flags[updated] = True
    if spec.kind == "t1":
        selected = np.empty(spec.Q, dtype=np.int32)
        for category in (False, True):
            mask = wanted == category
            if mask.any():
                selected[mask] = rng.choice(np.flatnonzero(flags == category), int(mask.sum()))
        chosen = facts[selected]
        return chosen[:, 0], chosen[:, 1], chosen[:, 2], np.ones(spec.Q, dtype=np.int32)
    hops = rng.choice(np.asarray(spec.hops, dtype=np.int32), spec.Q)
    starts = np.empty(spec.Q, dtype=np.int32)
    for h in spec.hops:
        chains = np.flatnonzero(layout.lengths >= h)
        affected = np.asarray([
            flags[layout.edge_offsets[c] + layout.lengths[c] - h:
                  layout.edge_offsets[c] + layout.lengths[c]].any() for c in chains
        ])
        for category in (False, True):
            mask = (hops == h) & (wanted == category)
            if mask.any():
                selected = rng.choice(chains[affected == category], int(mask.sum()))
                starts[mask] = layout.offsets[selected] + layout.lengths[selected] - h
    return starts, np.zeros(spec.Q, dtype=np.int32), starts + hops, hops


def generate(spec: TaskSpec, split: str, start: int, count: int, base_seed: int) -> EpisodeArrays:
    """Generate independent episodes with disjoint fact-set partitions by split.

    For seen splits, the minimum fact entity (relative to the pool start) has
    residue 0/1/2 for train/val/test. This prevents cross-split duplicates even
    in small graph spaces, without retaining a dataset-wide rejection cache.
    """
    if split not in _SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    _integer("start", start, 0)
    _integer("count", count, 0)
    _integer("base_seed", base_seed, 0)
    layout = _layout(spec)
    low, high = spec.unseen if split == "test_unseen" else spec.seen
    split_offset = {"train": 0, "val": 1, "test": 2, "test_unseen": 0}[split]
    required = layout.node_count + split_offset
    if high - low < required:
        raise ValueError(f"{split} pool needs at least {required} entities for this spec and split partition")
    arrays = _empty(spec, count, {
        "generator_version": GENERATOR_VERSION, "spec": asdict(spec),
        "split": split, "base_seed": base_seed, "start": start,
        "split_partition": "minimum_fact_entity_modulo_3",
    })
    simple_chain = spec.name in ("t2_chain_end", "t2_walk")
    all_nodes_used = spec.name != "t3_missing" and (spec.kind != "t1" or spec.head_sharing == 0)
    for row, index in enumerate(range(start, start + count)):
        seed = derive_seed(base_seed, "relations", GENERATOR_VERSION, layout.digest, split, index)
        arrays.episode_seed[row] = seed
        rng = np.random.default_rng(seed)
        nodes = rng.choice(high - low - split_offset, layout.node_count, replace=False).astype(np.int32)
        nodes += low + split_offset
        if simple_chain:
            starts, answers, hops = _chain_queries(spec, layout, rng)
            facts = layout.facts
        else:
            facts = _t1_facts(spec, layout, rng) if spec.kind == "t1" else layout.facts.copy()
            if spec.name == "t3_update":
                updated = _update_keys(spec, layout, rng)
                starts, relations, answers, hops = _updated_queries(spec, layout, facts, updated, rng)
                arrays.q_rel[row] = relations
                obsolete = facts[updated].copy()
                obsolete[:, 2] = np.arange(layout.node_count - spec.n_updates, layout.node_count)
                facts = np.concatenate((facts, obsolete))
            elif spec.kind == "t1":
                selected = facts[rng.integers(len(facts), size=spec.Q)]
                starts, answers = selected[:, 0], selected[:, 2]
                arrays.q_rel[row] = selected[:, 1]
                hops = np.ones(spec.Q, dtype=np.int32)
            else:
                starts, answers, hops = _chain_queries(spec, layout, rng)
            if spec.name == "t2_typed" and spec.n_distract:
                keys = rng.choice(layout.chain_nodes * (spec.n_rel - 1), spec.n_distract, replace=False)
                distract = np.empty((spec.n_distract, 3), dtype=np.int32)
                distract[:, 0] = keys // (spec.n_rel - 1)
                distract[:, 1] = keys % (spec.n_rel - 1) + 1
                distract[:, 2] = np.arange(layout.chain_nodes, layout.node_count)
                facts = np.concatenate((facts, distract))
            if spec.name == "t3_missing" and rng.random() < spec.missing_rate:
                key_head = int(starts[0])
                if spec.kind == "t2_walk":
                    key_head += int(rng.integers(int(hops[0])))
                key_rel = arrays.q_rel[row, 0]
                match = np.flatnonzero((facts[:, 0] == key_head) & (facts[:, 1] == key_rel))[0]
                facts[match] = layout.node_count - 2, 0, layout.node_count - 1
                for q in range(spec.Q):
                    if spec.kind == "t1":
                        missing = starts[q] == key_head and arrays.q_rel[row, q] == key_rel
                    else:
                        missing = starts[q] <= key_head < starts[q] + hops[q]
                    if missing:
                        answers[q] = -1
        if split != "test_unseen":
            if all_nodes_used:
                minimum = int(nodes.argmin())
            else:
                used = facts[:, (0, 2)].ravel()
                minimum = int(used[nodes[used].argmin()])
            old = int(nodes[minimum])
            new = old - (old - low - split_offset) % 3
            if new != old:
                # An unused local node may own this label; swap to keep the
                # renaming injective, including missing queries' orphan nodes.
                if not all_nodes_used:
                    nodes[nodes == new] = old
                nodes[minimum] = new
        order = rng.permutation(spec.E)
        if spec.name == "t3_update" and spec.n_updates:
            positions = np.argsort(order)
            for j, edge in enumerate(updated):
                old = len(layout.facts) + j
                if positions[old] > positions[edge]:
                    a, b = positions[old], positions[edge]
                    order[a], order[b] = order[b], order[a]
        ordered = facts[order]
        arrays.facts[row, :, 0] = nodes[ordered[:, 0]]
        arrays.facts[row, :, 1] = ordered[:, 1]
        arrays.facts[row, :, 2] = nodes[ordered[:, 2]]
        arrays.q_start[row] = nodes[starts]
        arrays.q_hop_true[row] = hops
        if spec.kind in ("t1", "t2_walk"):
            arrays.q_hop_given[row] = hops
        arrays.answer[row] = nodes[answers]
        arrays.answer[row, answers == -1] = spec.V
    return arrays


def _walk(
    facts: NDArray[np.int32], valid: NDArray[np.bool_], start: int, relation: int,
    hops: int, insufficient: int,
) -> tuple[int, list[tuple[int, int]]]:
    latest = {(int(h), int(r)): int(t) for (h, r, t), ok in zip(facts, valid) if ok}
    path: list[tuple[int, int]] = []
    visited: set[int] = set()
    node = start
    while hops < 0 or len(path) < hops:
        if node in visited:
            raise ValueError("counterfactual source contains a cycle")
        visited.add(node)
        key = (node, relation)
        if key not in latest:
            return (node if hops < 0 else insufficient), path
        path.append(key)
        node = latest[key]
    return node, path


def _fresh(
    facts: NDArray[np.int32], queries: NDArray[np.int32], pool: tuple[int, int],
    count: int, rng: np.random.Generator,
) -> NDArray[np.int32]:
    used = np.concatenate((facts[:, 0], facts[:, 2], queries))
    choices = np.setdiff1d(np.arange(*pool, dtype=np.int32), used)
    if len(choices) < count:
        raise ValueError(f"counterfactual needs {count} unused entities in its split pool")
    return rng.choice(choices, count, replace=False)


def counterfactual(
    arrays: EpisodeArrays, kind: str, seed: int,
) -> tuple[EpisodeArrays, NDArray[np.int32]]:
    """Fork answerable questions for path edits, or derange whole-episode memories.

    Path edits return N*Q rows with one query each, in episode-major order.
    ``meta['source_episode']`` and ``meta['source_query']`` identify the originals.
    Swap keeps the original shape and records ``meta['pairing']``. Its returned
    labels are the new oracle answers; the control must be scored against the
    *original* ``arrays.answer``. No input arrays or metadata are mutated.
    """
    _integer("seed", seed, 0)
    if kind not in ("replace_key_fact", "remove_required_fact", "swap_memory"):
        raise ValueError(f"unknown counterfactual: {kind!r}")
    spec = TaskSpec(**arrays.meta["spec"])
    n, q = arrays.answer.shape
    pool = spec.unseen if arrays.meta["split"] == "test_unseen" else spec.seen
    meta = {**arrays.meta, "counterfactual": {"kind": kind, "seed": seed}}
    if kind == "swap_memory":
        if n < 2:
            raise ValueError("swap_memory needs at least two episodes")
        rng = np.random.default_rng(derive_seed(seed, "counterfactual", kind))
        order = rng.permutation(n)
        pairing = np.empty(n, dtype=np.int64)
        pairing[order] = np.roll(order, 1)
        meta["pairing"] = pairing.tolist()
        result = _empty(spec, n, meta)
        result.facts[:] = arrays.facts[pairing]
        result.fact_valid[:] = arrays.fact_valid[pairing]
        for field in ("q_start", "q_rel", "q_hop_given", "q_hop_true"):
            getattr(result, field)[:] = getattr(arrays, field)
        for i in range(n):
            result.episode_seed[i] = derive_seed(seed, "counterfactual", kind, int(arrays.episode_seed[i]))
            for j in range(q):
                answer, path = _walk(result.facts[i], result.fact_valid[i], int(result.q_start[i, j]),
                                     int(result.q_rel[i, j]), int(result.q_hop_given[i, j]), spec.V)
                result.answer[i, j] = answer
                if result.q_hop_given[i, j] < 0:
                    result.q_hop_true[i, j] = len(path)
        return result, result.answer.copy()
    fork_spec = replace(spec, Q=1)
    meta["spec"] = asdict(fork_spec)
    meta["source_episode"] = np.repeat(np.arange(n), q).tolist()
    meta["source_query"] = np.tile(np.arange(q), n).tolist()
    result = _empty(fork_spec, n * q, meta)
    for i in range(n):
        for j in range(q):
            row = i * q + j
            derived = derive_seed(seed, "counterfactual", kind, int(arrays.episode_seed[i]), j)
            rng = np.random.default_rng(derived)
            result.episode_seed[row] = derived
            facts = arrays.facts[i].copy()
            valid = arrays.fact_valid[i].copy()
            start, relation, hops = (int(getattr(arrays, name)[i, j])
                                     for name in ("q_start", "q_rel", "q_hop_given"))
            original, path = _walk(facts, valid, start, relation, hops, spec.V)
            if original == spec.V or not path:
                raise ValueError(f"path counterfactual needs an answerable nonempty path (episode {i}, query {j})")
            key = path[-1] if hops >= 0 else path[int(rng.integers(len(path)))]
            matches = np.flatnonzero(valid & (facts[:, 0] == key[0]) & (facts[:, 1] == key[1]))
            if kind == "replace_key_fact":
                fresh = _fresh(facts, arrays.q_start[i], pool, 1, rng)
                facts[matches[-1], 2] = fresh[0]
            else:
                fresh = _fresh(facts, arrays.q_start[i], pool, 2 * len(matches), rng)
                for k, position in enumerate(matches):
                    facts[position] = fresh[2 * k], relation, fresh[2 * k + 1]
            answer, new_path = _walk(facts, valid, start, relation, hops, spec.V)
            result.facts[row] = facts
            result.fact_valid[row] = valid
            result.q_start[row, 0] = start
            result.q_rel[row, 0] = relation
            result.q_hop_given[row, 0] = hops
            result.q_hop_true[row, 0] = len(new_path) if hops < 0 else arrays.q_hop_true[i, j]
            result.answer[row, 0] = answer
    return result, result.answer.copy()
