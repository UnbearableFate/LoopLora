"""P1L-C: read-position and bridge-state patching of trained oracle readers (evaluation only).

Registered in docs/prereg/P1L-C.md. Nothing here changes the deployed read path and nothing in
training, the P1-L confirmation or its analysis imports this module. ``read_control`` wraps ONE
adapter instance for the duration of one scoring call (the adapter classes, ``core`` and
``llm.inject`` are untouched) and restores it on exit.

Terms. A row is one scored question (episode-major B*Q flattening). Its final position is its last
real question token, the one whose hidden state predicts the answer. z_s is the reader state after
read s; the module reads the fixed memory M at every real question token.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from looplora.data.relations import EpisodeArrays, TaskSpec
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed, stable_hash
from looplora.sym.evaluate import tensor_hash

from .batching import tokenize_batch
from .evaluate import _slice

# docs/prereg/P1L-C.md: test [102000, 104000), eval base seed 4242; P1-L reads [100000, 102000).
REGISTERED_RANGE = dict(split="test", start=102000, count=2000, seed=4242)
# Timing/pipeline smoke: validation episodes only, never the registered range.
SMOKE_RANGE = dict(split="val", start=0, count=64, seed=4242)
READ_HOPS = (1, 2, 3)
PATCH_HOP = 2
PATCH_EPISODES = 256
SMOKE_PATCH_EPISODES = 16
PATCH_CONDITIONS = ("world_A", "world_B", "world_C", "B_final", "C_final", "R_final",
                    "B_all", "C_all", "R_all")
RANDOM_POSITIONS = 128


# ---------------------------------------------------------------- reader states and read control

def read_states(adapter, u, memory, steps, *, replace_first=None):
    """States z_1..z_steps of an M1/B4 (iterq) or B8 (unshared) reader on memory ``memory``.

    The loop is ``IterativeQueryAdapter._read`` / ``UnsharedAdapter._read`` with the same tensor
    operations, so the last state equals ``adapter.read`` bit for bit (tests/test_p1lc.py).
    ``replace_first`` maps z_1 to the state that steps 2..steps start from.
    """
    variant = adapter.cfg.variant
    if variant not in {"iterq", "unshared"} or steps < 1:
        raise ValueError("state patching is defined for M1/B4/B8 readers with at least one read")
    z, states = u, []
    for step in range(steps):
        R, U, Q = ((adapter.R[step], adapter.U[step], adapter.Q[step]) if variant == "unshared"
                   else (adapter.R, adapter.U, adapter.Q))
        z = u + adapter.cfg.rho * adapter._phi(F.linear(z, R) + adapter._query(z, memory, Q, U))
        if step == 0 and replace_first is not None:
            z = replace_first(z)
        states.append(z)
    return states


def last_positions(mask):
    """Index of the last real token of every row of a [B, Q, T] or [R, T] question mask."""
    flat = mask.reshape(-1, mask.shape[-1])
    index = torch.arange(flat.shape[1], device=flat.device)
    return torch.where(flat, index, -1).amax(-1)


@contextmanager
def read_control(adapter, last, *, last_only=False, capture=None, donor=None, where="final"):
    """Control the adapter's memory reads during ONE scoring forward of the rows ``last`` describes.

    last_only  memory is zero at every position except each row's final one (the module still runs
               its memory-free branch there). Analysis A, condition (b).
    capture    a dict that receives, per position t, a [rows, r] tensor with z_1 of every row that
               is real at t (zeros elsewhere); deployed outputs are unchanged. See stack_states.
    donor      a [rows, T, r] tensor: z_1 is replaced by donor[:, t] at the final position of every
               row (where="final") or at every real position (where="all"); steps 2..L continue
               from the replaced state on the recipient's own memory.
    Entering the context with none of these options only counts positions. On exit the adapter has
    no instance attributes left over from this context.
    """
    if where not in {"final", "all"}:
        raise ValueError("where must be 'final' or 'all'")
    if sum(option is not None and option is not False for option in (capture, donor, last_only or None)) > 1:
        raise ValueError("last_only, capture and donor are separate passes")
    final_positions = {int(position) for position in last.detach().cpu().tolist()}
    cursor = SimpleNamespace(t=-1, final=None, rows=None)
    process, masked, read = adapter.process_sequence, adapter._masked_branch, adapter._read
    needs_rows = capture is not None or donor is not None

    def process_sequence(x, M0=None, loop_steps=1, valid=None, write=None, reset=None):
        if x.shape[0] != len(last):
            raise ValueError("read control was built for a different number of question rows")
        if donor is not None and tuple(donor.shape[:2]) != tuple(x.shape[:2]):
            raise ValueError("donor states do not match the scored question tokens")
        cursor.t = -1
        if capture is not None:
            capture.clear()
        return process(x, M0, loop_steps, valid, write, reset)

    def masked_branch(u, memory, loop_steps, valid):
        if u.ndim != 2:
            raise ValueError("read control needs the per-position reader path (process_sequence)")
        cursor.t += 1
        cursor.final = last == cursor.t
        # The same row selection _masked_branch makes; u[:, t] rows are the flat row indices.
        cursor.rows = valid.reshape(-1).nonzero(as_tuple=True)[0] if needs_rows else None
        if last_only and memory is not None:
            memory = torch.where(cursor.final[:, None, None], memory, torch.zeros_like(memory))
        return masked(u, memory, loop_steps, valid)

    def replicated_read(u, memory, loop_steps):
        t, rows = cursor.t, cursor.rows
        if capture is None and where == "final" and t not in final_positions:
            return read(u, memory, loop_steps)

        def first(z):
            if capture is not None:
                capture[t] = z.new_zeros((len(last), z.shape[-1])).index_copy(0, rows, z)
                return z
            hit = (cursor.final[rows] if where == "final"
                   else torch.ones(len(rows), dtype=torch.bool, device=z.device))
            return torch.where(hit[:, None], donor[:, t].index_select(0, rows), z)

        return read_states(adapter, u, memory, loop_steps, replace_first=first)[-1]

    with ExitStack() as stack:
        stack.enter_context(patch.object(adapter, "process_sequence", process_sequence))
        stack.enter_context(patch.object(adapter, "_masked_branch", masked_branch))
        if needs_rows:
            stack.enter_context(patch.object(adapter, "_read", replicated_read))
        yield


def stack_states(captured, rows, length, width, device):
    """[rows, length, width] first-read states from the dict ``read_control`` filled."""
    zero = torch.zeros((rows, width), dtype=torch.float32, device=device)
    return torch.stack([captured.get(t, zero) for t in range(length)], dim=1)


def random_units(indices, length, width, base_seed):
    """Unit random directions [len(indices), length, width] for the random-direction control.

    Row i depends on the absolute episode index i, the position and ``width`` only (a fixed table of
    RANDOM_POSITIONS positions is drawn and cut), not on the batch the episode happens to be in.
    """
    if length > RANDOM_POSITIONS:
        raise ValueError(f"questions longer than {RANDOM_POSITIONS} tokens need a larger random-direction table")
    result = np.empty((len(indices), length, width), dtype=np.float32)
    for row, index in enumerate(indices):
        rng = np.random.default_rng(derive_seed(base_seed, "p1lc", "random-direction", int(index)))
        draw = rng.standard_normal((RANDOM_POSITIONS, width))[:length]
        result[row] = draw / np.linalg.norm(draw, axis=-1, keepdims=True)
    return torch.from_numpy(result)


@torch.no_grad()
def score_logits(experiment, prepared, depth, control=None):
    """Answer logits [B, Q, V]; ``control`` holds read_control options, None is the plain deployed path."""
    if control is None:
        return experiment.score(prepared, depth)[0]
    last = last_positions(prepared.question_mask)
    with read_control(experiment.adapter, last, **control):
        return experiment.score(prepared, depth)[0]


# ------------------------------------------------------------------------------ episodes and worlds

def keep_queries(arrays: EpisodeArrays, hops) -> EpisodeArrays:
    """Episodes restricted to their queries of the given hops (one query per hop, evaluation design)."""
    columns = []
    for hop in hops:
        found = np.flatnonzero(arrays.q_hop_true[0] == hop)
        if len(found) != 1:
            raise ValueError(f"evaluation episodes need exactly one hop-{hop} query")
        columns.append(int(found[0]))
    pick = lambda value: value[:, columns]  # noqa: E731
    meta = {**arrays.meta, "spec": {**arrays.meta["spec"], "Q": len(columns)}}
    return replace(arrays, q_start=pick(arrays.q_start), q_rel=pick(arrays.q_rel),
                   q_hop_given=pick(arrays.q_hop_given), q_hop_true=pick(arrays.q_hop_true),
                   answer=pick(arrays.answer), meta=meta)


def chain_paths(facts, valid):
    """Node lists of the relation-0 chains, root first (T2 episodes: one outgoing fact per node)."""
    edges = {int(h): int(t) for (h, r, t), ok in zip(facts, valid) if ok and int(r) == 0}
    tails = set(edges.values())
    paths = []
    for root in sorted(h for h in edges if h not in tails):
        path = [root]
        while path[-1] in edges:
            path.append(edges[path[-1]])
            if len(path) > len(edges) + 1:
                raise ValueError("episode graph contains a cycle")
        paths.append(path)
    return paths


@dataclass(frozen=True)
class BridgeWorlds:
    """Episodes with a hop-2 question on the long chain, and the memories of worlds B and C.

    World A is the original episode s -> b_A -> t_A. B and C differ from A only in the tail of the
    question's first edge (s -> b_A): B has s -> b_B, the head of the other chain's final edge
    b_B -> t_B (t_B is that chain's terminal, the answer of a hop-2 question in world B); C has
    s -> c, an entity that has no outgoing fact (and appears in no other fact). Every other fact,
    including b_A -> t_A and b_B -> t_B, is unchanged, so both continuations are in every memory.
    """

    rows: np.ndarray      # episode rows in the evaluation arrays
    index: np.ndarray     # absolute generator index of each episode
    start: np.ndarray
    bridge_a: np.ndarray
    target_a: np.ndarray
    bridge_b: np.ndarray
    target_b: np.ndarray
    unrelated: np.ndarray
    edge: np.ndarray      # fact slot of the first edge
    facts_b: np.ndarray   # [n, E, 3]
    facts_c: np.ndarray

    def __len__(self):
        return len(self.rows)

    def fingerprint(self):
        names = ("rows", "index", "start", "bridge_a", "target_a", "bridge_b", "target_b",
                 "unrelated", "edge", "facts_b", "facts_c")
        return stable_hash({name: getattr(self, name).tolist() for name in names})


def select_bridge_worlds(arrays: EpisodeArrays, spec: TaskSpec, *, first_index, base_seed,
                         hop=PATCH_HOP, limit=PATCH_EPISODES) -> BridgeWorlds:
    """The first ``limit`` episodes (generation order) whose hop-``hop`` question is on the long chain.

    b_B is the only candidate (the other chain's final edge); the unrelated entity c is drawn from
    the seen pool, excluding every entity of the episode, with a seed derived from the episode's
    absolute index. Both depend on the episode only, not on the reader or its training seed.
    """
    if (spec.name != "t2_chain_end" or spec.n_distract or len(spec.chain_lengths) != 2
            or spec.chain_lengths[0] == spec.chain_lengths[1]):
        raise ValueError("bridge worlds need t2_chain_end with two chains of different length")
    found = np.flatnonzero(arrays.q_hop_true[0] == hop)
    if len(found) != 1:
        raise ValueError(f"episodes need exactly one hop-{hop} query")
    column, pool = int(found[0]), np.arange(*spec.seen)
    names = ("rows", "start", "bridge_a", "target_a", "bridge_b", "target_b", "unrelated", "edge")
    kept = {name: [] for name in names}
    worlds_b, worlds_c = [], []
    for row in range(len(arrays.facts)):
        if len(kept["rows"]) == limit:
            break
        facts, valid = arrays.facts[row], arrays.fact_valid[row]
        paths = chain_paths(facts, valid)
        if len(paths) != 2:
            raise ValueError(f"episode row {row} does not have two chains")
        long_path, other = max(paths, key=len), min(paths, key=len)
        start = int(arrays.q_start[row, column])
        if start != long_path[-1 - hop] or int(arrays.q_rel[row, column]) != 0:
            continue
        edge = np.flatnonzero(valid & (facts[:, 0] == start) & (facts[:, 1] == 0))
        if len(edge) != 1 or int(facts[edge[0], 2]) != long_path[-hop]:
            raise ValueError(f"episode row {row}: the first edge of the question is not unique")
        index = first_index + row
        used = np.unique(facts[valid][:, (0, 2)])
        rng = np.random.default_rng(derive_seed(base_seed, "p1lc", "unrelated-entity", int(index)))
        unrelated = int(rng.choice(np.setdiff1d(pool, used)))
        bridge_b, target_b = other[-2], other[-1]
        world_b, world_c = facts.copy(), facts.copy()
        world_b[edge[0], 2], world_c[edge[0], 2] = bridge_b, unrelated
        for name, value in zip(names, (row, start, long_path[-hop], long_path[-1], bridge_b,
                                       target_b, unrelated, int(edge[0]))):
            kept[name].append(value)
        worlds_b.append(world_b)
        worlds_c.append(world_c)
    if len(kept["rows"]) < limit:
        raise ValueError(f"only {len(kept['rows'])} episodes have a hop-{hop} question on the long chain")
    rows = np.asarray(kept["rows"], dtype=np.int64)
    return BridgeWorlds(rows=rows, index=first_index + rows,
                        **{name: np.asarray(kept[name], dtype=np.int64) for name in names[1:]},
                        facts_b=np.stack(worlds_b), facts_c=np.stack(worlds_c))


# ------------------------------------------------------------------------------------ analyses

@torch.no_grad()
def read_position_samples(experiment, arrays, depth, *, batch_size, split="p1lc"):
    """Analysis A: deployed reads versus memory only at each question's final position.

    Scores every query of every episode in the confirmation batch shape and returns flat sample
    arrays in the looplora.sym.stats format for the conditions memory_only/deployed (the plain
    evaluation path) and memory_only/final_only, plus an audit dict. The registered analysis reads
    hops 1-3 of these samples. Raises if the control wrapper without options does not reproduce the
    deployed logits exactly.
    """
    episodes, queries = arrays.answer.shape
    chunks, memory_hashes = {}, []
    for begin in range(0, episodes, batch_size):
        part = _slice(arrays, slice(begin, min(episodes, begin + batch_size)))
        prepared = experiment.prepare(part)
        memory_hashes.append(tensor_hash(prepared.snapshot.states[experiment.layer]))
        answers = prepared.answer_ids.cpu().numpy()
        shape = answers.shape
        deployed = score_logits(experiment, prepared, depth)
        if not torch.equal(deployed, score_logits(experiment, prepared, depth, dict())):
            raise AssertionError("the read-control wrapper changed the deployed logits")
        final_only = score_logits(experiment, prepared, depth, dict(last_only=True))
        for name, logits in (("deployed", deployed), ("final_only", final_only)):
            prediction = logits.argmax(-1).cpu().numpy()
            values = dict(split=np.full(shape, split), condition=np.full(shape, f"memory_only/{name}"),
                          episode_id=np.broadcast_to(part.episode_seed[:, None], shape),
                          query_id=np.broadcast_to(np.arange(queries), shape), hop=part.q_hop_true,
                          L=np.full(shape, depth, dtype=np.int64), answer=answers,
                          prediction=prediction, correct=prediction == answers)
            for key, value in values.items():
                chunks.setdefault(key, []).append(np.asarray(value).ravel().copy())
    samples = {key: np.concatenate(parts) for key, parts in chunks.items()}
    return samples, dict(memory_hashes=memory_hashes)


@torch.no_grad()
def bridge_patch_samples(experiment, arrays, worlds, depth, *, batch_size, base_seed):
    """Analysis B: patch the reader state after the first read (z_1) at the final question position.

    Every hop-2 question is run on world A's memory (deployed), and again with z_1 replaced by z_1 of
    the same row and position computed on world B's memory (B), on world C's memory (C), or by z_1
    of world A plus a random direction of the norm of z_1^B - z_1^A at that position (R). Steps 2..L
    keep world A's memory. "_final" patches the final question position, "_all" every real position.
    Also returns the deployed predictions in worlds A, B and C. Audits (raise on failure): the
    captured-state replay of every world and the identity patch reproduce the plain logits exactly.
    """
    device, width = experiment.device, experiment.adapter.cfg.r
    token_of = np.asarray(load_name_pool().token_ids, dtype=np.int64)
    predictions = {name: [] for name in PATCH_CONDITIONS}
    norms = {name: [] for name in ("norm_a", "norm_b", "norm_c")}
    memory_hashes = []
    for begin in range(0, len(worlds), batch_size):
        end = min(len(worlds), begin + batch_size)
        part = keep_queries(_slice(arrays, worlds.rows[begin:end]), (PATCH_HOP,))
        tokens = tokenize_batch(part, experiment.tokenizer, device=device, include_support=False)
        prepared = {name: experiment.prepare(replace(part, facts=facts), tokens=tokens)
                    for name, facts in (("A", part.facts), ("B", worlds.facts_b[begin:end]),
                                        ("C", worlds.facts_c[begin:end]))}
        base = prepared["A"]
        last = last_positions(base.question_mask)
        rows, length = len(last), base.question_mask.shape[-1]
        if rows != end - begin:
            raise AssertionError("bridge patching scores one question per episode")
        plain = {name: score_logits(experiment, batch, depth) for name, batch in prepared.items()}
        z1 = {}
        for name, batch in prepared.items():
            captured = {}
            logits = score_logits(experiment, batch, depth, dict(capture=captured))
            if not torch.equal(logits, plain[name]):
                raise AssertionError(f"captured first-read replay differs from the deployed read (world {name})")
            z1[name] = stack_states(captured, rows, length, width, device)
        norm = (z1["B"] - z1["A"]).norm(dim=-1)
        units = random_units(worlds.index[begin:end], length, width, base_seed).to(device)
        z1["R"] = z1["A"] + units * norm[..., None]
        for where in ("final", "all"):
            logits = score_logits(experiment, base, depth, dict(donor=z1["A"], where=where))
            if not torch.equal(logits, plain["A"]):
                raise AssertionError(f"patching the deployed state changed the answer (where={where})")
        for name in "ABC":
            predictions[f"world_{name}"].append(plain[name].argmax(-1)[:, 0].cpu().numpy())
        for label in "BCR":
            for where in ("final", "all"):
                logits = score_logits(experiment, base, depth, dict(donor=z1[label], where=where))
                predictions[f"{label}_{where}"].append(logits.argmax(-1)[:, 0].cpu().numpy())
        position = torch.arange(rows, device=device)
        at_end = {name: z1[name][position, last] for name in "ABC"}
        norms["norm_a"].append(at_end["A"].norm(dim=-1).cpu().numpy())
        norms["norm_b"].append((at_end["B"] - at_end["A"]).norm(dim=-1).cpu().numpy())
        norms["norm_c"].append((at_end["C"] - at_end["A"]).norm(dim=-1).cpu().numpy())
        memory_hashes.append(dict(batch_start=begin, **{
            name: tensor_hash(batch.snapshot.states[experiment.layer]) for name, batch in prepared.items()}))
    samples = dict(
        episode_id=np.asarray(arrays.episode_seed[worlds.rows], dtype=np.int64), episode_index=worlds.index,
        start=worlds.start, bridge_a=worlds.bridge_a, target_a=worlds.target_a, bridge_b=worlds.bridge_b,
        target_b=worlds.target_b, unrelated=worlds.unrelated,
        token_t_A=token_of[worlds.target_a], token_t_B=token_of[worlds.target_b],
        token_b_A=token_of[worlds.bridge_a], token_b_B=token_of[worlds.bridge_b],
        token_c=token_of[worlds.unrelated],
        conditions=np.asarray(PATCH_CONDITIONS),
        prediction=np.stack([np.concatenate(predictions[name]) for name in PATCH_CONDITIONS]).astype(np.int64),
        **{name: np.concatenate(values) for name, values in norms.items()})
    return samples, dict(memory_hashes=memory_hashes, worlds_hash=worlds.fingerprint())

