"""Behavioural checks against a small, independent latest-fact dictionary oracle."""

from collections import Counter
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
import random

import numpy as np
import pytest

from looplora.data.relations import INSUFFICIENT, TaskSpec, counterfactual, generate
from looplora.infra.seeds import stable_hash


ARRAY_FIELDS = (
    "facts", "fact_valid", "q_start", "q_rel", "q_hop_given",
    "q_hop_true", "answer", "episode_seed",
)


def _specs():
    return [
        TaskSpec("t1", E=16, Q=8, head_sharing=0.75),
        TaskSpec("t2_chain_end", E=16, Q=8, chain_lengths=(6, 5, 5)),
        TaskSpec("t2_walk", E=16, Q=8, chain_lengths=(6, 5, 5)),
        TaskSpec("t2_typed", E=20, Q=8, chain_lengths=(6, 5, 5), n_distract=4),
        TaskSpec("t3_update", E=18, Q=8, base="t1", n_updates=2),
        TaskSpec("t3_update", E=18, Q=8, base="t2_chain_end",
                 chain_lengths=(6, 5, 5), n_updates=2),
        TaskSpec("t3_missing", E=16, Q=8, base="t1"),
        TaskSpec("t3_missing", E=16, Q=8, base="t2_walk",
                 chain_lengths=(6, 5, 5)),
    ]


def _graph(arrays, episode):
    # Iterating in presentation order gives last-write-wins without any generator helper.
    graph = {}
    for fact, valid in zip(arrays.facts[episode], arrays.fact_valid[episode]):
        if valid:
            head, rel, tail = map(int, fact)
            graph[head, rel] = tail
    return graph


def _is_terminal_query(spec):
    return spec.name in {"t2_chain_end", "t2_typed"} or (
        spec.name == "t3_update" and spec.base == "t2_chain_end"
    )


def _oracle(arrays, spec, episode, query):
    graph = _graph(arrays, episode)
    node = int(arrays.q_start[episode, query])
    rel = int(arrays.q_rel[episode, query])
    path = []
    if _is_terminal_query(spec):
        visited = set()
        while (node, rel) in graph:
            assert node not in visited, "a terminal query entered a cycle"
            visited.add(node)
            path.append((node, rel))
            node = graph[node, rel]
        return node, path
    hops = int(arrays.q_hop_given[episode, query])
    if spec.name == "t1" or spec.base == "t1":
        hops = 1
    assert hops > 0
    for _ in range(hops):
        key = node, rel
        path.append(key)
        if key not in graph:
            return spec.V, path
        node = graph[key]
    return node, path


def _assert_oracle(arrays, spec):
    for episode in range(len(arrays.facts)):
        for query in range(arrays.q_start.shape[1]):
            expected, _ = _oracle(arrays, spec, episode, query)
            assert int(arrays.answer[episode, query]) == expected


def _assert_acyclic(graph, rel=0):
    for head, relation in graph:
        if relation != rel:
            continue
        node, visited = head, set()
        while (node, rel) in graph:
            assert node not in visited, "relation graph contains a cycle"
            visited.add(node)
            node = graph[node, rel]


def _chains(graph, rel=0):
    heads = {head for head, relation in graph if relation == rel}
    tails = {tail for (_, relation), tail in graph.items() if relation == rel}
    chains = {}
    for root in sorted(heads - tails):
        node, path = root, []
        while (node, rel) in graph:
            path.append((node, rel))
            node = graph[node, rel]
        chains[root] = node, path
    return chains


def _assert_arrays_equal(left, right):
    for field in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def _signature(facts):
    canonical = sorted(tuple(map(int, row)) for row in facts)
    return hashlib.blake2b(json.dumps(canonical).encode(), digest_size=16).hexdigest()


@pytest.mark.parametrize("spec", _specs(), ids=lambda spec: f"{spec.name}-{spec.base}")
def test_every_task_matches_independent_oracle_and_array_contract(spec):
    arrays = generate(spec, "train", start=0, count=32, base_seed=37)
    assert arrays.facts.shape == (32, spec.E, 3)
    assert arrays.fact_valid.shape == (32, spec.E)
    assert arrays.facts.dtype == np.int32
    assert arrays.fact_valid.dtype == np.bool_
    assert arrays.fact_valid.all()
    for field in ("q_start", "q_rel", "q_hop_given", "q_hop_true", "answer"):
        assert getattr(arrays, field).shape == (32, spec.Q)
        assert getattr(arrays, field).dtype == np.int32
    assert arrays.episode_seed.shape == (32,)
    assert arrays.episode_seed.dtype == np.int64
    assert (arrays.episode_seed >= 0).all()
    assert ((arrays.answer >= 0) & (arrays.answer <= spec.V)).all()
    assert ((arrays.facts[:, :, 1] >= 0) & (arrays.facts[:, :, 1] < spec.n_rel)).all()
    assert json.loads(json.dumps(arrays.meta["spec"])) == json.loads(json.dumps(asdict(spec)))
    assert arrays.meta["split"] == "train"
    assert arrays.meta["base_seed"] == 37
    assert arrays.meta["start"] == 0
    assert arrays.meta["generator_version"]
    json.dumps(arrays.meta)
    for episode, facts in enumerate(arrays.facts):
        if spec.name != "t3_update":
            assert len(_graph(arrays, episode)) == spec.E
        if spec.name == "t3_missing" and spec.base == "t2_walk":
            _assert_acyclic(_graph(arrays, episode))
    _assert_oracle(arrays, spec)


@pytest.mark.parametrize("spec", _specs(), ids=lambda spec: f"{spec.name}-{spec.base}")
def test_episode_identity_slices_and_split_isolation(spec):
    whole = generate(spec, "train", 0, 24, 81)
    _assert_arrays_equal(whole, generate(spec, "train", 0, 24, 81))
    for start, count in ((0, 1), (3, 7), (23, 1)):
        part = generate(spec, "train", start, count, 81)
        for field in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(part, field), getattr(whole, field)[start:start + count])
        assert part.meta["start"] == start
    all_seeds, all_facts = set(), set()
    for split in ("train", "val", "test", "test_unseen"):
        arrays = generate(spec, split, 0, 24, 81)
        seeds = set(map(int, arrays.episode_seed))
        signatures = {_signature(facts) for facts in arrays.facts}
        assert len(seeds) == 24
        assert all_seeds.isdisjoint(seeds)
        assert all_facts.isdisjoint(signatures)
        all_seeds.update(seeds)
        all_facts.update(signatures)
        low, high = spec.unseen if split == "test_unseen" else spec.seen
        for entities in (arrays.facts[:, :, 0], arrays.facts[:, :, 2], arrays.q_start):
            assert ((entities >= low) & (entities < high)).all()
        answerable = arrays.answer[arrays.answer != spec.V]
        assert ((answerable >= low) & (answerable < high)).all()
    assert not np.array_equal(whole.episode_seed, generate(spec, "train", 0, 24, 82).episode_seed)


def test_tiny_pool_fact_sets_are_partitioned_across_splits():
    spec = TaskSpec("t1", E=1, Q=1, V=16, n_rel=1, seen=(0, 8), unseen=(8, 16))
    signatures = {}
    for split in ("train", "val", "test", "test_unseen"):
        arrays = generate(spec, split, 0, 128, 89)
        signatures[split] = {_signature(facts) for facts in arrays.facts}
        _assert_oracle(arrays, spec)
    for split, facts in signatures.items():
        for other_split, other_facts in signatures.items():
            if split != other_split:
                assert facts.isdisjoint(other_facts)


@pytest.mark.parametrize("name,base,head_sharing", [
    ("t1", None, 1.0),
    ("t3_missing", "t1", 1.0),
    ("t3_missing", "t2_walk", 0.0),
])
def test_partition_uses_present_facts_with_shared_heads_and_missing_edges(name, base, head_sharing):
    spec = TaskSpec(name, E=12, Q=6, V=160, seen=(7, 87), unseen=(90, 160),
                    base=base, head_sharing=head_sharing, missing_rate=1,
                    chain_lengths=(6, 6) if base == "t2_walk" else None)
    signatures = set()
    for offset, split in enumerate(("train", "val", "test")):
        arrays = generate(spec, split, 0, 128, 97)
        minimum_entities = arrays.facts[:, :, (0, 2)].min(axis=(1, 2))
        assert (((minimum_entities - spec.seen[0]) % 3) == offset).all()
        current_signatures = {_signature(facts) for facts in arrays.facts}
        assert signatures.isdisjoint(current_signatures)
        signatures.update(current_signatures)
        for entities in (arrays.facts[:, :, 0], arrays.facts[:, :, 2], arrays.q_start):
            assert ((entities >= spec.seen[0]) & (entities < spec.seen[1])).all()
        _assert_oracle(arrays, spec)


@pytest.mark.parametrize("name", ("t2_chain_end", "t2_walk", "t2_typed"))
def test_disjoint_chains_and_exact_terminal_distance(name):
    lengths = (3, 5, 8)
    n_distract = 5 if name == "t2_typed" else 0
    spec = TaskSpec(name, E=sum(lengths) + n_distract, Q=20,
                    chain_lengths=lengths, hops=(1, 3, 5, 8), n_distract=n_distract)
    arrays = generate(spec, "test", 0, 48, 127)
    saw_nonterminal_walk_answer = False
    saw_shuffled_chain = False
    for episode, facts in enumerate(arrays.facts):
        graph = _graph(arrays, episode)
        assert len(graph) == spec.E
        _assert_acyclic(graph)
        edges = [(int(h), int(t)) for h, r, t in facts if r == 0]
        assert len(edges) == sum(lengths)
        heads, tails = zip(*edges)
        assert len(set(heads)) == len(heads)
        assert len(set(tails)) == len(tails)
        chain_nodes = set(heads) | set(tails)
        assert len(chain_nodes) == sum(lengths) + len(lengths)
        observed_lengths = []
        for root in set(heads) - set(tails):
            node, distance = root, 0
            while (node, 0) in graph:
                node = graph[node, 0]
                distance += 1
            observed_lengths.append(distance)
        assert sorted(observed_lengths) == sorted(lengths)
        distractors = [(int(h), int(r), int(t)) for h, r, t in facts if r != 0]
        assert len(distractors) == n_distract
        assert all(head in chain_nodes and 1 <= rel < spec.n_rel for head, rel, _ in distractors)
        assert (arrays.q_rel[episode] == 0).all()
        positions = {int(h): i for i, (h, r, _) in enumerate(facts) if r == 0}
        saw_shuffled_chain |= any(
            tail in positions and positions[head] > positions[tail] for head, tail in edges
        )
        for query in range(spec.Q):
            answer, path = _oracle(arrays, spec, episode, query)
            assert len(path) == int(arrays.q_hop_true[episode, query])
            if name == "t2_walk":
                assert arrays.q_hop_given[episode, query] == arrays.q_hop_true[episode, query]
                saw_nonterminal_walk_answer |= (answer, 0) in graph
            else:
                assert arrays.q_hop_given[episode, query] == -1
                assert (answer, 0) not in graph
    assert saw_shuffled_chain
    if name == "t2_walk":
        assert saw_nonterminal_walk_answer
    _assert_oracle(arrays, spec)


@pytest.mark.parametrize("name", ("t2_chain_end", "t2_walk", "t2_typed"))
def test_hops_are_uniform_and_do_not_change_context_size(name):
    distractors = 4 if name == "t2_typed" else 0
    spec = TaskSpec(name, E=24 + distractors, Q=8, chain_lengths=(8, 8, 8),
                    hops=(1, 3, 8), n_distract=distractors)
    arrays = generate(spec, "train", 0, 800, 211)
    counts = Counter(map(int, arrays.q_hop_true.flat))
    assert set(counts) == set(spec.hops)
    for count in counts.values():
        assert abs(count / arrays.q_hop_true.size - 1 / len(spec.hops)) < 0.035
    assert (arrays.fact_valid.sum(axis=1) == spec.E).all()


def test_t1_distinct_keys_and_head_sharing_knob():
    for sharing in (0.0, 1.0):
        spec = TaskSpec("t1", E=24, Q=12, head_sharing=sharing)
        arrays = generate(spec, "train", 0, 16, 343)
        for facts in arrays.facts:
            assert len({(int(h), int(r)) for h, r, _ in facts}) == spec.E
            n_heads = len(set(map(int, facts[:, 0])))
            if sharing == 0:
                assert n_heads == spec.E
            else:
                assert n_heads < spec.E
        assert (arrays.q_hop_true == 1).all()
        _assert_oracle(arrays, spec)


@pytest.mark.parametrize("base", ("t1", "t2_chain_end"))
@pytest.mark.parametrize("rate", (0.0, 0.5, 1.0))
def test_updates_preserve_history_and_query_rate(base, rate):
    spec = TaskSpec("t3_update", E=18, Q=8, base=base, n_updates=2,
                    chain_lengths=(8, 8) if base == "t2_chain_end" else None,
                    update_query_rate=rate)
    arrays = generate(spec, "train", 0, 400, 353)
    updated_queries = 0
    for episode, facts in enumerate(arrays.facts):
        history = {}
        for position, (head, rel, tail) in enumerate(facts):
            history.setdefault((int(head), int(rel)), []).append((position, int(tail)))
        updated = {key for key, values in history.items() if len(values) == 2}
        assert len(history) == spec.E - spec.n_updates
        assert len(updated) == spec.n_updates
        assert all(len(values) in (1, 2) for values in history.values())
        final_graph = _graph(arrays, episode)
        for key in updated:
            (old_position, old_tail), (new_position, new_tail) = history[key]
            assert old_position < new_position
            assert old_tail != new_tail
            assert final_graph[key] == new_tail
            assert old_tail not in {head for head, _ in final_graph}
        if base == "t2_chain_end":
            _assert_acyclic(final_graph)
        for query in range(spec.Q):
            _, path = _oracle(arrays, spec, episode, query)
            updated_queries += bool(updated.intersection(path))
    observed_rate = updated_queries / arrays.answer.size
    if rate in (0, 1):
        assert observed_rate == rate
    else:
        assert abs(observed_rate - rate) < 0.05
    _assert_oracle(arrays, spec)


@pytest.mark.parametrize("base", ("t1", "t2_walk"))
@pytest.mark.parametrize("rate", (0.0, 0.5, 1.0))
def test_missing_rate_is_per_episode_and_relabels_all_queries(base, rate):
    spec = TaskSpec("t3_missing", E=16, Q=8, base=base, missing_rate=rate,
                    chain_lengths=(8, 8) if base == "t2_walk" else None)
    arrays = generate(spec, "train", 0, 600, 367)
    observed_rate = float(np.mean(arrays.answer[:, 0] == spec.V))
    if rate in (0, 1):
        assert observed_rate == rate
    else:
        assert abs(observed_rate - rate) < 0.06
    assert (arrays.fact_valid.sum(axis=1) == spec.E).all()
    expected_hops = {1} if base == "t1" else set(spec.hops)
    assert set(map(int, arrays.q_hop_true.flat)) == expected_hops
    _assert_oracle(arrays, spec)


@pytest.mark.parametrize("spec", [replace(spec, missing_rate=0) if spec.name == "t3_missing" else spec
                                  for spec in _specs()],
                         ids=lambda spec: f"{spec.name}-{spec.base}")
@pytest.mark.parametrize("kind", ("replace_key_fact", "remove_required_fact"))
def test_path_counterfactuals_fork_queries_and_recompute_answers(spec, kind):
    original = generate(spec, "test", 0, 12, 421)
    saved = {field: getattr(original, field).copy() for field in ARRAY_FIELDS}
    saved_meta = json.dumps(original.meta, sort_keys=True)
    changed, answers = counterfactual(original, kind, seed=431)
    assert changed.facts.shape == (12 * spec.Q, spec.E, 3)
    assert changed.answer.shape == (12 * spec.Q, 1)
    np.testing.assert_array_equal(answers, changed.answer)
    assert changed.fact_valid.all()
    for entities in (changed.facts[:, :, 0], changed.facts[:, :, 2]):
        assert ((entities >= spec.seen[0]) & (entities < spec.seen[1])).all()
    assert len(changed.meta["source_episode"]) == 12 * spec.Q
    assert len(changed.meta["source_query"]) == 12 * spec.Q
    seen_sources = set()
    for episode in range(len(changed.facts)):
        source_episode = int(changed.meta["source_episode"][episode])
        source_query = int(changed.meta["source_query"][episode])
        seen_sources.add((source_episode, source_query))
        for field in ("q_start", "q_rel", "q_hop_given"):
            assert getattr(changed, field)[episode, 0] == getattr(original, field)[source_episode, source_query]
        expected, new_path = _oracle(changed, spec, episode, 0)
        assert answers[episode, 0] == expected
        if _is_terminal_query(spec):
            assert changed.q_hop_true[episode, 0] == len(new_path)
        else:
            assert changed.q_hop_true[episode, 0] == original.q_hop_true[source_episode, source_query]
        if kind == "replace_key_fact":
            assert expected != original.answer[source_episode, source_query]
            assert expected != spec.V
        elif _is_terminal_query(spec):
            assert expected != original.answer[source_episode, source_query]
            assert expected != spec.V
            _, old_path = _oracle(original, spec, source_episode, source_query)
            assert expected in {head for head, _ in old_path}
        else:
            assert expected == spec.V
        if _is_terminal_query(spec) or spec.name == "t2_walk":
            _assert_acyclic(_graph(changed, episode))
    assert seen_sources == {(episode, query) for episode in range(12) for query in range(spec.Q)}
    for field in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(original, field), saved[field])
    assert json.dumps(original.meta, sort_keys=True) == saved_meta
    repeated, repeated_answers = counterfactual(original, kind, seed=431)
    _assert_arrays_equal(changed, repeated)
    np.testing.assert_array_equal(answers, repeated_answers)


@pytest.mark.parametrize("spec", _specs(), ids=lambda spec: f"{spec.name}-{spec.base}")
def test_swap_memory_is_a_derangement_with_recomputed_labels(spec):
    original = generate(spec, "test", 0, 16, 443)
    saved = {field: getattr(original, field).copy() for field in ARRAY_FIELDS}
    saved_meta = json.dumps(original.meta, sort_keys=True)
    changed, answers = counterfactual(original, "swap_memory", seed=449)
    pairing = list(map(int, changed.meta["pairing"]))
    assert sorted(pairing) == list(range(16))
    assert all(episode != source for episode, source in enumerate(pairing))
    np.testing.assert_array_equal(changed.facts, original.facts[pairing])
    np.testing.assert_array_equal(changed.fact_valid, original.fact_valid[pairing])
    for field in ("q_start", "q_rel", "q_hop_given"):
        np.testing.assert_array_equal(getattr(changed, field), getattr(original, field))
    if _is_terminal_query(spec):
        for episode in range(16):
            for query in range(spec.Q):
                _, path = _oracle(changed, spec, episode, query)
                assert changed.q_hop_true[episode, query] == len(path)
    else:
        np.testing.assert_array_equal(changed.q_hop_true, original.q_hop_true)
    np.testing.assert_array_equal(answers, changed.answer)
    _assert_oracle(changed, spec)
    repeated, _ = counterfactual(original, "swap_memory", seed=449)
    _assert_arrays_equal(changed, repeated)
    for field in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(original, field), saved[field])
    assert json.dumps(original.meta, sort_keys=True) == saved_meta


@pytest.mark.parametrize("name", ("t2_chain_end", "t2_typed"))
@pytest.mark.parametrize("split", ("test", "test_unseen"))
def test_swap_successors_preserves_global_counts_and_changes_each_answer(name, split):
    distractors = 5 if name == "t2_typed" else 0
    spec = TaskSpec(name, E=16 + distractors, Q=8, chain_lengths=(1, 3, 5, 7),
                    hops=(1, 3, 5, 7), n_distract=distractors)
    original = generate(spec, split, 0, 16, 953)
    saved = {field: getattr(original, field).copy() for field in ARRAY_FIELDS}
    saved_meta = json.dumps(original.meta, sort_keys=True)
    changed, answers = counterfactual(original, "swap_successors", seed=967)
    assert changed.facts.shape == (16 * spec.Q, spec.E, 3)
    assert changed.answer.shape == (16 * spec.Q, 1)
    np.testing.assert_array_equal(answers, changed.answer)
    assert changed.meta["spec"] == asdict(replace(spec, Q=1))
    assert changed.meta["counterfactual"] == {"kind": "swap_successors", "seed": 967}
    assert changed.meta["source_episode"] == np.repeat(np.arange(16), spec.Q).tolist()
    assert changed.meta["source_query"] == np.tile(np.arange(spec.Q), 16).tolist()
    saw_new_distance = False
    for row in range(len(changed.facts)):
        episode, query = divmod(row, spec.Q)
        before, after = original.facts[episode], changed.facts[row]
        np.testing.assert_array_equal(changed.fact_valid[row], original.fact_valid[episode])
        np.testing.assert_array_equal(after[:, :2], before[:, :2])
        np.testing.assert_array_equal(after[before[:, 1] != 0], before[before[:, 1] != 0])
        chain_mask = before[:, 1] == 0
        assert Counter(map(int, after[chain_mask, 0])) == Counter(map(int, before[chain_mask, 0]))
        assert Counter(map(int, after[chain_mask, 2])) == Counter(map(int, before[chain_mask, 2]))
        edits = np.flatnonzero(after[:, 2] != before[:, 2])
        assert len(edits) == 2
        assert (before[edits, 1] == 0).all()
        np.testing.assert_array_equal(after[edits, 2], before[edits[::-1], 2])
        graph = _graph(changed, row)
        assert len(graph) == int(changed.fact_valid[row].sum())
        _assert_acyclic(graph)
        old_chains, new_chains = _chains(_graph(original, episode)), _chains(graph)
        assert new_chains.keys() == old_chains.keys()
        assert {terminal for terminal, _ in new_chains.values()} == {
            terminal for terminal, _ in old_chains.values()
        }
        expected, new_path = _oracle(changed, spec, row, 0)
        old_answer, old_path = _oracle(original, spec, episode, query)
        assert answers[row, 0] == expected != old_answer
        assert changed.q_hop_true[row, 0] == len(new_path)
        saw_new_distance |= len(new_path) != len(old_path)
        edited_keys = {tuple(map(int, before[position, :2])) for position in edits}
        assert len(edited_keys.intersection(old_path)) == 1
        other_key = (edited_keys - set(old_path)).pop()
        other_terminal = next(terminal for terminal, path in old_chains.values() if other_key in path)
        assert expected == other_terminal
        for field in ("q_start", "q_rel", "q_hop_given"):
            assert getattr(changed, field)[row, 0] == getattr(original, field)[episode, query]
    assert saw_new_distance
    for field in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(original, field), saved[field])
    assert json.dumps(original.meta, sort_keys=True) == saved_meta
    repeated, repeated_answers = counterfactual(original, "swap_successors", seed=967)
    _assert_arrays_equal(changed, repeated)
    assert repeated.meta == changed.meta
    np.testing.assert_array_equal(repeated_answers, answers)


def test_swap_successors_samples_chains_and_each_edge_uniformly():
    spec = TaskSpec("t2_chain_end", E=12, Q=1, chain_lengths=(4, 2, 6), hops=(4,))
    original = generate(spec, "train", 0, 1, 971)
    chains = _chains(_graph(original, 0))
    root, (terminal, source_path) = next((root, chain) for root, chain in chains.items() if len(chain[1]) == 4)
    original.q_start[0, 0], original.answer[0, 0] = root, terminal
    other_paths = [path for other_terminal, path in chains.values() if other_terminal != terminal]
    partner_counts, source_counts = Counter(), Counter()
    partner_edges = [Counter() for _ in other_paths]
    for seed in range(1536):
        changed, _ = counterfactual(original, "swap_successors", seed)
        edits = np.flatnonzero(changed.facts[0, :, 2] != original.facts[0, :, 2])
        keys = [tuple(map(int, original.facts[0, position, :2])) for position in edits]
        source_key = next(key for key in keys if key in source_path)
        partner_key = next(key for key in keys if key not in source_path)
        partner = next(i for i, path in enumerate(other_paths) if partner_key in path)
        partner_counts[partner] += 1
        source_counts[source_path.index(source_key)] += 1
        partner_edges[partner][other_paths[partner].index(partner_key)] += 1
    for partner, path in enumerate(other_paths):
        assert abs(partner_counts[partner] / 1536 - 0.5) < 0.07
        for edge in range(len(path)):
            assert abs(partner_edges[partner][edge] / partner_counts[partner] - 1 / len(path)) < 0.07
    for edge in range(len(source_path)):
        assert abs(source_counts[edge] / 1536 - 1 / len(source_path)) < 0.07


@pytest.mark.parametrize("name", ("t2_chain_end", "t2_typed"))
def test_swap_successors_needs_no_unused_entities(name):
    spec = TaskSpec(name, E=2, Q=2, V=8, seen=(0, 4), unseen=(4, 8),
                    chain_lengths=(1, 1), hops=(1,))
    original = generate(spec, "train", 0, 1, 977)
    changed, answers = counterfactual(original, "swap_successors", seed=983)
    for row in range(spec.Q):
        np.testing.assert_array_equal(changed.facts[row, :, 2], original.facts[0, ::-1, 2])
        assert answers[row, 0] != original.answer[0, row]
    assert (changed.q_hop_true == 1).all()
    _assert_oracle(changed, spec)


@pytest.mark.parametrize("name", ("t2_chain_end", "t2_typed"))
def test_swap_successors_rejects_single_chain(name):
    spec = TaskSpec(name, E=3, Q=1, chain_lengths=(3,))
    original = generate(spec, "test", 0, 1, 991)
    with pytest.raises(ValueError, match="swap_successors.*at least two chains"):
        counterfactual(original, "swap_successors", seed=997)


@pytest.mark.parametrize("spec", [spec for spec in _specs() if spec.name not in ("t2_chain_end", "t2_typed")],
                         ids=lambda spec: f"{spec.name}-{spec.base}")
def test_swap_successors_rejects_other_tasks(spec):
    original = generate(spec, "test", 0, 1, 1009)
    with pytest.raises(ValueError, match="swap_successors.*t2_chain_end.*t2_typed"):
        counterfactual(original, "swap_successors", seed=1013)


@pytest.mark.parametrize("spec", _specs(), ids=lambda spec: f"{spec.name}-{spec.base}")
def test_local_randomness_preserves_global_python_and_numpy_states(spec):
    python_saved, numpy_saved = random.getstate(), np.random.get_state()
    try:
        if spec.name == "t3_missing":
            spec = replace(spec, missing_rate=0)
        random.seed(811)
        np.random.seed(821)
        python_before, numpy_before = random.getstate(), np.random.get_state()
        first = generate(spec, "train", 0, 8, 827)
        first_cf, _ = counterfactual(first, "replace_key_fact", seed=829)
        counterfactual(first, "remove_required_fact", seed=829)
        counterfactual(first, "swap_memory", seed=829)
        if spec.name in ("t2_chain_end", "t2_typed"):
            counterfactual(first, "swap_successors", seed=829)
        assert random.getstate() == python_before
        numpy_after = np.random.get_state()
        assert numpy_before[0] == numpy_after[0]
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        assert numpy_before[2:] == numpy_after[2:]
        random.seed(839)
        np.random.seed(853)
        second = generate(spec, "train", 0, 8, 827)
        second_cf, _ = counterfactual(second, "replace_key_fact", seed=829)
        _assert_arrays_equal(first, second)
        _assert_arrays_equal(first_cf, second_cf)
    finally:
        random.setstate(python_saved)
        np.random.set_state(numpy_saved)


def test_spec_is_frozen_hash_stable_and_supports_nondefault_pools():
    assert INSUFFICIENT == 1024
    spec = TaskSpec("t1", E=6, Q=3, V=80, seen=(5, 45), unseen=(50, 80))
    with pytest.raises(FrozenInstanceError):
        spec.E = 8
    assert stable_hash(spec) == stable_hash(TaskSpec(**asdict(spec)))
    assert stable_hash(spec) != stable_hash(TaskSpec("t1", E=7, Q=3, V=80,
                                                   seen=(5, 45), unseen=(50, 80)))
    assert spec.insufficient == spec.V
    arrays = generate(spec, "test_unseen", 0, 16, 859)
    assert ((arrays.facts[:, :, (0, 2)] >= 50) & (arrays.facts[:, :, (0, 2)] < 80)).all()
    missing, answers = counterfactual(arrays, "remove_required_fact", seed=863)
    assert (answers == 80).all()
    _assert_oracle(missing, spec)


def test_zero_count_keeps_all_array_shapes():
    spec = TaskSpec("t1", E=6, Q=3)
    arrays = generate(spec, "train", 7, 0, 877)
    assert arrays.facts.shape == (0, 6, 3)
    assert arrays.fact_valid.shape == (0, 6)
    assert arrays.answer.shape == (0, 3)
    assert arrays.episode_seed.shape == (0,)


@pytest.mark.parametrize("base", ("t1", "t2_chain_end"))
def test_zero_updates_and_minimal_contexts(base):
    spec = TaskSpec("t3_update", base=base, E=1, Q=1, n_updates=0,
                    update_query_rate=0, hops=(1,))
    arrays = generate(spec, "train", 0, 8, 878)
    assert (arrays.q_hop_true == 1).all()
    assert arrays.facts.shape == (8, 1, 3)
    assert not (arrays.answer == spec.V).any()
    _assert_oracle(arrays, spec)


def test_default_interference_bases_and_single_update_boundary():
    for spec in (TaskSpec("t3_update"), TaskSpec("t3_missing"),
                 TaskSpec("t3_update", E=2, Q=1, n_updates=1, update_query_rate=1),
                 TaskSpec("t3_update", E=2, Q=1, base="t2_chain_end", hops=(1,),
                          n_updates=1, update_query_rate=1)):
        arrays = generate(spec, "train", 0, 12, 879)
        _assert_oracle(arrays, spec)


@pytest.mark.parametrize("kwargs", [
    {"name": "unknown"}, {"E": 0}, {"Q": 0}, {"n_rel": 0},
    {"seen": (4, 4)}, {"seen": (-1, 768)}, {"unseen": (768, 1025)},
    {"seen": (0, 800)}, {"head_sharing": -0.1}, {"head_sharing": 1.1},
    {"name": "t2_walk", "hops": ()}, {"name": "t2_walk", "hops": (0,)},
    {"name": "t2_walk", "chain_lengths": (12, 11)},
    {"name": "t2_walk", "chain_lengths": (0, 24)},
    {"name": "t2_walk", "hops": (25,)},
    {"name": "t2_typed", "n_rel": 1, "n_distract": 2},
    {"name": "t3_update", "base": "t2_walk"},
    {"name": "t3_update", "n_updates": -1},
    {"name": "t3_update", "n_updates": 24},
    {"name": "t3_update", "n_updates": 0, "update_query_rate": 1},
    {"name": "t3_update", "base": "t2_chain_end", "chain_lengths": (22,)},
    {"name": "t3_update", "update_query_rate": 1.1},
    {"name": "t3_missing", "base": "t2_chain_end"},
    {"name": "t3_missing", "missing_rate": -0.1},
])
def test_invalid_specifications_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        spec = TaskSpec(**({"name": "t1"} | kwargs))
        generate(spec, "train", 0, 1, 881)


@pytest.mark.parametrize("split,start,count", [("validation", 0, 1), ("train", -1, 1), ("train", 0, -1)])
def test_invalid_generation_requests(split, start, count):
    with pytest.raises(ValueError):
        generate(TaskSpec("t1"), split, start, count, 883)


def test_pool_capacity_failure_is_explicit():
    with pytest.raises(ValueError):
        generate(TaskSpec("t2_walk", E=8, V=20, seen=(0, 8), unseen=(8, 20)),
                 "train", 0, 1, 885)


def test_counterfactual_rejects_missing_paths_unknown_kind_and_singleton_swap():
    missing = generate(TaskSpec("t3_missing", base="t1", missing_rate=1), "train", 0, 2, 887)
    for kind in ("replace_key_fact", "remove_required_fact"):
        with pytest.raises(ValueError):
            counterfactual(missing, kind, seed=907)
    original = generate(TaskSpec("t1"), "train", 0, 1, 911)
    for kind in ("swap_memory", "unknown"):
        with pytest.raises(ValueError):
            counterfactual(original, kind, seed=919)
