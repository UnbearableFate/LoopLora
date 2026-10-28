"""NCR: neighbour-confusion classification of single-read errors from saved samples.

Run only inside a Miyabi-G compute allocation, including the CPU cases.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from looplora.data.relations import generate
from looplora.sym import ncr
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.evaluate import evaluate, stratify_queries
from looplora.sym.model import SymbolicModel, make_task_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_neighbour_confusion as script  # noqa: E402

HOPS = [1, 2, 3, 4]


def _config(writer="oracle_random", **updates):
    base = {
        "name": "ncr-tiny", "device": "cpu", "seed": 17, "writer": writer, "variant": "M1",
        "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 8},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 96, "n_rel": 4,
                 "seen": [0, 72], "unseen": [72, 96], "chain_lengths": [4, 4], "hops": HOPS},
        "train": {"steps": 1, "batch_size": 2, "warmup_steps": 0, "depths": [1, 4], "log_every": 1},
        "eval": {"episodes": 8, "batch_size": 4, "splits": ["test"], "depths": [1, 4], "hops": HOPS,
                 "counterfactuals": False, "seed": 777},
    }
    return resolve_config(recursive_merge(base, updates))


def _episodes(config, count=8, seed=777):
    spec = make_task_spec(config, evaluation=True)
    return stratify_queries(generate(spec, "test", 0, count, seed), spec, config["eval"]["hops"], seed)


def _path(graph, start, hop):
    nodes = [start]
    for _ in range(hop):
        nodes.append(graph[nodes[-1]])
    return nodes


def test_classify_on_a_hand_made_graph():
    graph = {0: 1, 1: 2, 2: 3, 3: 4, 10: 11, 11: 12}
    terminals = {4, 12}
    candidates = 8
    for prediction, expected in [(4, None), (3, "first_hop"), (2, "start"), (12, "terminal_other"),
                                 (0, "other"), (1, "other"), (10, "other"), (11, "other")]:
        assert ncr.classify(graph, terminals, candidates, 2, 2, 4, prediction)[0] == expected
    sizes = ncr.classify(graph, terminals, candidates, 2, 2, 4, 3)[1]
    assert sizes == dict(first_hop=1, one_short=0, start=1, terminal_other=1, other=4)
    for prediction, expected in [(4, None), (2, "first_hop"), (3, "one_short"), (1, "start"),
                                 (12, "terminal_other"), (0, "other"), (10, "other"), (11, "other")]:
        assert ncr.classify(graph, terminals, candidates, 1, 3, 4, prediction)[0] == expected
    assert ncr.classify(graph, terminals, candidates, 1, 3, 4, 4)[1] == dict(
        first_hop=1, one_short=1, start=1, terminal_other=1, other=3)


def test_classify_rejects_hop_one_inconsistent_answers_and_broken_paths():
    graph = {0: 1, 1: 2, 2: 3}
    with pytest.raises(ValueError, match="at least 2"):
        ncr.classify(graph, {3}, 4, 0, 1, 1, 1)
    with pytest.raises(ValueError, match="does not end in the saved answer"):
        ncr.classify(graph, {3}, 4, 0, 2, 3, 1)
    with pytest.raises(ValueError, match="leaves the episode graph"):
        ncr.classify(graph, {3}, 4, 3, 2, 3, 0)


@pytest.mark.parametrize("hop", [2, 3, 4])
def test_categories_partition_the_wrong_candidates_of_every_generated_query(hop):
    config = _config()
    arrays = _episodes(config)
    column = config["eval"]["hops"].index(hop)
    for episode, (graph, terminals, n_candidates) in enumerate(ncr.episode_tables(arrays)):
        valid = arrays.facts[episode][arrays.fact_valid[episode]]
        candidates = sorted({int(entity) for entity in valid[:, (0, 2)].ravel()})
        assert len(candidates) == n_candidates
        start, answer = int(arrays.q_start[episode, column]), int(arrays.answer[episode, column])
        counts = {}
        for prediction in candidates:
            label, sizes = ncr.classify(graph, terminals, n_candidates, start, hop, answer, prediction)
            counts[label] = counts.get(label, 0) + 1
        assert counts.pop(None) == 1
        assert {name: counts.get(name, 0) for name in ncr.CATEGORIES} == sizes
        assert sum(sizes.values()) == n_candidates - 1


def _synthetic_samples(arrays, config, rule):
    """Samples for every (episode, query) at L=1 and L=4; ``rule(graph, terminals, start, hop, answer, e) -> prediction``."""
    tables = ncr.episode_tables(arrays)
    hops = config["eval"]["hops"]
    columns = {}
    for depth in (1, 4):
        for episode, (graph, terminals, n_candidates) in enumerate(tables):
            for query, hop in enumerate(hops):
                start, answer = int(arrays.q_start[episode, query]), int(arrays.answer[episode, query])
                prediction = answer if depth == 4 else (rule(graph, terminals, start, hop, answer, episode)
                                                        if hop >= 2 else answer)
                row = dict(split="test", condition="original", episode_id=int(arrays.episode_seed[episode]),
                           query_id=query, hop=hop, L=depth, answer=answer, prediction=prediction,
                           correct=prediction == answer, candidate_size=n_candidates)
                for name, value in row.items():
                    columns.setdefault(name, []).append(value)
    samples = {name: np.asarray(values) for name, values in columns.items()}
    samples["split"] = samples["split"].astype(str)
    samples["condition"] = samples["condition"].astype(str)
    return samples


def _intended(graph, terminals, start, hop, answer, kind):
    nodes = _path(graph, start, hop)
    if kind == "correct":
        return answer
    if kind == "first_hop":
        return nodes[1]
    if kind == "start":
        return start
    if kind == "terminal_other":
        return sorted(terminals - {answer})[0]
    candidates = {n for pair in graph.items() for n in pair}
    excluded = {answer, start, nodes[1], nodes[hop - 1], *terminals}
    return sorted(candidates - excluded)[0]


KINDS = ("correct", "first_hop", "start", "terminal_other", "other")


def test_confusion_rows_assign_each_prediction_to_its_category():
    config = _config()
    arrays = _episodes(config)
    kinds = {}

    def rule(graph, terminals, start, hop, answer, episode):
        kind = KINDS[(episode + hop) % len(KINDS)]
        kinds[episode, hop] = kind
        return _intended(graph, terminals, start, hop, answer, kind)

    samples = _synthetic_samples(arrays, config, rule)
    order = np.argsort(arrays.episode_seed)  # rows are ordered by episode id, not by generation index
    for hop in (2, 3, 4):
        rows = ncr.confusion_rows(samples, arrays, hop=hop, depth=1, split="test")
        expected = [kinds[int(episode), hop] for episode in order]
        assert np.array_equal(rows["episode_id"], arrays.episode_seed[order])
        assert np.array_equal(rows["correct"], np.asarray(expected) == "correct")
        assert np.array_equal(rows["error"], ~rows["correct"])
        for name in ncr.CATEGORIES:
            assert np.array_equal(rows["category"][name], np.asarray(expected) == name)
        assert np.allclose(rows["uniform_correct"], 1 / 10)
        assert np.allclose(sum(rows["uniform_rate"].values()), 9 / 10)
        for index in np.flatnonzero(rows["error"]):
            assert sum(values[index] for values in rows["uniform_error"].values()) == pytest.approx(1.0)
        assert not any(values[rows["correct"]].any() for values in rows["uniform_error"].values())
    assert ncr.confusion_rows(samples, arrays, hop=2, depth=4, split="test")["correct"].all()


def test_confusion_rows_reject_tampered_or_mismatched_samples():
    config = _config()
    arrays = _episodes(config)
    samples = _synthetic_samples(arrays, config, lambda graph, terminals, start, hop, answer, episode:
                                 _intended(graph, terminals, start, hop, answer, "first_hop"))

    def tampered(name, mutate):
        copy = {key: value.copy() for key, value in samples.items()}
        mutate(copy[name])
        return copy

    with pytest.raises(ValueError, match="no samples for hop"):
        ncr.confusion_rows(samples, arrays, hop=7, depth=1, split="test")
    with pytest.raises(ValueError, match="at least 2"):
        ncr.confusion_rows(samples, arrays, hop=1, depth=1, split="test")
    with pytest.raises(ValueError, match="differ from the saved samples"):
        ncr.confusion_rows(tampered("answer", lambda value: value.__setitem__(slice(None), value[::-1].copy())),
                           arrays, hop=2, depth=1, split="test")
    with pytest.raises(ValueError, match="candidate count"):
        ncr.confusion_rows(tampered("candidate_size", lambda value: value.__iadd__(1)),
                           arrays, hop=2, depth=1, split="test")
    with pytest.raises(ValueError, match="correctness disagrees"):
        ncr.confusion_rows(tampered("correct", lambda value: value.__ixor__(True)),
                           arrays, hop=2, depth=1, split="test")
    with pytest.raises(ValueError, match="absent from the regenerated"):
        ncr.confusion_rows(tampered("episode_id", lambda value: value.__iadd__(1)),
                           arrays, hop=2, depth=1, split="test")


def _summaries(arrays, config, rules, hop=2):
    return {seed: ncr.confusion_rows(_synthetic_samples(arrays, config, rule), arrays, hop=hop, depth=1,
                                     split="test") for seed, rule in rules.items()}


def test_summarize_pools_rates_and_shares_with_exact_reference_values():
    config = _config()
    count = 8
    arrays = _episodes(config, count)

    def always(kind):
        return lambda graph, terminals, start, hop, answer, episode: _intended(
            graph, terminals, start, hop, answer, kind)

    def alternating(graph, terminals, start, hop, answer, episode):
        return _intended(graph, terminals, start, hop, answer, "first_hop" if episode % 2 == 0 else "start")

    per_seed = _summaries(arrays, config, {17: always("first_hop"), 29: alternating})
    result = ncr.summarize(per_seed, resamples=2000, label=("test",))
    assert result["training_seeds"] == [17, 29]
    assert result["accuracy"] == 0.0 and result["below_chance"] is True
    assert result["chance_accuracy"] == pytest.approx(1 / 10)
    assert result["errors_per_seed"] == {17: count, 29: count}
    first, start = result["categories"]["first_hop"], result["categories"]["start"]
    assert first["share"] == pytest.approx(0.75) and start["share"] == pytest.approx(0.25)
    assert first["rate"] == pytest.approx(0.75) and start["rate"] == pytest.approx(0.25)
    assert first["per_seed_share"] == pytest.approx([1.0, 0.5])
    assert first["uniform_share"] == pytest.approx(1 / 9) and first["uniform_rate"] == pytest.approx(1 / 10)
    assert result["categories"]["one_short"]["share"] == 0.0
    assert result["categories"]["one_short"]["uniform_share"] == 0.0
    assert sum(entry["share"] for entry in result["categories"].values()) == pytest.approx(1.0)
    assert sum(entry["uniform_share"] for entry in result["categories"].values()) == pytest.approx(1.0)
    assert result["categories"]["other"]["uniform_share"] == pytest.approx(6 / 9)
    for entry in result["categories"].values():
        assert entry["share_ci_low"] <= entry["share_ci_high"] and entry["rate_ci_low"] <= entry["rate_ci_high"]


def test_summarize_mixture_invariants_and_error_cases():
    config = _config()
    count = 60
    arrays = _episodes(config, count)

    def mixed(offset):
        def rule(graph, terminals, start, hop, answer, episode):
            kind = ("correct", "first_hop", "start", "first_hop", "other", "terminal_other")[(episode + offset) % 6]
            return _intended(graph, terminals, start, hop, answer, kind)
        return rule

    per_seed = _summaries(arrays, config, {17: mixed(0), 29: mixed(1), 43: mixed(3)}, hop=3)
    result = ncr.summarize(per_seed, resamples=2000, label=("mixture",))
    categories = result["categories"]
    assert sum(entry["share"] for entry in categories.values()) == pytest.approx(1.0)
    assert sum(entry["rate"] for entry in categories.values()) == pytest.approx(1 - result["accuracy"])
    assert sum(entry["uniform_rate"] for entry in categories.values()) == pytest.approx(
        1 - result["chance_accuracy"])
    assert categories["first_hop"]["share"] > categories["first_hop"]["uniform_share"]
    assert result["accuracy"] == pytest.approx(np.mean([row["correct"].mean() for row in per_seed.values()]))
    repeat = ncr.summarize(per_seed, resamples=2000, label=("mixture",))
    assert repeat == result

    errorless = dict(per_seed)
    errorless[17] = _summaries(arrays, config, {17: lambda *args: args[4]}, hop=3)[17]
    with pytest.raises(ncr.UndefinedShares, match="no errors"):
        ncr.summarize(errorless, resamples=2000)
    shifted = dict(per_seed)
    shifted[17] = {**per_seed[17], "episode_id": per_seed[17]["episode_id"][::-1].copy()}
    with pytest.raises(ValueError, match="disagree on the evaluation episodes") as caught:
        ncr.summarize(shifted, resamples=2000)
    assert not isinstance(caught.value, ncr.UndefinedShares)  # a data-integrity failure is not an undefined group


@pytest.mark.parametrize("writer", ["oracle_random", "joint_fact"])
def test_saved_evaluator_samples_are_reproduced_and_classified(writer, tmp_path):
    config = _config(writer)
    evaluate(SymbolicModel(config), config, tmp_path, samples_only=True)
    samples, recorded = ncr.load_directory(tmp_path)
    arrays = ncr.regenerate(recorded, "test")
    assert np.array_equal(arrays.facts, _episodes(config).facts)
    for hop in (2, 3, 4):
        rows = ncr.confusion_rows(samples, arrays, hop=hop, depth=1, split="test")
        assert len(rows["episode_id"]) == config["eval"]["episodes"]
        assert sum(int(values.sum()) for values in rows["category"].values()) == int(rows["error"].sum())
    cell = (samples["L"] == 4) & (samples["hop"] == 2)
    order = np.lexsort((samples["query_id"][cell], samples["episode_id"][cell]))
    assert np.array_equal(ncr.confusion_rows(samples, arrays, hop=2, depth=4, split="test")["correct"],
                          samples["correct"][cell][order])


def test_load_directory_requires_config_and_prediction_arrays(tmp_path):
    with pytest.raises(FileNotFoundError):
        ncr.load_directory(tmp_path)
    config = _config()
    evaluate(SymbolicModel(config), config, tmp_path, samples_only=True)
    with np.load(tmp_path / "samples.npz", allow_pickle=False) as archive:
        stripped = {key: archive[key] for key in archive.files if key != "prediction"}
    np.savez_compressed(tmp_path / "samples.npz", **stripped)
    with pytest.raises(ValueError, match="lack the 'prediction' array"):
        ncr.load_directory(tmp_path)


def _tiny_confirmation_runs(root, writer="oracle_random", variant="M1"):
    for seed in (17, 29, 43):
        config = _config(writer, seed=seed)
        directory = root / "runs" / f"primary-{writer}-{variant}-r32-s{seed}" / "confirmation"
        evaluate(SymbolicModel(config), config, directory, samples_only=True)


def test_analysis_script_summarizes_saved_evaluations_and_reports_missing_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "require_compute", lambda: None)
    p1s, out = tmp_path / "p1s", tmp_path / "out"
    _tiny_confirmation_runs(p1s)
    arguments = ["--p1s-root", str(p1s), "--f2-root", str(tmp_path / "f2"), "--writers", "oracle_random",
                 "--variants", "M1", "--datasets", "confirmation", "--hops", "2", "3"]
    assert script.main(arguments + ["--out", str(out)]) == 0
    with (out / "accuracy.csv").open(newline="") as stream:
        accuracy = list(csv.DictReader(stream))
    with (out / "categories.csv").open(newline="") as stream:
        categories = list(csv.DictReader(stream))
    assert [row["hop"] for row in accuracy] == ["2", "3"]
    assert len(categories) == 2 * len(ncr.CATEGORIES)
    for hop in ("2", "3"):
        shares = [float(row["share"]) for row in categories if row["hop"] == hop]
        assert sum(shares) == pytest.approx(1.0)
        assert sum(float(row["uniform_share"]) for row in categories if row["hop"] == hop) == pytest.approx(1.0)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["missing"] == [] and summary["undefined"] == [] and summary["groups"] == 2
    assert summary["verified_rows"] == 3 * 2 * 8 and summary["verified_directories"] == 3
    assert "oracle_random" in (out / "summary.md").read_text()
    # a missing training seed leaves the group out and is reported, never filled in
    for path in (p1s / "runs" / "primary-oracle_random-M1-r32-s43" / "confirmation").iterdir():
        path.unlink()
    again = tmp_path / "again"
    assert script.main(arguments + ["--out", str(again)]) == 0
    summary = json.loads((again / "summary.json").read_text())
    assert summary["groups"] == 0 and len(summary["missing"]) == 1 and "s43" in summary["missing"][0]
