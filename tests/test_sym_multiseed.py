"""Pooled inference keeps training seeds paired on common episode clusters."""
from __future__ import annotations

import json

import numpy as np
import pytest

from looplora.sym.stats import main, pooled_compare_runs


def _samples(correct, *, depth=4, episodes=None, hops=None):
    count = len(correct)
    return {
        "split": np.full(count, "val"), "condition": np.full(count, "original"),
        "episode_id": np.arange(count) if episodes is None else np.asarray(episodes),
        "query_id": np.arange(count),
        "hop": np.full(count, 2) if hops is None else np.asarray(hops),
        "L": np.full(count, depth), "answer": np.full(count, 7),
        "correct": np.asarray(correct, dtype=np.int64),
    }


def _concat(*groups):
    return {key: np.concatenate([group[key] for group in groups]) for key in groups[0]}


def test_pooled_estimate_is_equal_mean_of_seed_effects_on_explicit_hops():
    a, b = {}, {}
    # Each episode has four selected queries. Known seed effects are .25,
    # .50 and .75; all episode-bootstrap draws have a pooled effect of .50.
    for seed, successes in ((17, 1), (29, 2), (43, 3)):
        selected = ([1] * successes + [0] * (4 - successes)) * 8
        values = selected + [0] * 8
        episodes = np.repeat(np.arange(8), 4).tolist() + list(range(8))
        hops = [2, 3, 2, 3] * 8 + [1] * 8
        a[seed] = _samples(values, episodes=episodes, hops=hops)
        b[seed] = _samples([0] * 32 + [1] * 8, depth=1, episodes=episodes, hops=hops)
    result = pooled_compare_runs(a, b, 4, 1, hops=[3, 2])
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0.5
    assert [r["difference"] for r in result["per_seed"]] == [0.25, 0.5, 0.75]
    assert result["training_seeds"] == [17, 29, 43]
    assert result["n_seeds"] == 3
    assert result["n_episodes"] == 8
    assert result["n_queries_per_seed"] == 32
    assert result["hops"] == [2, 3]
    assert result["resamples"] == 2000
    assert result["joint_episode_resampling"] is True
    assert result["statistic"] == "mean_per_seed_paired_accuracy_difference"


def test_joint_episode_draws_preserve_crossseed_anticorrelation():
    # Independent per-seed episode draws would invent nonzero uncertainty.
    a = {17: _samples([1, 0]), 29: _samples([0, 0])}
    b = {17: _samples([0, 0]), 29: _samples([1, 0])}
    result = pooled_compare_runs(a, b, 4, 4, hops=[2])
    assert [r["difference"] for r in result["per_seed"]] == [0.5, -0.5]
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0


def test_unequal_episode_sizes_use_query_accuracy_not_mean_episode_accuracy():
    a = {17: _samples([1] * 100 + [0] * 300, episodes=[5] * 100 + [9] * 300)}
    b = {17: _samples([0] * 400, episodes=[5] * 100 + [9] * 300)}
    result = pooled_compare_runs(a, b, 4, 4, hops=[2])
    assert result["difference"] == 0.25
    assert result["ci_low"] == 0
    assert result["ci_high"] == 1
    assert result["n_episodes"] == 2


def test_same_checkpoint_depth_comparison_and_row_seed_order_are_deterministic():
    samples = {
        17: _concat(_samples([0, 0, 0, 0], depth=1), _samples([1, 0, 1, 0])),
        29: _concat(_samples([1, 0, 0, 0], depth=1), _samples([1, 1, 1, 0])),
    }
    result = pooled_compare_runs(samples, samples, 4, 1, hops=[2], seed=7)
    reordered = {s: {k: v[::-1] for k, v in samples[s].items()} for s in (29, 17)}
    assert pooled_compare_runs(reordered, reordered, 4, 1, hops=[2], seed=7) == result
    assert result["difference"] == 0.5


@pytest.mark.parametrize("field,value", [
    ("answer", 99), ("hop", 3), ("episode_id", 99), ("query_id", 99),
])
@pytest.mark.parametrize("side", ["a", "b"])
def test_rejects_crossseed_label_mismatch_even_when_within_seed_pairing_matches(field, value, side):
    a = {s: _samples([1, 0]) for s in (17, 29)}
    b = {s: _samples([0, 0]) for s in (17, 29)}
    sources = a if side == "a" else b
    sources[29][field] = np.asarray([value, value])
    # Episode/query ids remain unique because their complementary id differs.
    with pytest.raises(ValueError, match="Pooled paired samples"):
        pooled_compare_runs(a, b, 4, 4, hops=[2])


def test_rejects_unmatched_seed_sets_or_queries_instead_of_intersecting():
    a = {17: _samples([1, 0]), 29: _samples([1, 0])}
    with pytest.raises(ValueError, match="identical training seed keys"):
        pooled_compare_runs(a, {17: _samples([0, 0])}, 4, 4, hops=[2])
    b = {17: _samples([0, 0]), 29: _samples([0])}
    with pytest.raises(ValueError, match="identical.*episode/query"):
        pooled_compare_runs(a, b, 4, 4, hops=[2])
    with pytest.raises(ValueError, match="nonempty mappings"):
        pooled_compare_runs({}, {}, 4, 4, hops=[2])


@pytest.mark.parametrize("hops", [[], [2, 2], [0], [True], [2.0], [2, 3]])
def test_rejects_invalid_or_absent_hop_subsets(hops):
    samples = {17: _samples([1, 0])}
    with pytest.raises(ValueError, match="hop|Hop"):
        pooled_compare_runs(samples, samples, 4, 4, hops=hops)


@pytest.mark.parametrize("split,condition", [("test", "original"), ("val", "swap_memory")])
def test_does_not_fall_back_to_another_split_or_condition(split, condition):
    samples = {17: _samples([1, 0])}
    with pytest.raises(ValueError, match="No samples"):
        pooled_compare_runs(samples, samples, 4, 4, hops=[2], split=split, condition=condition)


def test_explicit_split_condition_selection_does_not_pool_other_groups():
    chosen = _samples([1, 0])
    other = _samples([0, 1])
    other["split"] = np.full(2, "test")
    another_condition = _samples([0, 1])
    another_condition["condition"] = np.full(2, "swap_memory")
    a = {17: _concat(chosen, other, another_condition)}
    b = {17: _samples([0, 0])}
    result = pooled_compare_runs(a, b, 4, 4, hops=[2])
    assert result["n_queries_per_seed"] == 2
    assert result["difference"] == 0.5


def test_pooled_cli_parses_seed_keys_and_manifest_relative_paths(tmp_path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    manifest = {}
    arrays = {}
    for seed in (17, 29):
        arrays[seed] = _concat(_samples([0, 0], depth=1), _samples([1, 0]))
        np.savez(folder / f"seed{seed}.npz", **arrays[seed])
        manifest[str(seed)] = f"artifacts/seed{seed}.npz"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "pooled.json"
    main(["--pooled-a", str(path), "--pooled-b", str(path), "--depth-a", "4", "--depth-b", "1",
          "--hops", "2", "--label-a", "L4", "--label-b", "L1", "--out", str(output)])
    result = json.loads(output.read_text())
    expected = pooled_compare_runs(arrays, arrays, 4, 1, hops=[2])
    expected["comparison"] = "L4 - L1"
    assert result == expected


@pytest.mark.parametrize("contents", [
    '{"17": "a.npz", "17": "b.npz"}',
    '{"17": "a.npz", "017": "b.npz"}',
    '{"bad": "a.npz"}', '{"-1": "a.npz"}', '{"17": 17}', '{}', '[]',
])
def test_pooled_cli_rejects_ambiguous_or_malformed_manifests(tmp_path, contents):
    path = tmp_path / "manifest.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match="seed|manifest|path"):
        main(["--pooled-a", str(path), "--pooled-b", str(path), "--depth-a", "4", "--depth-b", "1",
              "--hops", "2", "--out", str(tmp_path / "out.json")])


def test_pooled_requires_at_least_2000_resamples():
    samples = {17: _samples([1, 0])}
    with pytest.raises(ValueError, match="2000"):
        pooled_compare_runs(samples, samples, 4, 4, hops=[2], resamples=1999)
