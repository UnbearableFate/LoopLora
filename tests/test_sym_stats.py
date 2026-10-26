"""Inference regressions: known effects, episode dependence and strict pairing."""
from __future__ import annotations

import json

import numpy as np
import pytest

from looplora.infra.seeds import derive_seed
from looplora.sym.stats import compare_runs, depth_comparisons, main


def _samples(correct, *, depth=4, episode_ids=None):
    values = np.asarray(correct, dtype=np.int64)
    count = len(values)
    episodes = np.arange(count) if episode_ids is None else np.asarray(episode_ids)
    return {
        "split": np.full(count, "test"), "condition": np.full(count, "original"),
        "episode_id": episodes, "query_id": np.arange(count),
        "hop": np.ones(count, dtype=np.int64), "L": np.full(count, depth),
        "answer": np.full(count, 7), "correct": values,
    }


def _concat(*groups):
    return {key: np.concatenate([group[key] for group in groups]) for key in groups[0]}


def test_known_paired_effect_is_covered_and_order_does_not_matter():
    rng = np.random.default_rng(derive_seed(42, "test", "known_effect"))
    uniform = rng.random(4000)
    episodes = np.repeat(np.arange(400), 10)
    a = _samples(uniform < 0.55, episode_ids=episodes)
    b = _samples(uniform < 0.35, depth=1, episode_ids=episodes)
    result = compare_runs(a, b, 4, 1, seed=21)
    pooled = result[0]
    assert pooled["ci_low"] <= 0.20 <= pooled["ci_high"]
    assert abs(pooled["difference"] - 0.20) < 0.025
    assert pooled["n_episodes"] == 400
    assert pooled["n_queries"] == 4000
    assert pooled["resamples"] >= 2000
    assert pooled["cluster_unit"] == "episode"
    order = rng.permutation(len(uniform))
    shuffled = {key: values[order] for key, values in b.items()}
    assert compare_runs(a, shuffled, 4, 1, seed=21) == result
    assert compare_runs(a, b, 4, 1, seed=21) == result


def test_bootstrap_preserves_complete_episodes_and_query_weighting():
    # Two perfectly correlated clusters must yield a wide [0, 1] interval,
    # even though they contain many queries. Unequal size tests the estimand.
    a = _samples([1] * 100 + [0] * 300, episode_ids=[0] * 100 + [1] * 300)
    b = _samples([0] * 400, depth=1, episode_ids=[0] * 100 + [1] * 300)
    result = compare_runs(a, b, 4, 1)[0]
    assert result["difference"] == 0.25
    assert result["ci_low"] == 0
    assert result["ci_high"] == 1
    assert result["n_episodes"] == 2


def test_zero_effect_and_single_episode_are_well_defined():
    a = _samples([1, 0, 1], episode_ids=[9, 9, 9])
    b = _samples([1, 0, 1], depth=1, episode_ids=[9, 9, 9])
    result = compare_runs(a, b, 4, 1)[0]
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0


@pytest.mark.parametrize("field,replacement", [
    ("answer", 8), ("hop", 2), ("episode_id", 99), ("query_id", 99),
    ("condition", "swap_memory"), ("split", "val"),
])
def test_rejects_unpaired_or_differently_labeled_data(field, replacement):
    a = _samples([1, 0])
    b = _samples([0, 0], depth=1)
    # Avoid fixed-width string truncation obscuring a condition mismatch.
    b[field] = np.asarray([replacement, replacement])
    with pytest.raises(ValueError, match="Paired samples"):
        compare_runs(a, b, 4, 1)


def test_rejects_missing_queries_duplicate_keys_and_invalid_correctness():
    a = _samples([1, 0])
    b = _samples([0], depth=1)
    with pytest.raises(ValueError, match="identical"):
        compare_runs(a, b, 4, 1)
    duplicate = _concat(a, a)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_runs(duplicate, duplicate, 4, 4)
    invalid = dict(a, correct=np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="binary"):
        compare_runs(invalid, a, 4, 4)


@pytest.mark.parametrize("resamples", [0, 1999, 2000.5, True])
def test_requires_at_least_2000_resamples(resamples):
    a = _samples([1, 0])
    with pytest.raises(ValueError, match="2000"):
        compare_runs(a, a, 4, 4, resamples=resamples)


def test_depth_comparisons_keep_splits_conditions_and_hops_separate():
    groups = []
    for split in ("val", "test"):
        for condition in ("original", "swap_memory"):
            for depth, correct in ((1, [0, 0]), (4, [1, 0]), (8, [1, 1])):
                group = _samples(correct, depth=depth, episode_ids=[0, 0])
                group["split"] = np.full(2, split)
                group["condition"] = np.full(2, condition)
                group["hop"] = np.array([1, 2])
                groups.append(group)
    results = depth_comparisons(_concat(*groups))
    assert len(results) == 2 * 2 * 2 * 3
    assert {r["comparison"] for r in results} == {"L=4 - L=1", "L=8 - L=4"}
    assert all(r["difference"] == 0.5 for r in results if r["hop"] is None)
    assert depth_comparisons(_samples([1, 0], depth=1)) == []


def test_partial_depth_sweep_raises_instead_of_intersecting_queries():
    samples = _concat(_samples([0, 0], depth=1), _samples([1], depth=4))
    with pytest.raises(ValueError, match="identical"):
        depth_comparisons(samples)


def test_stats_cli_reads_npz_and_writes_json(tmp_path):
    samples = _concat(_samples([0, 0], depth=1), _samples([1, 0], depth=4))
    path = tmp_path / "samples.npz"
    output = tmp_path / "stats.json"
    np.savez(path, **samples)
    main(["--samples", str(path), "--out", str(output)])
    results = json.loads(output.read_text())
    assert results[0]["difference"] == 0.5
    assert results[0]["depth_a"] == 4
    assert results[0]["depth_b"] == 1


def test_figures_write_all_requested_pngs_and_exact_statistics(tmp_path):
    from looplora.sym.figures import create_figures

    samples = _concat(_samples([0, 0], depth=1), _samples([1, 0], depth=4))
    paths = create_figures(samples, tmp_path)
    assert len([p for p in paths if p.suffix == ".png"]) == 3
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in paths if p.suffix == ".png")
    assert json.loads((tmp_path / "paired_differences.json").read_text()) == depth_comparisons(samples)
