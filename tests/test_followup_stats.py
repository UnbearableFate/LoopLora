"""Episode-paired linear contrasts used by the S9 follow-up studies (NumPy only).

Run only inside a Miyabi-G compute allocation.
"""
from __future__ import annotations

import numpy as np
import pytest

from looplora.sym import followup_stats as fs
from looplora.sym.stats import pooled_compare_runs


def _samples(seed, *, episodes=30, queries=4, depths=(1, 4), conditions=("original",), split="test"):
    """Flat evaluator-style arrays; labels depend only on (episode, query)."""
    rng = np.random.default_rng(seed)
    columns = {name: [] for name in ("split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct")}
    for condition in conditions:
        for depth in depths:
            for episode in range(episodes):
                for query in range(queries):
                    columns["split"].append(split)
                    columns["condition"].append(condition)
                    columns["episode_id"].append(1000 + 7 * episode)
                    columns["query_id"].append(query)
                    columns["hop"].append(query + 1)
                    columns["L"].append(depth)
                    columns["answer"].append((3 * episode + query) % 11)
                    columns["correct"].append(int(rng.random() < 0.35 + 0.1 * depth / 4))
    result = {name: np.asarray(values) for name, values in columns.items()}
    order = rng.permutation(len(result["correct"]))
    return {name: values[order] for name, values in result.items()}


def test_pooled_mean_contrast_reproduces_pooled_compare_runs_exactly():
    by_seed = {17: _samples(1), 29: _samples(2), 43: _samples(3)}
    reference = pooled_compare_runs(by_seed, by_seed, 4, 1, hops=[2, 3], split="test", condition="original",
                                    resamples=2000, seed=2718, confidence=0.975)
    aligned = fs.aligned_terms(by_seed, [("test", "original", 4), ("test", "original", 1)])
    keep = np.isin(aligned["hop"], [2, 3])
    contrast = {seed: (values[0] - values[1])[keep] for seed, values in aligned["values"].items()}
    result = fs.pooled_mean_contrast(contrast, aligned["episode_id"][keep], resamples=2000, confidence=0.975,
                                     bootstrap_seed=reference["bootstrap_seed"])
    assert result["difference"] == pytest.approx(reference["difference"], abs=1e-12)
    assert result["ci_low"] == pytest.approx(reference["ci_low"], abs=1e-12)
    assert result["ci_high"] == pytest.approx(reference["ci_high"], abs=1e-12)
    assert result["n_episodes"] == reference["n_episodes"]
    assert result["n_queries_per_seed"] == reference["n_queries_per_seed"]
    assert [row["difference"] for row in result["per_seed"]] == pytest.approx(
        [row["difference"] for row in reference["per_seed"]], abs=1e-12)


def test_default_bootstrap_seed_depends_on_label_and_is_reproducible():
    values = {17: np.linspace(0, 1, 40)}
    episodes = np.repeat(np.arange(10), 4)
    first = fs.pooled_mean_contrast(values, episodes, label=("a",))
    again = fs.pooled_mean_contrast(values, episodes, label=("a",))
    other = fs.pooled_mean_contrast(values, episodes, label=("b",))
    assert first == again
    assert first["bootstrap_seed"] != other["bootstrap_seed"]


def test_constant_contrast_has_a_degenerate_interval():
    episodes = np.repeat(np.arange(12), 3)
    result = fs.pooled_mean_contrast({17: np.full(36, 0.1), 29: np.full(36, 0.1)}, episodes, label=("const",))
    assert result["difference"] == pytest.approx(0.1)
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.1)
    assert fs.passes_gain(result, 0.05)
    assert not fs.passes_gain(result, 0.11)


def test_every_seed_uses_the_same_episode_draws():
    # With identical per-seed values, a joint episode draw makes the pooled
    # bootstrap distribution equal the single-seed distribution draw by draw;
    # independent per-seed draws would not.
    rng = np.random.default_rng(3)
    episodes = np.repeat(np.arange(25), 2)
    values = rng.random(50)
    single = fs.pooled_mean_contrast({17: values}, episodes, bootstrap_seed=123, return_draws=True)
    pooled = fs.pooled_mean_contrast({17: values, 29: values, 43: values}, episodes, bootstrap_seed=123,
                                     return_draws=True)
    np.testing.assert_allclose(pooled["draws"], single["draws"], rtol=1e-13, atol=0)
    assert pooled["joint_episode_resampling"] is True
    assert fs.public(pooled).keys() == pooled.keys() - {"draws"}


def test_validation_rejects_malformed_inputs():
    episodes = np.repeat(np.arange(4), 2)
    good = {17: np.zeros(8)}
    with pytest.raises(ValueError, match="resamples"):
        fs.pooled_mean_contrast(good, episodes, resamples=1999)
    with pytest.raises(ValueError, match="confidence"):
        fs.pooled_mean_contrast(good, episodes, confidence=1.0)
    with pytest.raises(ValueError, match="finite"):
        fs.pooled_mean_contrast({17: np.array([np.nan] + [0.0] * 7)}, episodes)
    with pytest.raises(ValueError, match="aligned"):
        fs.pooled_mean_contrast({17: np.zeros(7)}, episodes)
    with pytest.raises(ValueError, match="at least one"):
        fs.pooled_mean_contrast({}, episodes)


def test_aligned_terms_requires_identical_query_keys():
    by_seed = {17: _samples(1), 29: _samples(2)}
    aligned = fs.aligned_terms(by_seed, [("test", "original", 1), ("test", "original", 4)])
    assert set(aligned["values"]) == {17, 29}
    assert all(len(values) == 2 for values in aligned["values"].values())
    order = np.lexsort((aligned["query_id"], aligned["episode_id"]))
    assert np.array_equal(order, np.arange(len(order)))
    broken = {seed: dict(arrays) for seed, arrays in by_seed.items()}
    broken[29]["answer"] = broken[29]["answer"] + 1
    with pytest.raises(ValueError, match="disagree"):
        fs.aligned_terms(broken, [("test", "original", 1)])
    with pytest.raises(ValueError, match="no samples"):
        fs.aligned_terms(by_seed, [("test", "missing", 1)])


def test_select_rows_orders_by_episode_then_query():
    arrays = _samples(5, depths=(1,), conditions=("x", "y"))
    index = fs.select_rows(arrays, "test", "y", 1)
    assert len(index) == 30 * 4
    assert np.all(arrays["condition"][index] == "y")
    keys = list(zip(arrays["episode_id"][index].tolist(), arrays["query_id"][index].tolist()))
    assert keys == sorted(keys)


def test_independent_difference_combines_streams_and_checks_inputs():
    episodes = np.repeat(np.arange(20), 2)
    rng = np.random.default_rng(0)
    first = fs.pooled_mean_contrast({17: rng.random(40)}, episodes, label=("x",), return_draws=True)
    second = fs.pooled_mean_contrast({17: rng.random(40)}, episodes, label=("y",), return_draws=True)
    combined = fs.independent_difference(first, second, confidence=0.95)
    assert combined["difference"] == pytest.approx(first["difference"] - second["difference"])
    assert combined["ci_low"] < combined["difference"] < combined["ci_high"]
    with pytest.raises(ValueError, match="different bootstrap"):
        fs.independent_difference(first, first)
    with pytest.raises(ValueError, match="draws"):
        fs.independent_difference(fs.public(first), second)


def test_passes_gain_needs_both_size_and_positive_lower_bound():
    assert fs.passes_gain({"difference": 0.05, "ci_low": 1e-9}, 0.05)
    assert not fs.passes_gain({"difference": 0.049, "ci_low": 0.01}, 0.05)
    assert not fs.passes_gain({"difference": 0.2, "ci_low": 0.0}, 0.05)
    assert not fs.passes_gain({"difference": float("nan"), "ci_low": 0.1}, 0.0)


def test_pooled_ratio_reduces_to_the_mean_contrast_with_unit_denominators():
    rng = np.random.default_rng(4)
    episodes = np.repeat(np.arange(20), 3)
    values = {17: rng.random(60), 29: rng.random(60)}
    ones = {seed: np.ones(60) for seed in values}
    ratio = fs.pooled_ratio(values, ones, episodes, bootstrap_seed=77, confidence=0.975)
    mean = fs.pooled_mean_contrast(values, episodes, bootstrap_seed=77, confidence=0.975)
    assert ratio["ratio"] == pytest.approx(mean["difference"], abs=1e-12)
    assert ratio["ci_low"] == pytest.approx(mean["ci_low"], abs=1e-12)
    assert ratio["ci_high"] == pytest.approx(mean["ci_high"], abs=1e-12)


def test_pooled_ratio_counts_events_and_rejects_empty_denominators():
    episodes = np.repeat(np.arange(10), 2)
    numerator = {17: np.array([1.0, 0.0] * 10)}
    denominator = {17: np.ones(20)}
    result = fs.pooled_ratio(numerator, denominator, episodes, label=("share",))
    assert result["ratio"] == pytest.approx(0.5)
    assert result["ci_low"] == pytest.approx(0.5) and result["ci_high"] == pytest.approx(0.5)
    assert result["per_seed"][0]["denominator"] == 20
    with pytest.raises(ValueError, match="no events"):
        fs.pooled_ratio({17: np.zeros(20)}, {17: np.zeros(20)}, episodes)
    sparse = np.zeros(20)
    sparse[0] = 1.0
    with pytest.raises(ValueError, match="resample"):
        fs.pooled_ratio({17: sparse}, {17: sparse}, episodes)
    with pytest.raises(ValueError, match="same nonempty"):
        fs.pooled_ratio({17: np.ones(20)}, {29: np.ones(20)}, episodes)
