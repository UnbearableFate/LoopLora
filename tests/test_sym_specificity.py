"""Walk specificity must retain episode and cross-seed covariance."""
from __future__ import annotations

import numpy as np
import pytest

from looplora.sym.stats import pooled_walk_specificity


def _samples(reach, far):
    rows = []
    for depth in (1, 4):
        for episode, (near_value, far_value) in enumerate(zip(reach, far)):
            for query, hop in enumerate((2, 3, 6, 7, 8)):
                correct = 0 if depth == 1 else near_value if hop in (2, 3) else far_value
                rows.append((episode, query, hop, depth, correct))
    return {"episode_id": np.asarray([r[0] for r in rows]),
            "query_id": np.asarray([r[1] for r in rows]),
            "hop": np.asarray([r[2] for r in rows]), "L": np.asarray([r[3] for r in rows]),
            "correct": np.asarray([r[4] for r in rows]),
            "answer": np.zeros(len(rows), dtype=np.int64),
            "split": np.full(len(rows), "test"), "condition": np.full(len(rows), "original")}


def test_known_specificity_and_per_seed_components():
    samples = {17: _samples([1, 1], [0, 0]), 29: _samples([1, 1], [1, 1])}
    result = pooled_walk_specificity(samples)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == .5
    assert result["reach_difference"] == 1
    assert result["far_difference"] == .5
    assert [r["difference"] for r in result["per_seed"]] == [1, 0]
    assert result["query_counts_by_subset"] == {"reach": 4, "far": 6}
    assert result["n_queries_per_seed"] == 10
    assert result["n_episodes"] == 2


def test_reach_and_far_covariance_is_preserved_within_each_episode():
    samples = {17: _samples([1, 0] * 10, [1, 0] * 10)}
    result = pooled_walk_specificity(samples)
    assert result["reach_difference"] == result["far_difference"] == .5
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0


def test_episode_draws_are_joint_across_training_seeds():
    samples = {17: _samples([1, 0], [0, 0]), 29: _samples([0, 0], [1, 0])}
    result = pooled_walk_specificity(samples)
    assert [r["difference"] for r in result["per_seed"]] == [.5, -.5]
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0


def test_per_episode_contrast_does_not_weight_episodes_by_query_count():
    samples = _samples([1, 0], [0, 0])
    # Add extra distinct reachable queries only in the zero-effect episode.
    extra = {key: value[(samples["episode_id"] == 1) & (samples["hop"] == 2)].copy()
             for key, value in samples.items()}
    extra["query_id"][:] = 99
    samples = {key: np.concatenate((value, extra[key])) for key, value in samples.items()}
    result = pooled_walk_specificity({17: samples})
    assert result["difference"] == .5
    assert result["episode_weighting"] == "equal"
    assert result["ci_low"] == 0 and result["ci_high"] == 1


def test_confidence_changes_cutoffs_only_and_row_seed_order_is_irrelevant(monkeypatch):
    samples = {17: _samples([1, 0] * 50, [0, 0] * 50),
               29: _samples([1, 0] * 50, [0, 0] * 50)}
    captured = []
    quantile = np.quantile

    def capture(values, cutoffs):
        captured.append((values.copy(), cutoffs))
        return quantile(values, cutoffs)

    monkeypatch.setattr(np, "quantile", capture)
    report = pooled_walk_specificity(samples)
    reverse = {s: {key: value[::-1] for key, value in samples[s].items()} for s in (29, 17)}
    gate = pooled_walk_specificity(reverse, confidence=.975)
    np.testing.assert_array_equal(captured[0][0], captured[1][0])
    assert captured[0][1] == pytest.approx([.025, .975])
    assert captured[1][1] == pytest.approx([.0125, .9875])
    assert report["bootstrap_seed"] == gate["bootstrap_seed"]
    assert gate["ci_low"] <= report["ci_low"] <= report["ci_high"] <= gate["ci_high"]
    assert report["per_seed"] == gate["per_seed"]


@pytest.mark.parametrize("field,value", [("answer", 99), ("hop", 9), ("query_id", 99), ("episode_id", 99)])
@pytest.mark.parametrize("across_seed", [False, True])
def test_rejects_different_query_labels_across_depths_or_seeds(field, value, across_seed):
    samples = {17: _samples([1, 0], [0, 0])}
    if across_seed:
        samples[29] = {key: array.copy() for key, array in samples[17].items()}
        target = samples[29]
        target[field][(target["episode_id"] == 0) & (target["query_id"] == 0)] = value
    else:
        target = samples[17]
        target[field][(target["L"] == 4) & (target["episode_id"] == 0) & (target["query_id"] == 0)] = value
    with pytest.raises(ValueError, match="identical.*labels"):
        pooled_walk_specificity(samples)


def test_every_episode_must_cover_each_requested_hop():
    samples = _samples([1, 0], [0, 0])
    keep = ~((samples["episode_id"] == 1) & (samples["hop"] == 8))
    with pytest.raises(ValueError, match="Every episode.*hop 8"):
        pooled_walk_specificity({17: {key: value[keep] for key, value in samples.items()}})


@pytest.mark.parametrize("reach,far", [([], [6]), ([2, 2], [6]), ([0], [6]),
                                              ([True], [6]), ([2], [2]), ([2.0], [6])])
def test_invalid_hop_subsets_are_rejected(reach, far):
    with pytest.raises(ValueError, match="hop subsets"):
        pooled_walk_specificity({17: _samples([1], [0])}, reach_hops=reach, far_hops=far)


@pytest.mark.parametrize("confidence", [True, np.bool_(True), 0, 1, float("nan"), float("inf"), "0.95"])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError, match="Confidence"):
        pooled_walk_specificity({17: _samples([1], [0])}, confidence=confidence)


def test_missing_split_condition_and_too_few_resamples_are_rejected():
    samples = {17: _samples([1], [0])}
    with pytest.raises(ValueError, match="No specificity samples"):
        pooled_walk_specificity(samples, split="val")
    with pytest.raises(ValueError, match="No specificity samples"):
        pooled_walk_specificity(samples, condition="swap_memory")
    with pytest.raises(ValueError, match="2000"):
        pooled_walk_specificity(samples, resamples=1999)
