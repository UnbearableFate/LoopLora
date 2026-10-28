"""Episode-paired linear contrasts for the S9 follow-up studies (IVD, CVH, TSM, UNS).

Everything here is NumPy only. ``pooled_mean_contrast`` generalizes
``looplora.sym.stats.pooled_compare_runs`` from "A minus B" to any per-query
contrast (for example a difference of differences): the statistic is the equal-
weight mean over training seeds of the per-seed query-weighted mean of the
contrast, the CI is a percentile interval of whole-episode bootstrap draws, and
every draw uses the same episodes for every seed. The resampling procedure
(draw matrix shape, batch size and ratio-of-sums statistic) is identical to
``pooled_compare_runs``, so passing that function's ``bootstrap_seed`` and the
per-query difference reproduces its interval exactly (see tests).

Independent contrasts (different episode sets, e.g. seen versus unseen pools)
use ``independent_difference`` on the draws of two separate calls.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from looplora.infra.seeds import derive_seed

MIN_RESAMPLES = 2000
Term = tuple[str, str, int]  # (split, condition, depth)
# Descriptive hop groups of the S9 reports ('reach' and 'far' are the project's named endpoints).
HOP_GROUPS: Mapping[str, tuple[int, ...]] = {"hop1": (1,), "reach": (2, 3), "mid": (4, 5), "far": (6, 7, 8)}


def _check(resamples: int, confidence: float) -> float:
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < MIN_RESAMPLES:
        raise ValueError(f"episode bootstrap requires at least {MIN_RESAMPLES} integer resamples")
    if isinstance(confidence, bool) or not isinstance(confidence, Real) or not 0 < float(confidence) < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    return float(confidence)


def select_rows(arrays: Mapping[str, np.ndarray], split: str, condition: str, depth: int) -> np.ndarray:
    """Row indices of one (split, condition, depth) cell, ordered by (episode_id, query_id)."""
    mask = (arrays["split"] == split) & (arrays["condition"] == condition) & (arrays["L"] == depth)
    index = np.flatnonzero(mask)
    if not index.size:
        raise ValueError(f"no samples for split={split!r}, condition={condition!r}, L={depth}")
    order = np.lexsort((arrays["query_id"][index], arrays["episode_id"][index]))
    return index[order]


def aligned_terms(samples_by_seed: Mapping[int, Mapping[str, np.ndarray]],
                  terms: Sequence[Term], field: str = "correct") -> dict[str, Any]:
    """Per-seed ``field`` arrays for several cells with identical query keys.

    Every seed and every term must contain exactly the same (episode, query)
    keys with the same hop and answer labels; nothing is intersected or filled.
    Returns episode_id/query_id/hop/answer (shared) and ``values[seed][k]``
    (float64) for term k.
    """
    if not samples_by_seed or not terms:
        raise ValueError("aligned_terms requires samples and at least one term")
    reference: dict[str, np.ndarray] | None = None
    values: dict[int, list[np.ndarray]] = {}
    for seed in sorted(samples_by_seed):
        arrays = samples_by_seed[seed]
        values[int(seed)] = []
        for term in terms:
            index = select_rows(arrays, *term)
            labels = {name: np.asarray(arrays[name])[index] for name in ("episode_id", "query_id", "hop", "answer")}
            if reference is None:
                reference = labels
            else:
                for name, expected in reference.items():
                    if not np.array_equal(labels[name], expected):
                        raise ValueError(f"paired cells disagree on {name} (seed {seed}, term {term})")
            values[int(seed)].append(np.asarray(arrays[field])[index].astype(np.float64))
    assert reference is not None
    return dict(**reference, values=values)


def pooled_mean_contrast(values_by_seed: Mapping[int, np.ndarray], episode_id: np.ndarray, *,
                         resamples: int = MIN_RESAMPLES, seed: int = 2718,
                         label: Sequence[Any] = (), confidence: float = 0.975,
                         bootstrap_seed: int | None = None,
                         return_draws: bool = False) -> dict[str, Any]:
    """Mean over seeds of per-seed query-weighted mean contrast, joint episode bootstrap.

    ``values_by_seed[s]`` holds one per-query contrast value per selected query
    (already restricted to the wanted hops) aligned with ``episode_id``.
    """
    confidence = _check(resamples, confidence)
    if not values_by_seed:
        raise ValueError("at least one training seed is required")
    training_seeds = sorted(int(s) for s in values_by_seed)
    episode_id = np.asarray(episode_id)
    arrays = []
    for s in training_seeds:
        value = np.asarray(values_by_seed[s], dtype=np.float64)
        if value.shape != episode_id.shape or value.ndim != 1:
            raise ValueError("contrast values must be 1-D and aligned with episode_id")
        if not np.all(np.isfinite(value)):
            raise ValueError("contrast values must be finite")
        arrays.append(value)
    if episode_id.size == 0:
        raise ValueError("no queries selected")
    _, clusters = np.unique(episode_id, return_inverse=True)
    counts = np.bincount(clusters).astype(np.float64)
    totals = np.stack([np.bincount(clusters, weights=value) for value in arrays])
    n_episodes = len(counts)
    if bootstrap_seed is None:
        bootstrap_seed = derive_seed(seed, "sym", "followup_pooled_contrast", training_seeds, *label)
    rng = np.random.default_rng(int(bootstrap_seed))
    boot = np.empty(resamples, dtype=np.float64)
    batch_size = max(1, min(256, 1_000_000 // (n_episodes * len(training_seeds))))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        draws = rng.integers(0, n_episodes, size=(stop - start, n_episodes))
        per_seed_boot = totals[:, draws].sum(axis=-1) / counts[draws].sum(axis=-1)
        boot[start:stop] = per_seed_boot.mean(axis=0)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    per_seed = [{"seed": s, "difference": float(value.mean())} for s, value in zip(training_seeds, arrays)]
    result: dict[str, Any] = {
        "difference": float(np.mean([row["difference"] for row in per_seed])),
        "ci_low": float(low), "ci_high": float(high), "confidence": confidence,
        "per_seed": per_seed, "training_seeds": training_seeds,
        "n_episodes": int(n_episodes), "n_queries_per_seed": int(episode_id.size),
        "resamples": int(resamples), "seed": int(seed), "bootstrap_seed": int(bootstrap_seed),
        "cluster_unit": "episode", "joint_episode_resampling": True,
        "method": "pooled_percentile_bootstrap_of_linear_contrast",
    }
    if return_draws:
        result["draws"] = boot
    return result


def pooled_ratio(numerator_by_seed: Mapping[int, np.ndarray], denominator_by_seed: Mapping[int, np.ndarray],
                 episode_id: np.ndarray, *, resamples: int = MIN_RESAMPLES, seed: int = 2718,
                 label: Sequence[Any] = (), confidence: float = 0.95,
                 bootstrap_seed: int | None = None) -> dict[str, Any]:
    """Mean over seeds of the per-seed ratio of summed numerators to summed denominators.

    The resampling is that of ``pooled_mean_contrast`` (whole episodes, the same
    draws for every seed); with a denominator of one per query it is identical to
    it. Every seed must have a positive denominator, and so must every resample
    (otherwise the share is undefined and an error is raised).
    """
    confidence = _check(resamples, confidence)
    training_seeds = sorted(int(s) for s in numerator_by_seed)
    if not training_seeds or sorted(int(s) for s in denominator_by_seed) != training_seeds:
        raise ValueError("numerators and denominators need the same nonempty training seeds")
    episode_id = np.asarray(episode_id)
    nums, dens = [], []
    for s in training_seeds:
        num = np.asarray(numerator_by_seed[s], dtype=np.float64)
        den = np.asarray(denominator_by_seed[s], dtype=np.float64)
        if num.shape != episode_id.shape or den.shape != episode_id.shape or num.ndim != 1:
            raise ValueError("numerators and denominators must be 1-D and aligned with episode_id")
        if not (np.all(np.isfinite(num)) and np.all(np.isfinite(den)) and np.all(den >= 0)):
            raise ValueError("numerators and denominators must be finite and denominators nonnegative")
        if den.sum() <= 0:
            raise ValueError(f"seed {s} has no events in the denominator")
        nums.append(num)
        dens.append(den)
    _, clusters = np.unique(episode_id, return_inverse=True)
    num_totals = np.stack([np.bincount(clusters, weights=value) for value in nums])
    den_totals = np.stack([np.bincount(clusters, weights=value) for value in dens])
    n_episodes = num_totals.shape[1]
    if bootstrap_seed is None:
        bootstrap_seed = derive_seed(seed, "sym", "followup_pooled_ratio", training_seeds, *label)
    rng = np.random.default_rng(int(bootstrap_seed))
    boot = np.empty(resamples, dtype=np.float64)
    batch_size = max(1, min(256, 1_000_000 // (n_episodes * len(training_seeds))))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        draws = rng.integers(0, n_episodes, size=(stop - start, n_episodes))
        denominator = den_totals[:, draws].sum(axis=-1)
        if np.any(denominator <= 0):
            raise ValueError("a bootstrap resample contains no events in the denominator")
        boot[start:stop] = (num_totals[:, draws].sum(axis=-1) / denominator).mean(axis=0)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    per_seed = [{"seed": s, "ratio": float(num.sum() / den.sum()), "numerator": float(num.sum()),
                 "denominator": float(den.sum())} for s, num, den in zip(training_seeds, nums, dens)]
    return {"ratio": float(np.mean([row["ratio"] for row in per_seed])), "ci_low": float(low),
            "ci_high": float(high), "confidence": confidence, "per_seed": per_seed,
            "training_seeds": training_seeds, "n_episodes": int(n_episodes), "resamples": int(resamples),
            "seed": int(seed), "bootstrap_seed": int(bootstrap_seed), "cluster_unit": "episode",
            "joint_episode_resampling": True, "method": "pooled_percentile_bootstrap_of_ratio_of_sums"}


def independent_difference(a: Mapping[str, Any], b: Mapping[str, Any], *,
                           confidence: float = 0.975) -> dict[str, Any]:
    """A minus B for two contrasts computed on independent episode sets.

    Both inputs must come from ``pooled_mean_contrast(..., return_draws=True)``
    with the same number of resamples and different bootstrap seeds.
    """
    if "draws" not in a or "draws" not in b or len(a["draws"]) != len(b["draws"]):
        raise ValueError("independent_difference needs draws of equal length from both sides")
    if a["bootstrap_seed"] == b["bootstrap_seed"]:
        raise ValueError("independent contrasts must use different bootstrap streams")
    confidence = _check(len(a["draws"]), confidence)
    boot = np.asarray(a["draws"]) - np.asarray(b["draws"])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    return {"difference": float(a["difference"] - b["difference"]),
            "ci_low": float(low), "ci_high": float(high), "confidence": confidence,
            "resamples": int(len(boot)), "method": "independent_stream_difference"}


def public(result: Mapping[str, Any]) -> dict[str, Any]:
    """A JSON-ready copy without the raw bootstrap draws."""
    return {key: value for key, value in result.items() if key != "draws"}


def passes_gain(result: Mapping[str, Any], minimum: float) -> bool:
    """Registered pattern: point estimate at least ``minimum`` and CI lower bound above zero."""
    value = float(result["difference"])
    return math.isfinite(value) and value >= minimum and float(result["ci_low"]) > 0
