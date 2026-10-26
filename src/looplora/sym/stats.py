"""Episode-paired accuracy inference for symbolic evaluations.

Run on a compute node with ``PYTHONPATH=src``::

    python -m looplora.sym.stats --samples runs/model/samples.npz --out stats.json
    python -m looplora.sym.stats --samples runs/m1/samples.npz \
        --reference runs/b4/samples.npz --depth-a 4 --depth-b 1 \
        --label-a M1 --label-b B4 --out paired.json
    python -m looplora.sym.stats --pooled-a m1.json --pooled-b b5.json \
        --depth-a 4 --depth-b 4 --hops 2 3 --split val \
        --condition original --confidence 0.975 --label-a M1 --label-b B5 --out pooled.json

Pooled manifests are JSON objects such as ``{"17": "runs/seed17/samples.npz",
"29": "runs/seed29/samples.npz"}``; relative paths resolve against the manifest
directory. Pass the same manifest on both sides for a within-checkpoint depth
contrast. Pooled seeds are fixed experimental replicates: bootstrap uncertainty
comes from shared evaluation episodes, without independently resampling seeds.
Pooled confidence defaults to 0.95; preregistered gates use 0.975.

All comparisons retain split and condition labels. Bootstrap draws resample
whole episodes, retaining every query in each selected episode. The estimand
is query accuracy (a ratio of query totals), including unequal episode sizes.
"""
from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from looplora.infra.seeds import derive_seed

SampleSource = Mapping[str, Any] | str | Path
_REQUIRED = ("split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct")


def load_samples(source: SampleSource) -> dict[str, np.ndarray]:
    """Load flat evaluator arrays, rejecting ambiguous or malformed records."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / "samples.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    else:
        arrays = {key: np.asarray(value) for key, value in source.items()}
    missing = set(_REQUIRED) - arrays.keys()
    if missing:
        raise ValueError(f"Missing sample arrays: {sorted(missing)}")
    count = arrays["correct"].size
    if count == 0:
        raise ValueError("Samples must contain at least one query")
    for key in _REQUIRED:
        if arrays[key].ndim != 1 or len(arrays[key]) != count:
            raise ValueError(f"Sample array {key!r} must have shape ({count},)")
    for key in ("split", "condition"):
        if arrays[key].dtype.kind not in "US" or np.any(arrays[key] == ""):
            raise ValueError(f"Sample array {key!r} must contain nonempty text labels")
        arrays[key] = arrays[key].astype(str)
    for key in ("episode_id", "query_id", "answer", "hop", "L"):
        if arrays[key].dtype.kind not in "iu":
            raise ValueError(f"Sample array {key!r} must contain integers")
    if np.any(arrays["hop"] < 1) or np.any(arrays["L"] < 0):
        raise ValueError("Hop labels must be positive and depths must be nonnegative")
    if arrays["correct"].dtype.kind not in "biuf" or not np.all(
        (arrays["correct"] == 0) | (arrays["correct"] == 1)
    ):
        raise ValueError("Correctness must contain only binary zero/one values")
    keys = zip(*(arrays[key].tolist() for key in ("split", "condition", "L", "episode_id", "query_id")))
    if len(set(keys)) != count:
        raise ValueError("Duplicate episode/query records within a split, condition and depth")
    return arrays


def _depth_rows(arrays: Mapping[str, np.ndarray], depth: int) -> dict[tuple[Any, ...], int]:
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 0:
        raise ValueError("Comparison depths must be nonnegative integers")
    indices = np.flatnonzero(arrays["L"] == depth)
    if not len(indices):
        raise ValueError(f"No samples at L={depth}")
    return {
        (str(arrays["split"][i]), str(arrays["condition"][i]),
         int(arrays["episode_id"][i]), int(arrays["query_id"][i])): int(i)
        for i in indices
    }


def _check_resamples(resamples: int) -> None:
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 2000:
        raise ValueError("Episode bootstrap requires at least 2000 integer resamples")


def compare_runs(
    samples_a: SampleSource,
    samples_b: SampleSource,
    depth_a: int,
    depth_b: int,
    resamples: int = 2000,
    seed: int = 2718,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> list[dict[str, Any]]:
    """Return A-minus-B paired CIs, pooled and by hop for every split/condition.

    The two selected depths must have exactly the same episode/query keys and
    answer/hop labels. Record ordering may differ. No intersection, filtering
    or fallback to an unpaired test is performed when datasets do not match.
    """
    _check_resamples(resamples)
    a, b = load_samples(samples_a), load_samples(samples_b)
    rows_a, rows_b = _depth_rows(a, depth_a), _depth_rows(b, depth_b)
    if rows_a.keys() != rows_b.keys():
        missing_a = len(rows_b.keys() - rows_a.keys())
        missing_b = len(rows_a.keys() - rows_b.keys())
        raise ValueError(
            "Paired samples require identical split/condition/episode/query labels "
            f"(missing from A: {missing_a}; missing from B: {missing_b})"
        )
    keys = sorted(rows_a)
    ia = np.array([rows_a[key] for key in keys], dtype=np.int64)
    ib = np.array([rows_b[key] for key in keys], dtype=np.int64)
    for field in ("hop", "answer"):
        if not np.array_equal(a[field][ia], b[field][ib]):
            raise ValueError(f"Paired samples have different {field} labels")
    results: list[dict[str, Any]] = []
    groups = sorted({key[:2] for key in keys})
    for split, condition in groups:
        group = np.array([key[:2] == (split, condition) for key in keys], dtype=bool)
        hops: list[int | None] = [None, *sorted(int(h) for h in np.unique(a["hop"][ia[group]]))]
        for hop in hops:
            keep = group if hop is None else group & (a["hop"][ia] == hop)
            selected_a, selected_b = ia[keep], ib[keep]
            episode_ids = a["episode_id"][selected_a]
            _, clusters = np.unique(episode_ids, return_inverse=True)
            accuracy_a = a["correct"][selected_a].astype(np.float64)
            accuracy_b = b["correct"][selected_b].astype(np.float64)
            totals = np.bincount(clusters, weights=accuracy_a - accuracy_b)
            counts = np.bincount(clusters).astype(np.float64)
            n_episodes = len(counts)
            bootstrap_seed = derive_seed(seed, "sym", "paired_episode_bootstrap", split,
                                         condition, hop, int(depth_a), int(depth_b))
            rng = np.random.default_rng(bootstrap_seed)
            boot = np.empty(resamples, dtype=np.float64)
            # Bound the intermediate draw matrix even for large eval sets.
            batch_size = max(1, min(256, 1_000_000 // n_episodes))
            for start in range(0, resamples, batch_size):
                stop = min(start + batch_size, resamples)
                draws = rng.integers(0, n_episodes, size=(stop - start, n_episodes))
                boot[start:stop] = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            results.append({
                "comparison": f"{label_a} - {label_b}",
                "split": split, "condition": condition, "hop": hop,
                "depth_a": int(depth_a), "depth_b": int(depth_b),
                "n_queries": len(selected_a), "n_episodes": n_episodes,
                "accuracy_a": float(accuracy_a.mean()), "accuracy_b": float(accuracy_b.mean()),
                "difference": float((accuracy_a - accuracy_b).mean()),
                "ci_low": float(low), "ci_high": float(high), "confidence": 0.95,
                "resamples": int(resamples), "seed": int(seed),
                "bootstrap_seed": bootstrap_seed, "cluster_unit": "episode",
                "method": "paired_percentile_bootstrap",
            })
    return results


def depth_comparisons(
    samples: SampleSource, resamples: int = 2000, seed: int = 2718,
) -> list[dict[str, Any]]:
    """Compute the preregistered L=4 minus 1 and L=8 minus 4 contrasts.

    Omit a contrast when either depth is entirely absent (e.g. fixed-L B4).
    When both depths exist, missing groups or queries are pairing errors.
    """
    _check_resamples(resamples)
    arrays = load_samples(samples)
    depths = set(arrays["L"].tolist())
    results: list[dict[str, Any]] = []
    for higher, lower in ((4, 1), (8, 4)):
        if higher in depths and lower in depths:
            results.extend(compare_runs(arrays, arrays, higher, lower, resamples, seed,
                                        label_a=f"L={higher}", label_b=f"L={lower}"))
    return results


def pooled_compare_runs(
    samples_a_by_seed: Mapping[int, SampleSource],
    samples_b_by_seed: Mapping[int, SampleSource],
    depth_a: int,
    depth_b: int,
    *,
    hops: list[int] | tuple[int, ...],
    split: str = "val",
    condition: str = "original",
    resamples: int = 2000,
    seed: int = 2718,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Mean per-seed A-minus-B accuracy with a joint episode bootstrap.

    Every selected split/condition must have identical episode/query keys and
    hop/answer labels across both sides and all training seeds. Explicit hop
    selection is applied after this pairing check. Each bootstrap draw uses
    the same episodes, including all their selected queries, for every seed;
    it then averages the per-seed accuracy differences with equal seed weights.
    This CI conditions on the supplied training seeds; seeds are not clusters.
    Confidence changes only the percentile cutoffs, retaining identical draws.
    """
    _check_resamples(resamples)
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not 0 < confidence < 1):
        raise ValueError("Confidence must be a finite real number strictly between zero and one")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Confidence must be a finite real number strictly between zero and one")
    for sources in (samples_a_by_seed, samples_b_by_seed):
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError("Pooled comparisons require nonempty mappings keyed by training seed")
        if any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0
               for s in sources):
            raise ValueError("Training seed keys must be nonnegative integers")
    if samples_a_by_seed.keys() != samples_b_by_seed.keys():
        raise ValueError("Pooled comparisons require identical training seed keys on both sides")
    if not isinstance(split, str) or not split or not isinstance(condition, str) or not condition:
        raise ValueError("Pooled comparisons require explicit nonempty split and condition labels")
    if not isinstance(hops, (list, tuple)) or not hops:
        raise ValueError("Pooled comparisons require a nonempty explicit hop subset")
    if any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1 for h in hops):
        raise ValueError("Hop subset must contain positive integers")
    selected_hops = sorted(int(h) for h in hops)
    if len(set(selected_hops)) != len(selected_hops):
        raise ValueError("Hop subset must contain distinct hops")
    training_seeds = sorted(int(s) for s in samples_a_by_seed)
    reference_keys: list[tuple[Any, ...]] | None = None
    reference_hops: np.ndarray | None = None
    reference_answers: np.ndarray | None = None
    per_seed: list[dict[str, Any]] = []
    differences: list[np.ndarray] = []
    episode_ids: np.ndarray | None = None
    for training_seed in training_seeds:
        selected: list[np.ndarray] = []
        for side, sources, depth in (("A", samples_a_by_seed, depth_a),
                                     ("B", samples_b_by_seed, depth_b)):
            arrays = load_samples(sources[training_seed])
            rows = {key: index for key, index in _depth_rows(arrays, depth).items()
                    if key[:2] == (split, condition)}
            if not rows:
                raise ValueError(f"No samples for seed {training_seed}, side {side}, "
                                 f"split={split}, condition={condition}, L={depth}")
            keys = sorted(rows)
            indices = np.asarray([rows[key] for key in keys], dtype=np.int64)
            if reference_keys is None:
                reference_keys = keys
                reference_hops = arrays["hop"][indices]
                reference_answers = arrays["answer"][indices]
            elif keys != reference_keys:
                raise ValueError("Pooled paired samples require identical split/condition/episode/query "
                                 f"labels across all seeds and sides (seed {training_seed}, side {side})")
            elif not np.array_equal(arrays["hop"][indices], reference_hops):
                raise ValueError(f"Pooled paired samples have different hop labels "
                                 f"(seed {training_seed}, side {side})")
            elif not np.array_equal(arrays["answer"][indices], reference_answers):
                raise ValueError(f"Pooled paired samples have different answer labels "
                                 f"(seed {training_seed}, side {side})")
            keep = np.isin(arrays["hop"][indices], selected_hops)
            present = set(int(h) for h in arrays["hop"][indices][keep])
            if present != set(selected_hops):
                raise ValueError(f"Requested hops are absent from seed {training_seed}, side {side}: "
                                 f"{sorted(set(selected_hops) - present)}")
            indices = indices[keep]
            selected.append(arrays["correct"][indices].astype(np.float64))
            if episode_ids is None:
                episode_ids = arrays["episode_id"][indices]
        a, b = selected
        delta = a - b
        differences.append(delta)
        per_seed.append({"seed": training_seed, "accuracy_a": float(a.mean()),
                         "accuracy_b": float(b.mean()), "difference": float(delta.mean())})
    # Every side/seed has the same ordered query labels, so cluster sizes are
    # common. Unequal episode sizes still use query-weighted accuracy ratios.
    assert episode_ids is not None
    _, clusters = np.unique(episode_ids, return_inverse=True)
    counts = np.bincount(clusters).astype(np.float64)
    totals = np.stack([np.bincount(clusters, weights=delta) for delta in differences])
    n_episodes = len(counts)
    bootstrap_seed = derive_seed(seed, "sym", "pooled_paired_episode_bootstrap", training_seeds,
                                 split, condition, selected_hops, int(depth_a), int(depth_b))
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.empty(resamples, dtype=np.float64)
    batch_size = max(1, min(256, 1_000_000 // (n_episodes * len(training_seeds))))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        draws = rng.integers(0, n_episodes, size=(stop - start, n_episodes))
        per_seed_boot = totals[:, draws].sum(axis=-1) / counts[draws].sum(axis=-1)
        boot[start:stop] = per_seed_boot.mean(axis=0)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    return {
        "comparison": "A - B", "split": split, "condition": condition, "hops": selected_hops,
        "depth_a": int(depth_a), "depth_b": int(depth_b),
        "training_seeds": training_seeds, "n_seeds": len(training_seeds), "per_seed": per_seed,
        "n_episodes": n_episodes, "n_queries_per_seed": len(episode_ids),
        "accuracy_a": float(np.mean([row["accuracy_a"] for row in per_seed])),
        "accuracy_b": float(np.mean([row["accuracy_b"] for row in per_seed])),
        "difference": float(np.mean([row["difference"] for row in per_seed])),
        "ci_low": float(low), "ci_high": float(high), "confidence": confidence,
        "resamples": int(resamples), "seed": int(seed), "bootstrap_seed": bootstrap_seed,
        "cluster_unit": "episode", "joint_episode_resampling": True,
        "statistic": "mean_per_seed_paired_accuracy_difference",
        "method": "paired_multiseed_percentile_bootstrap",
    }


def pooled_walk_specificity(
    samples_by_seed: Mapping[int, SampleSource],
    depth_a: int = 4,
    depth_b: int = 1,
    *,
    reach_hops: list[int] | tuple[int, ...] = (2, 3),
    far_hops: list[int] | tuple[int, ...] = (6, 7, 8),
    split: str = "test",
    condition: str = "original",
    resamples: int = 2000,
    seed: int = 2718,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap the per-episode reachable-minus-far depth-effect contrast.

    Query labels must agree across depths and training seeds. Every episode
    must cover every requested hop. Within each episode, average query effects
    separately in the two hop subsets, subtract, then average episodes and
    seeds equally. Complete episode contrasts are resampled jointly across
    seeds, preserving covariance between the hop subsets. Confidence changes
    only percentile cutoffs. No confirmation files are selected by this API.
    """
    _check_resamples(resamples)
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not 0 < confidence < 1):
        raise ValueError("Confidence must be a finite real number strictly between zero and one")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Confidence must be a finite real number strictly between zero and one")
    if not isinstance(samples_by_seed, Mapping) or not samples_by_seed:
        raise ValueError("Specificity requires a nonempty mapping keyed by training seed")
    if any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in samples_by_seed):
        raise ValueError("Training seed keys must be nonnegative integers")
    if not isinstance(split, str) or not split or not isinstance(condition, str) or not condition:
        raise ValueError("Specificity requires nonempty split and condition labels")
    subsets = []
    for hops in (reach_hops, far_hops):
        if (not isinstance(hops, (list, tuple)) or not hops
                or any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1 for h in hops)
                or len(set(hops)) != len(hops)):
            raise ValueError("Specificity hop subsets must contain distinct positive integers")
        subsets.append(sorted(int(h) for h in hops))
    reach, far = subsets
    if set(reach) & set(far):
        raise ValueError("Specificity reachable and far hop subsets must be disjoint")
    training_seeds = sorted(int(s) for s in samples_by_seed)
    reference_keys = None
    reference_hops = reference_answers = None
    deltas = []
    for training_seed in training_seeds:
        arrays = load_samples(samples_by_seed[training_seed])
        paired = []
        for depth in (depth_a, depth_b):
            rows = {key: index for key, index in _depth_rows(arrays, depth).items()
                    if key[:2] == (split, condition)}
            if not rows:
                raise ValueError(f"No specificity samples for seed {training_seed}, L={depth}, {split}/{condition}")
            keys = sorted(rows)
            indices = np.asarray([rows[key] for key in keys], dtype=np.int64)
            if reference_keys is None:
                reference_keys = keys
                reference_hops = arrays["hop"][indices]
                reference_answers = arrays["answer"][indices]
            elif (keys != reference_keys
                  or not np.array_equal(arrays["hop"][indices], reference_hops)
                  or not np.array_equal(arrays["answer"][indices], reference_answers)):
                raise ValueError("Specificity requires identical episode/query/hop/answer labels across depths and seeds")
            paired.append(arrays["correct"][indices].astype(np.float64))
        deltas.append(paired[0] - paired[1])
    assert reference_keys is not None and reference_hops is not None
    episode_ids = np.asarray([key[2] for key in reference_keys])
    _, clusters = np.unique(episode_ids, return_inverse=True)
    n_episodes = int(clusters.max()) + 1
    subset_effects = []
    query_counts = {}
    for label, hops in (("reach", reach), ("far", far)):
        for hop in hops:
            hop_counts = np.bincount(clusters[reference_hops == hop], minlength=n_episodes)
            if np.any(hop_counts == 0):
                raise ValueError(f"Every episode must cover specificity hop {hop}")
        mask = np.isin(reference_hops, hops)
        counts = np.bincount(clusters[mask], minlength=n_episodes)
        query_counts[label] = int(mask.sum())
        subset_effects.append(np.stack([
            np.bincount(clusters[mask], weights=delta[mask], minlength=n_episodes) / counts
            for delta in deltas]))
    reach_effect, far_effect = subset_effects
    contrasts = reach_effect - far_effect
    bootstrap_seed = derive_seed(seed, "sym", "pooled_episode_walk_specificity", training_seeds,
                                 split, condition, reach, far, int(depth_a), int(depth_b))
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.empty(resamples, dtype=np.float64)
    batch_size = max(1, min(256, 1_000_000 // (n_episodes * len(training_seeds))))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        draws = rng.integers(0, n_episodes, size=(stop - start, n_episodes))
        boot[start:stop] = contrasts[:, draws].mean(axis=-1).mean(axis=0)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    return {
        "comparison": "reachable depth effect - far depth effect", "split": split, "condition": condition,
        "depth_a": int(depth_a), "depth_b": int(depth_b), "reach_hops": reach, "far_hops": far,
        "training_seeds": training_seeds, "n_seeds": len(training_seeds),
        "n_episodes": n_episodes, "n_queries_per_seed": sum(query_counts.values()),
        "query_counts_by_subset": query_counts, "episode_weighting": "equal",
        "reach_difference": float(reach_effect.mean()), "far_difference": float(far_effect.mean()),
        "difference": float(contrasts.mean()), "ci_low": float(low), "ci_high": float(high),
        "confidence": confidence, "resamples": int(resamples), "seed": int(seed),
        "bootstrap_seed": bootstrap_seed, "cluster_unit": "episode", "joint_episode_resampling": True,
        "statistic": "mean_per_seed_episode_walk_specificity", "method": "paired_multiseed_percentile_bootstrap",
        "per_seed": [{"seed": training_seed, "reach_difference": float(reach_effect[i].mean()),
                      "far_difference": float(far_effect[i].mean()), "difference": float(contrasts[i].mean())}
                     for i, training_seed in enumerate(training_seeds)],
    }


def _load_manifest(path: Path) -> dict[int, Path]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate training seed in pooled manifest: {key}")
            result[key] = value
        return result

    manifest = json.loads(path.read_text(), object_pairs_hook=unique_pairs)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Pooled manifest must be a nonempty JSON object of seed: sample-path entries")
    result: dict[int, Path] = {}
    for key, value in manifest.items():
        try:
            training_seed = int(key)
        except ValueError as exc:
            raise ValueError(f"Invalid training seed in pooled manifest: {key!r}") from exc
        if training_seed < 0 or training_seed in result:
            raise ValueError(f"Negative or duplicate training seed in pooled manifest: {key!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Sample path for training seed {key} must be nonempty text")
        sample_path = Path(value)
        result[training_seed] = sample_path if sample_path.is_absolute() else path.parent / sample_path
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", type=Path)
    source.add_argument("--pooled-a", type=Path)
    parser.add_argument("--pooled-b", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--depth-a", type=int)
    parser.add_argument("--depth-b", type=int)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--hops", nargs="+", type=int)
    parser.add_argument("--split", default="val")
    parser.add_argument("--condition", default="original")
    parser.add_argument("--confidence", type=float, help="Pooled CI confidence (default: 0.95; gates: 0.975)")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.pooled_a is not None:
        if args.pooled_b is None or args.depth_a is None or args.depth_b is None or args.hops is None:
            parser.error("--pooled-a requires --pooled-b, --depth-a, --depth-b and --hops")
        if args.reference is not None:
            parser.error("--reference cannot be combined with pooled manifests")
        result = pooled_compare_runs(_load_manifest(args.pooled_a), _load_manifest(args.pooled_b),
                                     args.depth_a, args.depth_b, hops=args.hops,
                                     split=args.split, condition=args.condition,
                                     resamples=args.resamples, seed=args.seed,
                                     confidence=0.95 if args.confidence is None else args.confidence)
        result["comparison"] = f"{args.label_a} - {args.label_b}"
    elif (args.pooled_b is not None or args.hops is not None or args.split != "val"
          or args.condition != "original" or args.confidence is not None):
        parser.error("--pooled-b, --hops, --split, --condition and --confidence require --pooled-a")
    elif args.reference is not None:
        if args.depth_a is None or args.depth_b is None:
            parser.error("--reference requires both --depth-a and --depth-b")
        result = compare_runs(args.samples, args.reference, args.depth_a, args.depth_b,
                              args.resamples, args.seed, label_a=args.label_a, label_b=args.label_b)
    else:
        if args.depth_a is not None or args.depth_b is not None:
            parser.error("Depth arguments require --reference")
        result = depth_comparisons(args.samples, args.resamples, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
