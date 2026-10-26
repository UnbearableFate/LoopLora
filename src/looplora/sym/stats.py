"""Episode-paired accuracy inference for symbolic evaluations.

Run on a compute node with ``PYTHONPATH=src``::

    python -m looplora.sym.stats --samples runs/model/samples.npz --out stats.json
    python -m looplora.sym.stats --samples runs/m1/samples.npz \
        --reference runs/b4/samples.npz --depth-a 4 --depth-b 1 \
        --label-a M1 --label-b B4 --out paired.json

All comparisons retain split and condition labels. Bootstrap draws resample
whole episodes, retaining every query in each selected episode. The estimand
is query accuracy (a ratio of query totals), including unequal episode sizes.
"""
from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--depth-a", type=int)
    parser.add_argument("--depth-b", type=int)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.reference is not None:
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
