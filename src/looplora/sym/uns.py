"""UNS (F11): audit of the unseen-pool evaluations that P1-S, P1-S-F1 and P1-S-F2 already ran.

Nothing is trained or evaluated here and the module needs only NumPy. Every
registered reader checkpoint was evaluated once, in its screening evaluation
(eval seed 4242, 2000 episodes, hops 1-8, depths 0-16), on both the seen pool
(split ``test``, entities 0-767) and the unseen pool (split ``test_unseen``,
entities 768-1023). The audit (a) checks that those saved samples follow the
registered protocol, (b) recomputes the committed unseen-pool M1 depth contrasts
from the samples and compares them with the exported rows, (c) tabulates the
seen-pool versus unseen-pool accuracies and contrasts with episode-bootstrap
intervals, and (d) transcribes the remaining committed rows for context.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from looplora.infra.seeds import derive_seed, stable_hash
from . import followup_stats as fs
from .stats import load_samples, pooled_compare_runs

POOLS = (("seen", "test"), ("unseen", "test_unseen"))
CONDITION = "original"
EVAL_SEED = 4242
EPISODES = 2000
HOPS = tuple(range(1, 9))
DEPTHS = (0, 1, 2, 3, 4, 6, 8, 12, 16)
ENTITY_RANGE = {"test": (0, 768), "test_unseen": (768, 1024)}
CANDIDATES = 14
GROUPS = {"hop1": (1,), "reach": (2, 3), "far": (6, 7, 8)}
CELLS = (("M1", 1), ("M1", 4), ("B4", 1))
ENDPOINTS = {"m1_depth_gain": (("M1", 4), ("M1", 1)), "m1_l4_minus_b4": (("M1", 4), ("B4", 1))}
WRITERS = ("joint", "joint_fact")
CONTROL = "oracle_random"
READERS = ("M1", "B4")
SEEDS = (17, 29, 43)
FACTORIZED = ("joint_fact", "t1_frozen_fact")
REPRODUCED = {"hop1": (1,), "reach": (2, 3), "far": (6, 7, 8)}
CONTEXT_ENDPOINTS = ("hop1", "reach", "far", "specificity", "vs_fixed", "vs_loop")
BOOTSTRAP_SEED = 2718  # the seed of the committed P1-S/F1/F2 analyses
KEEP = ("split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct", "prediction",
        "candidate_size")
# Part B (docs/prereg/UNS.md): the screening protocol on new episodes, for the readers that the committed
# unseen-pool rows do not cover (the P2-S-F1 M1 trained at the fixed depth 4) and their B4 references.
EXTENSION_SEED = 999006  # never used by another evaluation
EXTENSION = dict(seed=EXTENSION_SEED, episodes=EPISODES, splits=[split for _, split in POOLS], hops=list(HOPS),
                 depths=list(DEPTHS), batch_size=64)


def extension_name(writer: str, reader: str, seed: int) -> str:
    """Name of one Part B evaluation; there the reader 'M1' is the one trained at L = 4."""
    return f"{writer}-{'M1L4' if reader == 'M1' else reader}-r32-s{seed}"


def run_root(writer: str, reader: str, seed: int, *, p1s_root: str | Path, f2_root: str | Path) -> Path:
    """Directory of one released r = 32 run (checkpoint.pt, screening/ and confirmation/ evaluations)."""
    if writer in FACTORIZED:
        return Path(f2_root) / "evaluation" / f"f2-main-{writer}-{reader}-s{seed}"
    return Path(p1s_root) / "runs" / f"primary-{writer}-{reader}-r32-s{seed}"


def run_directory(writer: str, reader: str, seed: int, *, p1s_root: str | Path, f2_root: str | Path,
                  dataset: str = "screening") -> Path:
    """Directory of one run's ``dataset`` evaluation (screening or confirmation); it holds samples.npz.

    The screening directory also holds metrics.json. A confirmation directory holds
    metrics.json only after its aggregation, so only samples.npz is required there.
    """
    if dataset not in ("screening", "confirmation"):
        raise ValueError("dataset must be 'screening' or 'confirmation'")
    path = run_root(writer, reader, seed, p1s_root=p1s_root, f2_root=f2_root) / dataset
    for name in ("samples.npz",) + (("metrics.json",) if dataset == "screening" else ()):
        if not (path / name).is_file():
            raise FileNotFoundError(f"missing {path / name}")
    return path


def file_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restrict(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The original-condition rows of the two pools, reduced to the columns the audit uses."""
    keep = (np.asarray(arrays["condition"]) == CONDITION) & np.isin(np.asarray(arrays["split"]),
                                                                     [split for _, split in POOLS])
    return {name: np.asarray(arrays[name])[keep] for name in KEEP if name in arrays}


def load_original(path: str | Path) -> dict[str, np.ndarray]:
    """Validated samples of one evaluation, restricted as in ``restrict``."""
    return restrict(load_samples(Path(path)))


def audit_samples(samples: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Structural checks of one screening evaluation; every violated check is listed in ``failures``."""
    failures: list[str] = []
    pools: dict[str, Any] = {}
    episodes: dict[str, np.ndarray] = {}
    for name in ("prediction", "candidate_size"):
        if name not in samples:
            failures.append(f"samples lack the {name!r} array")
    for pool, split in POOLS:
        rows = np.flatnonzero(np.asarray(samples["split"]) == split)
        if not rows.size:
            failures.append(f"{split}: no original-condition rows")
            continue
        low, high = ENTITY_RANGE[split]
        answer = np.asarray(samples["answer"])[rows]
        if answer.min() < low or answer.max() >= high:
            failures.append(f"{split}: answers outside the entity pool [{low}, {high})")
        if "prediction" in samples:
            prediction = np.asarray(samples["prediction"])[rows]
            if prediction.min() < low or prediction.max() >= high:
                failures.append(f"{split}: predictions outside the entity pool [{low}, {high})")
        depths = sorted({int(value) for value in np.asarray(samples["L"])[rows]})
        if depths != list(DEPTHS):
            failures.append(f"{split}: depths {depths} differ from the registered {list(DEPTHS)}")
        hops = sorted({int(value) for value in np.asarray(samples["hop"])[rows]})
        if hops != list(HOPS):
            failures.append(f"{split}: hops {hops} differ from the registered {list(HOPS)}")
        episode_ids = np.unique(np.asarray(samples["episode_id"])[rows])
        episodes[pool] = episode_ids
        if len(episode_ids) != EPISODES:
            failures.append(f"{split}: {len(episode_ids)} episodes instead of {EPISODES}")
        reference = None
        for depth in depths:
            cell = rows[np.asarray(samples["L"])[rows] == depth]
            order = np.lexsort((np.asarray(samples["query_id"])[cell], np.asarray(samples["episode_id"])[cell]))
            cell = cell[order]
            labels = [np.asarray(samples[name])[cell] for name in ("episode_id", "query_id", "hop", "answer")]
            if reference is None:
                reference = labels
            elif not all(np.array_equal(a, b) for a, b in zip(labels, reference)):
                failures.append(f"{split}: query labels differ between depths (L={depth})")
            hop_counts = np.bincount(np.asarray(samples["hop"])[cell].astype(np.int64), minlength=max(HOPS) + 1)
            if any(hop_counts[hop] != EPISODES for hop in HOPS) or len(cell) != EPISODES * len(HOPS):
                failures.append(f"{split}: L={depth} lacks one query per hop per episode")
        if "prediction" in samples and not np.array_equal(
                np.asarray(samples["correct"])[rows].astype(bool),
                np.asarray(samples["prediction"])[rows] == answer):
            failures.append(f"{split}: saved correctness differs from prediction == answer")
        sizes = sorted({int(value) for value in np.asarray(samples["candidate_size"])[rows]}) \
            if "candidate_size" in samples else []
        if "candidate_size" in samples and sizes != [CANDIDATES]:
            failures.append(f"{split}: candidate sizes {sizes} differ from {CANDIDATES}")
        pools[pool] = dict(split=split, rows=int(rows.size), episodes=int(len(episode_ids)), depths=depths,
                           hops=hops, candidate_sizes=sizes)
    if len(episodes) == 2 and np.intersect1d(episodes["seen"], episodes["unseen"]).size:
        failures.append("seen and unseen pools share episode ids")
    return dict(pools=pools, failures=failures)


def audit_run(directory: str | Path, *, writer: str, reader: str, seed: int,
              evaluation_seed: int = EVAL_SEED) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Protocol, hash and sample audit of one evaluation; returns (report, restricted samples).

    The registered protocol is the screening one (seed 4242); ``evaluation_seed`` selects another
    seed for the UNS extension, whose evaluations follow the same protocol on new episodes.
    """
    directory = Path(directory)
    failures: list[str] = []
    metadata = json.loads((directory / "metrics.json").read_text())
    config = metadata["config"]
    settings, task = config["eval"], config["task"]

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    check(metadata.get("evaluation_seed") == evaluation_seed and settings["seed"] == evaluation_seed,
          f"evaluation seed is not {evaluation_seed}")
    check(metadata.get("config_hash") == stable_hash(config), "config hash differs from the recorded one")
    check(settings["splits"] == [split for _, split in POOLS], "evaluation splits differ")
    check(settings["episodes"] == EPISODES, "evaluation episode count differs")
    check(list(settings["hops"]) == list(HOPS) and list(settings["depths"]) == list(DEPTHS),
          "evaluation hops or depths differ")
    check(task["name"] == "t2_chain_end" and task["E"] == 12 and list(task["chain_lengths"]) == [8, 4]
          and task["V"] == 1024, "task differs from t2_chain_end E=12 chains (8, 4) over V=1024")
    check(list(task["seen"]) == list(ENTITY_RANGE["test"]) and list(task["unseen"]) == list(ENTITY_RANGE["test_unseen"]),
          "entity pools differ")
    check(config["writer"] == writer and config["variant"] == reader and config["seed"] == seed
          and config["model"]["r"] == 32, "run identity differs from the requested writer/reader/seed/r=32")
    check(file_hash(directory / "samples.npz") == metadata.get("samples_hash"),
          "sample file hash differs from the recorded one")
    samples = load_original(directory / "samples.npz")
    structure = audit_samples(samples)
    failures.extend(structure["failures"])
    report = dict(writer=writer, reader=reader, seed=seed, directory=str(directory),
                  samples_hash=metadata.get("samples_hash"), pools=structure["pools"], failures=failures)
    return report, samples


def _cell(samples: Mapping[str, np.ndarray], split: str, depth: int) -> dict[str, np.ndarray]:
    index = fs.select_rows(samples, split, CONDITION, depth)
    cell = {name: np.asarray(samples[name])[index] for name in ("episode_id", "query_id", "hop", "answer")}
    cell["correct"] = np.asarray(samples["correct"])[index].astype(np.float64)
    cell["candidate_size"] = np.asarray(samples["candidate_size"])[index].astype(np.float64)
    return cell


def _paired(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], hops: Sequence[int]):
    for name in ("episode_id", "query_id", "hop", "answer"):
        if not np.array_equal(a[name], b[name]):
            raise ValueError(f"paired cells disagree on {name}")
    keep = np.isin(a["hop"], list(hops))
    if not keep.any():
        raise ValueError(f"no queries at hops {list(hops)}")
    return a["episode_id"][keep], (a["correct"] - b["correct"])[keep]


def _pool_results(values_by_seed: Mapping[int, np.ndarray], episode_id: np.ndarray, *, label: Sequence[Any],
                  resamples: int, confidence: float, seed: int) -> dict[str, Any]:
    return fs.pooled_mean_contrast(values_by_seed, episode_id, resamples=resamples, confidence=confidence,
                                   bootstrap_seed=derive_seed(seed, "uns", *label), return_draws=True)


def _row(label: Mapping[str, Any], pool: str, result: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(label, pool=pool, difference=result["difference"], ci_low=result["ci_low"],
               ci_high=result["ci_high"], confidence=result["confidence"], n_episodes=result.get("n_episodes", ""))
    for entry in result["per_seed"]:
        row[f"seed_{entry['seed']}"] = entry["difference"]
    return row


def _gap_row(label: Mapping[str, Any], unseen: Mapping[str, Any], seen: Mapping[str, Any], *,
             confidence: float) -> dict[str, Any]:
    gap = fs.independent_difference(unseen, seen, confidence=confidence)
    row = dict(label, pool="unseen_minus_seen", difference=gap["difference"], ci_low=gap["ci_low"],
               ci_high=gap["ci_high"], confidence=gap["confidence"], n_episodes="")
    for u, s in zip(unseen["per_seed"], seen["per_seed"]):
        row[f"seed_{u['seed']}"] = u["difference"] - s["difference"]
    return row


def _stack(samples: Mapping[tuple[str, int], Mapping[str, np.ndarray]], seeds: Sequence[int], split: str,
           cell_a: tuple[str, int], cell_b: tuple[str, int], hops: Sequence[int]):
    episode_id, values = None, {}
    for s in seeds:
        ids, value = _paired(_cell(samples[cell_a[0], s], split, cell_a[1]),
                             _cell(samples[cell_b[0], s], split, cell_b[1]), hops)
        if episode_id is None:
            episode_id = ids
        elif not np.array_equal(episode_id, ids):
            raise ValueError("training seeds disagree on the evaluation episodes")
        values[s] = value
    return episode_id, values


def endpoint_rows(samples: Mapping[tuple[str, int], Mapping[str, np.ndarray]], writer: str, *,
                  resamples: int = 2000, confidence: float = 0.95, seed: int = BOOTSTRAP_SEED) -> list[dict[str, Any]]:
    """Seen/unseen M1 depth gains and M1@L4 minus B4@L1, by hop group, with unseen-minus-seen rows.

    ``samples[(reader, training_seed)]`` holds the restricted samples of one run.
    The pools hold different (independent) episodes, so the pool difference uses
    independent bootstrap streams; each pool's interval is a joint episode bootstrap.
    """
    seeds = sorted({s for _, s in samples})
    rows = []
    for endpoint, (cell_a, cell_b) in ENDPOINTS.items():
        for group, hops in GROUPS.items():
            results = {}
            for pool, split in POOLS:
                episode_id, values = _stack(samples, seeds, split, cell_a, cell_b, hops)
                results[pool] = _pool_results(values, episode_id, label=(writer, endpoint, group, pool),
                                              resamples=resamples, confidence=confidence, seed=seed)
            label = dict(writer=writer, endpoint=endpoint, hops=",".join(str(h) for h in hops), group=group)
            for pool, _ in POOLS:
                rows.append(_row(label, pool, results[pool]))
            rows.append(_gap_row(label, results["unseen"], results["seen"], confidence=confidence))
    return rows


def accuracy_rows(samples: Mapping[tuple[str, int], Mapping[str, np.ndarray]], writer: str, *,
                  resamples: int = 2000, confidence: float = 0.95, seed: int = BOOTSTRAP_SEED) -> list[dict[str, Any]]:
    """Per-hop accuracy of the registered cells on each pool, and the unseen-minus-seen difference."""
    seeds = sorted({s for _, s in samples})
    rows = []
    for reader, depth in CELLS:
        for hop in HOPS:
            results, chance = {}, {}
            for pool, split in POOLS:
                cells = {s: _cell(samples[reader, s], split, depth) for s in seeds}
                reference = cells[seeds[0]]
                keep = reference["hop"] == hop
                values = {}
                for s in seeds:
                    for name in ("episode_id", "query_id", "hop", "answer"):
                        if not np.array_equal(cells[s][name], reference[name]):
                            raise ValueError("training seeds disagree on the evaluation episodes")
                    values[s] = cells[s]["correct"][keep]
                chance[pool] = float(np.mean(1.0 / reference["candidate_size"][keep]))
                results[pool] = _pool_results(values, reference["episode_id"][keep],
                                              label=(writer, "accuracy", reader, depth, hop, pool),
                                              resamples=resamples, confidence=confidence, seed=seed)
            label = dict(writer=writer, reader=reader, L=depth, hop=hop)
            for pool, _ in POOLS:
                row = _row(label, pool, results[pool])
                row["accuracy"] = row.pop("difference")
                row["chance"] = chance[pool]
                rows.append(row)
            gap = _gap_row(label, results["unseen"], results["seen"], confidence=confidence)
            gap["accuracy"] = gap.pop("difference")
            gap["chance"] = ""
            rows.append(gap)
    return rows


def _pooled_seeds(row: Mapping[str, Any]) -> str:
    """Training seeds pooled by an exported row, e.g. ``"17,29,43"``.

    The P1-S export has a ``training_seeds`` column; the F1 and F2 exports do not, so the
    ``seed`` entries of the JSON ``per_seed`` column are used (empty when neither is usable).
    """
    if row.get("training_seeds"):
        return str(row["training_seeds"])
    try:
        return ",".join(str(int(entry["seed"])) for entry in json.loads(row.get("per_seed") or "[]"))
    except (KeyError, TypeError, ValueError):
        return ""


def load_committed(path: str | Path, study: str, r_by_group: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Every row of a committed ``reported_contrasts.csv`` with numeric fields parsed.

    The P1-S export has no rank column (its ``secondary`` group is r = 16); pass
    ``r_by_group`` for such files. Rows without either default to r = 32. Every row keeps the
    training seeds it pools: the committed ``screening`` rows pool seed 17 only, while the
    ``unseen`` rows of the primary groups pool 17, 29 and 43.
    """
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for row in rows:
        group = row.get("group", "primary")
        out.append(dict(study=study, group=group, writer=row["writer"],
                        r=row.get("r") or (r_by_group or {}).get(group, "32"),
                        reader=row["reader"], dataset=row["dataset"], endpoint=row["endpoint"],
                        confidence=float(row["confidence"]), difference=float(row["difference"]),
                        ci_low=float(row["ci_low"]), ci_high=float(row["ci_high"]),
                        training_seeds=_pooled_seeds(row)))
    return out


def reproduce_reported(samples_by_seed: Mapping[int, Mapping[str, np.ndarray]], committed: Sequence[Mapping[str, Any]],
                       *, writer: str, group: str = "primary", r: str = "32",
                       tolerance: float = 1e-9) -> list[dict[str, Any]]:
    """Recompute the committed unseen-pool M1 contrasts (L4 minus L1) and compare them with the exported rows."""
    checks = []
    for endpoint, hops in REPRODUCED.items():
        for confidence in (0.95, 0.975):
            match = [row for row in committed
                     if row["dataset"] == "unseen" and row["writer"] == writer and row["reader"] == "M1"
                     and row["endpoint"] == endpoint and row["group"] == group and str(row["r"]) == str(r)
                     and abs(row["confidence"] - confidence) < 1e-9]
            if len(match) != 1:
                raise ValueError(f"{len(match)} committed rows for {writer}/M1/{endpoint}/{confidence}")
            new = pooled_compare_runs(samples_by_seed, samples_by_seed, 4, 1, hops=hops, split="test_unseen",
                                      condition=CONDITION, resamples=2000, seed=BOOTSTRAP_SEED,
                                      confidence=confidence)
            old = match[0]
            deviation = max(abs(new[key] - old[key]) for key in ("difference", "ci_low", "ci_high"))
            checks.append(dict(writer=writer, endpoint=endpoint, confidence=confidence, recomputed=new["difference"],
                               committed=old["difference"], max_abs_deviation=float(deviation),
                               reproduced=bool(deviation <= tolerance)))
    return checks


def context_table(rows: Iterable[Mapping[str, Any]], *, confidence: float = 0.95) -> list[dict[str, Any]]:
    """Committed rows of the six named endpoints, one line per (study, group, writer, r, reader, endpoint).

    Each dataset's values are listed with the training seeds that row pools (``<dataset>_training_seeds``):
    the committed ``screening`` rows pool seed 17 only, so they are not like-for-like with the
    three-seed ``unseen`` rows.
    """
    table: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        if row["endpoint"] not in CONTEXT_ENDPOINTS or abs(row["confidence"] - confidence) > 1e-9:
            continue
        key = tuple(row[name] for name in ("study", "group", "writer", "r", "reader", "endpoint"))
        entry = table.setdefault(key, dict(zip(("study", "group", "writer", "r", "reader", "endpoint"), key)))
        for field in ("difference", "ci_low", "ci_high", "training_seeds"):
            entry[f"{row['dataset']}_{field}"] = row[field]
    return list(table.values())
