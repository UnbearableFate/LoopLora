#!/usr/bin/env python3
"""Release-gated P1-S final-v1 analysis. Run inside a compute allocation.

PYTHONPATH=src python scripts/analyze_p1s_main.py --plan runs/p1s/main/plan.json

The metadata-only release check finishes before any confirmation sample is
opened. Raw archives are streamed once; only compact original-query fields at
L=1,4,8 are cached on disk, with a bounded in-memory cache for paired inference.
No checkpoint selection or training decisions are made here.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from looplora.infra.seeds import stable_hash

WRITERS = ("oracle_random", "oracle_orth", "t1_frozen", "joint")
VARIANTS = ("M1", "M1-lin", "B5", "B5-lin", "B3", "B4")
READERS = ("M1", "M1-lin")
SEEDS = (17, 29, 43)
DEPTHS = (0, 1, 2, 3, 4, 6, 8, 12, 16)
HOPS = tuple(range(1, 9))
FIELDS = ("x", "u", "memory", "write_count")
CF = ("replace_key_fact", "remove_required_fact", "swap_memory")
A5 = "swap_successors"
BOOTSTRAP_SEED = 2718


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(run: dict) -> tuple:
    return run["group"], run["writer"], run["variant"], run["seed"], run["r"]


def verify_release(plan_path: Path) -> tuple[dict, dict]:
    """Validate the entire manifest and current completion state without npz I/O."""
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    root = Path(plan["root"]).resolve()
    require(plan_path == root / "plan.json", "analysis plan must be the root's plan.json")
    require(plan["steps"] in (15000, 20000), "unexpected main training budget")
    runs = plan["runs"]
    require(isinstance(runs, list) and len(runs) == 105, "main manifest must contain 105 runs")
    names = [run["name"] for run in runs]
    require(len(set(names)) == 105, "duplicate main run names")
    require(Counter(run["group"] for run in runs) ==
            {"writer": 3, "primary": 72, "replication": 18, "secondary": 12},
            "main manifest group counts differ from final v1")
    expected = {
        *(('primary', w, v, s, 32) for w in WRITERS for v in VARIANTS for s in SEEDS),
        *(('replication', w, v, s, 32) for w in ("oracle_random", "t1_frozen", "joint")
          for v in READERS for s in SEEDS),
        *(('secondary', w, v, 17, 16) for w in ("oracle_random", "oracle_orth", "joint")
          for v in ("M1", "M1-lin", "B5", "B5-lin")),
    }
    actual = [_identity(run) for run in runs if run["group"] != "writer"]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
            "reader matrix differs from final v1")
    require(sorted(run["seed"] for run in runs if run["group"] == "writer") == list(SEEDS),
            "auxiliary writers must cover exactly the three training seeds")
    barrier = read_json(root / "barrier.json")
    require(barrier.get("status") == "passed", "global confirmation release barrier has not passed")
    require(barrier.get("plan_hash") == stable_hash(plan), "barrier belongs to a different plan")
    require(isinstance(barrier.get("created_at"), str) and bool(barrier["created_at"]),
            "barrier has no creation timestamp")
    datetime.fromisoformat(barrier["created_at"].replace("Z", "+00:00"))
    require(isinstance(barrier.get("runs"), list) and len(barrier["runs"]) == 105
            and set(barrier["runs"]) == set(names), "barrier does not release all 105 runs")
    expected_samples = {str(Path(run["run_dir"]).resolve() / "confirmation" / "samples.npz")
                        for run in runs if run["group"] != "writer"}
    released = barrier.get("confirmation_samples")
    require(isinstance(released, list) and len(released) == 102
            and set(released) == expected_samples, "barrier does not release all 102 confirmation files")
    for run in runs:
        folder = Path(run["run_dir"]).resolve()
        require(folder == root / "runs" / run["name"], f"run outside manifest root: {run['name']}")
        record = read_json(folder / "record.json")
        require(record.get("status") == "complete" and record.get("training_complete") is True,
                f"training incomplete: {run['name']}")
        effective = record["config"]
        require(effective["train"]["steps"] == run["config"]["train"]["steps"] == plan["steps"],
                f"run training budget differs: {run['name']}")
        envelope = read_json(folder / "config.json")
        require(envelope["config"] == effective and envelope["config_hash"] == record["config_hash"] == stable_hash(effective),
                f"effective run config/hash differs: {run['name']}")
        require(record["planned_config_hash"] == stable_hash(run["config"]),
                f"planned config hash differs: {run['name']}")
        if effective != run["config"]:
            attempts = record.get("attempts", [])
            expected_config = copy.deepcopy(run["config"])
            lower = {.003: .001, .001: .0003}.get(expected_config["train"]["lr"])
            require(lower is not None and len(attempts) == 2
                    and attempts[0]["status"] == "nonfinite_loss" and attempts[1]["status"] == "complete"
                    and attempts[0]["lr"] == expected_config["train"]["lr"] and attempts[1]["lr"] == lower,
                    f"unapproved effective config change: {run['name']}")
            expected_config["train"]["lr"] = lower
            require(effective == expected_config, f"effective config changed beyond permitted LR retry: {run['name']}")
        for filename in ("checkpoint.pt", "train_log.jsonl"):
            require((folder / filename).is_file(), f"missing completed artifact: {folder / filename}")
        if run["group"] == "writer":
            require((folder / "writer.pt").is_file(), f"missing auxiliary writer: {run['name']}")
        else:
            require(record.get("confirmation_samples_written") is True,
                    f"confirmation samples incomplete: {run['name']}")
            require((folder / "confirmation" / "samples.npz").is_file(),
                    f"missing released samples: {run['name']}")
            metadata = read_json(folder / "confirmation" / "evaluation.json")
            expected_config = copy.deepcopy(effective)
            expected_config["eval"].update(splits=["test"], seed=777001, counterfactuals=False)
            require(metadata["config"] == expected_config
                    and metadata["config_hash"] == stable_hash(expected_config)
                    and metadata["checkpoint_config"] == effective
                    and metadata["checkpoint_config_hash"] == stable_hash(effective),
                    f"confirmation evaluation/checkpoint config differs: {run['name']}")
            settings = metadata["config"]["eval"]
            require(metadata.get("aggregation_deferred") is True and "table" not in metadata
                    and "diagnostics" not in metadata, f"confirmation was aggregated before release: {run['name']}")
            require(metadata.get("evaluation_seed") == settings["seed"] == 777001
                    and settings["splits"] == ["test"] and settings["episodes"] == 2000
                    and settings["counterfactuals"] is False,
                    f"confirmation set differs from preregistration: {run['name']}")
            require(settings["depths"] == list(DEPTHS) and settings["hops"] == list(HOPS),
                    f"confirmation hop/depth sweep differs: {run['name']}")
    return plan, barrier


class CompactCache:
    """Disk-backed required fields only; at most ``limit`` run/set arrays in RAM."""

    def __init__(self, root: Path, limit: int = 12):
        require(limit >= 1, "cache limit must be positive")
        self.root = root
        self.limit = limit
        self.memory: OrderedDict[tuple[str, str], dict] = OrderedDict()

    def save(self, name: str, dataset: str, arrays: dict, split: str) -> None:
        mask = ((arrays["split"] == split) & (arrays["condition"] == "original")
                & np.isin(arrays["L"], (1, 4, 8)))
        require(bool(mask.any()), f"empty compact dataset: {name}/{dataset}")
        path = self.root / dataset / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        small = {key: arrays[key][mask].astype(dtype, copy=False) for key, dtype in
                 (("episode_id", np.int64), ("query_id", np.int16), ("hop", np.int16),
                  ("L", np.int16), ("answer", np.int32), ("correct", bool))}
        np.savez_compressed(path, **small, split=np.asarray(split))

    def get(self, name: str, dataset: str) -> dict:
        key = name, dataset
        if key not in self.memory:
            with np.load(self.root / dataset / f"{name}.npz", allow_pickle=False) as archive:
                arrays = {field: archive[field] for field in archive.files if field != "split"}
                split = str(archive["split"].item())
            count = len(arrays["correct"])
            arrays["split"] = np.broadcast_to(np.asarray(split), (count,))
            arrays["condition"] = np.broadcast_to(np.asarray("original"), (count,))
            self.memory[key] = arrays
            while len(self.memory) > self.limit:
                self.memory.popitem(last=False)
        self.memory.move_to_end(key)
        return self.memory[key]


def audit_snapshots(metadata: dict, config: dict, *, writer_run: bool = False) -> dict:
    require(metadata.get("gate0_passed") is True, "evaluation did not pass immutable-depth Gate 0")
    settings = config["eval"]
    hops = [1] if writer_run else list(HOPS)
    depths = [1] if writer_run else list(DEPTHS)
    count = settings["episodes"]
    conditions = [(split, "original") for split in settings["splits"]]
    if settings.get("counterfactuals") and "test" in settings["splits"]:
        conditions += [("test", condition) for condition in CF]
        if settings.get("swap_successors"):
            conditions.append(("test", A5))
    expected = {(split, condition, begin, depth) for split, condition in conditions
                for begin in range(0, count * (len(hops) if condition in (*CF[:2], A5) else 1), settings["batch_size"])
                for depth in ((1, 4, 8) if condition == A5 else depths)}
    seen = set()
    snapshots = {}
    for row in metadata["memory_hashes"]:
        key = row["split"], row["condition"], row["batch_start"], row["L"]
        require(key in expected and key not in seen, f"unexpected/duplicate depth snapshot: {key}")
        seen.add(key)
        values = tuple(row[field] for field in FIELDS)
        require(all(isinstance(value, str) and len(value) == 32 for value in values), "invalid snapshot digest")
        group = key[:3]
        require(group not in snapshots or snapshots[group] == values, f"depth-dependent x/u/M/write_count: {group}")
        snapshots[group] = values
    require(seen == expected, "missing batch/depth snapshots")
    return snapshots


def _data_digest(arrays: dict, indices: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for field in ("episode_id", "query_id", "hop", "answer", "candidate_size", "memory_episode_id"):
        value = np.ascontiguousarray(arrays[field][indices])
        digest.update(field.encode() + str(value.dtype).encode() + value.tobytes())
    if "counterfactual_hop_true" in arrays:
        digest.update(np.ascontiguousarray(arrays["counterfactual_hop_true"][indices]).tobytes())
    return digest.hexdigest()


def audit_samples(arrays: dict, config: dict, *, writer_run: bool = False) -> tuple[list[dict], dict]:
    """Check coverage, label correctness, depth pairing and reproduce every cell."""
    from looplora.sym.stats import load_samples
    arrays = load_samples(arrays)
    settings = config["eval"]
    episodes = settings["episodes"]
    hops = [1] if writer_run else list(HOPS)
    depths = [1] if writer_run else list(DEPTHS)
    groups = {(split, "original") for split in settings["splits"]}
    if settings.get("counterfactuals") and "test" in settings["splits"]:
        groups.update(("test", condition) for condition in CF)
        if settings.get("swap_successors"):
            groups.add(("test", A5))
    require(set(zip(arrays["split"], arrays["condition"])) == groups, "unexpected evaluation sample groups")
    expected_rows = episodes * len(hops) * sum(3 if condition == A5 else len(depths) for _, condition in groups)
    require(len(arrays["correct"]) == expected_rows, "wrong sample row count")
    require(np.array_equal(arrays["correct"], arrays["prediction"] == arrays["answer"]), "incorrect saved correctness")
    require(np.array_equal(arrays["full_correct"], arrays["full_prediction"] == arrays["answer"]), "incorrect full correctness")
    require(bool(np.all(arrays["candidate_size"] > 0)), "invalid candidate size")
    table, fingerprints, original_rows = [], {}, {}
    for split in settings["splits"]:
        for depth in depths:
            indices = np.flatnonzero((arrays["split"] == split) & (arrays["condition"] == "original") & (arrays["L"] == depth))
            original_rows[split, depth] = indices[np.lexsort((arrays["query_id"][indices], arrays["episode_id"][indices]))]
    for split, condition in sorted(groups):
        fingerprint = None
        for depth in ((1, 4, 8) if condition == A5 else depths):
            selected = np.flatnonzero((arrays["split"] == split) & (arrays["condition"] == condition) & (arrays["L"] == depth))
            require(len(selected) == episodes * len(hops), "incomplete depth sample coverage")
            ordered = selected[np.lexsort((arrays["query_id"][selected], arrays["episode_id"][selected]))]
            episode_ids, counts = np.unique(arrays["episode_id"][ordered], return_counts=True)
            require(len(episode_ids) == episodes and bool(np.all(counts == len(hops))), "incomplete episode query coverage")
            require(np.array_equal(arrays["query_id"][ordered], np.tile(np.arange(len(hops)), episodes)), "missing or repeated query ids")
            require(np.array_equal(arrays["hop"][ordered], np.tile(hops, episodes)), "queries do not cover every hop")
            original = original_rows[split, depth]
            for field in ("episode_id", "query_id", "hop"):
                require(np.array_equal(arrays[field][ordered], arrays[field][original]), f"counterfactual source {field} differs")
            require(np.array_equal(arrays["original_answer"][ordered], arrays["answer"][original]), "counterfactual original answers differ")
            require(np.array_equal(arrays["original_prediction"][ordered], arrays["prediction"][original]), "counterfactual original predictions differ")
            current = _data_digest(arrays, ordered)
            require(fingerprint is None or current == fingerprint, "query/data labels differ across depths")
            fingerprint = current
            if condition == "swap_memory":
                require(bool(np.all(arrays["memory_episode_id"][selected] != arrays["episode_id"][selected])), "swap_memory contains self donors")
                require(np.array_equal(arrays["answer"][ordered], arrays["answer"][original])
                        and np.array_equal(arrays["candidate_size"][ordered], arrays["candidate_size"][original]),
                        "swap_memory changes source answer/candidates")
            if condition in ("replace_key_fact", A5):
                require(bool(np.all(arrays["answer"][selected] != arrays["original_answer"][selected])), "replacement answer did not change")
            for hop in hops:
                chosen = selected[arrays["hop"][selected] == hop]
                correct = arrays["correct"][chosen]
                row = dict(split=split, condition=condition, hop=hop, L=depth, n=len(chosen),
                           accuracy=float(correct.mean()), full_accuracy=float(arrays["full_correct"][chosen].mean()),
                           chance_accuracy=float((1.0 / arrays["candidate_size"][chosen]).mean()))
                if condition == "replace_key_fact":
                    previous = arrays["original_prediction"][chosen] == arrays["original_answer"][chosen]
                    row.update(prediction_changed_rate=float((arrays["prediction"][chosen] != arrays["original_prediction"][chosen]).mean()),
                               n_original_correct=int(previous.sum()),
                               new_answer_accuracy_given_original_correct=float(correct[previous].mean()) if previous.any() else None)
                table.append(row)
        fingerprints[f"{split}/{condition}"] = fingerprint
    # The evaluator sorts hop before depth in its public table.
    table.sort(key=lambda row: (row["split"], row["condition"], row["hop"], row["L"]))
    return table, {"rows": len(arrays["correct"]), "episodes": episodes, "depths": depths,
                   "hops": hops, "groups": sorted(fingerprints), "data_fingerprints": fingerprints}


def successor_follow_rates(arrays: dict) -> list[dict]:
    """A5 diagnostic paired with its exact unswapped query, never a gate input."""
    selected = arrays["condition"] == A5
    if not selected.any():
        return []
    require("counterfactual_hop_true" in arrays, "A5 samples lack actual modified hop labels")
    rows = []
    for depth in (1, 4, 8):
        for hop in HOPS:
            base = selected & (arrays["L"] == depth) & (arrays["hop"] == hop)
            reachable = arrays["counterfactual_hop_true"] <= depth
            for label, mask in (("all", base), ("true", base & reachable), ("false", base & ~reachable)):
                count = int(mask.sum())
                if not count:
                    continue
                follows = arrays["prediction"][mask] == arrays["answer"][mask]
                unswapped = arrays["original_prediction"][mask] == arrays["answer"][mask]
                rows.append({"L": depth, "original_hop": hop, "new_hop_le_L": label, "n": count,
                             "follow_count": int(follows.sum()), "unswapped_follow_count": int(unswapped.sum()),
                             "follow_rate": float(follows.mean()), "unswapped_follow_rate": float(unswapped.mean()),
                             "paired_follow_difference": float(follows.mean() - unswapped.mean()), "gating": False})
    return rows


def _frozen_writer_check(run: dict, writers: dict[int, dict], record: dict) -> None:
    import torch
    source_run = writers[run["seed"]]
    source = Path(source_run["run_dir"]) / "writer.pt"
    config = record["config"]
    require(Path(config["pretrain"]["writer_checkpoint"]).resolve() == source.resolve(), "frozen reader uses a non-main writer")
    require(Path(record["consumed_writer_checkpoint"]).resolve() == source.resolve(), "recorded frozen writer source differs")
    source_writer = torch.load(source, map_location="cpu", weights_only=True)
    local = torch.load(Path(run["run_dir"]) / "writer.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(Path(run["run_dir"]) / "checkpoint.pt", map_location="cpu", weights_only=True)
    require(checkpoint["config"] == config, "frozen reader checkpoint config differs from its record")
    require(source_writer["signature"] == local["signature"], "frozen writer signature differs")
    require(source_writer["training_config"] == read_json(Path(source_run["run_dir"]) / "record.json")["config"], "writer source training config differs")
    require(source_writer["training"]["steps"] == config["train"]["steps"], "source writer step budget differs")
    require(set(source_writer["writer"]) == set(local["writer"]), "frozen writer tensor names differ")
    summary = checkpoint["summary"]
    require(summary["writer_frozen"] is True and summary["pretrain"] is None and summary["pretrain_seconds"] == 0,
            "reader locally retrained its frozen writer")
    for name, value in source_writer["writer"].items():
        for actual in (local["writer"][name], checkpoint["model"]["writer." + name]):
            require(value.dtype == actual.dtype and value.shape == actual.shape and torch.equal(value, actual),
                    f"frozen writer tensor changed: {run['name']}/{name}")


def _training_log(run: dict, steps: int) -> list[dict]:
    points = []
    with (Path(run["run_dir"]) / "train_log.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                point = json.loads(line)
                require(point["stage"] == "train" and math.isfinite(point["loss"]), f"invalid train log: {run['name']}")
                require(not points or point["step"] > points[-1]["step"], f"nonmonotone train log: {run['name']}")
                points.append(point)
    require(bool(points) and points[0]["step"] == 1 and points[-1]["step"] == steps, f"incomplete train log: {run['name']}")
    return points


def collect(plan: dict, barrier: dict, out: Path, cache: CompactCache) -> tuple[list[dict], dict, dict]:
    """Stream exactly one raw run archive at a time, after verified release."""
    from looplora.sym.evaluate import _aggregate_diagnostics
    tables, logs, successors = [], {}, []
    audit = {"status": "running", "plan_hash": stable_hash(plan), "barrier_created_at": barrier["created_at"],
             "runs_checked": 0, "reader_runs_checked": 0, "frozen_writers_checked": 0,
             "sample_rows": 0, "evaluations": [], "controlled_writer_groups": []}
    writers = {run["seed"]: run for run in plan["runs"] if run["group"] == "writer"}
    controlled = {}
    try:
        for run in plan["runs"]:
            folder = Path(run["run_dir"])
            record = read_json(folder / "record.json")
            logs[run["name"]] = _training_log(run, plan["steps"])
            if run["writer"] == "t1_frozen":
                _frozen_writer_check(run, writers, record)
                audit["frozen_writers_checked"] += 1
            writer_run = run["group"] == "writer"
            for dataset in (["writer_validation"] if writer_run else ["screening", "confirmation"]):
                location = folder / dataset
                metadata = read_json(location / ("evaluation.json" if dataset == "confirmation" else "metrics.json"))
                config = metadata.get("config") or read_json(location / "config.json")["config"]
                require(metadata["config_hash"] == stable_hash(config)
                        and metadata["checkpoint_config"] == record["config"]
                        and metadata["checkpoint_config_hash"] == stable_hash(record["config"]), "evaluation/checkpoint config hashes differ")
                require(metadata.get("evaluation_seed") == (777001 if dataset == "confirmation" else 4242), "evaluation generator seed differs")
                settings = config["eval"]
                require(settings["episodes"] == (1000 if writer_run else 2000), "evaluation episode count differs")
                require(settings["splits"] == (["val"] if writer_run else ["test"] if dataset == "confirmation" else ["test", "test_unseen"]), "evaluation split scope differs")
                require(bool(settings.get("counterfactuals")) == (dataset == "screening"), "evaluation counterfactual scope differs")
                require(bool(settings.get("swap_successors") and settings.get("counterfactuals")) ==
                        (dataset == "screening" and run["group"] in ("primary", "replication")),
                        "A5 diagnostic scope differs")
                snapshots = audit_snapshots(metadata, config, writer_run=writer_run)
                sample_path = location / "samples.npz"
                if "samples_hash" in metadata:
                    require(file_hash(sample_path) == metadata["samples_hash"], "evaluation sample file hash differs")
                with np.load(sample_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                table, counts = audit_samples(arrays, config, writer_run=writer_run)
                if settings.get("swap_successors"):
                    successors.extend({"name": run["name"], "group": run["group"], "writer": run["writer"],
                                       "variant": run["variant"], "seed": run["seed"], **row}
                                      for row in successor_follow_rates(arrays))
                if dataset == "confirmation":
                    require(not (location / "metrics.json").exists(), f"confirmation already finalized: {run['name']}")
                    write_json(location / "metrics.json", {**metadata, "aggregation_deferred": False,
                               "finalized_after_global_barrier": True, "barrier_created_at": barrier["created_at"],
                               "table": table, "diagnostics": _aggregate_diagnostics(arrays)})
                    cache.save(run["name"], "confirmation", arrays, "test")
                else:
                    require(table == metadata["table"], f"samples do not reproduce saved metrics: {run['name']}/{dataset}")
                    if not writer_run:
                        cache.save(run["name"], "unseen", arrays, "test_unseen")
                        if run["seed"] == 17:
                            cache.save(run["name"], "screening", arrays, "test")
                tables.extend({"name": run["name"], "group": run["group"], "writer": run["writer"],
                               "variant": run["variant"], "seed": run["seed"], "r": run["r"],
                               "E": config["task"]["E"], "dataset": dataset, **row} for row in table)
                if not writer_run and run["writer"] != "joint":
                    group_spec = {"writer": run["writer"], "task": config["task"], "seed": run["seed"], "r": run["r"],
                                  "dataset": dataset, "backbone_seed": config["backbone_seed"], "eval": settings}
                    # Reader depth, label and nonlinearities do not affect preparation.
                    key = stable_hash(group_spec)
                    fingerprint = stable_hash({"snapshots": [[list(k), list(v)] for k, v in sorted(snapshots.items())],
                                               "data": counts["data_fingerprints"]})
                    if key in controlled:
                        require(controlled[key]["fingerprint"] == fingerprint,
                                f"controlled variants have different x/u/M/data: {run['name']}")
                        controlled[key]["members"].append(run["name"])
                    else:
                        controlled[key] = {"writer": run["writer"], "seed": run["seed"], "dataset": dataset,
                                           "r": run["r"], "fingerprint": fingerprint, "members": [run["name"]]}
                audit["sample_rows"] += counts["rows"]
                audit["evaluations"].append({"name": run["name"], "dataset": dataset, **counts})
                del arrays
            audit["runs_checked"] += 1
            audit["reader_runs_checked"] += int(not writer_run)
        audit.update(status="passed", all_sample_tables_reproduced=True, all_depth_snapshots_identical=True,
                     all_controlled_groups_identical=True, all_frozen_writers_match_source=True,
                     controlled_writer_groups=list(controlled.values()))
        write_json(out / "swap_successors_per_seed.json", successors)
        write_csv(out / "swap_successors_per_seed.csv", successors)
        pooled = defaultdict(lambda: {"n": 0, "follow_count": 0, "unswapped_follow_count": 0, "seeds": []})
        for row in successors:
            key = tuple(row[field] for field in ("group", "writer", "variant", "L", "original_hop", "new_hop_le_L"))
            for field in ("n", "follow_count", "unswapped_follow_count"):
                pooled[key][field] += row[field]
            pooled[key]["seeds"].append(row["seed"])
        pooled_rows = []
        for key, counts in sorted(pooled.items()):
            follow, original = counts["follow_count"] / counts["n"], counts["unswapped_follow_count"] / counts["n"]
            pooled_rows.append(dict(zip(("group", "writer", "variant", "L", "original_hop", "new_hop_le_L"), key)) |
                               counts | {"follow_rate": follow, "unswapped_follow_rate": original,
                                         "paired_follow_difference": follow - original, "gating": False})
        write_json(out / "swap_successors.json", pooled_rows)
        write_csv(out / "swap_successors.csv", pooled_rows)
        return tables, logs, audit
    except BaseException as error:
        audit.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "artifact_audit.json", audit)


def paired(cache: CompactCache, runs_a: list[dict], runs_b: list[dict], dataset: str,
           depth_a: int, depth_b: int, hops: tuple[int, ...], *, specificity: bool = False) -> dict:
    from looplora.sym.stats import pooled_compare_runs, pooled_walk_specificity
    a = {run["seed"]: cache.get(run["name"], dataset) for run in runs_a}
    b = {run["seed"]: cache.get(run["name"], dataset) for run in runs_b}
    options = dict(split="test_unseen" if dataset == "unseen" else "test", condition="original",
                   resamples=2000, seed=BOOTSTRAP_SEED)
    output = {}
    for key, confidence in (("report", .95), ("gate", .975)):
        if specificity:
            output[key] = pooled_walk_specificity(a, depth_a, depth_b, **options, confidence=confidence)
        else:
            output[key] = pooled_compare_runs(a, b, depth_a, depth_b, hops=hops, **options, confidence=confidence)
    return output


def statistics(plan: dict, cache: CompactCache) -> tuple[list[dict], list[dict]]:
    from looplora.sym.gates import evaluate_gate_table
    endpoints, reported = {}, []
    index = defaultdict(list)
    for run in plan["runs"]:
        index[run["group"], run["writer"], run["variant"]].append(run)

    def select(group: str, writer: str, variant: str, dataset: str) -> list[dict]:
        return [run for run in index[group, writer, variant] if dataset != "screening" or run["seed"] == 17]

    def record(group: str, writer: str, reader: str, dataset: str, name: str, value: dict) -> None:
        reported.append(dict(group=group, writer=writer, reader=reader, dataset=dataset, endpoint=name,
                             gating=False, statistics=value))

    for dataset in ("confirmation", "screening", "unseen"):
        for writer in WRITERS:
            for reader in READERS:
                p = select("primary", writer, reader, dataset)
                f = select("primary", writer, "B5" if reader == "M1" else "B5-lin", dataset)
                loop = select("primary", writer, "B3", dataset)
                rep_writer = "oracle_random" if writer.startswith("oracle_") else writer
                rep = select("replication", rep_writer, reader, dataset)
                values = {}
                for prefix, runs in (("", p), ("replication_", rep)):
                    for name, hops in (("reach", (2, 3)), ("far", (6, 7, 8)), ("specificity", (2, 3))):
                        values[prefix + name] = paired(cache, runs, runs, dataset, 4, 1, hops,
                                                       specificity=name == "specificity")
                values["hop1"] = paired(cache, p, p, dataset, 4, 1, (1,))
                values["vs_fixed"] = paired(cache, p, f, dataset, 4, 4, (2, 3))
                values["vs_loop"] = paired(cache, p, loop, dataset, 4, 4, (2, 3))
                if dataset == "confirmation":
                    endpoints[writer, reader] = values
                else:
                    for name, value in values.items():
                        record("primary", writer, reader, dataset, name, value)
            for variant in VARIANTS:
                runs = select("primary", writer, variant, dataset)
                record("primary", writer, variant, dataset, "L8_minus_L4_hops4to8",
                       paired(cache, runs, runs, dataset, 8, 4, (4, 5, 6, 7, 8)))
            m1, linear = (select("primary", writer, v, dataset) for v in READERS)
            b4 = select("primary", writer, "B4", dataset)
            for label, hops in (("reach", (2, 3)), ("all_hops", HOPS)):
                record("primary", writer, "M1", dataset, f"M1_minus_M1lin_L4_{label}",
                       paired(cache, m1, linear, dataset, 4, 4, hops))
                record("primary", writer, "B4", dataset, f"B4_L1_minus_M1_L1_{label}",
                       paired(cache, b4, m1, dataset, 1, 1, hops))
        for writer in ("oracle_random", "oracle_orth", "joint"):
            for reader in READERS:
                p = select("secondary", writer, reader, dataset)
                f = select("secondary", writer, "B5" if reader == "M1" else "B5-lin", dataset)
                for name, da, db, hops, specific in (
                        ("reach", 4, 1, (2, 3), False), ("far", 4, 1, (6, 7, 8), False),
                        ("specificity", 4, 1, (2, 3), True), ("hop1", 4, 1, (1,), False),
                        ("L8_minus_L4_hops4to8", 8, 4, (4, 5, 6, 7, 8), False)):
                    record("secondary", writer, reader, dataset, name,
                           paired(cache, p, p, dataset, da, db, hops, specificity=specific))
                record("secondary", writer, reader, dataset, "vs_fixed",
                       paired(cache, p, f, dataset, 4, 4, (2, 3)))
            record("secondary", writer, "M1", dataset, "M1_minus_M1lin_L4_reach",
                   paired(cache, select("secondary", writer, "M1", dataset),
                          select("secondary", writer, "M1-lin", dataset), dataset, 4, 4, (2, 3)))
            record("secondary", writer, "M1", dataset, "M1_minus_M1lin_L4_all_hops",
                   paired(cache, select("secondary", writer, "M1", dataset),
                          select("secondary", writer, "M1-lin", dataset), dataset, 4, 4, HOPS))
            for variant in ("B5", "B5-lin"):
                runs = select("secondary", writer, variant, dataset)
                record("secondary", writer, variant, dataset, "L8_minus_L4_hops4to8",
                       paired(cache, runs, runs, dataset, 8, 4, (4, 5, 6, 7, 8)))
        for writer in ("oracle_random", "t1_frozen", "joint"):
            for reader in READERS:
                runs = select("replication", writer, reader, dataset)
                record("replication", writer, reader, dataset, "L8_minus_L4_hops4to8",
                       paired(cache, runs, runs, dataset, 8, 4, (4, 5, 6, 7, 8)))
            for label, hops in (("reach", (2, 3)), ("all_hops", HOPS)):
                record("replication", writer, "M1", dataset, f"M1_minus_M1lin_L4_{label}",
                       paired(cache, select("replication", writer, "M1", dataset),
                              select("replication", writer, "M1-lin", dataset), dataset, 4, 4, hops))
    return evaluate_gate_table(endpoints, evaluation_seed=777001), reported


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"cannot write empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def text(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(text(cell) for cell in row) + " |" for row in rows)]) + "\n\n"


def _interval(stat: dict) -> str:
    return f"{100*stat['difference']:+.2f} [{100*stat['ci_low']:+.2f}, {100*stat['ci_high']:+.2f}]"


def render_tables(out: Path, tables: list[dict], gates: list[dict], reported: list[dict], plan: dict) -> None:
    write_json(out / "accuracy_tables.json", tables)
    write_csv(out / "accuracy_tables.csv", tables)
    write_json(out / "gates.json", gates)
    write_json(out / "reported_contrasts.json", reported)
    flat = [{key: row[key] for key in ("group", "writer", "reader", "dataset", "endpoint")} |
            {"confidence": stat["confidence"], "difference": stat["difference"], "ci_low": stat["ci_low"],
             "ci_high": stat["ci_high"], "training_seeds": ",".join(map(str, stat["training_seeds"])),
             "per_seed": json.dumps(stat["per_seed"])} for row in reported for stat in row["statistics"].values()]
    write_csv(out / "reported_contrasts.csv", flat)
    (out / "reported_contrasts.md").write_text("# Reported contrasts (never gating)\n\n" +
        "Intervals and estimates are percentage points. Screening uses training seed 17 only; "
        "secondary runs also have one training seed. Unseen uses the independent entity pool.\n\n" +
        _table(["Group", "Writer", "Reader", "Set", "Endpoint", "Estimate [95% CI] (pp)", "Estimate [97.5% CI] (pp)"],
               [[row["group"], row["writer"], row["reader"], row["dataset"], row["endpoint"],
                 _interval(row["statistics"]["report"]), _interval(row["statistics"]["gate"])] for row in reported]))
    lines = ["# P1-S confirmation gates", "", "Accuracy differences and intervals are percentage points. "
             "Decisions use unrounded 97.5% intervals and final-v1 thresholds; 95% intervals are descriptive.", "",
             _table(["Writer", "Reader", "G1", "G2", "Pass"], [[row["writer"], row["reader"], row["g1_passed"], row["g2_passed"], row["passed"]] for row in gates])]
    for row in gates:
        lines += [f"## {row['writer']} / {row['reader']}", "",
                  _table(["Endpoint", "Estimate [95% CI] (pp)", "Estimate [97.5% CI] (pp)", "Per-seed estimate (pp)"],
                         [[name, _interval(pair["report"]), _interval(pair["gate"]),
                           ", ".join(f"{item['seed']}: {100*item['difference']:+.2f}" for item in pair["report"]["per_seed"])]
                          for name, pair in row["statistics"].items()]),
                  _table(["Criterion", "Raw value", "Threshold", "Pass"],
                         [[name, criterion["value"], f"{criterion['operator']} {criterion['threshold']}", criterion["passed"]]
                          for name, criterion in row["criteria"].items()])]
    (out / "gates.md").write_text("\n".join(lines))
    groups = defaultdict(list)
    for row in tables:
        if row["dataset"] in ("confirmation", "writer_validation"):
            groups[row["name"], row["dataset"]].append(row)
    lines = ["# Complete confirmation hop × depth tables", "", "Candidate-set accuracy (%). "
             "The CSV/JSON include every screening, unseen and counterfactual cell, full-vocabulary accuracy, "
             "chance accuracy and counterfactual consistency components.", ""]
    for (name, dataset), rows in groups.items():
        cells = {(row["hop"], row["L"]): row["accuracy"] for row in rows}
        depths, hops = sorted({row["L"] for row in rows}), sorted({row["hop"] for row in rows})
        lines += [f"## {name} / {dataset}", "", _table(["Hop", *(f"L={depth}" for depth in depths)],
                  [[hop, *(f"{100*cells[hop, depth]:.2f}" for depth in depths)] for hop in hops])]
    (out / "accuracy_tables.md").write_text("\n".join(lines))
    original = {(row["name"], row["dataset"], row["split"], row["hop"], row["L"]): row for row in tables if row["condition"] == "original"}
    cf = []
    for row in tables:
        if row["condition"] in (*CF, A5):
            baseline = original[row["name"], row["dataset"], row["split"], row["hop"], row["L"]]
            cf.append({**row, "original_accuracy": baseline["accuracy"],
                       "accuracy_change_from_original": row["accuracy"] - baseline["accuracy"],
                       "accuracy_minus_chance": row["accuracy"] - row["chance_accuracy"], "gating": False})
    write_json(out / "counterfactual_consistency.json", cf)
    write_csv(out / "counterfactual_consistency.csv", cf)
    writer_scores = [row for row in tables if row["dataset"] == "writer_validation"]
    write_json(out / "writer_t1_scores.json", writer_scores)
    write_csv(out / "writer_t1_scores.csv", writer_scores)
    (out / "writer_t1_scores.md").write_text("# Main shared writers: T1 validation\n\n" +
        _table(["Seed", "T1 accuracy (%)", "Chance (%)", "Steps"],
               [[row["seed"], f"{100*row['accuracy']:.2f}", f"{100*row['chance_accuracy']:.2f}", plan["steps"]] for row in writer_scores]))


def figures(figures_dir: Path, tables: list[dict], gates: list[dict], reported: list[dict], logs: dict, plan: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    primary = [row for row in tables if row["group"] == "primary" and row["dataset"] == "confirmation"]

    def save(fig: Any, name: str) -> None:
        path = figures_dir / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path.resolve()))

    for writer in WRITERS:
        for reader in READERS:
            fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
            for hop, ax in zip(HOPS, axes.flat):
                curves = []
                for seed in SEEDS:
                    cells = {row["L"]: row["accuracy"] for row in primary if row["writer"] == writer and row["variant"] == reader and row["seed"] == seed and row["hop"] == hop}
                    values = [100*cells[depth] for depth in DEPTHS]
                    curves.append(values)
                    ax.plot(DEPTHS, values, alpha=.3, lw=1, label=f"seed {seed}")
                ax.plot(DEPTHS, np.mean(curves, axis=0), color="black", lw=2, label="seed mean")
                ax.set(title=f"Hop {hop}", ylim=(0, 100), xlabel="Read depth L", ylabel="Accuracy (%)")
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f"{writer} / {reader}: confirmation, r=32")
            fig.tight_layout()
            save(fig, f"accuracy-depth-{writer}-{reader}")
        fig, axes = plt.subplots(1, 5, figsize=(19, 4), sharey=True)
        for variant, ax in zip(("M1", "M1-lin", "B5", "B5-lin", "B3"), axes):
            matrix = [[100*np.mean([row["accuracy"] for row in primary if row["writer"] == writer and row["variant"] == variant and row["hop"] == hop and row["L"] == depth]) for depth in DEPTHS] for hop in HOPS]
            image = ax.imshow(matrix, vmin=0, vmax=100, aspect="auto", origin="lower", cmap="viridis")
            ax.set(title=variant, xticks=range(len(DEPTHS)), xticklabels=DEPTHS, yticks=range(8), yticklabels=HOPS, xlabel="Read depth L")
        axes[0].set_ylabel("Hop")
        fig.colorbar(image, ax=list(axes), label="Accuracy (%)", shrink=.8)
        fig.suptitle(f"{writer}: confirmation seed mean, r=32")
        save(fig, f"hop-depth-{writer}")
        fig, ax = plt.subplots(figsize=(9, 5))
        differences = []
        for gate in gates:
            if gate["writer"] == writer:
                for endpoint, comparator in (("vs_fixed", gate["fixed_query_comparator"]), ("vs_loop", "B3")):
                    differences.append((f"{gate['reader']} − {comparator}", gate["statistics"][endpoint]))
        linear = next(row for row in reported if row["group"] == "primary" and row["writer"] == writer and row["dataset"] == "confirmation" and row["endpoint"] == "M1_minus_M1lin_L4_reach")
        differences.append(("M1 − M1-lin", linear["statistics"]))
        for y, (label, pair) in enumerate(differences):
            for role, width, color in (("gate", 1, "#9aa3af"), ("report", 3, "#225ea8")):
                stat = pair[role]
                ax.plot([100*stat["ci_low"], 100*stat["ci_high"]], [y, y], color=color, lw=width)
            ax.plot(100*pair["report"]["difference"], y, "o", color="black")
        ax.axvline(0, color="black", lw=.7)
        ax.set(yticks=range(len(differences)), yticklabels=[label for label, _ in differences], xlabel="Paired accuracy difference (pp)", title=f"{writer}: L=4, hops 2–3; 95% / 97.5% CIs")
        save(fig, f"paired-differences-{writer}")
    for reader in READERS:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
        for hop, ax in zip(HOPS, axes.flat):
            for group, facts, color in (("primary", 12, "#225ea8"), ("replication", 18, "#cb181d")):
                rows = [row for row in tables if row["dataset"] == "confirmation" and row["writer"] == "oracle_random" and row["variant"] == reader and row["group"] == group and row["hop"] == hop]
                values = [100*np.mean([row["accuracy"] for row in rows if row["L"] == depth]) for depth in DEPTHS]
                ax.plot(DEPTHS, values, color=color, label=f"E={facts}")
            ax.set(title=f"Hop {hop}", ylim=(0, 100), xlabel="Read depth L", ylabel="Accuracy (%)")
        axes.flat[0].legend()
        fig.suptitle(f"oracle_random / {reader}: load × depth, confirmation seed mean")
        fig.tight_layout()
        save(fig, f"load-depth-oracle_random-{reader}")
    log_groups = defaultdict(list)
    for run in plan["runs"]:
        log_groups[run["group"], run["writer"]].append(run)
    for (group, writer), runs in log_groups.items():
        variants = [variant for variant in VARIANTS if any(run["variant"] == variant for run in runs)]
        if not variants:
            variants = sorted({run["variant"] for run in runs})
        fig, axes = plt.subplots(1 if len(variants) <= 3 else 2, min(3, len(variants)), figsize=(15, 4 if len(variants) <= 3 else 8), squeeze=False)
        for variant, ax in zip(variants, axes.flat):
            for run in runs:
                if run["variant"] == variant:
                    points = logs[run["name"]]
                    ax.plot([p["step"] for p in points], [p["loss"] for p in points], label=f"seed {run['seed']}", lw=.8)
            ax.set(title=variant, xlabel="Training step", ylabel="Minibatch CE")
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(variants):]:
            ax.set_visible(False)
        fig.suptitle(f"{group} / {writer}: training loss")
        fig.tight_layout()
        save(fig, f"training-loss-{group}-{writer}")
    return paths


def analyze(plan_path: Path, out: Path, figures_dir: Path, *, cache_limit: int = 12) -> dict:
    # Keep this first: no sample, checkpoint tensor or metric inference before release.
    plan, barrier = verify_release(plan_path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = out / "analysis_started.json"
    require(not started.exists(), f"analysis already started; inspect retained artifacts before restarting: {started}")
    # Refuse partial/repeated confirmation finalization before the first raw read.
    require(not any((Path(run["run_dir"]) / "confirmation" / "metrics.json").exists()
                    for run in plan["runs"] if run["group"] != "writer"), "confirmation metrics already exist")
    write_json(started, {"plan_hash": stable_hash(plan), "barrier_created_at": barrier["created_at"],
                         "created_at": datetime.now(timezone.utc).isoformat(), "confirmation_metrics_previously_read": False})
    cache = CompactCache(out / "compact_samples", cache_limit)
    tables, logs, audit = collect(plan, barrier, out, cache)
    gates, reported = statistics(plan, cache)
    render_tables(out, tables, gates, reported, plan)
    paths = figures(figures_dir, tables, gates, reported, logs, plan)
    summary = {"status": "complete", "plan_hash": stable_hash(plan), "barrier_created_at": barrier["created_at"],
               "completed_at": datetime.now(timezone.utc).isoformat(), "runs": audit["runs_checked"],
               "confirmation_runs": audit["reader_runs_checked"], "gate_rows": len(gates),
               "passing_readers": [{"writer": row["writer"], "reader": row["reader"]} for row in gates if row["passed"]],
               "reported_contrasts": len(reported), "accuracy_cells": len(tables), "figures": paths,
               "cache_limit": cache_limit, "bootstrap_resamples": 2000, "gate_confidence": .975,
               "report_confidence": .95, "steps": plan["steps"], "prereg_commit": plan.get("prereg_commit"),
               "selection_source": plan.get("selection_source"), "cost_and_registry": "maintained by the main runner"}
    write_json(out / "analysis_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/p1s"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures/p1s"))
    parser.add_argument("--cache-limit", type=int, default=12)
    args = parser.parse_args(argv)
    result = analyze(args.plan, args.out, args.figures, cache_limit=args.cache_limit)
    print(json.dumps({key: result[key] for key in ("status", "runs", "confirmation_runs", "gate_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
