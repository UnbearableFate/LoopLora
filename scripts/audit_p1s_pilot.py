#!/usr/bin/env python3
"""Audit completed P1-S validation artifacts; run only in a compute allocation.

Submit through scripts/submit.py, with persistent absolute paths for --runs,
--plans and --reports. Each root contains pilot1/, pilot2/ and replication/;
plan stage directories contain manifest.json and shard-*.yaml. If --plans is
omitted, use <runs>/plans. No training, scientific contrasts or gate conclusions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

STAGES = {"pilot1": 60, "pilot2": 60, "replication": 15}
FIELDS = ("x", "u", "memory", "write_count")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"expected JSON mapping: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _locked_config(config: dict[str, Any], stage: str, name: str) -> None:
    _require(config["name"] == name, f"config name differs: {name}")
    _require(config["seed"] == 17, f"forbidden training seed: {name}")
    settings = config["eval"]
    _require(settings["splits"] == ["val"] and settings["seed"] == 4242,
             f"forbidden evaluation seed/split: {name}")
    _require(settings["episodes"] == 1000 and not settings["counterfactuals"],
             f"unexpected validation episode count/conditions: {name}")
    _require(config["train"]["steps"] == 5000 and config["train"]["batch_size"] == 64,
             f"unexpected training budget: {name}")
    expected_hops = [1] if stage == "pilot1" else list(range(1, 9))
    _require(settings["hops"] == expected_hops, f"unexpected evaluation hops: {name}")
    if stage == "pilot1":
        _require(config["task"]["name"] == "t1", f"Pilot-1 is not T1: {name}")
    else:
        _require(config["model"]["r"] == config["model"]["r_mem"] == 32,
                 f"unexpected Pilot-2/replication rank: {name}")


def _snapshots(metrics: dict[str, Any], config: dict[str, Any], name: str) -> dict[int, tuple[str, ...]]:
    """Require every evaluation batch/depth, then remove the depth dimension."""
    _require(metrics.get("gate0_passed") is True, f"saved Gate 0 flag failed: {name}")
    settings = config["eval"]
    depths = set(settings["depths"])
    batches = set(range(0, settings["episodes"], settings["batch_size"]))
    seen: set[tuple[int, int]] = set()
    snapshots: dict[int, tuple[str, ...]] = {}
    for row in metrics["memory_hashes"]:
        _require(row["split"] == "val" and row["condition"] == "original",
                 f"unexpected snapshot condition: {name}")
        key = (row["batch_start"], row["L"])
        _require(key not in seen and key[0] in batches and key[1] in depths,
                 f"duplicate or unexpected batch/depth snapshot: {name}: {key}")
        seen.add(key)
        value = tuple(row[field] for field in FIELDS)
        _require(all(isinstance(part, str) and len(part) == 32 for part in value),
                 f"invalid snapshot hashes: {name}: {key}")
        if key[0] in snapshots:
            _require(snapshots[key[0]] == value, f"depth-dependent x/u/M/write_count: {name}: {key}")
        snapshots[key[0]] = value
    _require(seen == {(batch, depth) for batch in batches for depth in depths},
             f"missing batch/depth snapshots: {name}")
    return snapshots


def _sample_counts(samples: dict, metrics: dict, config: dict, record: dict, name: str) -> tuple[dict, str]:
    import numpy as np
    from looplora.sym.evaluate import aggregate_samples

    settings = config["eval"]
    depths, hops = settings["depths"], settings["hops"]
    count = len(samples["answer"])
    _require(count == 1000 * len(hops) * len(depths), f"wrong sample row count: {name}")
    _require(record["eval_samples"] == count, f"record/sample count differs: {name}")
    _require(set(samples["split"]) == {"val"} and set(samples["condition"]) == {"original"},
             f"forbidden sample split/condition: {name}")
    _require(set(samples["L"].tolist()) == set(depths), f"sample depths differ: {name}")
    _require(np.array_equal(samples["correct"], samples["prediction"] == samples["answer"]),
             f"incorrect saved correctness labels: {name}")
    _require(np.array_equal(samples["full_correct"], samples["full_prediction"] == samples["answer"]),
             f"incorrect full-vocabulary correctness labels: {name}")
    _require(bool(np.all(samples["candidate_size"] > 0)), f"invalid candidate count: {name}")
    rebuilt = aggregate_samples(samples)
    _require(rebuilt == metrics["table"], f"samples do not reproduce metrics table: {name}")
    _require(record["scores"] == metrics["table"], f"record scores differ from metrics: {name}")
    _require(len(rebuilt) == len(depths) * len(hops) and all(row["n"] == 1000 for row in rebuilt),
             f"hop/depth table is incomplete: {name}")
    fingerprint: str | None = None
    for depth in depths:
        chosen = np.flatnonzero(samples["L"] == depth)
        _require(len(np.unique(samples["episode_id"][chosen])) == 1000,
                 f"wrong unique episode count: {name}: L={depth}")
        query_ids = samples["query_id"][chosen]
        _require(bool(np.all((query_ids >= 0) & (query_ids < len(hops)))),
                 f"unexpected query ids: {name}: L={depth}")
        _require(np.array_equal(samples["hop"][chosen], np.asarray(hops)[query_ids]),
                 f"queries do not supply one query per hop: {name}: L={depth}")
        ordered = chosen[np.lexsort((query_ids, samples["episode_id"][chosen]))]
        digest = hashlib.blake2b(digest_size=16)
        for field in ("episode_id", "query_id", "hop", "answer", "candidate_size"):
            value = np.ascontiguousarray(samples[field][ordered])
            digest.update(field.encode() + str(value.dtype).encode() + str(value.shape).encode())
            digest.update(value.tobytes())
        current = digest.hexdigest()
        if fingerprint is not None:
            _require(current == fingerprint, f"query/data labels differ across depths: {name}")
        fingerprint = current
    return {"rows": count, "episodes": 1000, "queries_per_episode": len(hops),
            "depths": len(depths), "unique_queries": 1000 * len(hops)}, str(fingerprint)


def _same_writer(left: dict, right: dict, name: str) -> None:
    import torch

    _require(left["signature"] == right["signature"], f"writer signature differs: {name}")
    _require(set(left["writer"]) == set(right["writer"]), f"writer tensor keys differ: {name}")
    for key, value in left["writer"].items():
        reference = right["writer"][key]
        _require(value.dtype == reference.dtype and value.shape == reference.shape
                 and torch.equal(value, reference), f"writer tensor differs: {name}: {key}")


def audit(runs: Path, plans: Path, reports: Path, selection1: Path | None = None) -> dict[str, Any]:
    """Stream run artifacts; retain only hashes, counts and small provenance data."""
    import torch
    from looplora.infra.seeds import stable_hash
    from looplora.sym.run import load_suite
    from looplora.sym.stats import load_samples

    runs, plans, reports = runs.resolve(), plans.resolve(), reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "running", "scope": "validation artifact integrity only; no scientific contrasts or gate conclusions",
        "runs_root": str(runs), "plans_root": str(plans), "stages": {},
        "sample_rows": 0, "runs_checked": 0, "controlled_writer_groups": [],
        "frozen_writers_checked": 0, "failure_reason": None,
    }
    report = reports / "p1s_pilot_audit.json"
    groups: dict[str, dict] = {}
    pilot1_sources: dict[Path, dict] = {}
    selected_source: Path | None = None
    try:
        if selection1 is not None:
            selection = _read(selection1)
            selected = selection["selected"]["joint/r32"]
            _require(selected["E"] == selection["E_star"], "selected Pilot-1 writer is not at E*")
            selected_source = Path(selected["writer_checkpoint"]).resolve()
            result["selection1"] = str(selection1.resolve())
            result["selected_writer_checkpoint"] = str(selected_source)
        for stage, expected_count in STAGES.items():
            stage_runs, stage_plans = runs / stage, plans / stage
            manifest = _read(stage_plans / "manifest.json")
            expected_names = set(manifest["names"])
            _require(len(manifest["names"]) == len(expected_names) == manifest["runs"] == expected_count,
                     f"wrong manifest run count/names: {stage}")
            shard_plans = sorted(stage_plans.glob("shard-*.yaml"))
            _require(len(shard_plans) == manifest["shards"], f"wrong shard manifest count: {stage}")
            expected_shards = {path.stem for path in shard_plans}
            actual_shards = {path.name for path in stage_runs.glob("shard-*") if path.is_dir()}
            _require(actual_shards == expected_shards, f"run shards differ from plan: {stage}")
            observed: set[str] = set()
            stage_rows: dict[str, dict] = {}
            for shard_plan in shard_plans:
                shard = stage_runs / shard_plan.stem
                jobs = load_suite(shard_plan, shard)
                job_names = {job["name"] for job in jobs}
                suite = _read(shard / "suite.json")
                _require(set(suite["runs"]) == job_names, f"suite run names differ: {stage}/{shard.name}")
                actual_directories = {path.name for path in shard.iterdir()
                                      if path.is_dir() and not path.name.startswith("_")}
                _require(actual_directories == job_names, f"run directories differ: {stage}/{shard.name}")
                for job in jobs:
                    name, config = job["name"], job["config"]
                    _require(name not in observed, f"duplicate run across shards: {stage}/{name}")
                    observed.add(name)
                    output = shard / name
                    envelope = _read(output / "config.json")
                    _require(envelope["config"] == config, f"resolved run config differs from plan: {stage}/{name}")
                    _locked_config(config, stage, name)
                    record = _read(output / "record.json")
                    _require(record["experiment_id"] == name and record["status"] == "complete",
                             f"incomplete or misidentified run: {stage}/{name}")
                    _require(suite["runs"][name]["status"] == "complete"
                             and suite["runs"][name]["returncode"] == 0,
                             f"failed suite child: {stage}/{name}")
                    _require(stable_hash(config) == envelope["config_hash"] == record["config_hash"],
                             f"config hash differs: {stage}/{name}")
                    for filename in ("checkpoint.pt", "timing.json", "train_log.jsonl"):
                        _require((output / filename).is_file(), f"missing artifact: {output / filename}")
                    with (output / "train_log.jsonl").open() as stream:
                        log_rows = [json.loads(line) for line in stream if line.strip()]
                    steps = [row["step"] for row in log_rows]
                    _require(bool(log_rows) and all(row["stage"] == "train" for row in log_rows)
                             and steps[0] == 1 and steps[-1] == 5000
                             and all(left < right for left, right in zip(steps, steps[1:])),
                             f"unexpected local training stages/steps: {name}")
                    metrics = _read(output / "metrics.json")
                    snapshots = _snapshots(metrics, config, name)
                    samples = load_samples(output / "samples.npz")
                    counts, data_hash = _sample_counts(samples, metrics, config, record, name)
                    del samples
                    stage_rows[name] = counts
                    result["sample_rows"] += counts["rows"]
                    result["runs_checked"] += 1
                    if config["writer"] in {"oracle_random", "oracle_orth", "t1_frozen"}:
                        group_spec = {"writer": config["writer"], "task": config["task"],
                                      "r": config["model"]["r"], "r_mem": config["model"]["r_mem"],
                                      "d_e": config["model"]["d_e"], "backbone_seed": config["backbone_seed"],
                                      "eval": {key: config["eval"][key] for key in
                                               ("seed", "episodes", "batch_size", "hops")}}
                        key = stable_hash(group_spec)
                        if key in groups:
                            anchor = groups[key]
                            _require(anchor["snapshots"] == snapshots and anchor["data_hash"] == data_hash,
                                     f"controlled writer x/u/M/write or evaluation data differ: {anchor['members'][0]} vs {name}")
                            anchor["members"].append(name)
                        else:
                            groups[key] = {"spec": group_spec, "snapshots": snapshots,
                                           "data_hash": data_hash, "members": [name]}
                    if stage == "pilot1" and config["writer"] == "joint":
                        source_path = (output / "writer.pt").resolve()
                        _require(source_path.is_file(), f"Pilot-1 joint writer export missing: {name}")
                        pilot1_sources[source_path] = config
                    if config["writer"] == "t1_frozen":
                        source_path = Path(config["pretrain"]["writer_checkpoint"]).resolve()
                        _require(source_path in pilot1_sources, f"frozen source is not an audited Pilot-1 joint writer: {name}")
                        source_config = pilot1_sources[source_path]
                        _require(source_config["model"]["r"] == 32 and source_config["task"]["E"] == config["task"]["E"],
                                 f"frozen source rank/fact load differs: {name}")
                        if selected_source is None:
                            selected_source = source_path
                            result["selected_writer_checkpoint"] = str(selected_source)
                        _require(source_path == selected_source, f"frozen writer differs from selected shared source: {name}")
                        _require(Path(record["consumed_writer_checkpoint"]).resolve() == source_path,
                                 f"recorded consumed writer path differs: {name}")
                        _require(record["pretrain_seconds"] == record["pretrain_tokens"] == 0,
                                 f"unexpected local frozen-writer pretraining cost: {name}")
                        timing = _read(output / "timing.json")
                        _require(timing["pretrain_seconds"] == 0, f"local pretraining timing is nonzero: {name}")
                        source_writer = torch.load(source_path, map_location="cpu", weights_only=True)
                        local_writer = torch.load(output / "writer.pt", map_location="cpu", weights_only=True)
                        _require(source_writer["training_config"] == source_config,
                                 f"selected source writer training config differs from Pilot-1 record: {name}")
                        _require(source_writer["training"]["steps"] == 5000,
                                 f"selected source writer training is incomplete: {name}")
                        _same_writer(local_writer, source_writer, name)
                        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
                        summary = checkpoint["summary"]
                        _require(summary["pretrain"] is None and summary["pretrain_seconds"] == 0
                                 and summary["writer_frozen"] is True, f"checkpoint writer-freeze/pretraining summary differs: {name}")
                        for tensor_name, value in source_writer["writer"].items():
                            actual_tensor = checkpoint["model"]["writer." + tensor_name]
                            _require(actual_tensor.dtype == value.dtype and actual_tensor.shape == value.shape
                                     and torch.equal(actual_tensor, value),
                                     f"reader checkpoint writer changed: {name}: {tensor_name}")
                        result["frozen_writers_checked"] += 1
                    result["stages"][stage] = {"runs": len(stage_rows), "expected_runs": expected_count,
                                               "sample_rows": sum(row["rows"] for row in stage_rows.values()),
                                               "sample_counts": stage_rows}
                _require(len(job_names) == len(jobs), f"duplicate plan names: {stage}/{shard.name}")
            _require(observed == expected_names, f"completed stage differs from expected manifest: {stage}")
        result["controlled_writer_groups"] = [
            {"writer": group["spec"]["writer"], "task": group["spec"]["task"]["name"],
             "r": group["spec"]["r"], "E": group["spec"]["task"]["E"],
             "members": sorted(group["members"]), "data_hash": group["data_hash"]}
            for _, group in sorted(groups.items())]
        result.update(status="passed", all_sample_tables_reproduced=True,
                      all_depth_snapshots_identical=True, all_controlled_groups_identical=True,
                      all_frozen_writers_match_source=True)
    except BaseException as error:
        result.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        _write(report, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--selection1", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.runs, args.plans or args.runs / "plans", args.reports, args.selection1)
    except Exception as error:
        print(f"Pilot artifact audit failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in
                      ("status", "runs_checked", "sample_rows", "frozen_writers_checked")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
