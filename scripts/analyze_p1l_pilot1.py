#!/usr/bin/env python3
"""Audit and summarize completed P1-L Pilot-1 writer artifacts on compute.

This script only reads existing runs. It neither trains nor evaluates episodes.
Policy selection is completed before any capacity run artifacts are opened.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm.evaluate import aggregate_samples
from looplora.llm.pilot1_select import choose_steps, select_capacity, select_policy, validate_writer_config
from looplora.sym.stats import load_samples

_REQUIRED = ("config.json", "train_log.jsonl", "metrics.json", "samples.npz",
             "timing.json", "record.json", "checkpoint.pt", "writer.pt")
_TREND_FRACTION = .20
_RELATIVE_TREND_THRESHOLD = .01


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value), f"{name} must be finite")
    result = float(value)
    _require(result > 0 if positive else result >= 0, f"invalid {name}")
    return result


def _run_paths(root: Path, expected_count: int, procs: int) -> list[Path]:
    root = root.resolve()
    _require(root.is_dir(), f"run root does not exist: {root}")

    def direct_paths(directory: Path) -> list[Path]:
        return sorted(path for path in directory.iterdir() if path.is_dir()
                      and any((path / name).exists() for name in ("config.json", "record.json")))

    def suite_paths(directory: Path, *, allow_redirect: bool) -> list[Path]:
        suite = _json(directory / "suite.json")
        _require(suite.get("status") == "complete", f"suite is not complete: {directory}")
        _require(suite.get("procs") == procs, f"suite worker count differs from throughput selection: {directory}")
        runs = Path(suite.get("run_root", directory))
        if not runs.is_absolute():
            runs = directory / runs
        _require(allow_redirect or runs.resolve() == directory.resolve(),
                 f"shard writer directories must be directly under the shard: {directory}")
        _require(runs.is_dir(), f"suite run directory does not exist: {runs}")
        paths = direct_paths(runs)
        _require(bool(paths) and set(suite["runs"]) == {path.name for path in paths},
                 f"suite run names differ from direct writer directories: {directory}")
        _require(all(row.get("status") == "complete" for row in suite["runs"].values()),
                 f"suite contains incomplete writer runs: {directory}")
        return paths

    if (root / "suite.json").is_file():
        paths = suite_paths(root, allow_redirect=True)
    else:
        children = sorted(path for path in root.iterdir() if path.is_dir())
        shards = [path for path in children if (path / "suite.json").is_file()]
        if shards:
            maximum_shards = 6 if expected_count == 6 else 2 if expected_count == 3 else expected_count
            _require(len(shards) == len(children) and len(shards) <= maximum_shards,
                     f"expected only one level of at most {maximum_shards} suite shards under {root}")
            paths = [path for shard in shards for path in suite_paths(shard, allow_redirect=False)]
        else:
            paths = direct_paths(root)
    _require(len(paths) == expected_count, f"expected {expected_count} writer directories under {root}, got {len(paths)}")
    _require(len({path.name for path in paths}) == len(paths), f"duplicate writer directory names across shards: {root}")
    return paths


def _loss_rows(path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    previous = 0
    steps = config["train"]["steps"]
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            step = item["step"]
            _require(isinstance(step, int) and not isinstance(step, bool) and previous < step <= steps,
                     f"invalid logged step at {path}:{line_number}")
            _require(item.get("stage") == "writer" and item.get("L") == 1,
                     f"non-writer training log at {path}:{line_number}")
            expected_seed = derive_seed(17, "llm", "writer-data")
            _require(item.get("data_seed") == expected_seed, f"training data seed differs at {path}:{line_number}")
            loss = _finite(item["loss"], f"training loss at {path}:{line_number}")
            rows.append(dict(run=path.parent.name, run_path=str(path.parent), E=config["task"]["E"],
                             write_policy=config["write_policy"], lr=config["train"]["lr"],
                             step=step, loss=loss, logged_lr=_finite(item["lr"], "logged learning rate")))
            previous = step
    _require(bool(rows) and rows[0]["step"] == 1 and rows[-1]["step"] == steps,
             f"training log does not span all requested steps: {path}")
    tail = [row for row in rows if row["step"] > (1 - _TREND_FRACTION) * steps]
    trend: dict[str, Any] = dict(loss_first=rows[0]["loss"], loss_final=rows[-1]["loss"],
                                 tail_start_step=(1 - _TREND_FRACTION) * steps,
                                 tail_logged_points=len(tail), tail_slope=None,
                                 tail_observed_drop=None, tail_fitted_drop=None,
                                 tail_relative_fitted_drop=None,
                                 convergence_description="insufficient_logged_points")
    if len(tail) >= 2:
        x = np.asarray([row["step"] for row in tail], dtype=np.float64)
        y = np.asarray([row["loss"] for row in tail], dtype=np.float64)
        centered = x - x.mean()
        slope = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
        fitted_drop = -slope * float(x[-1] - x[0])
        relative = fitted_drop / max(float(y.mean()), np.finfo(np.float64).eps)
        description = ("still_falling" if relative >= _RELATIVE_TREND_THRESHOLD else
                       "rising" if relative <= -_RELATIVE_TREND_THRESHOLD else "converged_descriptively")
        trend.update(tail_slope=slope, tail_observed_drop=float(y[0] - y[-1]),
                     tail_fitted_drop=fitted_drop, tail_relative_fitted_drop=relative,
                     convergence_description=description)
    return rows, trend


def _audit_run(path: Path, *, batch: int, steps: int, procs: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(all((path / name).is_file() for name in _REQUIRED), f"missing writer artifacts: {path}")
    envelope, record = _json(path / "config.json"), _json(path / "record.json")
    config = envelope["config"]
    prefix = f"{path.name}: "
    _require(record.get("status") == "complete", prefix + "record is not complete")
    _require(record.get("source_dirty") is False and envelope.get("source_dirty") is False,
             prefix + "source must be clean")
    validate_writer_config(config)
    _require(config["train"]["steps"] == steps and config["train"]["batch_size"] == batch
             and config["train"]["depths"] == [1], prefix + "training budget differs from throughput selection")
    config_hash = stable_hash(config)
    _require(envelope["config_hash"] == config_hash == record["config_hash"], prefix + "configuration hash mismatch")
    _require(record["stage"] == "writer" and record["variant"] == "M1" and record["train_seed"] == 17
             and record["eval_data_seed"] == 4242
             and record["data_seed"] == derive_seed(17, "llm", "writer-data"), prefix + "record metadata differs from configuration")
    samples = load_samples(path / "samples.npz")
    _require(all(value.ndim == 1 and len(value) == 1000 for value in samples.values()), prefix + "expected exactly 1000 flat sample rows")
    _require(len(np.unique(samples["episode_id"])) == 1000 and np.all(samples["query_id"] == 0),
             prefix + "expected one query from each of 1000 distinct episodes")
    for field, expected in (("split", "val"), ("condition", "memory_only/original"),
                            ("protocol", "memory_only"), ("hop", 1), ("L", 1)):
        _require(bool(np.all(samples[field] == expected)), prefix + f"incorrect sample {field}")
    for field, expected in (("correct", samples["prediction"] == samples["answer"]),
                            ("full_prediction", samples["prediction"]), ("full_correct", samples["correct"]),
                            ("candidate_correct", samples["candidate_prediction"] == samples["answer"]),
                            ("original_answer", samples["answer"]), ("original_prediction", samples["prediction"]),
                            ("memory_episode_id", samples["episode_id"])):
        _require(np.array_equal(samples[field], expected), prefix + f"incorrect sample metric/label {field}")
    _require(bool(np.all(samples["candidate_size"] > 0)), prefix + "empty candidate set")
    metrics, timing = _json(path / "metrics.json"), _json(path / "timing.json")
    _require(aggregate_samples(samples) == metrics["table"], prefix + "samples do not reproduce metrics table")
    _require(metrics["primary_metric"] == "full_vocabulary_first_token_exact_match"
             and metrics["secondary_metric"] == "episode_candidate_token_accuracy"
             and metrics["gate0_passed"] is True, prefix + "incorrect metric semantics or failed Gate 0")
    _require(metrics["episode_ranges"] == {"val": {"split": "val", "start": 0, "count": 1000}},
             prefix + "incorrect episode range")
    _require(record["eval_samples"] == 1000, prefix + "record evaluation count differs")
    sets = timing["eval_sets"]
    _require(len(sets) == 1 and sets[0]["split"] == "val" and sets[0]["protocol"] == "memory_only"
             and sets[0]["episodes"] == 1000 and sets[0]["queries"] == 1000, prefix + "incorrect evaluation timing coverage")
    loss_rows, trend = _loss_rows(path / "train_log.jsonl", config)
    runtime = record["runtime"]
    _require(runtime.get("deterministic_algorithms") is True and runtime.get("cudnn_deterministic") is True
             and runtime.get("cublas_workspace_config") == ":4096:8", prefix + "deterministic runtime settings differ")
    train_seconds = _finite(record["train_seconds"], "training seconds", positive=True)
    total_seconds = _finite(timing["total_seconds"], "total seconds", positive=True)
    _require(total_seconds <= 1800, prefix + "run exceeds the 1800-second Pilot-1 budget")
    row = dict(run=path.name, run_path=str(path), stage="writer", seed=17,
               E=int(config["task"]["E"]), write_policy=config["write_policy"],
               lr=float(config["train"]["lr"]), rank=32, layer=14, steps=steps, batch_size=batch,
               procs=procs, training_episodes=steps * batch,
               training_queries=steps * batch * config["task"]["Q"], episodes=1000, n=1000,
               correct=int(samples["correct"].sum()), candidate_correct=int(samples["candidate_correct"].sum()),
               em=float(samples["correct"].mean()), candidate_accuracy=float(samples["candidate_correct"].mean()),
               chance_accuracy=float(np.mean(1 / samples["candidate_size"])),
               train_seconds=train_seconds,
               evaluation_seconds=_finite(timing["evaluation_seconds"], "evaluation seconds", positive=True),
               eval_set_seconds=_finite(sets[0]["seconds"], "evaluation set seconds", positive=True),
               total_seconds=total_seconds,
               seconds_per_1000_steps=_finite(timing["seconds_per_1000_steps"], "seconds per 1000 steps", positive=True),
               train_tokens=int(record["train_tokens"]), host=runtime["host"], job_id=runtime["job_id"],
               device=runtime["device"], gpu=runtime["gpu"], dtype=runtime["dtype"],
               source_dirty=record["source_dirty"], deterministic_algorithms=runtime["deterministic_algorithms"],
               cudnn_deterministic=runtime["cudnn_deterministic"],
               cublas_workspace_config=runtime["cublas_workspace_config"],
               peak_memory_bytes=record["peak_memory_bytes"], git_commit=record["git_commit"],
               **trend)
    return row, loss_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, summaries: list[dict[str, Any]], losses: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(14, 10), squeeze=False)
    for axis, summary in zip(axes.flat, summaries):
        points = [row for row in losses if row["run_path"] == summary["run_path"]]
        axis.plot([row["step"] for row in points], [row["loss"] for row in points], linewidth=1.4)
        axis.axvspan(summary["tail_start_step"], summary["steps"], color="gray", alpha=.12)
        axis.set_title(f"E={summary['E']} {summary['write_policy']} lr={summary['lr']:g}\n"
                       f"{summary['convergence_description']}", fontsize=9)
        axis.set_xlabel("Optimization step")
        axis.set_ylabel("Logged training answer CE")
        axis.grid(alpha=.2)
    figure.suptitle("P1-L Pilot-1 writer training loss; shaded final 20% of steps")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(policy_root: Path, capacity_root: Path, throughput_path: Path, reports: Path,
            jobs_path: Path | None = None) -> dict[str, Any]:
    throughput = _json(throughput_path)
    selected_throughput = throughput["selected"]
    _require(selected_throughput["E"] == 24 and selected_throughput["write_policy"] == "all_tokens",
             "throughput selection must measure E=24/all_tokens")
    batch, procs = selected_throughput["batch_size"], selected_throughput["procs"]
    _require(batch in (8, 16, 32) and procs in (1, 2), "invalid selected batch/worker count")
    _finite(selected_throughput["episodes_per_second"], "throughput episodes/second", positive=True)
    _finite(selected_throughput["worst_worker_p95_step_seconds"], "throughput p95 step seconds", positive=True)
    steps = choose_steps(selected_throughput)
    policy_rows, losses = [], []
    # Validate the entire six-run matrix before selecting or opening capacity results.
    for path in _run_paths(policy_root, 6, procs):
        row, points = _audit_run(path, batch=batch, steps=steps, procs=procs)
        policy_rows.append(row)
        losses.extend(points)
    selected_policy = select_policy(policy_rows)
    capacity_rows = [selected_policy]
    extra_rows = []
    for path in _run_paths(capacity_root, 3, procs):
        row, points = _audit_run(path, batch=batch, steps=steps, procs=procs)
        _require(row["write_policy"] == selected_policy["write_policy"] and row["lr"] == selected_policy["lr"],
                 f"capacity configuration differs from selected policy/lr: {path}")
        extra_rows.append(row)
        losses.extend(points)
    capacity_rows.extend(extra_rows)
    capacity = select_capacity(capacity_rows)
    rows = sorted(policy_rows, key=lambda row: (row["write_policy"], row["lr"])) + sorted(extra_rows, key=lambda row: row["E"])
    jobs = _json(jobs_path) if jobs_path is not None else None
    for row in rows + [selected_policy]:
        metadata = None if jobs is None else jobs.get(str(row["job_id"]))
        _require(jobs is None or metadata is not None, f"missing PBS job metadata: {row['job_id']}")
        row.update(job_nodes=None, job_elapsed_seconds=None, job_node_hours=None)
        if metadata is not None:
            nodes = metadata["nodes"]
            _require(isinstance(nodes, int) and not isinstance(nodes, bool) and nodes > 0,
                     "PBS job nodes must be a positive integer")
            elapsed = _finite(metadata["elapsed_seconds"], "PBS job elapsed seconds", positive=True)
            row.update(job_nodes=nodes, job_elapsed_seconds=elapsed, job_node_hours=nodes * elapsed / 3600)
    capacities_below_095 = all(row["correct"] < .95 * row["n"] for row in capacity_rows)
    summary = dict(policy_runs=6, additional_capacity_runs=3, all_runs_complete=True,
                   all_sample_tables_reproduced=True, evaluation_protocol="memory_only/original",
                   evaluation_split="val", evaluation_seed=4242, train_seed=17,
                   validation_episodes_per_run=1000, steps_per_run=steps, batch_size=batch, procs=procs,
                   throughput_path=str(throughput_path.resolve()), throughput_selected=selected_throughput,
                   policy_selection=selected_policy, capacity=capacity,
                   capacity_runs=sorted(capacity_rows, key=lambda row: row["E"]), runs=rows,
                   all_capacities_below_095=capacities_below_095,
                   writer_statement=("Validation EM is below 0.95 at every tested E for the selected policy/lr."
                                     if capacities_below_095 else
                                     "At least one tested E meets validation EM >= 0.95 for the selected policy/lr."),
                   policy_rule="best full-vocabulary EM; within 5/1000 of best prefer fact_final, then smaller lr",
                   capacity_rule="largest of E=16/24 with >=950/1000; else E=12 with >=950; else E=8 with >=800 flagged; else diagnosis",
                   loss_trend_rule=dict(final_step_fraction=_TREND_FRACTION,
                       fit="OLS loss against logged optimization step in the final 20% of requested steps",
                       relative_fitted_drop="negative slope times observed tail step span divided by mean tail loss",
                       threshold=_RELATIVE_TREND_THRESHOLD,
                       labels=">=1% still_falling; <=-1% rising; otherwise converged_descriptively; <2 points insufficient",
                       interpretation="descriptive only; not evidence of optimization completion and never used for selection"),
                   jobs_path=None if jobs_path is None else str(jobs_path.resolve()),
                   cost_semantics="job_node_hours repeats per shared job; sum unique job IDs, never per-run rows",
                   scope="nine completed writer runs only; no additional runs or tuning")
    if jobs is not None:
        unique_jobs = {str(row["job_id"]): dict(nodes=row["job_nodes"], elapsed_seconds=row["job_elapsed_seconds"],
                                               node_hours=row["job_node_hours"]) for row in rows}
        summary["jobs"] = unique_jobs
        summary["writer_jobs_node_hours"] = sum(job["node_hours"] for job in unique_jobs.values())
    reports.mkdir(parents=True, exist_ok=True)
    _write_csv(reports / "p1l_pilot1_runs.csv", rows)
    _write_csv(reports / "p1l_pilot1_loss_curves.csv", losses)
    _plot(reports / "p1l_pilot1_loss_curves.png", rows, losses)
    (reports / "p1l_pilot1_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--throughput", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, help="Optional JSON mapping PBS job IDs to nodes and elapsed_seconds")
    args = parser.parse_args()
    summary = analyze(args.policy_root, args.capacity_root, args.throughput, args.reports, args.jobs)
    print(json.dumps({key: summary[key] for key in ("policy_runs", "additional_capacity_runs", "capacity", "writer_statement")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
