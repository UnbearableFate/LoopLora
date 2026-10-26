#!/usr/bin/env python3
"""Describe completed Pilot-1 validation errors after its writer stop rule fires.

Run on a Miyabi-G compute node with PYTHONPATH=src, for example::

    python scripts/diagnose_p1s_writer.py \
        --selection1 reports/p1s/pilot1/selection.json \
        --out reports/p1s/pilot1/writer_diagnosis.json

Reads full_table.json beside selection.json and selects the recorded best LR
per writer/rank/E (accuracy, then smaller LR). Reconstructs existing val1000
queries solely to annotate saved predictions; it never trains or calls a model.
Strata are descriptive associations, not intervention or tuning results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _run_path(row: dict[str, Any], report_dir: Path) -> Path:
    path = Path(row["run_dir"])
    return (path if path.is_absolute() else report_dir / path).resolve()


def _identity(row: dict[str, Any], report_dir: Path) -> tuple[Any, ...]:
    return (row["name"], row["writer"], row["r"], row["E"], row["lr"],
            row["accuracy"], str(_run_path(row, report_dir)))


def _checkpoint_gates(run: Path, config: dict[str, Any]) -> dict[str, Any]:
    if config["writer"].startswith("oracle_"):
        return {"source": "fixed oracle rule", "lambda": 1.0, "beta": 1.0}
    import torch

    checkpoint = run / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"] != config:
        raise AssertionError(f"checkpoint/config mismatch: {run}")
    state = saved["model"]
    gates = {}
    for name, parameter in (("lambda", "a_lambda"), ("beta", "a_beta")):
        value = state[f"writer.{parameter}"]
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise AssertionError(f"invalid scalar writer gate {parameter}: {run}")
        shared = state.get(f"adapter.{parameter}")
        if shared is not None and not torch.equal(value, shared):
            raise AssertionError(f"checkpoint writer/reader gate mismatch: {run}")
        gates[name] = float(value.sigmoid().item())
    return {"source": str(checkpoint), **gates}


def _diagnose_run(row: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    import numpy as np

    from looplora.data.relations import generate
    from looplora.sym.evaluate import aggregate_samples, stratify_queries
    from looplora.sym.model import make_task_spec
    from looplora.sym.pilot import loss_diagnostic
    from looplora.sym.stats import load_samples

    run = _run_path(row, report_dir)
    record = _read(run / "record.json")
    config_record = _read(run / "config.json")
    config = config_record["config"]
    if record["status"] != "complete":
        raise ValueError(f"diagnosis only accepts completed runs: {run}")
    expected = {"name": row["name"], "writer": row["writer"], "variant": "B4", "seed": 17}
    if any(config[key] != value for key, value in expected.items()):
        raise AssertionError(f"run/table identity mismatch: {run}")
    if (config["task"]["name"] != "t1" or config["task"]["n_rel"] != 4
            or config["task"]["head_sharing"] != .3 or config["model"]["r"] != row["r"]
            or config["model"]["r_mem"] != row["r"] or config["task"]["E"] != row["E"]
            or config["train"]["lr"] != row["lr"] or config["train"]["steps"] != 5000):
        raise AssertionError(f"run is not the recorded Pilot-1 setup: {run}")
    settings = config["eval"]
    if (settings["splits"] != ["val"] or settings["seed"] != 4242
            or settings["episodes"] != 1000 or settings["hops"] != [1]
            or settings["counterfactuals"] or row["depth"] != 1):
        raise ValueError(f"diagnosis is restricted to existing Pilot-1 val1000/seed4242: {run}")
    metrics = _read(run / "metrics.json")
    if not metrics["gate0_passed"] or metrics["query_protocol"] != "one_query_per_hop_from_support_v1":
        raise AssertionError(f"unverified evaluation protocol: {run}")
    samples = load_samples(run / "samples.npz")
    if (set(samples["split"].tolist()) != {"val"}
            or set(samples["condition"].tolist()) != {"original"}
            or set(samples["L"].tolist()) != {1}):
        raise AssertionError(f"samples contain unexpected splits/conditions/depths: {run}")
    if aggregate_samples(samples) != metrics["table"]:
        raise AssertionError(f"saved predictions do not reproduce metrics: {run}")
    if not np.array_equal(samples["prediction"] == samples["answer"], samples["correct"]):
        raise AssertionError(f"saved correctness does not match prediction/answer: {run}")
    if len(samples["answer"]) != row["n"] or not np.isclose(
            samples["correct"].mean(), row["accuracy"], rtol=0, atol=1e-12):
        raise AssertionError(f"full-table accuracy/count differs from samples: {run}")

    spec = make_task_spec(config, evaluation=True)
    arrays = stratify_queries(generate(spec, "val", 0, 1000, 4242), spec, settings["hops"], 4242)
    generated = {(int(episode), query): (i, query)
                 for i, episode in enumerate(arrays.episode_seed)
                 for query in range(arrays.answer.shape[1])}
    saved_keys = list(zip(samples["episode_id"].tolist(), samples["query_id"].tolist()))
    if len(generated) != len(saved_keys) or set(generated) != set(saved_keys):
        raise AssertionError(f"reconstructed episode/query labels differ: {run}")
    head_count, last_head_write, age = [], [], []
    for k, key in enumerate(saved_keys):
        i, q = generated[key]
        if (int(samples["answer"][k]) != int(arrays.answer[i, q])
                or int(samples["hop"][k]) != int(arrays.q_hop_true[i, q])):
            raise AssertionError(f"reconstructed answer/hop mismatch: {run}, {key}")
        facts, valid = arrays.facts[i], arrays.fact_valid[i]
        positions = np.flatnonzero(valid)
        start, relation = int(arrays.q_start[i, q]), int(arrays.q_rel[i, q])
        same_head = positions[facts[positions, 0] == start]
        queried = same_head[facts[same_head, 1] == relation]
        if len(queried) != 1 or len(np.unique(facts[same_head, 1])) != len(same_head):
            raise AssertionError(f"T1 keys must be unique and the queried fact must exist: {run}, {key}")
        position = int(queried[0])
        if int(facts[position, 2]) != int(samples["answer"][k]):
            raise AssertionError(f"queried fact does not contain the saved answer: {run}, {key}")
        head_count.append(len(same_head))
        last_head_write.append(position == int(same_head[-1]))
        age.append(int(np.count_nonzero(positions > position)))
    head_count = np.asarray(head_count)
    last_head_write = np.asarray(last_head_write, dtype=bool)
    age = np.asarray(age)
    correct = samples["correct"].astype(bool)
    cosine, memory = samples["first_read_cosine"], samples["memory_norm"]
    if cosine.shape != correct.shape or memory.shape != correct.shape or not np.isfinite(memory).all():
        raise AssertionError(f"invalid saved diagnostic shapes or memory norms: {run}")

    def summarize(mask: Any) -> dict[str, Any]:
        n = int(np.count_nonzero(mask))
        finite_cosine = mask & np.isfinite(cosine)
        n_cosine = int(np.count_nonzero(finite_cosine))
        return {"n_queries": n, "n_correct": int(np.count_nonzero(correct & mask)),
                "accuracy": float(correct[mask].mean()) if n else None,
                "chance_accuracy": float((1.0 / samples["candidate_size"][mask]).mean()) if n else None,
                "mean_first_read_cosine": float(cosine[finite_cosine].mean()) if n_cosine else None,
                "n_finite_first_read_cosine": n_cosine,
                "mean_memory_norm": float(memory[mask].mean()) if n else None,
                "mean_write_age": float(age[mask].mean()) if n else None,
                "min_write_age": int(age[mask].min()) if n else None,
                "max_write_age": int(age[mask].max()) if n else None}

    cuts = np.quantile(age, [.25, .5, .75])
    quartile = np.searchsorted(cuts, age, side="left")
    convergence = loss_diagnostic(run / "train_log.jsonl")
    for key, value in convergence.items():
        if row.get(key) != value:
            raise AssertionError(f"recorded convergence diagnostic differs: {run}, {key}")
    groups = {"one_fact": head_count == 1, "multiple_relation_specific_facts": head_count > 1}
    return {
        "name": row["name"], "writer": row["writer"], "r": row["r"], "E": row["E"], "lr": row["lr"],
        "run_dir": str(run), "samples": str(run / "samples.npz"),
        "config_hash": config_record.get("config_hash"), "runtime": record.get("runtime"),
        "reconstructed_episode_count": len(arrays.episode_seed), "joined_answers_equivalent": True,
        "selection_accuracy": row["accuracy"], "convergence": convergence,
        "writer_gates": _checkpoint_gates(run, config),
        "overall": summarize(np.ones(len(correct), dtype=bool)),
        "head_multiplicity": {name: summarize(mask) for name, mask in groups.items()},
        "queried_fact_is_last_write_for_head": {
            "yes": summarize(last_head_write), "no": summarize(~last_head_write)},
        "head_multiplicity_and_last_write": {
            f"{name}/{label}": summarize(mask & last)
            for name, mask in groups.items()
            for label, last in (("last", last_head_write), ("not_last", ~last_head_write))},
        "write_age_quartiles": {
            "definition": "number of valid fact writes after the queried fact; lower age is more recent",
            "cutpoints": cuts.tolist(),
            "boundary_rule": "Q1 <= p25; Q2 > p25 and <= p50; Q3 > p50 and <= p75; Q4 > p75; ties stay together",
            "groups": {f"Q{i + 1}": summarize(quartile == i) for i in range(4)}},
    }


def diagnose(selection_path: Path) -> dict[str, Any]:
    from looplora.sym.pilot import best

    selection_path = selection_path.resolve()
    selection = _read(selection_path)
    table_path = selection_path.with_name("full_table.json")
    table = _read(table_path)
    if not isinstance(table, list) or not table or len({row["name"] for row in table}) != len(table):
        raise ValueError("full_table.json must contain nonempty distinct completed Pilot-1 runs")
    keys = sorted({(row["writer"], row["r"], row["E"]) for row in table})
    selected_rows = [best([row for row in table if (row["writer"], row["r"], row["E"]) == key])
                     for key in keys]
    joint32 = {row["E"]: row for row in selected_rows if row["writer"] == "joint" and row["r"] == 32}
    if set(joint32) != {8, 12, 16, 24}:
        raise ValueError("Pilot-1 stop-rule diagnosis requires joint/r32 results at every registered E")
    for E, row in joint32.items():
        recorded = selection["joint_r32_best_by_E"][str(E)]
        if _identity(row, selection_path.parent) != _identity(recorded, selection_path.parent):
            raise AssertionError(f"selection/full-table best LR differs at joint/r32/E{E}")
    if not selection["writer_diagnosis_required"] or not all(row["accuracy"] < .95 for row in joint32.values()):
        raise ValueError("Pilot-1 writer stop rule has not fired; this script does not add tuning or evaluation")
    for row in selection["selected"].values():
        if row is not None and not any(_identity(row, selection_path.parent) == _identity(best_row, selection_path.parent)
                                       for best_row in selected_rows):
            raise AssertionError("selected run is not a recorded best LR for its writer/rank/E")
    return {
        "selection1": str(selection_path), "full_table": str(table_path),
        "scope": "completed Pilot-1 validation artifacts only; no training or model evaluation",
        "selection_rule": "best validation LR per writer/rank/E; smaller LR breaks accuracy ties",
        "stop_rule": "joint/r32 accuracy < 0.95 at each E in {8,12,16,24}",
        "stop_rule_triggered": True,
        "query_protocol": "generate(spec,val,0,1000,4242), then stratify_queries(spec,[1],4242)",
        "interpretation": [
            "Compare one-fact and shared-head strata within each run; report counts alongside accuracy.",
            "Oracle keys use only the head, so relation-specific facts for the same head overwrite the same key; last-head-write strata expose this representational limitation.",
            "A learned-writer shortfall on one-fact heads also involves capacity, optimization, query/readout alignment or random-code interference; these descriptive strata do not establish a cause.",
            "Write-age and gate associations are observational; plateau status is the recorded loss diagnostic, not proof of convergence.",
            "Do not use this report to retune learning rates, alter preregistered thresholds, or make a T2 mechanism conclusion.",
        ],
        "runs": [_diagnose_run(row, selection_path.parent) for row in selected_rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection1", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="JSON report path")
    args = parser.parse_args()
    result = diagnose(args.selection1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "diagnosed_runs": len(result["runs"]), "stop_rule_triggered": True}))


if __name__ == "__main__":
    main()
