#!/usr/bin/env python3
"""LSR offline analysis (numpy only; docs/prereg/LSR.md).

  design   write reports/lsr/design.json: the registered prereg digest, grid, rules, ranges and the six checkpoints
  test     read the test traces of both backbones, apply the locked thresholds and write the registered endpoints, the paired
           episode bootstrap (10,000 draws, 99.17%), the per-seed conditions, the descriptive tables and the oracles
  select   read the validation traces of both backbones and write reports/lsr/thresholds.json (status locked): the six
           thresholds, L_c, the O1 depths, and every grid value's validation numbers. Validation traces only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import csv

import numpy as np

from looplora.llm import lsr_rules as rules, lsr_stats as stats
from run_p1l_main import ROOT, digest, save

PREREG = "docs/prereg/LSR.md"
DESIGN = "reports/lsr/design.json"
# Clarification C1 (master ad1a9b5) changed the prereg file after the validation runs started; it changes no rule and no code. The
# validation records carry the hash of the prereg as registered at 23112ba.
EARLIER_PREREG_HASHES = {"91235d80a38f200e3a7b3bec907c1f1e": "registered 23112ba, before Clarification C1"}
THRESHOLDS = "reports/lsr/thresholds.json"
OBJECTIVE = ("per rule R1/R3 and backbone: maximise the summed correct count on hops 4-8 over the pooled validation questions of the "
             "three seeds, over grid values with pooled hop-1 change against fixed L = 4 >= -8 pp and >= -10 pp in every seed; "
             "ties: fewer mean reads, then smaller k; no feasible value: the rule fails and its test statistics are never computed")


def checkpoints() -> dict:
    result = {}
    for backbone, spec in rules.BACKBONES.items():
        path = (spec["root"] or ROOT) / spec["manifest"]
        manifest = json.loads(path.read_text())
        runs = {}
        for seed in rules.SEEDS:
            name = f"M1-L4-s{seed}"
            record = json.loads((Path(manifest["runs"][name]) / "record.json").read_text())
            runs[name] = dict(path=manifest["runs"][name], checkpoint_hash=record["checkpoint_hash"], status=record["status"])
        result[backbone] = dict(label=spec["label"], manifest=str(path), manifest_hash=digest(path),
                                backbone=spec["backbone"], runs=runs)
    return result


def design() -> dict:
    return dict(study="lsr", prereg=dict(path=PREREG, hash=digest(ROOT / PREREG)),
                earlier_prereg_hashes=EARLIER_PREREG_HASHES, rules=dict(
        R1="||z^s - z^(s-1)|| / ||z^(s-1)|| <= tau1, first s in 1..8, s = 8 forced; cost s reads",
        R2="p^s = p^(s-1) = p^(s-2) (p^0 = L = 0 prediction), s >= 2, s = 8 forced; cost s reads and s + 1 upper-layer passes",
        R3="||r^s|| / ||r^0|| <= tau2 with r = U M Q z, s = 1..7, s = 8 forced; cost s + 1 reads"),
        grid_k=list(rules.GRID_K), grid="tau = 2^(k/2)", objective=OBJECTIVE,
        margins_pp=dict(pooled=rules.MARGIN_POOLED, per_seed=rules.MARGIN_SEED),
        ranges=dict(validation=rules.VAL_RANGE, smoke=rules.SMOKE_RANGE, latency=rules.LATENCY_RANGE,
                    test=dict(split="test", start=106000, count=2000, seed=rules.SEED, window=rules.WINDOW)),
        checkpoints=checkpoints())


def load_backbone(directory: Path, backbone: str, locked: dict) -> tuple[dict, dict]:
    """Validation traces of the three seeds, checked against their records and the design lock."""
    traces, hashes, commits = {}, {}, set()
    prereg_hash = digest(ROOT / PREREG)
    for seed in rules.SEEDS:
        name = f"M1-L4-s{seed}"
        record = json.loads((directory / name / "record.json").read_text())
        trace_path = directory / name / "trace.npz"
        if (record["status"] != "complete" or record["trace_hash"] != digest(trace_path) or record["backbone"] != backbone
                or record["split"] != "val" or record["window"] != rules.VAL_RANGE
                or record["checkpoint_hash"] != locked["checkpoints"][backbone]["runs"][name]["checkpoint_hash"]
                or record["prereg_hash"] not in {prereg_hash, *locked["earlier_prereg_hashes"]} or record["source_dirty"]):
            raise ValueError(f"{name}: the validation record does not match the design lock")
        commits.add(record["git_commit"])
        with np.load(trace_path) as data:
            trace = {key: data[key] for key in data.files}
        if trace["pred"].shape != (rules.VAL_RANGE["count"], 8, 9) or not np.array_equal(
                trace["episode_index"], rules.VAL_RANGE["start"] + np.arange(rules.VAL_RANGE["count"])):
            raise ValueError(f"{name}: unexpected validation trace shape or episodes")
        traces[seed], hashes[name] = trace, record["trace_hash"]
    if len(commits) != 1:
        raise ValueError("the six validation runs must share one source commit")
    return traces, dict(trace_hashes=hashes, source_commit=commits.pop())


def select(args) -> dict:
    locked = json.loads((ROOT / DESIGN).read_text())
    if locked["prereg"]["hash"] != digest(ROOT / PREREG):
        raise ValueError("the design lock names another prereg")
    result = dict(study="lsr", status="locked", prereg=locked["prereg"], design=dict(path=DESIGN, hash=digest(ROOT / DESIGN)),
                  objective=OBJECTIVE, grid_k=list(rules.GRID_K), validation=rules.VAL_RANGE, backbones={})
    for backbone, directory in (("0.6b", args.val_06b), ("1.7b", args.val_17b)):
        traces, provenance = load_backbone(directory, backbone, locked)
        chosen = rules.select_thresholds(traces)
        summary = dict(provenance, baseline_em_by_hop=chosen["baseline_em_by_hop"],
                       oracle_best_fixed_L_by_hop=chosen["oracle_best_fixed_L_by_hop"],
                       fixed_em_by_depth_hop=chosen["fixed_em_by_depth_hop"], fixed=chosen["fixed"],
                       rules=chosen["rules"])
        result["backbones"][backbone] = summary
    return result


def load_test(directory: Path, backbone: str, locked: dict, prereg_hash: str) -> tuple[dict, dict]:
    """Test traces of the three seeds, checked against their records and the design lock."""
    traces, hashes, commits, latency = {}, {}, set(), {}
    window = dict(split="test", start=106000, count=2000, seed=rules.SEED)
    for seed in rules.SEEDS:
        name = f"M1-L4-s{seed}"
        record = json.loads((directory / name / "record.json").read_text())
        trace_path = directory / name / "trace.npz"
        if (record["status"] != "complete" or record["trace_hash"] != digest(trace_path) or record["backbone"] != backbone
                or record["split"] != "test" or record["window"] != window or record["prereg_hash"] != prereg_hash
                or record["checkpoint_hash"] != locked["checkpoints"][backbone]["runs"][name]["checkpoint_hash"]
                or record["source_dirty"]):
            raise ValueError(f"{name}: the test record does not match the design lock")
        commits.add(record["git_commit"])
        with np.load(trace_path) as data:
            trace = {key: data[key] for key in data.files}
        if trace["pred"].shape != (2000, 8, 9) or not np.array_equal(trace["episode_index"], 106000 + np.arange(2000)):
            raise ValueError(f"{name}: unexpected test trace shape or episodes")
        traces[seed], hashes[name] = trace, record["trace_hash"]
        if "latency" in record:
            latency[name] = record["latency"]
    if len(commits) != 1:
        raise ValueError("the three test runs of a backbone must share one source commit")
    return traces, dict(trace_hashes=hashes, source_commit=commits.pop(), latency=latency)


def write_tables(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    endpoints, accuracy, depths = [], [], []
    for backbone, value in result["backbones"].items():
        for rule, entry in value["rules"].items():
            row = dict(backbone=backbone, rule=rule, status=entry["status"], passes=entry["passes"])
            if entry["status"] == "evaluated":
                row.update(tau=entry["tau"], gain48_pp=entry["gain48"][0], gain48_lo=entry["gain48"][1], gain48_hi=entry["gain48"][2],
                           hop1_pp=entry["hop1"][0], hop1_lo=entry["hop1"][1], hop1_hi=entry["hop1"][2],
                           **{f"cond_{k}": v for k, v in entry["conditions"].items()},
                           **{f"seed{s}_gain48": v["gain48"][0] for s, v in entry["per_seed"].items()},
                           **{f"seed{s}_hop1": v["hop1"][0] for s, v in entry["per_seed"].items()},
                           mean_reads=entry["mean_reads"], mean_depth=entry["mean_depth"],
                           upper_layer_passes=entry["upper_layer_passes"])
                for hop, em in enumerate(entry["em_by_hop"], 1):
                    accuracy.append(dict(backbone=backbone, series=rule, seed="pooled", hop=hop, em=em))
                for seed, ems in entry["em_by_hop_per_seed"].items():
                    for hop, em in enumerate(ems, 1):
                        accuracy.append(dict(backbone=backbone, series=rule, seed=seed, hop=hop, em=em))
                for hop, counts in enumerate(entry["stop_depth_counts_by_hop"], 1):
                    for depth, count in enumerate(counts, 1):
                        depths.append(dict(backbone=backbone, rule=rule, hop=hop, depth=depth, count=count,
                                           mean_reads=entry["mean_reads_by_hop"][hop - 1]))
            endpoints.append(row)
        for name, entry in value["fixed"].items():
            if "em_by_hop" in entry:
                for hop, em in enumerate(entry["em_by_hop"], 1):
                    accuracy.append(dict(backbone=backbone, series=name, seed="pooled", hop=hop, em=em))
                for seed, ems in entry["em_by_hop_per_seed"].items():
                    for hop, em in enumerate(ems, 1):
                        accuracy.append(dict(backbone=backbone, series=name, seed=seed, hop=hop, em=em))
        for name, entry in value["oracles"].items():
            if isinstance(entry, dict):
                for hop, em in enumerate(entry["em_by_hop"], 1):
                    accuracy.append(dict(backbone=backbone, series=name.split("_")[0], seed="pooled", hop=hop, em=em))
    for name, table in (("endpoints.csv", endpoints), ("accuracy.csv", accuracy), ("stop-depth.csv", depths)):
        keys = list(dict.fromkeys(key for row in table for key in row))
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table)


def test(args) -> dict:
    locked = json.loads((ROOT / DESIGN).read_text())
    thresholds_path = ROOT / THRESHOLDS
    thresholds = json.loads(thresholds_path.read_text())
    prereg_hash = digest(ROOT / PREREG)
    if (thresholds["status"] != "locked" or thresholds["prereg"]["hash"] != prereg_hash
            or locked["prereg"]["hash"] != prereg_hash or thresholds["design"]["hash"] != digest(ROOT / DESIGN)):
        raise ValueError("the thresholds are not locked under this prereg and design")
    result = dict(study="lsr", prereg=dict(path=PREREG, hash=prereg_hash),
                  thresholds=dict(path=THRESHOLDS, hash=digest(thresholds_path)), comparisons=stats.COMPARISONS,
                  confidence=1 - stats.ALPHA, draws=stats.DRAWS, backbones={}, provenance={})
    for backbone, directory in (("0.6b", args.test_06b), ("1.7b", args.test_17b)):
        traces, provenance = load_test(directory, backbone, locked, prereg_hash)
        result["backbones"][backbone] = stats.evaluate_backbone(traces, thresholds["backbones"][backbone], backbone)
        result["provenance"][backbone] = provenance
    result["passing"] = {backbone: {rule: entry["passes"] for rule, entry in value["rules"].items()}
                         for backbone, value in result["backbones"].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("design").add_argument("--out", type=Path, default=ROOT / DESIGN)
    chosen = sub.add_parser("select")
    chosen.add_argument("--val-06b", type=Path, required=True)
    chosen.add_argument("--val-17b", type=Path, required=True)
    chosen.add_argument("--out", type=Path, required=True, help="absolute path: a job's ROOT is a throw-away snapshot")
    tested = sub.add_parser("test")
    tested.add_argument("--test-06b", type=Path, required=True)
    tested.add_argument("--test-17b", type=Path, required=True)
    tested.add_argument("--out", type=Path, required=True, help="absolute directory for results.json and the CSV tables")
    args = parser.parse_args()
    if args.command == "test":
        if (args.out / "results.json").exists():
            raise ValueError("results.json exists: the test analysis is written once")
        result = test(args)
        save(args.out / "results.json", result)
        write_tables(result, args.out)
        print(json.dumps(result["passing"]), flush=True)
        return
    if args.command == "design":
        save(args.out, design())
    else:
        out = args.out
        if out.exists():
            raise ValueError("thresholds.json exists: a lock is written once")
        save(out, select(args))
        value = json.loads(out.read_text())
        for backbone, entry in value["backbones"].items():
            print(backbone, {rule: entry["rules"][rule]["selected"] for rule in rules.RULES},
                  "L_c", entry["fixed"]["selected"], flush=True)


if __name__ == "__main__":
    main()
