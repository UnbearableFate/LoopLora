#!/usr/bin/env python3
"""LSR offline analysis (numpy only; docs/prereg/LSR.md).

  design   write reports/lsr/design.json: the registered prereg digest, grid, rules, ranges and the six checkpoints
  select   read the validation traces of both backbones and write reports/lsr/thresholds.json (status locked): the six
           thresholds, L_c, the O1 depths, and every grid value's validation numbers. Validation traces only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from looplora.llm import lsr_rules as rules
from run_p1l_main import ROOT, digest, save

PREREG = "docs/prereg/LSR.md"
DESIGN = "reports/lsr/design.json"
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
    return dict(study="lsr", prereg=dict(path=PREREG, hash=digest(ROOT / PREREG)), rules=dict(
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
                or record["prereg_hash"] != prereg_hash or record["source_dirty"]):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("design").add_argument("--out", type=Path, default=ROOT / DESIGN)
    chosen = sub.add_parser("select")
    chosen.add_argument("--val-06b", type=Path, required=True)
    chosen.add_argument("--val-17b", type=Path, required=True)
    chosen.add_argument("--out", type=Path, required=True, help="absolute path: a job's ROOT is a throw-away snapshot")
    args = parser.parse_args()
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
