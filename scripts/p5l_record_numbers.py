#!/usr/bin/env python3
"""Recompute the descriptive numbers quoted in reports/decisions/P5L.md and docs/reports/P5L.md (stdlib only).

Reads reports/p5l/r128 (1.7B) and reports/p1l/r128 (0.6B). Rates are percent; seed means over the seeds present.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def mean(v):
    return sum(v) / len(v)


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def table(directory, out):
    cell = defaultdict(dict)
    for r in load(directory / "accuracy.csv"):
        cell[(r["setting"], r["protocol"], r["reader"], int(r["L"]), int(r["hop"]))][int(r["seed"])] = float(r["accuracy"])
    out.append(f"== {directory} accuracy (full-vocabulary EM %, seed mean; n_seeds; per seed) ==")
    for key in sorted(cell):
        setting, protocol, reader, depth, hop = key
        if setting == "primary":
            seeds = cell[key]
            out.append(f"  {protocol:16s} {reader:6s} L={depth} hop{hop}: {100 * mean(list(seeds.values())):6.2f} "
                       f"(n={len(seeds)}; " + ", ".join(f"{s}:{100 * a:.2f}" for s, a in sorted(seeds.items())) + ")")
    for reader, depth in (("M1-L4", 4), ("B4", 1), ("B8", 4), ("B5-L4", 4), ("B3-L4", 4)):
        out.append(f"  mix hops2,3 {reader}@{depth}: primary "
                   f"{100 * mean([mean(list(cell[('primary', 'memory_only', reader, depth, h)].values())) for h in (2, 3)]):.2f}%, typed "
                   f"{100 * mean([mean(list(cell[('typed', 'memory_only', reader, depth, h)].values())) for h in (2, 3)]):.2f}%")
    return cell


def main():
    out = []
    for directory in (Path("reports/p5l/r128"), Path("reports/p1l/r128")):
        ep = json.loads((directory / "endpoints.json").read_text())
        out.append(f"== {directory} endpoints: G1 {ep['G1_L']} G2 {ep['G2_L']} B7 {ep['memory_use_passed']} ==")
        for name, c in sorted(ep["contrasts"].items()):
            acc = f"({100 * c['accuracy_a']:.2f} vs {100 * c['accuracy_b']:.2f}) " if "accuracy_a" in c else ""
            out.append(f"  {name}: {100 * c['difference']:.2f} [{100 * c['ci_low']:.2f},{100 * c['ci_high']:.2f}] {acc}per seed "
                       + ", ".join(f"{s['seed']}:{100 * s['difference']:.2f}" for s in c["per_seed"]))
        for name, c in sorted(ep["memory_use"]["contrasts"].items()):
            out.append(f"  B7 {name}: {100 * c['difference']:.2f} [{100 * c['ci_low']:.2f},{100 * c['ci_high']:.2f}] "
                       f"({100 * c['accuracy_a']:.2f} vs {100 * c['accuracy_b']:.2f})")
        table(directory, out)
        params = {r["reader"]: int(r["adapter_parameters"]) for r in load(directory / "inference.csv")}
        out.append("  adapter parameters: " + ", ".join(f"{k} {v}" for k, v in sorted(params.items())))
        sw = defaultdict(list)
        for r in load(directory / "swap-successors.csv"):
            sw[(r["reader"], int(r["L"]), int(r["hop"]))].append((float(r["follows_new"]), float(r["original_predicts_new"])))
        for key, v in sorted(sw.items()):
            if key[2] <= 3 and key[1] == 4:
                out.append(f"  swap {key[0]} L=4 hop{key[2]}: follows {100 * mean([x[0] for x in v]):.2f}% unswapped {100 * mean([x[1] for x in v]):.2f}%")
        times = defaultdict(list)
        for r in load(directory / "inference.csv"):
            if r["protocol"] == "memory_only":
                times[r["reader"]].append(float(r["mean_seconds_per_question"]))
        out.append("  mean inference ms/question: " + ", ".join(f"{k} {1000 * mean(v):.3f}" for k, v in sorted(times.items())))
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
