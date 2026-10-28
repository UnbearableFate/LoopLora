#!/usr/bin/env python3
"""Recompute the descriptive numbers quoted in reports/decisions/P1L.md from the committed P1-L tables (stdlib only).

Reads reports/p1l (r = 32) and reports/p1l/r128; writes reports/p1l/readout-pooled.csv and prints the numbers.
Rates are fractions in the CSVs and are printed here in percent. Pooled = unweighted mean over the seeds present
(every cell has n = 2000 per seed).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def mean(v):
    return sum(v) / len(v)


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def accuracy(rows, out):
    cell = defaultdict(dict)  # (setting, protocol, reader, L, hop) -> {seed: acc}
    for r in rows:
        cell[(r["setting"], r["protocol"], r["reader"], int(r["L"]), int(r["hop"]))][int(r["seed"])] = float(r["accuracy"])
    out.append("accuracy (full-vocabulary EM, % ; mean over seeds present; per-seed values in brackets)")
    for key in sorted(cell):
        setting, protocol, reader, depth, hop = key
        if setting != "primary":
            continue
        seeds = cell[key]
        out.append(f"  {protocol:16s} {reader:6s} L={depth} hop{hop}: {100 * mean(list(seeds.values())):6.2f} "
                   f"(n_seeds={len(seeds)}; " + ", ".join(f"{s}:{100 * a:.2f}" for s, a in sorted(seeds.items())) + ")")
    return cell


def mix(cell, setting, protocol, reader, depth, hops):
    return 100 * mean([mean(list(cell[(setting, protocol, reader, depth, h)].values())) for h in hops])


def readout(rows, rank, csv_rows, out):
    t = defaultdict(list)
    for r in rows:
        t[(r["method"], r["reader"], int(r["hop"]), int(r["entity_step"]), int(r["reader_step"]))].append(r)
    out.append(f"per-step readout, rank {rank}: top-1 rate (%) of the hop-s entity, entity_step e x reader_step s, "
               f"pooled over seeds [min seed, max seed]; mean rank in brackets")
    for key in sorted(t):
        method, reader, hop, e, s = key
        rs = t[key]
        rate = [float(x["top1_rate"]) for x in rs]
        csv_rows.append({"rank": rank, "method": method, "reader": reader, "hop": hop, "entity_step": e, "reader_step": s,
                         "n_seeds": len(rs), "n_per_seed": rs[0]["n"], "top1_rate": mean(rate), "top1_min": min(rate),
                         "top1_max": max(rate), "mean_rank": mean([float(x["mean_rank"]) for x in rs])})
    for method in ("logit_lens", "contrastive", "random", "raw_logit"):
        for reader in ("M1-L4", "B8"):
            out.append(f"  {method} {reader}")
            for hop in (1, 2, 3):
                for e in range(1, hop + 1):
                    cells = []
                    for s in range(1, 5):
                        rs = t[(method, reader, hop, e, s)]
                        rate = [float(x["top1_rate"]) for x in rs]
                        cells.append(f"s{s} {100 * mean(rate):5.2f} [{100 * min(rate):.2f},{100 * max(rate):.2f}] "
                                     f"({mean([float(x['mean_rank']) for x in rs]):.1f})")
                    out.append(f"    hop{hop} e{e}: " + " | ".join(cells))
            cells = [r for r in csv_rows if r["rank"] == rank and r["method"] == method and r["reader"] == reader]
            top = max(cells, key=lambda r: r["top1_rate"])
            out.append(f"    max pooled cell over hops 1-8: {100 * top['top1_rate']:.2f}% (hop{top['hop']} e{top['entity_step']} s{top['reader_step']}); "
                       f"max single seed {100 * max(r['top1_max'] for r in cells):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/p1l")
    a = ap.parse_args()
    base = Path(a.reports)
    out, csv_rows = [], []
    for rank, d in ((32, base), (128, base / "r128")):
        out.append(f"==== rank {rank} ====")
        ep = json.loads((d / "endpoints.json").read_text())
        out.append(f"G1-L {ep['G1_L']} G2-L {ep['G2_L']} B7 {ep['memory_use_passed']}")
        for name, c in sorted(ep["contrasts"].items()):
            out.append(f"  {name}: {100 * c['difference']:.2f} [{100 * c['ci_low']:.2f},{100 * c['ci_high']:.2f}] "
                       + (f"({100 * c['accuracy_a']:.2f} vs {100 * c['accuracy_b']:.2f}) " if "accuracy_a" in c else "")
                       + "per seed " + ", ".join(f"{s['seed']}:{100 * s['difference']:.2f}" for s in c["per_seed"]))
        for name, c in sorted(ep["memory_use"]["contrasts"].items()):
            out.append(f"  B7 {name}: {100 * c['difference']:.2f} [{100 * c['ci_low']:.2f},{100 * c['ci_high']:.2f}] "
                       f"({100 * c['accuracy_a']:.2f} vs {100 * c['accuracy_b']:.2f}) per seed "
                       + ", ".join(f"{x['seed']}:{100 * x['difference']:.2f}" for x in c["per_seed"]))
        cell = accuracy(load(d / "accuracy.csv"), out)
        for reader, depth in (("M1-L4", 4), ("B4", 1), ("B8", 4), ("B5-L4", 4), ("B3-L4", 4)):
            out.append(f"  mix hops2,3 {reader}@{depth}: memory_only primary {mix(cell, 'primary', 'memory_only', reader, depth, [2, 3]):.2f}%, "
                       f"typed {mix(cell, 'typed', 'memory_only', reader, depth, [2, 3]):.2f}%")
        params = {r["reader"]: int(r["adapter_parameters"]) for r in load(d / "inference.csv")}
        out.append("  adapter parameters: " + ", ".join(f"{k} {v}" for k, v in sorted(params.items())))
        sw = defaultdict(list)
        for r in load(d / "swap-successors.csv"):
            sw[(r["reader"], int(r["L"]), int(r["hop"]))].append((float(r["follows_new"]), float(r["original_predicts_new"])))
        for key, v in sorted(sw.items()):
            if key[2] <= 3:
                out.append(f"  swap {key[0]} L={key[1]} hop{key[2]}: follows {100 * mean([x[0] for x in v]):.2f}% unswapped {100 * mean([x[1] for x in v]):.2f}% (n_seeds={len(v)})")
        readout(load(d / "per-step-readout.csv"), rank, csv_rows, out)
    rc = json.loads((base / "rank-comparison.json").read_text())["contrasts"]["B8r128_vs_B8r32"]
    out.append(f"rank comparison B8@4 hops2,3 r128-r32: {100 * rc['difference']:.2f} [{100 * rc['ci_low']:.2f},{100 * rc['ci_high']:.2f}] per seed "
               + ", ".join(f"{s['seed']}:{100 * s['difference']:.2f}" for s in rc["per_seed"]))
    with open(base / "readout-pooled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print("\n".join(out))


if __name__ == "__main__":
    main()
