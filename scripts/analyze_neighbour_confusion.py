#!/usr/bin/env python3
"""NCR (review item C4): neighbour-confusion rate of single-read errors (docs/prereg/NCR.md).

Descriptive analysis of saved evaluation samples only: nothing is trained or
re-evaluated. The evaluation episodes are regenerated from each directory's
recorded config and verified against the saved answers, hops, candidate counts
and correctness row by row. Run through scripts/submit.py (CPU work, a few minutes).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from audit_uns import DEFAULT_F2, DEFAULT_P1S
from s9_common import now, require, require_compute, table_lines, write_csv, write_json
from looplora.infra.seeds import stable_hash
from looplora.sym import ncr, uns

WRITERS = ("t1_frozen_fact", "joint_fact", "joint", "t1_frozen", "oracle_random", "oracle_orth")
VARIANTS = ("M1", "B4")
DATASETS = ("screening", "confirmation")
HOPS = (2, 3)
DEPTH = 1
SPLIT = "test"


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}"


def analyse(args: argparse.Namespace) -> dict:
    cache: dict[str, object] = {}
    rows_accuracy, rows_categories, missing, verified, undefined = [], [], [], [], []
    for writer in args.writers:
        for variant in args.variants:
            for dataset in args.datasets:
                per_seed = {hop: {} for hop in args.hops}
                for seed in uns.SEEDS:
                    try:
                        directory = uns.run_directory(writer, variant, seed, p1s_root=args.p1s_root,
                                                      f2_root=args.f2_root, dataset=dataset)
                    except FileNotFoundError as error:
                        missing.append(str(error))
                        continue
                    samples, config = ncr.load_directory(directory)
                    key = stable_hash([config["task"], config["eval"], SPLIT])
                    if key not in cache:
                        cache[key] = ncr.regenerate(config, SPLIT)
                    for hop in args.hops:
                        per_seed[hop][seed] = ncr.confusion_rows(samples, cache[key], hop=hop, depth=DEPTH,
                                                                 split=SPLIT)
                        verified.append(dict(directory=str(directory), hop=hop,
                                             rows=int(len(per_seed[hop][seed]["episode_id"]))))
                for hop in args.hops:
                    if sorted(per_seed[hop]) != list(uns.SEEDS):
                        continue  # incomplete group: reported through ``missing``
                    label = dict(writer=writer, variant=variant, dataset=dataset, hop=hop)
                    try:
                        result = ncr.summarize(per_seed[hop], confidence=0.95, resamples=args.resamples,
                                               label=(writer, variant, dataset, hop))
                    except ncr.UndefinedShares as error:  # the registered case; any other error stops the analysis
                        undefined.append({**label, "reason": str(error)})
                        continue
                    rows_accuracy.append({**label, "L": DEPTH, "accuracy": result["accuracy"],
                                          "ci_low": result["accuracy_ci_low"], "ci_high": result["accuracy_ci_high"],
                                          "chance_accuracy": result["chance_accuracy"],
                                          "below_chance": result["below_chance"], "n_episodes": result["n_episodes"],
                                          **{f"errors_seed_{s}": n for s, n in result["errors_per_seed"].items()}})
                    for name, entry in result["categories"].items():
                        rows_categories.append({
                            **label, "L": DEPTH, "category": name, "rate": entry["rate"],
                            "rate_ci_low": entry["rate_ci_low"], "rate_ci_high": entry["rate_ci_high"],
                            "uniform_rate": entry["uniform_rate"], "share": entry["share"],
                            "share_ci_low": entry["share_ci_low"], "share_ci_high": entry["share_ci_high"],
                            "uniform_share": entry["uniform_share"],
                            **{f"share_seed_{s}": v for s, v in zip(uns.SEEDS, entry["per_seed_share"])}})
    return dict(accuracy=rows_accuracy, categories=rows_categories, missing=missing, verified=verified,
                undefined=undefined)


def render(result: dict) -> list[str]:
    lines = ["# NCR summary", "",
             f"Single read (L = {DEPTH}), split {SPLIT}, pooled over the training seeds; percent. Intervals are 95% "
             "percentile intervals of a joint episode bootstrap and are descriptive (docs/prereg/NCR.md). 'Chance' "
             "columns are the values when the guess (rate) or the error (share) is uniform over the candidates.", ""]
    for hop in HOPS:
        categories = {(r["writer"], r["variant"], r["dataset"], r["category"]): r
                      for r in result["categories"] if r["hop"] == hop}
        rows = []
        for acc in (r for r in result["accuracy"] if r["hop"] == hop):
            first = categories[acc["writer"], acc["variant"], acc["dataset"], "first_hop"]
            row = [acc["writer"], acc["variant"], acc["dataset"], pct(acc["accuracy"]), pct(acc["chance_accuracy"]),
                   "yes" if acc["below_chance"] else "no", pct(first["rate"]), pct(first["uniform_rate"]),
                   f"{pct(first['share'])} [{pct(first['share_ci_low'])}, {pct(first['share_ci_high'])}]",
                   pct(first["uniform_share"])]
            for name in ("start", "terminal_other", "other") + (("one_short",) if hop >= 3 else ()):
                entry = categories[acc["writer"], acc["variant"], acc["dataset"], name]
                row.append(f"{pct(entry['share'])} ({pct(entry['uniform_share'])})")
            rows.append(row)
        tail = ["start", "terminal_other", "other"] + (["one_short"] if hop >= 3 else [])
        lines += [f"## Hop {hop}", ""]
        lines += table_lines(["writer", "reader", "evaluation", "accuracy", "chance", "below chance",
                              "first-hop rate", "chance", "first-hop share of errors", "chance",
                              *[f"{name} share (chance)" for name in tail]], rows)
        lines.append("")
    if result["undefined"]:
        lines += ["## Undefined groups", ""] + [f"- {u['writer']}/{u['variant']}/{u['dataset']}/hop {u['hop']}: "
                                                  f"{u['reason']}" for u in result["undefined"]] + [""]
    if result["missing"]:
        lines += ["## Missing evaluation directories", ""] + [f"- {m}" for m in result["missing"]] + [""]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1s-root", default=DEFAULT_P1S, help="P1-S main run root")
    parser.add_argument("--f2-root", default=DEFAULT_F2, help="P1-S-F2 root")
    parser.add_argument("--out", required=True, help="output directory (absolute path)")
    parser.add_argument("--writers", nargs="+", default=list(WRITERS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--hops", nargs="+", type=int, default=list(HOPS))
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    require_compute()
    require(args.resamples >= 2000 and all(h >= 2 for h in args.hops), "invalid resamples or hops")
    started = time.perf_counter()
    result = analyse(args)
    out = Path(args.out)
    write_csv(out / "accuracy.csv", result["accuracy"])
    write_csv(out / "categories.csv", result["categories"])
    (out / "summary.md").write_text("\n".join(render(result)) + "\n")
    write_json(out / "summary.json", {
        "created": now(), "seconds": time.perf_counter() - started, "resamples": args.resamples,
        "groups": len(result["accuracy"]), "undefined": result["undefined"], "missing": result["missing"],
        "verified_directories": len({v["directory"] for v in result["verified"]}),
        "verified_rows": sum(v["rows"] for v in result["verified"])})
    print(f"[ncr] {len(result['accuracy'])} groups, {len(result['missing'])} missing directories; outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
