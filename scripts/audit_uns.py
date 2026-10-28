#!/usr/bin/env python3
"""UNS (F11): audit of unseen-pool evaluations (docs/prereg/UNS.md).

Part A (default): reads the saved screening evaluations (seen pool ``test`` and unseen pool
``test_unseen`` of the same runs, eval seed 4242) and the committed ``reported_contrasts.csv``
exports; no confirmation archive is opened, nothing is trained or evaluated.

Part B (``--extension-root``): the same tables for the evaluations of ``scripts/run_uns.py`` (M1 trained at
L = 4 and the B4 references, eval seed 999006) after verifying the run's seals and Git-locked design.

Torch is never imported. Run through scripts/submit.py in a Miyabi-G allocation (a few minutes of one node).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from s9_common import (
    git_locked_file, interval, no_confirmation_path, now, percent, require, require_compute, table_lines,
    verify_git_lock, write_csv, write_json,
)
from looplora.infra.seeds import stable_hash
from looplora.sym import uns

DEFAULT_P1S = "/work/xg24i002/x10041/LoopLora/runs/p1s/main/20261026"
DEFAULT_F2 = "/work/xg24i002/x10041/LoopLora-wt/symf/runs/p1s/f2"
P1S_RANKS = {"primary": "32", "replication": "32", "secondary": "16"}


def load_committed(reports: Path) -> dict[str, list[dict]]:
    return {
        "P1-S": uns.load_committed(reports / "p1s" / "reported_contrasts.csv", "P1-S", P1S_RANKS),
        "P1-S-F1": uns.load_committed(reports / "p1s-f1" / "reported_contrasts.csv", "P1-S-F1"),
        "P1-S-F2": uns.load_committed(reports / "p1s-f2" / "reported_contrasts.csv", "P1-S-F2"),
    }


def load_extension(root: Path) -> dict:
    """Manifest of a completed Part B run, after its design lock, settings and coverage are verified."""
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["status"] == "complete" and manifest.get("partial") is False
            and manifest.get("checkpoints_unchanged") is True, "the extension run is not a completed full run")
    design = manifest["design"]
    require(manifest["design_hash"] == stable_hash(design), "the manifest design hash differs")
    design_path = Path(manifest["design_path"])
    verify_git_lock(design_path, manifest["design_lock_commit"], "the design")
    require(json.loads(git_locked_file(design_path, manifest["design_lock_commit"])) == design,
            "the manifest design differs from its Git lock")
    require(design["part"] == "B" and design["extension"] == uns.EXTENSION, "unregistered extension settings")
    expected = {uns.extension_name(writer, reader, seed) for writer in (*uns.WRITERS, uns.CONTROL)
                for reader in uns.READERS for seed in uns.SEEDS}
    runs = {row["name"]: row for row in manifest["runs"]}
    require(set(runs) == expected and {entry["name"] for entry in design["checkpoints"]} == expected,
            "the extension run does not cover the registered checkpoints")
    manifest["_runs_by_name"] = runs
    return manifest


def extension_directory(root: Path, manifest: dict, writer: str, reader: str, seed: int) -> Path:
    name = uns.extension_name(writer, reader, seed)
    directory = root / "evaluation" / name
    for file in ("samples.npz", "metrics.json"):
        if not (directory / file).is_file():
            raise FileNotFoundError(f"missing {directory / file}")
    require(uns.file_hash(directory / "samples.npz") == manifest["_runs_by_name"][name]["samples_blake2b"],
            f"samples changed after the run: {name}")
    return directory


def process(writer: str, args: argparse.Namespace, committed: dict[str, list[dict]] | None,
            extension: dict | None = None) -> dict:
    """Audit the six evaluations of one writer, then compute its tables if all pass."""
    samples, audits, failures = {}, [], []
    seed_of_evaluation = uns.EVAL_SEED if extension is None else extension["design"]["extension"]["seed"]
    for reader in uns.READERS:
        for seed in uns.SEEDS:
            try:
                if extension is None:
                    directory = uns.run_directory(writer, reader, seed, p1s_root=args.p1s_root, f2_root=args.f2_root)
                else:
                    directory = extension_directory(Path(args.extension_root), extension, writer, reader, seed)
            except FileNotFoundError as error:  # a missing cell is a finding; nothing is re-evaluated
                failures.append(f"{writer}/{reader}/s{seed}: {error}")
                continue
            no_confirmation_path(directory)
            report, restricted = uns.audit_run(directory, writer=writer, reader=reader, seed=seed,
                                               evaluation_seed=seed_of_evaluation)
            audits.append(report)
            samples[reader, seed] = restricted
            failures.extend(f"{writer}/{reader}/s{seed}: {message}" for message in report["failures"])
    result = {"writer": writer, "audits": audits, "failures": failures}
    if failures:
        return result
    options = dict(resamples=args.resamples, confidence=args.confidence)
    if extension is None:  # the committed exports belong to the screening evaluations only
        source = committed["P1-S-F2" if writer in uns.FACTORIZED else "P1-S"]
        result["reproduction"] = uns.reproduce_reported({s: samples["M1", s] for s in uns.SEEDS}, source, writer=writer)
    result["endpoints"] = uns.endpoint_rows(samples, writer, **options)
    result["accuracy"] = uns.accuracy_rows(samples, writer, **options)
    return result


def render(results: list[dict], confidence: float, *, extension: bool = False) -> list[str]:
    reader = "M1 trained at L = 4" if extension else "M1"
    lines = ["# UNS audit summary" + (" (Part B: new episodes, seed 999006)" if extension else ""), "",
             f"Intervals are {100 * confidence:.0f}% percentile intervals of a joint episode bootstrap; differences in "
             "percentage points. Descriptive only (docs/prereg/UNS.md)."
             + (f" Here M1 is the {reader} reader (P2-S-F1)." if extension else ""), "",
             "## Audit" + ("" if extension else " and reproduction"), ""]
    rows = []
    for result in results:
        checks = result.get("reproduction", [])
        row = [result["writer"], len(result["audits"]), len(result["failures"])]
        if not extension:
            row.append(f"{sum(c['reproduced'] for c in checks)}/{len(checks)}")
        rows.append(row)
    columns = ["writer", "runs audited", "audit failures"] + ([] if extension else ["committed rows reproduced"])
    lines += table_lines(columns, rows)
    for result in results:
        lines += ["", f"## {result['writer']}", ""]
        if result["failures"]:
            lines += ["Audit failed; no statistics were computed:", ""] + [f"- {message}" for message in result["failures"]]
            continue
        by_key: dict[tuple, dict] = {}
        for row in result["endpoints"]:
            by_key.setdefault((row["endpoint"], row["group"]), {})[row["pool"]] = row
        rows = [[endpoint, group, interval(pools["seen"]), interval(pools["unseen"]),
                 interval(pools["unseen_minus_seen"])] for (endpoint, group), pools in by_key.items()]
        lines += table_lines(["endpoint", "hops", "seen pool", "unseen pool", "unseen minus seen"], rows)
        lines += ["", f"Per-hop accuracy of {reader} at L = 4 (percent):", ""]
        accuracy = {(r["reader"], r["L"], r["hop"], r["pool"]): r for r in result["accuracy"]}
        rows = [[hop, f"{100 * accuracy['M1', 4, hop, 'seen']['accuracy']:.2f}",
                 f"{100 * accuracy['M1', 4, hop, 'unseen']['accuracy']:.2f}",
                 percent(accuracy["M1", 4, hop, "unseen_minus_seen"]["accuracy"])] for hop in uns.HOPS]
        lines += table_lines(["hop", "seen", "unseen", "unseen minus seen (pp)"], rows)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1s-root", default=DEFAULT_P1S, help="P1-S main run root (holds runs/primary-*)")
    parser.add_argument("--f2-root", default=DEFAULT_F2, help="P1-S-F2 root (holds evaluation/f2-main-*)")
    parser.add_argument("--reports", default="reports", help="directory with the committed reported_contrasts exports")
    parser.add_argument("--extension-root", help="audit the Part B evaluations under this scripts/run_uns.py root "
                                                 "instead of the committed screening evaluations")
    parser.add_argument("--out", required=True, help="output directory (absolute path)")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    require_compute()
    require(args.resamples >= 2000, "at least 2000 resamples are registered")
    started = time.perf_counter()
    out = Path(args.out)
    extension = None if args.extension_root is None else load_extension(Path(args.extension_root))
    committed = None if extension is not None else load_committed(Path(args.reports))
    results = [process(writer, args, committed, extension) for writer in (*uns.WRITERS, uns.CONTROL)]
    failures = [message for result in results for message in result["failures"]]
    write_json(out / "audit.json", {"created": now(), "status": "failed" if failures else "passed",
                                    "failures": failures, "runs": [a for r in results for a in r["audits"]]})
    if not failures:
        if extension is None:
            write_csv(out / "reproduction.csv", [row for r in results for row in r["reproduction"]])
        write_csv(out / "endpoints.csv", [row for r in results for row in r["endpoints"]])
        write_csv(out / "accuracy.csv", [row for r in results for row in r["accuracy"]])
    if extension is None:
        write_csv(out / "context.csv", uns.context_table([row for rows in committed.values() for row in rows]))
    (out / "summary.md").write_text("\n".join(render(results, args.confidence, extension=extension is not None)) + "\n")
    write_json(out / "summary.json", {
        "created": now(), "seconds": time.perf_counter() - started, "resamples": args.resamples,
        "confidence": args.confidence, "writers": [r["writer"] for r in results],
        "part": "A" if extension is None else "B", "audit_failures": len(failures),
        "reproduced": {r["writer"]: [c["reproduced"] for c in r.get("reproduction", [])] for r in results},
        **({} if extension is None else dict(design_hash=extension["design_hash"],
                                              design_lock_commit=extension["design_lock_commit"],
                                              evaluation_node_hours=extension["node_hours"]))})
    print(f"[uns] audit {'FAILED' if failures else 'passed'}; outputs in {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
