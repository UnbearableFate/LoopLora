#!/usr/bin/env python3
"""IVD (F8) analysis of a completed confirmation run (docs/prereg/IVD.md).

Every lock, seal and structure check runs before any statistic is computed. The registered
decision statistic is, per writer, the M1 depth gain (L4 minus L1) under low key overlap minus the
same under high key overlap, at load L24 on hops 2-3, paired by query, with a joint episode
bootstrap interval at confidence 1 - 0.05 / 2. Everything else is descriptive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import run_ivd
from s9_common import (
    digest_checkpoint, git_locked_file, interval, now, percent, require, table_lines, verify_git_lock, write_csv,
    write_json,
)
from looplora.infra.seeds import stable_hash
from looplora.sym import ivd
from looplora.sym.stats import load_samples

MIN_OVERLAP_GAP = 0.2  # flagged addition (docs/prereg/IVD.md): below this a null result is 'inconclusive'


def audit_run(run: Path) -> tuple[dict, dict]:
    """Verify the run's seals and structure; returns (manifest, samples[(writer, variant)][seed])."""
    manifest = json.loads((run / "manifest.json").read_text())
    require(manifest["status"] == "complete" and manifest["smoke"] is False
            and manifest.get("checkpoints_unchanged") is True, "the run is not a completed main run")
    design = manifest["design"]
    require(manifest["design_hash"] == stable_hash(design), "the manifest design hash differs")
    design_path = Path(manifest["design_path"])
    verify_git_lock(design_path, manifest["design_lock_commit"], "the design")
    require(json.loads(git_locked_file(design_path, manifest["design_lock_commit"])) == design,
            "the manifest design differs from its Git lock")
    entries = run_ivd.verify_design(design)
    require(manifest["evaluation"] == design["confirmation"], "unregistered evaluation set")
    timings = {row["name"]: row for row in manifest["runs"]}
    require(set(timings) == {entry["name"] for entry in entries}, "the run does not cover the registered checkpoints")
    samples: dict[tuple[str, str], dict[int, dict]] = {}
    episodes = None
    expected_rows = design["confirmation"]["episodes"] * len(design["hops"])
    for entry in entries:
        path = run / "samples" / f"{entry['name']}.npz"
        require(path.is_file() and digest_checkpoint(path) == timings[entry["name"]]["samples_blake2b"],
                f"samples missing or changed: {entry['name']}")
        arrays = load_samples(path)
        writer, variant = entry["writer"], entry["variant"]
        require(set(arrays["split"].tolist()) == {design["confirmation"]["split"]}, f"{entry['name']}: wrong split")
        require(set(arrays["condition"].tolist()) == set(ivd.cell_names(writer, design["loads"])),
                f"{entry['name']}: unexpected cells")
        require(sorted({int(v) for v in arrays["L"]}) == design["depths"][variant], f"{entry['name']}: wrong depths")
        require(sorted({int(v) for v in arrays["hop"]}) == design["hops"], f"{entry['name']}: wrong hops")
        for condition in ivd.cell_names(writer, design["loads"]):
            for depth in design["depths"][variant]:
                rows = int(((arrays["condition"] == condition) & (arrays["L"] == depth)).sum())
                require(rows == expected_rows, f"{entry['name']}: {condition} L={depth} has {rows} rows")
        ids = sorted({int(v) for v in arrays["episode_id"]})
        episodes = ids if episodes is None else episodes
        require(ids == episodes and len(ids) == design["confirmation"]["episodes"],
                f"{entry['name']}: episodes differ from the other checkpoints")
        samples.setdefault((writer, variant), {})[entry["seed"]] = arrays
    return manifest, samples


def decision_label(passes: bool, gap: float) -> str:
    if passes:
        return "supported"
    return "not_supported" if gap >= MIN_OVERLAP_GAP else "inconclusive"


def analyse(samples: dict, design: dict, resamples: int) -> dict:
    spec = design["decision"]
    hops = tuple(spec["hops"])
    interference, cells, accuracy, gaps = [], [], [], {}
    full_cells, full_accuracy = [], []  # the same tables on full-vocabulary exact match (no candidate restriction)
    for writer in ivd.WRITERS:
        m1, b4 = samples[writer, "M1"], samples[writer, "B4"]
        for group, group_hops in ivd.HOP_GROUPS.items():
            for load in design["loads"]:
                result = ivd.interference_contrast(m1, load=load, hops=group_hops, resamples=resamples,
                                                   confidence=spec["confidence"], label=(writer, group))
                decision = load == spec["load"] and tuple(group_hops) == hops
                interference.append(dict(
                    writer=writer, load=load, hop_group=group, hops=",".join(map(str, group_hops)),
                    role="decision" if decision else "reported", difference=result["difference"],
                    ci_low=result["ci_low"], ci_high=result["ci_high"], confidence=result["confidence"],
                    passes=result["passes"], n_episodes=result["n_episodes"],
                    **{f"difference_seed_{row['seed']}": row["difference"] for row in result["per_seed"]}))
            for row in ivd.cell_table(m1, writer, hops=group_hops, resamples=resamples):
                cells.append(dict(hop_group=group, **row))
            for row in ivd.accuracy_table(b4, writer, depth=1, hops=group_hops, resamples=resamples):
                accuracy.append(dict(hop_group=group, reader="B4", **row))
            for row in ivd.cell_table(m1, writer, hops=group_hops, resamples=resamples, field="full_correct"):
                full_cells.append(dict(hop_group=group, **row))
            for row in ivd.accuracy_table(b4, writer, depth=1, hops=group_hops, resamples=resamples,
                                          field="full_correct"):
                full_accuracy.append(dict(hop_group=group, reader="B4", **row))
        reach = {row["condition"]: row for row in cells if row["writer"] == writer and row["hop_group"] == "reach"}
        gaps[writer] = {load: reach[f"{load}-high"]["overlap_mean"] - reach[f"{load}-low"]["overlap_mean"]
                        for load in design["loads"]}
    decisions = {}
    for row in interference:
        if row["role"] == "decision":
            gap = gaps[row["writer"]][spec["load"]]
            decisions[row["writer"]] = dict(passes=row["passes"], overlap_gap=gap,
                                            label=decision_label(row["passes"], gap))
    return dict(interference=interference, cells=cells, accuracy=accuracy, gaps=gaps, decisions=decisions,
                full_cells=full_cells, full_accuracy=full_accuracy)


def render(result: dict, design: dict, manifest: dict) -> list[str]:
    spec = design["decision"]
    lines = ["# IVD summary", "",
             f"Confirmation episodes: split {design['confirmation']['split']}, seed {design['confirmation']['seed']}, "
             f"{design['confirmation']['episodes']} episodes per checkpoint; three training seeds pooled; "
             "differences in percentage points.", "",
             f"## Registered decision (M1, load {spec['load']}, hops {','.join(map(str, spec['hops']))}, "
             f"{100 * spec['confidence']:.1f}% interval)", "",
             "Interference contrast = (L4 - L1) under low overlap minus (L4 - L1) under high overlap.", ""]
    rows = []
    for row in result["interference"]:
        if row["role"] == "decision":
            decision = result["decisions"][row["writer"]]
            rows.append([row["writer"], interval(row), "yes" if row["passes"] else "no",
                         f"{decision['overlap_gap']:.3f}", decision["label"]])
    lines += table_lines(["writer", "contrast [interval]", f">= {100 * spec['minimum']:.0f} pp and interval > 0",
                          "overlap gap (high - low)", "decision"], rows)
    lines += ["", "## Interference contrast by load and hop group (reported)", ""]
    rows = [[r["writer"], r["load"], r["hop_group"], interval(r), "yes" if r["passes"] else "no"]
            for r in result["interference"]]
    lines += table_lines(["writer", "load", "hops", "contrast [interval]", "pattern"], rows)
    for writer in ivd.WRITERS:
        lines += ["", f"## {writer}: M1 depth gain (L4 - L1) by cell, hops {','.join(map(str, spec['hops']))}", ""]
        rows = [[r["condition"], percent(r["gain"]), f"[{percent(r['gain_ci_low'])}, {percent(r['gain_ci_high'])}]",
                 f"{100 * r['accuracy_l1']:.1f}", f"{100 * r['accuracy_l4']:.1f}",
                 "" if r["overlap_mean"] is None else f"{r['overlap_mean']:.3f}", f"{r['read_residual_l1']:.3f}"]
                for r in result["cells"] if r["writer"] == writer and r["hop_group"] == "reach"]
        lines += table_lines(["cell", "gain", "95% interval", "L1 acc", "L4 acc", "mean max |cos|",
                              "stored residual"], rows)
        lines += ["", f"B4 (single read) accuracy by cell, hops {','.join(map(str, spec['hops']))}:", ""]
        rows = [[r["condition"], f"{100 * r['accuracy']:.1f}", f"[{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"]
                for r in result["accuracy"] if r["writer"] == writer and r["hop_group"] == "reach"]
        lines += table_lines(["cell", "accuracy", "95% interval"], rows)
    lines += ["", f"Run: {manifest['total_seconds'] / 3600.0:.3f} node-h; design lock {manifest['design_lock_commit'][:12]}."]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="the completed run directory")
    parser.add_argument("--out", required=True, help="output directory (absolute path)")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    run_ivd.require_compute()
    require(args.resamples >= 2000, "at least 2000 resamples are registered")
    started = time.perf_counter()
    manifest, samples = audit_run(Path(args.run))
    design = manifest["design"]
    result = analyse(samples, design, args.resamples)
    out = Path(args.out)
    write_csv(out / "interference.csv", result["interference"])
    write_csv(out / "cells_m1.csv", result["cells"])
    write_csv(out / "accuracy_b4.csv", result["accuracy"])
    write_csv(out / "cells_m1_full_vocabulary.csv", result["full_cells"])
    write_csv(out / "accuracy_b4_full_vocabulary.csv", result["full_accuracy"])
    (out / "summary.md").write_text("\n".join(render(result, design, manifest)) + "\n")
    write_json(out / "summary.json", dict(
        created=now(), seconds=time.perf_counter() - started, resamples=args.resamples,
        design_hash=manifest["design_hash"], design_lock_commit=manifest["design_lock_commit"],
        decisions=result["decisions"], overlap_gaps=result["gaps"], run_node_hours=manifest["node_hours"],
        bootstrap=dict(design["decision"]), min_overlap_gap=MIN_OVERLAP_GAP, statistics=result["interference"]))
    print(json.dumps(dict(stage="analysis", decisions=result["decisions"], out=str(out))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
