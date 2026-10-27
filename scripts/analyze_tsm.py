#!/usr/bin/env python3
"""TSM (F10) analysis of a completed confirmation evaluation (docs/prereg/TSM.md).

Every lock, seal and structure check runs before any statistic is computed. For each writer and
each of the arms T1 and T2 the registered conditions are: hops 1-3 at L16 within 1 pp of L4; hops
4-8 at L8 no more than 1 pp below T0 at L8; hops 2-3 at L4 at least 3 pp above the B4 single read
with the interval above zero. An arm passes only if it passes for both writers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import run_tsm
from s9_common import (
    digest_checkpoint, git_locked_file, interval, now, require, table_lines, verify_git_lock, write_csv, write_json,
)
from looplora.infra.seeds import stable_hash
from looplora.sym import followup_stats as fs
from looplora.sym import tsm
from looplora.sym.stats import load_samples

ENDPOINTS = ("stay", "far", "gain")


def audit_run(root: Path) -> tuple[dict, dict[str, dict], dict]:
    """Verify seals and structure; returns (manifest, samples by name, gate profiles)."""
    out = root / "evaluation"
    manifest = json.loads((out / "manifest.json").read_text())
    require(manifest["status"] == "complete" and manifest.get("checkpoints_unchanged") is True,
            "the evaluation is not a completed main run")
    design = manifest["design"]
    require(manifest["design_hash"] == stable_hash(design), "the manifest design hash differs")
    design_path = Path(manifest["design_path"])
    verify_git_lock(design_path, manifest["design_lock_commit"], "the design")
    require(json.loads(git_locked_file(design_path, manifest["design_lock_commit"])) == design,
            "the manifest design differs from its Git lock")
    run_tsm.verify_design(design)
    for row in run_tsm.planned_runs():
        record = json.loads((root / "runs" / row["name"] / "record.json").read_text())
        config = run_tsm.make_config(design, row["writer"], row["seed"], row["arm"], device=record["runtime"]["device"])
        require(record["status"] == "complete" and record["config_hash"] == stable_hash(config)
                and record["steps"] == design["training"]["steps"], f"run {row['name']} is not a registered run")
        require(digest_checkpoint(Path(record["checkpoint"])) == record["checkpoint_blake2b"],
                f"checkpoint changed: {row['name']}")
    evaluation = design["evaluation"]
    expected = {run["name"] for run in run_tsm.planned_runs()} | {
        run_tsm.reference_name(w, s) for w in run_tsm.WRITERS for s in run_tsm.SEEDS}
    rows = {row["name"]: row for row in manifest["evaluations"]}
    require(set(rows) == expected, "the evaluation does not cover the registered runs and references")
    samples, reference = {}, None
    per_query = evaluation["episodes"] * len(evaluation["hops"])
    for name, row in sorted(rows.items()):
        path = out / "samples" / f"{name}.npz"
        require(path.is_file() and digest_checkpoint(path) == row["samples_blake2b"],
                f"samples missing or changed: {name}")
        arrays = load_samples(path)
        depths = evaluation["depths"] if row["arm"] != "B4" else [1]
        require(set(arrays["split"].tolist()) == {evaluation["split"]} and set(arrays["condition"].tolist()) == {"original"},
                f"{name}: wrong split or condition")
        require(sorted({int(v) for v in arrays["L"]}) == sorted(depths), f"{name}: wrong depths")
        require(sorted({int(v) for v in arrays["hop"]}) == evaluation["hops"], f"{name}: wrong hops")
        for depth in depths:
            count = int((arrays["L"] == depth).sum())
            require(count == per_query, f"{name}: L={depth} has {count} rows")
        episodes = sorted({int(v) for v in arrays["episode_id"]})
        reference = episodes if reference is None else reference
        require(episodes == reference and len(episodes) == evaluation["episodes"],
                f"{name}: episodes differ from the other evaluations")
        samples[name] = arrays
    profiles = json.loads((out / "gate_profiles.json").read_text())
    return manifest, samples, profiles


def accuracy_rows(samples: dict, design: dict, resamples: int) -> list[dict]:
    """Descriptive pooled accuracy by writer, method, depth and hop group."""
    rows = []
    for writer in run_tsm.WRITERS:
        methods = [(arm, lambda s, a=arm: run_tsm.run_name(writer, a, s), design["evaluation"]["depths"])
                   for arm in tsm.ARMS]
        methods.append(("B4", lambda s: run_tsm.reference_name(writer, s), [1]))
        for method, namer, depths in methods:
            by_seed = {s: samples[namer(s)] for s in run_tsm.SEEDS}
            for depth in depths:
                aligned = fs.aligned_terms(by_seed, [(design["evaluation"]["split"], "original", depth)])
                for group, hops in {**fs.HOP_GROUPS, "all": tuple(design["evaluation"]["hops"])}.items():
                    keep = np.isin(aligned["hop"], hops)
                    if not keep.any():
                        continue
                    values = {s: term[0][keep] for s, term in aligned["values"].items()}
                    result = fs.pooled_mean_contrast(values, aligned["episode_id"][keep], resamples=resamples,
                                                     confidence=0.95, label=("tsm", "accuracy", writer, method, depth,
                                                                             group))
                    rows.append(dict(writer=writer, method=method, L=depth, hop_group=group,
                                     accuracy=result["difference"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                                     confidence=0.95))
    return rows


def analyse(samples: dict, design: dict, resamples: int) -> dict:
    spec = design["decision"]
    endpoints, descriptive = [], []
    for writer in run_tsm.WRITERS:
        for arm in ("T1", "T2"):
            result = tsm.arm_endpoints(run_tsm.group_samples(samples, writer, arm), arm, writer=writer,
                                       confidence=spec["confidence"], resamples=resamples,
                                       split=design["evaluation"]["split"])
            row = dict(writer=writer, arm=arm, confidence=spec["confidence"], passes=result["passes"])
            for name in ENDPOINTS:
                row.update({f"{name}_{field}": result[name][field] for field in ("difference", "ci_low", "ci_high")},
                           **{f"{name}_passes": result["checks"][name]})
            endpoints.append(row)
            descriptive.extend(dict(writer=writer, arm=arm, contrast=name,
                                    **{field: value[field] for field in ("difference", "ci_low", "ci_high")})
                               for name, value in result["descriptive"].items())
    arms = {arm: all(r["passes"] for r in endpoints if r["arm"] == arm) for arm in ("T1", "T2")}
    return dict(endpoints=endpoints, descriptive=descriptive, arms=arms,
                accuracy=accuracy_rows(samples, design, resamples))


def training_rows(root: Path) -> list[dict]:
    rows = []
    for row in run_tsm.planned_runs():
        record = json.loads((root / "runs" / row["name"] / "record.json").read_text())
        summary = record["summary"]
        rows.append(dict(name=row["name"], writer=row["writer"], arm=row["arm"], seed=row["seed"],
                         steps=summary["steps"], final_loss=summary["final_loss"], mean_L=summary["mean_L"],
                         train_seconds=record["seconds"], step_seconds=summary["train_step_seconds"],
                         trainable_parameters=summary["trainable_parameters"],
                         peak_memory_bytes=record["peak_memory_bytes"]))
    return rows


def gate_rows(profiles: dict) -> list[dict]:
    rows = []
    for name, profile in sorted(profiles.items()):
        for hop, values in zip(profile["hops"], profile["mean_gate"]):
            rows.append(dict(name=name, hop=hop, **{f"step_{step}": value for step, value in zip(profile["steps"], values)}))
    return rows


def render(result: dict, design: dict, manifest: dict) -> list[str]:
    spec = design["decision"]
    lines = ["# TSM summary", "",
             f"Confirmation episodes: split {design['evaluation']['split']}, seed {design['evaluation']['seed']}, "
             f"{design['evaluation']['episodes']} episodes per run; three training seeds pooled; percentage points; "
             f"{100 * spec['confidence']:.2f}% intervals. T0 is the retrained reference, T1 trains at depth 8 with "
             "the stay loss, T2 adds the sticky gate; B4 is the released single-read reference.", "",
             "## Registered conditions", ""]
    rows = []
    for r in result["endpoints"]:
        rows.append([r["writer"], r["arm"],
                     interval(dict(difference=r["stay_difference"], ci_low=r["stay_ci_low"], ci_high=r["stay_ci_high"])),
                     "yes" if r["stay_passes"] else "no",
                     interval(dict(difference=r["far_difference"], ci_low=r["far_ci_low"], ci_high=r["far_ci_high"])),
                     "yes" if r["far_passes"] else "no",
                     interval(dict(difference=r["gain_difference"], ci_low=r["gain_ci_low"], ci_high=r["gain_ci_high"])),
                     "yes" if r["gain_passes"] else "no", "yes" if r["passes"] else "no"])
    lines += table_lines(["writer", "arm", "hops 1-3: L16 - L4", "<= 1 pp", "hops 4-8: arm@8 - T0@8", ">= -1 pp",
                          "hops 2-3: arm@4 - B4@1", ">= 3 pp, interval > 0", "passes"], rows)
    lines += ["", "An arm passes only if it passes for both writers: "
              + ", ".join(f"{arm}: {'yes' if ok else 'no'}" for arm, ok in result["arms"].items()), ""]
    lines += ["## Accuracy by depth (percent, hops 2-3)", ""]
    table = {(r["writer"], r["method"], r["L"], r["hop_group"]): r for r in result["accuracy"]}
    def cell(w, m, d, group):
        return f"{100 * table[w, m, d, group]['accuracy']:.1f}" if (w, m, d, group) in table else ""

    rows = [[w, m, d, cell(w, m, d, "reach"), cell(w, m, d, "far"), cell(w, m, d, "hop1")]
            for (w, m, d, g) in sorted(table) if g == "reach"]
    lines += table_lines(["writer", "method", "L", "hops 2-3", "hops 6-8", "hop 1"], rows)
    lines += ["", f"Evaluation: {manifest['node_hours']:.3f} node-h; design lock {manifest['design_lock_commit'][:12]}."]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="the completed run directory")
    parser.add_argument("--out", required=True, help="output directory (absolute path)")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    run_tsm.require_compute()
    require(args.resamples >= 2000, "at least 2000 resamples are registered")
    started = time.perf_counter()
    root, out = Path(args.root), Path(args.out)
    manifest, samples, profiles = audit_run(root)
    design = manifest["design"]
    result = analyse(samples, design, args.resamples)
    write_csv(out / "endpoints.csv", result["endpoints"])
    write_csv(out / "descriptive_contrasts.csv", result["descriptive"])
    write_csv(out / "accuracy.csv", result["accuracy"])
    write_csv(out / "training.csv", training_rows(root))
    write_csv(out / "gate_profiles.csv", gate_rows(profiles))
    (out / "summary.md").write_text("\n".join(render(result, design, manifest)) + "\n")
    write_json(out / "summary.json", dict(
        created=now(), seconds=time.perf_counter() - started, resamples=args.resamples,
        design_hash=manifest["design_hash"], design_lock_commit=manifest["design_lock_commit"],
        arms_pass=result["arms"], writer_passes={f"{r['writer']}/{r['arm']}": r["passes"] for r in result["endpoints"]},
        bootstrap=dict(design["decision"]), evaluation_node_hours=manifest["node_hours"]))
    print(json.dumps(dict(stage="analysis", arms_pass=result["arms"], out=str(out))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
