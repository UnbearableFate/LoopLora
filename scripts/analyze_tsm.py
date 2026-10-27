#!/usr/bin/env python3
"""TSM (F10) analysis of a completed confirmation evaluation (docs/prereg/TSM.md).

Every lock, seal and structure check runs before any statistic is computed. For each writer and
each of the arms T1 and T2 the registered conditions are: hops 1-3 at L16 within 1 pp of L4; hops
4-8 at L8 no more than 1 pp below T0 at L8; hops 2-3 at L4 at least 3 pp above the B4 single read
with the interval above zero. An arm passes only if it passes for both writers; an arm that passes
for exactly one writer is a writer-specific result, not a remedy.

A T1 or T2 run that stopped with a non-finite loss is a registered training failure: that arm fails for
that writer, its completed seeds are reported descriptively (they are not registered endpoints) and the
other arms are evaluated as registered. Every T0 run must be complete (a T0 failure stops the study, and so does a
code defect in any run). An evaluation that was restarted from scratch after an interruption is audited too: the
manifest's attempts must match the interrupted evaluations moved aside, its checkpoint seals must be those of the run
records, and it records the code (source_hash) and the orchestrator's decision (the path and blake2b digest of its file,
or none) of every attempt, and the code the runs were trained at. The summary always states both codes and flags a
difference: the first evaluation attempt runs at the training code, so a first attempt at another code is reported as
outside the rule (docs/prereg/TSM.md, Clarifications C1 and C2) unless a decision is recorded for it (Clarification
C4), and a restart at another evaluation code (Clarification C2) is reported with the code of each attempt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence

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
FIELDS = ("difference", "ci_low", "ci_high")
ARMS = ("T1", "T2")
OUTCOME_TEXT = dict(both="passes for both writers", writer_specific="writer-specific result, not a remedy",
                    none="passes for neither writer (a training failure counts as not passing)")


def arm_outcome(passing: Sequence[str]) -> str:
    """The registered reading of an arm from the writers it passes for: both, exactly some (writer-specific), none."""
    if len(passing) == len(run_tsm.WRITERS):
        return "both"
    return "writer_specific" if passing else "none"


def reading(endpoints: list[dict]) -> dict:
    """Per arm: whether it passes for both writers, the writers it passes for, and the registered outcome."""
    passing = {arm: [r["writer"] for r in endpoints if r["arm"] == arm and r["passes"]] for arm in ARMS}
    return dict(arms={arm: all(r["passes"] for r in endpoints if r["arm"] == arm) for arm in ARMS},
                passing_writers=passing, outcome={arm: arm_outcome(writers) for arm, writers in passing.items()})


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
    # T0 complete; T1/T2 complete or a registered training failure; registered configurations, unchanged checkpoints
    # and one code version for all 18 runs
    states = run_tsm.final_states(design, root)
    complete = sorted(name for name, state in states.items() if state == "complete")
    failed = sorted(name for name, state in states.items() if state == "training_failure")
    require(manifest["training_failures"] == [run_tsm.failure_row(root, name) for name in failed],
            "the manifest's training failures differ from the run records")
    require(manifest.get("checkpoints") == run_tsm.checkpoint_seals(root, states),
            "the manifest's checkpoint seals differ from the run records")
    require(manifest.get("training_source_hash") == run_tsm.training_source_hash(root),
            "the manifest's training source_hash differs from the run records")
    restarts = manifest.get("restarts")  # the interrupted evaluations are moved aside, never read
    require(isinstance(restarts, list) and manifest.get("attempt") == len(restarts) + 1 <= run_tsm.MAX_ATTEMPTS
            and [row.get("moved_to") for row in restarts]
            == [f"attempts/{path.name}" for path in run_tsm.moved_evaluations(root)],
            "the manifest's evaluation attempts differ from the interrupted evaluations moved aside under attempts/")
    require(isinstance(manifest.get("source", {}).get("source_hash"), str)
            and all("source_hash" in row for row in restarts),
            "the manifest does not record the code (source_hash) of every evaluation attempt")
    decisions = [row.get("decision", "missing") for row in restarts]  # by attempt: the moved ones, then this one
    decisions.append(manifest.get("decision", "missing"))
    require(all(d is None or (isinstance(d, dict) and set(d) == {"path", "blake2b"}) for d in decisions),
            "the manifest does not record the orchestrator's decision (the path and blake2b digest of its file, or "
            "none) of every evaluation attempt")
    evaluation = design["evaluation"]
    expected = set(complete) | {run_tsm.reference_name(w, s) for w in run_tsm.WRITERS for s in run_tsm.SEEDS}
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
    arm_of = {row["name"]: row["arm"] for row in run_tsm.planned_runs()}
    require(set(profiles) == {name for name in complete if arm_of[name] == "T2"},
            "the gate profiles do not match the completed T2 runs")
    return manifest, samples, profiles


def accuracy_rows(samples: dict, design: dict, resamples: int) -> list[dict]:
    """Descriptive pooled accuracy by writer, method, depth and hop group."""
    rows = []
    for writer in run_tsm.WRITERS:
        methods = [(arm, lambda s, a=arm: run_tsm.run_name(writer, a, s), design["evaluation"]["depths"])
                   for arm in tsm.ARMS]
        methods.append(("B4", lambda s: run_tsm.reference_name(writer, s), [1]))
        for method, namer, depths in methods:
            seeds = [s for s in run_tsm.SEEDS if namer(s) in samples]  # a failed T1/T2 run has no samples
            if not seeds:
                continue
            by_seed = {s: samples[namer(s)] for s in seeds}
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
                                     seeds=";".join(str(s) for s in seeds), accuracy=result["difference"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                                     confidence=0.95))
    return rows


def analyse(samples: dict, design: dict, resamples: int) -> dict:
    """Endpoints per writer and arm; an arm with a training failure has no registered endpoint and fails for the writer."""
    spec = design["decision"]
    endpoints, descriptive = [], []
    for writer in run_tsm.WRITERS:
        for arm in ARMS:
            done = [s for s in run_tsm.SEEDS if run_tsm.run_name(writer, arm, s) in samples]
            failed = [s for s in run_tsm.SEEDS if s not in done]
            result = None
            if done:  # T0 and B4 are restricted to the completed seeds of the arm
                result = tsm.arm_endpoints(run_tsm.group_samples(samples, writer, arm, done), arm, writer=writer,
                                           confidence=spec["confidence"], resamples=resamples,
                                           split=design["evaluation"]["split"])
            registered = result is not None and not failed
            row = dict(writer=writer, arm=arm, confidence=spec["confidence"], training_failed=bool(failed),
                       completed_seeds=";".join(str(s) for s in done), failed_seeds=";".join(str(s) for s in failed))
            for name in ENDPOINTS:
                row.update({f"{name}_{field}": result[name][field] if registered else None for field in FIELDS},
                           **{f"{name}_passes": result["checks"][name] if registered else None})
            row["passes"] = bool(registered and result["passes"])
            endpoints.append(row)
            if result is not None:  # the gating contrasts of a failed arm are descriptive only
                contrasts = dict(result["descriptive"]) if registered else {
                    **{name: result[name] for name in ENDPOINTS}, **result["descriptive"]}
                descriptive.extend(dict(writer=writer, arm=arm, contrast=name, seeds=row["completed_seeds"],
                                        training_failed=bool(failed), **{field: value[field] for field in FIELDS})
                                   for name, value in contrasts.items())
    return dict(endpoints=endpoints, descriptive=descriptive, **reading(endpoints),
                accuracy=accuracy_rows(samples, design, resamples))


def training_rows(root: Path) -> list[dict]:
    rows = []
    for row in run_tsm.planned_runs():
        record = run_tsm.read_record(root, row["name"])
        summary = record.get("summary", {})  # a registered training failure has no summary
        rows.append(dict(name=row["name"], writer=row["writer"], arm=row["arm"], seed=row["seed"],
                         status=record["status"], attempt=record.get("attempt"), failure=record.get("failure"),
                         steps=record["steps"], final_loss=summary.get("final_loss"), mean_L=summary.get("mean_L"),
                         train_seconds=record.get("seconds"), step_seconds=summary.get("train_step_seconds"),
                         trainable_parameters=summary.get("trainable_parameters"),
                         peak_memory_bytes=record.get("peak_memory_bytes")))
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
        if r["training_failed"]:
            continue
        rows.append([r["writer"], r["arm"],
                     interval(dict(difference=r["stay_difference"], ci_low=r["stay_ci_low"], ci_high=r["stay_ci_high"])),
                     "yes" if r["stay_passes"] else "no",
                     interval(dict(difference=r["far_difference"], ci_low=r["far_ci_low"], ci_high=r["far_ci_high"])),
                     "yes" if r["far_passes"] else "no",
                     interval(dict(difference=r["gain_difference"], ci_low=r["gain_ci_low"], ci_high=r["gain_ci_high"])),
                     "yes" if r["gain_passes"] else "no", "yes" if r["passes"] else "no"])
    if rows:
        lines += table_lines(["writer", "arm", "hops 1-3: L16 - L4", "<= 1 pp", "hops 4-8: arm@8 - T0@8", ">= -1 pp",
                              "hops 2-3: arm@4 - B4@1", ">= 3 pp, interval > 0", "passes"], rows)
    else:
        lines += ["No arm was trained completely for any writer: there is no registered endpoint."]
    lines += ["", "An arm passes only if it passes for both writers: "
              + ", ".join(f"{arm}: {'yes' if ok else 'no'}" for arm, ok in result["arms"].items()), "",
              "## Registered reading", ""]
    cells = {(r["writer"], r["arm"]): r for r in result["endpoints"]}

    def status(writer, arm):
        row = cells[writer, arm]
        return "training failure" if row["training_failed"] else "passes" if row["passes"] else "fails"

    lines += table_lines(["arm"] + list(run_tsm.WRITERS) + ["outcome"],
                         [[arm] + [status(w, arm) for w in run_tsm.WRITERS] + [OUTCOME_TEXT[result["outcome"][arm]]]
                          for arm in ARMS])
    lines += ["", "An arm that passes for exactly one writer is a writer-specific result, not a remedy. The statement "
              "that no arm passes is worded per writer; a training failure counts as not passing, and the table above "
              "says which cells failed in training:", ""]
    for writer in run_tsm.WRITERS:
        passing = [arm for arm in ARMS if cells[writer, arm]["passes"]]
        lost = [arm for arm in ARMS if cells[writer, arm]["training_failed"]]
        lines.append(f"- {writer}: " + (f"{' and '.join(passing)} passes" if passing else "no arm passes")
                     + (f"; training failure: {', '.join(lost)}" if lost else ""))
    lines += [""]
    if manifest["training_failures"]:
        lines += ["## Training failures", "",
                  "A non-finite loss in a T1 or T2 run is a registered training failure: no retry, no re-tuning. The "
                  "arm fails for that writer; its completed seeds are reported descriptively below (T0 and B4 "
                  "restricted to the same seeds) and are not registered endpoints; the other arms are evaluated as "
                  "registered.", ""]
        lines += table_lines(["run", "writer", "arm", "seed", "attempt", "failure"],
                             [[f["name"], f["writer"], f["arm"], f["seed"], f["attempt"],
                               str(f["failure"]).replace("|", "/")] for f in manifest["training_failures"]])
        descriptive = {(r["writer"], r["arm"], r["contrast"]): r for r in result["descriptive"]}
        rows = [[r["writer"], r["arm"], r["completed_seeds"]]
                + [interval(descriptive[r["writer"], r["arm"], name]) for name in ENDPOINTS]
                for r in result["endpoints"] if r["training_failed"] and r["completed_seeds"]]
        if rows:
            lines += ["", "Completed seeds of those arms (descriptive):", ""]
            lines += table_lines(["writer", "arm", "completed seeds", "hops 1-3: L16 - L4", "hops 4-8: arm@8 - T0@8",
                                  "hops 2-3: arm@4 - B4@1"], rows)
        lines += [""]
    lines += ["## Accuracy by depth (percent, hops 2-3)", ""]
    table = {(r["writer"], r["method"], r["L"], r["hop_group"]): r for r in result["accuracy"]}

    def cell(w, m, d, group):
        return f"{100 * table[w, m, d, group]['accuracy']:.1f}" if (w, m, d, group) in table else ""

    rows = [[w, m, d, table[w, m, d, g]["seeds"], cell(w, m, d, "reach"), cell(w, m, d, "far"), cell(w, m, d, "hop1")]
            for (w, m, d, g) in sorted(table) if g == "reach"]
    lines += table_lines(["writer", "method", "L", "seeds pooled", "hops 2-3", "hops 6-8", "hop 1"], rows)
    lines += ["", f"Evaluation: {manifest['node_hours']:.3f} node-h (the completed attempt); design lock "
                  f"{manifest['design_lock_commit'][:12]}."]
    trained, final = manifest["training_source_hash"], manifest["source"]["source_hash"]
    lines += ["", f"Code (source_hash): the 18 runs were trained at {trained[:12]}; the evaluation ran at "
                  f"{final[:12]} ({'the same code' if final == trained else 'a DIFFERENT code'})."]
    if manifest["restarts"]:
        lines += ["", f"The evaluation ran as attempt {manifest['attempt']}: the earlier attempt(s) ended before they "
                  "completed and were restarted from scratch with the same design, lock and checkpoints (docs/prereg/"
                  "TSM.md, Clarification C1); their partial output was moved aside unread: "
                  + ", ".join(row["moved_to"] for row in manifest["restarts"]) + "."]
    codes = [row["source_hash"] for row in manifest["restarts"]] + [final]  # by attempt; None: not known
    decisions = [row["decision"] for row in manifest["restarts"]] + [manifest["decision"]]  # None: none recorded
    differing = [number for number, code in enumerate(codes, 1) if code and code != trained]
    by_attempt = ", ".join(f"{number}: {(code or 'unknown')[:12]}" for number, code in enumerate(codes, 1))
    if 1 in differing and decisions[0]:
        lines += ["", "The first evaluation attempt ran at a code other than the training code, on the orchestrator's "
                  "recorded decision (docs/prereg/TSM.md, Clarification C4; decision file "
                  f"{decisions[0]['path']}, blake2b {decisions[0]['blake2b'][:12]}); source_hash by attempt: "
                  f"{by_attempt}; the 18 runs were trained at {trained[:12]}."]
    elif 1 in differing:
        lines += ["", "The first evaluation attempt ran at a code other than the training code and no decision of the "
                  "orchestrator is recorded for it: this is outside the rule that the first attempt runs at the "
                  "training code unless the orchestrator decides otherwise (docs/prereg/TSM.md, Clarifications C1, C2 "
                  f"and C4); source_hash by attempt: {by_attempt}; report it to the orchestrator before the result is "
                  "used."]
    elif differing:
        lines += ["", "The evaluation code changed at a restart, on the orchestrator's decision (docs/prereg/"
                  f"TSM.md, Clarification C2); source_hash by attempt: {by_attempt}; the 18 runs were trained at "
                  f"{trained[:12]}."]
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
        training_source_hash=run_tsm.training_source_hash(root),
        evaluation_source_hash=manifest["source"]["source_hash"], evaluation_decision=manifest["decision"],
        arms_pass=result["arms"], arm_outcome=result["outcome"], passing_writers=result["passing_writers"],
        writer_passes={f"{r['writer']}/{r['arm']}": r["passes"] for r in result["endpoints"]},
        training_failures=manifest["training_failures"], evaluation_attempt=manifest["attempt"],
        evaluation_restarts=manifest["restarts"],
        failed_cells=[f"{r['writer']}/{r['arm']}" for r in result["endpoints"] if r["training_failed"]],
        bootstrap=dict(design["decision"]), evaluation_node_hours=manifest["node_hours"]))
    print(json.dumps(dict(stage="analysis", arms_pass=result["arms"], arm_outcome=result["outcome"],
                          training_failures=[f["name"] for f in manifest["training_failures"]], out=str(out))),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
