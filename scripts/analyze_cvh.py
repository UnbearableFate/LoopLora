#!/usr/bin/env python3
"""CVH (F9) analysis of a completed confirmation (docs/prereg/CVH.md).

Every lock, seal and structure check runs before any statistic is computed. For each family
and each arm whose validation theta was feasible, the registered pattern is: hops 2-3 at least
3 pp above the single read (L1) with the interval above zero, hop 1 at most 1 pp below L1, and a
mean number of executed updates of at most 8. A latency improvement is claimed only if the arm's
measured end-to-end time beats the fixed 16-update read (primary clock: wall time).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import run_cvh
from s9_common import (
    digest_checkpoint, git_locked_file, interval, now, percent, require, table_lines, verify_git_lock, write_csv,
    write_json,
)
from looplora.infra.seeds import stable_hash
from looplora.sym import cvh

TRACE_KEYS = ("predictions", "losses", "answer", "hop", "episode_id", "p_A0", "p_A2")


def audit_run(root: Path) -> tuple[dict, dict, dict[str, dict[int, dict]]]:
    """Verify the seals and structure; returns (manifest, theta, traces[family][seed])."""
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["status"] == "complete" and manifest["smoke"] is False
            and manifest.get("checkpoints_unchanged") is True, "the run is not a completed main run")
    design = manifest["design"]
    require(manifest["design_hash"] == stable_hash(design), "the manifest design hash differs")
    design_path = Path(manifest["design_path"])
    verify_git_lock(design_path, manifest["design_lock_commit"], "the design")
    require(json.loads(git_locked_file(design_path, manifest["design_lock_commit"])) == design,
            "the manifest design differs from its Git lock")
    theta_path = Path(manifest["theta_path"])
    verify_git_lock(theta_path, manifest["theta_lock_commit"], "theta.json")
    require(digest_checkpoint(theta_path) == manifest["theta_blake2b"], "the theta seal differs")
    theta = json.loads(theta_path.read_text())
    require(theta == json.loads((root / "theta.json").read_text()), "theta.json differs from the training record")
    require(theta["design_hash"] == stable_hash(design) and theta["smoke"] is False and theta["confirmation_allowed"]
            and all(row["reproduced"] for row in theta["a0_reproduction"]), "theta.json is not a valid lock")
    require({row["name"] for row in run_cvh.verify_design(design)} == {row["name"] for row in manifest["readers"]},
            "the trained readers differ from the design")
    active = run_cvh.active_families(theta)
    require(manifest.get("active_families") == active, "the families looked at differ from the theta lock")
    readers = [row for row in manifest["readers"] if row["family"] in active]
    timings = {row["name"]: row for row in json.loads((root / "confirmation_timing.json").read_text())}
    require(set(timings) == {row["name"] for row in readers}, "the run does not cover the registered readers")
    confirmation = design["confirmation"]
    expected = (confirmation["episodes"], len(design["hops"]))
    traces: dict[str, dict[int, dict]] = {}
    reference = None
    for entry in readers:
        require(digest_checkpoint(Path(entry["a2_head_path"])) == entry["a2_head_blake2b"],
                f"A2 head changed: {entry['name']}")
        path = root / "traces" / entry["name"] / "test.npz"
        require(path.is_file() and digest_checkpoint(path) == timings[entry["name"]]["traces_blake2b"],
                f"traces missing or changed: {entry['name']}")
        with np.load(path, allow_pickle=False) as archive:
            trace = {key: np.array(archive[key], copy=True) for key in archive.files}
        require(set(trace) == set(TRACE_KEYS), f"{entry['name']}: unexpected trace arrays")
        require(trace["predictions"].shape == (*expected, cvh.MAX_STEPS + 1)
                and trace["losses"].shape == (*expected, cvh.MAX_STEPS)
                and trace["p_A0"].shape == trace["p_A2"].shape == (*expected, cvh.MAX_STEPS)
                and trace["answer"].shape == trace["hop"].shape == trace["episode_id"].shape == expected,
                f"{entry['name']}: unexpected trace shapes")
        require(np.array_equal(trace["hop"], np.broadcast_to(np.asarray(design["hops"]), expected)),
                f"{entry['name']}: hops are not one query per hop")
        for key in ("p_A0", "p_A2"):
            require(np.all(np.isfinite(trace[key])) and trace[key].min() >= 0 and trace[key].max() <= 1,
                    f"{entry['name']}: invalid halting probabilities")
        labels = (trace["episode_id"], trace["answer"])
        reference = labels if reference is None else reference
        require(all(np.array_equal(a, b) for a, b in zip(labels, reference)),
                f"{entry['name']}: episodes differ from the other readers")
        traces.setdefault(entry["family"], {})[entry["seed"]] = trace
    require(set(traces) == set(active) and all(len(v) == len(design["seeds"]) for v in traces.values()),
            "the traces do not cover every active family and seed")
    return manifest, theta, traces


def latency_summary(root: Path, design: dict) -> list[dict]:
    """Pooled end-to-end time per family and arm against the fixed L16 read of the same batches."""
    path = root / "latency.json"
    rows = json.loads(path.read_text()) if path.is_file() else []
    out = []
    for family in design["families"]:
        fixed = [r for r in rows if r["family"] == family and r["method"] == "fixed_16"]
        for arm in cvh.ARMS:
            learned = [r for r in rows if r["family"] == family and r["method"] == "learned" and r["arm"] == arm]
            if not learned or not fixed:
                out.append(dict(family=family, arm=arm, measured=False, claim=False))
                continue
            seeds = {r["seed"] for r in learned}
            require(len(learned) == len(fixed) == len(seeds), f"{family}/{arm}: incomplete latency rows")
            wall, wall16 = (float(np.mean([r["wall_ms_mean"] for r in group])) for group in (learned, fixed))
            cuda, cuda16 = (float(np.mean([r["cuda_ms_mean"] for r in group])) for group in (learned, fixed))
            out.append(dict(family=family, arm=arm, measured=True, wall_ms=wall, wall_ms_fixed_16=wall16,
                            cuda_ms=cuda, cuda_ms_fixed_16=cuda16,
                            mean_executed=float(np.mean([r["mean_executed_steps"] for r in learned])),
                            claim=bool(wall < wall16)))
    return out


def analyse(traces: dict, theta: dict, design: dict, resamples: int) -> dict:
    spec = design["decision"]
    endpoints, accuracy = [], []
    for family in design["families"]:
        chosen_rows = [row for row in theta["selected"] if row["family"] == family]

        def base_row(chosen: dict) -> dict:
            return dict(family=family, arm=chosen["arm"], feasible=chosen["feasible"], theta=chosen["theta"],
                        validation_accuracy=chosen.get("validation_accuracy"),
                        validation_hop1_delta=chosen.get("hop1_delta"),
                        validation_mean_executed_steps=chosen.get("validation_mean_executed_steps"))

        if family not in traces:  # both new arms failed on validation: the confirmation set is not looked at
            endpoints.extend(base_row(row) for row in chosen_rows if row["arm"] != "A0")
            continue
        by_seed = traces[family]
        methods = {f"L{depth}": cvh.fixed_correct(by_seed, depth) for depth in cvh.FIXED_DEPTHS}
        for chosen in chosen_rows:
            arm = chosen["arm"]
            base = base_row(chosen)
            if not chosen["feasible"]:
                endpoints.append(base)
                continue
            probabilities = {seed: trace["p_A2" if arm == "A2" else "p_A0"] for seed, trace in by_seed.items()}
            steps = {seed: cvh.policy_steps(value, chosen["theta"]) for seed, value in probabilities.items()}
            result = cvh.arm_endpoints(by_seed, steps, resamples=resamples, confidence=spec["confidence"],
                                       label=(family, arm))
            methods[arm] = cvh.arm_correct(by_seed, {seed: value[0] for seed, value in steps.items()})
            endpoints.append(dict(
                base, confidence=spec["confidence"],
                **{f"{name}_{field}": result[name][field] for name in ("reach_gain", "hop1_delta", "reach_vs_l4", "reach_vs_l16")
                   for field in ("difference", "ci_low", "ci_high")},
                mean_selected=result["mean_selected"], mean_executed=result["mean_executed"],
                passes_gain=result["passes_gain"], passes_hop1=result["passes_hop1"],
                passes_depth=result["passes_depth"], passes=result["passes"],
                gating=arm in ("A1", "A2")))
        accuracy.extend(dict(family=family, **row)
                        for row in cvh.accuracy_rows(by_seed, methods, resamples=resamples, label=(family,)))
    return dict(endpoints=endpoints, accuracy=accuracy)


def render(result: dict, latency: list[dict], theta: dict, manifest: dict) -> list[str]:
    design = manifest["design"]
    spec = design["decision"]
    lines = ["# CVH summary", "",
             f"Confirmation episodes: split {design['confirmation']['split']}, seed {design['confirmation']['seed']}, "
             f"{design['confirmation']['episodes']} episodes per reader; three training seeds pooled; percentage "
             f"points; {100 * spec['confidence']:.2f}% intervals. Arm A0 is the existing P4-S-L head at its locked "
             "theta (reference); A1 is that head with the hop-1-constrained theta; A2 is the continuation-value head.",
             "", "## Registered pattern", ""]
    rows = []
    claims = {(row["family"], row["arm"]): row for row in latency}
    for row in result["endpoints"]:
        if not row["feasible"]:
            rows.append([row["family"], row["arm"], "infeasible", "", "", "", "", "no validation theta keeps hop 1"])
            continue
        latency_row = claims.get((row["family"], row["arm"]), {})
        rows.append([row["family"], row["arm"], f"{row['theta']:.2f}",
                     interval(dict(difference=row["reach_gain_difference"], ci_low=row["reach_gain_ci_low"],
                                   ci_high=row["reach_gain_ci_high"])), percent(row["hop1_delta_difference"]),
                     f"{row['mean_executed']:.2f}", "yes" if row["passes"] else "no",
                     "yes" if latency_row.get("claim") else ("no" if latency_row.get("measured") else "not measured")])
    lines += table_lines(["family", "arm", "theta", "hops 2-3 minus L1 [interval]", "hop 1 minus L1",
                          "mean executed updates", "pattern holds", "latency claim"], rows)
    lines += ["", "## Latency (batch 64, ms; learned read against the fixed 16-update read)", ""]
    lines += table_lines(["family", "arm", "wall", "wall L16", "cuda", "cuda L16", "faster than L16"],
                         [[r["family"], r["arm"], f"{r['wall_ms']:.3f}", f"{r['wall_ms_fixed_16']:.3f}",
                           f"{r['cuda_ms']:.3f}", f"{r['cuda_ms_fixed_16']:.3f}", "yes" if r["claim"] else "no"]
                          for r in latency if r["measured"]])
    lines += ["", "## Accuracy by hop group (percent)", ""]
    groups = ("hop1", "reach", "mid", "far", "all")
    methods = ["L1", "L4", "L16", *cvh.ARMS]
    table = {(r["family"], r["method"], r["hop_group"]): r for r in result["accuracy"]}
    rows = []
    for family in design["families"]:
        for method in methods:
            if (family, method, "all") in table:
                rows.append([family, method] + [f"{100 * table[family, method, g]['accuracy']:.1f}"
                                                if (family, method, g) in table else "" for g in groups])
    lines += table_lines(["family", "method", *groups], rows)
    failures = theta["failures"]
    lines += ["", "## Validation lock", "",
              f"Infeasible (family, arm) choices: {', '.join(f['family'] + '/' + f['arm'] for f in failures) or 'none'}.",
              f"Run: {manifest['confirmation_seconds'] / 3600.0:.3f} node-h of confirmation; "
              f"design lock {manifest['design_lock_commit'][:12]}, theta lock {manifest['theta_lock_commit'][:12]}."]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="the completed run directory")
    parser.add_argument("--out", required=True, help="output directory (absolute path)")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    run_cvh.require_compute()
    require(args.resamples >= 2000, "at least 2000 resamples are registered")
    started = time.perf_counter()
    root, out = Path(args.root), Path(args.out)
    manifest, theta, traces = audit_run(root)
    design = manifest["design"]
    result = analyse(traces, theta, design, args.resamples)
    latency = latency_summary(root, design)
    write_csv(out / "endpoints.csv", result["endpoints"])
    write_csv(out / "accuracy.csv", result["accuracy"])
    write_csv(out / "latency.csv", latency)
    write_csv(out / "theta.csv", theta["selected"])
    (out / "summary.md").write_text("\n".join(render(result, latency, theta, manifest)) + "\n")
    write_json(out / "summary.json", dict(
        created=now(), seconds=time.perf_counter() - started, resamples=args.resamples,
        design_hash=manifest["design_hash"], design_lock_commit=manifest["design_lock_commit"],
        theta_lock_commit=manifest["theta_lock_commit"], bootstrap=dict(design["decision"]),
        passing=[f"{r['family']}/{r['arm']}" for r in result["endpoints"] if r.get("passes")],
        latency_claims=[f"{r['family']}/{r['arm']}" for r in latency if r["claim"]],
        failures=theta["failures"], confirmation_node_hours=manifest["confirmation_seconds"] / 3600.0))
    print(json.dumps(dict(stage="analysis", passing=[f"{r['family']}/{r['arm']}" for r in result["endpoints"]
                                                      if r.get("passes")], out=str(out))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
