#!/usr/bin/env python3
"""Audit the B8W confirmation artifacts and produce the registered decision (docs/prereg/B8W.md).

Reuses P1-L's paired-bootstrap endpoint (finalize_p1l_main.endpoint, pooled_compare_runs); the audit, the window (test
[108000, 110000)), the decision line and the descriptive tables are B8W's. Decision: untied@4 - shared@4 on hops 2 and 3,
95% CI, non-inferiority margin 2 pp (looplora.llm.b8w.keeps_solution).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_p4s_stopping import require_compute
from looplora.llm.b8w import decision_text, keeps_solution, NONINFERIORITY
from looplora.llm.config import CONFIRMATION_WINDOWS
from looplora.llm.evaluate import aggregate_samples
from looplora.sym.stats import load_samples

from evaluate_b8w import GROUPS, WINDOW, check_training
from finalize_p1l_main import endpoint, write_csv
from run_p1l_main import ROOT, SEEDS, digest, save

# (name, left, depth, right, depth, hops): the registered decision contrast first, then the reported ones
CONTRASTS = (
    ("DELTA_untied_vs_shared", "untied", 4, "shared", 4, (2, 3)),
    ("hop1_untied_vs_shared", "untied", 4, "shared", 4, (1,)),
    ("untied_vs_start", "untied", 4, "start", 4, (2, 3)),
    ("untied_vs_start_hop1", "untied", 4, "start", 4, (1,)),
    ("shared_vs_start", "shared", 4, "start", 4, (2, 3)),
    ("shared_vs_start_hop1", "shared", 4, "start", 4, (1,)),
    ("untied_vs_B8_original", "untied", 4, "B8", 4, (2, 3)),
    ("untied_vs_B4", "untied", 4, "B4", 1, (2, 3)),
    ("shared_vs_B4", "shared", 4, "B4", 1, (2, 3)),
    ("start_vs_B4", "start", 4, "B4", 1, (2, 3)),
)


def drift(manifest: dict) -> list[dict]:
    """Per seed and tensor: relative distance of each continued copy from the starting tensor, and the mean pairwise cosine
    distance between the four untied copies. Read from checkpoints; no test data."""
    rows = []
    for seed in SEEDS:
        loaded = lambda path: torch.load(path, map_location="cpu", weights_only=True)["adapter"]
        start = loaded(Path(manifest["runs"][f"start-s{seed}"]) / "checkpoint.pt")
        shared = loaded(Path(manifest["runs"][f"shared-s{seed}"]) / "checkpoint.pt")
        untied = loaded(Path(manifest["runs"][f"untied-s{seed}"]) / "checkpoint.pt")
        for key in ("R", "U", "Q"):
            base = start[key].double()
            relative = lambda tensor: float((tensor.double() - base).norm() / base.norm())
            copies = [untied[f"{key}.{step}"].double().flatten() for step in range(4)]
            cosine = [float(1 - torch.dot(copies[a], copies[b]) / (copies[a].norm() * copies[b].norm()))
                      for a in range(4) for b in range(a + 1, 4)]
            rows.append(dict(seed=seed, tensor=key, shared_relative_change=relative(shared[key]),
                             **{f"untied_copy{step}_relative_change": relative(untied[f"{key}.{step}"]) for step in range(4)},
                             untied_mean_pairwise_cosine_distance=sum(cosine) / len(cosine)))
    return rows


def train_loss(manifest: dict) -> list[dict]:
    rows = []
    for arm in ("shared", "untied"):
        for seed in SEEDS:
            for line in (Path(manifest["runs"][f"{arm}-s{seed}"]) / "train_log.jsonl").read_text().splitlines():
                row = json.loads(line)
                rows.append(dict(arm=arm, seed=seed, step=row["step"], loss=row["loss"], accuracy=row["accuracy"]))
    return rows


def finalize(args):
    manifest = json.loads(args.manifest.read_text())
    evaluations = json.loads(args.evaluations.read_text())["runs"]
    authorization = json.loads(args.authorization.read_text())
    check_training(manifest, authorization)
    start = CONFIRMATION_WINDOWS[WINDOW]
    if set(evaluations) != set(manifest["runs"]):
        raise ValueError("the evaluation manifest does not match the 15 registered runs")
    args.out.mkdir(parents=True, exist_ok=True)
    primary = {label: {} for label in GROUPS}
    counts, artifacts, reference_memory = [], [], None
    for name, directory in evaluations.items():
        out, train = Path(directory), Path(manifest["runs"][name])
        record = json.loads((out / "record.json").read_text())
        if (record["status"] != "complete" or record["checkpoint_hash"] != digest(train / "checkpoint.pt")
                or record["authorization_hash"] != digest(args.authorization)
                or record["manifest_hash"] != digest(args.manifest)):
            raise ValueError(f"invalid confirmation provenance: {name}")
        label, seed = name.rsplit("-s", 1)
        seed = int(seed)
        samples = load_samples(out / "primary" / "samples.npz")
        metrics = json.loads((out / "primary" / "metrics.json").read_text())
        cfg = json.loads((out / "primary" / "config.json").read_text())
        if (set(samples["split"]) != {"confirm"} or not cfg["eval"]["allow_confirmation"] or cfg["eval"]["episodes"] != 2000
                or cfg["eval"]["seed"] != 4242 or cfg["eval"].get("confirmation") != WINDOW
                or metrics["episode_ranges"] != {"confirm": dict(split="test", start=start, count=2000)}
                or not metrics["gate0_passed"]):
            raise ValueError(f"confirmation range/protocol mismatch: {name}")
        if not np.array_equal(samples["correct"], samples["prediction"] == samples["answer"]):
            raise ValueError("full-vocabulary correctness labels were changed")
        if metrics["table"] != aggregate_samples(samples):
            raise ValueError("metrics cannot be reproduced from sample arrays")
        primary[label][seed] = samples
        for row in metrics["table"]:
            keep = ((samples["condition"] == row["condition"]) & (samples["hop"] == row["hop"]) & (samples["L"] == row["L"]))
            if int(keep.sum()) != 2000:
                raise ValueError("each condition/depth/hop requires 2000 paired queries")
            counts.append(dict(reader=label, seed=seed, protocol=row["protocol"], hop=row["hop"], L=row["L"], n=row["n"],
                               correct=int(samples["correct"][keep].sum()), accuracy=row["accuracy"]))
        snapshot = {(item["split"], item["protocol"], item["condition"], item["batch_start"]): (item["memory"], item["write_count"])
                    for item in metrics["memory_hashes"]}
        if reference_memory is None:
            reference_memory = snapshot
        elif reference_memory != snapshot:
            raise ValueError("oracle support snapshots/write counts differ across runs")
        artifacts.append(dict(name=name, run=str(train), evaluation=str(out), checkpoint_hash=digest(train / "checkpoint.pt"),
                              evaluation_record_hash=digest(out / "record.json"),
                              samples_hash=digest(out / "primary" / "samples.npz"),
                              metrics_hash=digest(out / "primary" / "metrics.json")))
    result = {"decision_rule": f"untied@4 - shared@4, hops 2 and 3: keeps iff every seed >= -{100 * NONINFERIORITY:g} pp and the "
                               f"pooled 95% CI lower bound > -{100 * NONINFERIORITY:g} pp", "contrasts": {}, "contrasts_975": {}}
    for confidence, key in ((.95, "contrasts"), (.975, "contrasts_975")):
        for name, left, depth_left, right, depth_right, hops in CONTRASTS:
            result[key][name] = endpoint(primary, left, right, depth_left=depth_left, depth_right=depth_right, hops=hops,
                                         confidence=confidence)
    verdict = keeps_solution(result["contrasts"]["DELTA_untied_vs_shared"])
    result.update(study="b8w", confirmation_window=WINDOW, confidence=.95, keeps=verdict, decision=decision_text(verdict),
                  protocol_check_shared_below_start_by_more_than_2pp=bool(result["contrasts"]["shared_vs_start"]["difference"] < -NONINFERIORITY),
                  authorization=authorization)
    save(args.out / "endpoints.json", result)
    write_csv(args.out / "accuracy.csv", counts)
    write_csv(args.out / "drift.csv", drift(manifest))
    write_csv(args.out / "train-loss.csv", train_loss(manifest))
    save(args.out / "run-manifest.json", dict(status="audited", window=WINDOW, steps=manifest["steps"], lr=manifest["lr"],
        lock=manifest["lock"], runs=artifacts))
    (args.out / "results.md").write_text("\n".join(report(result, start)) + "\n")
    print(json.dumps(dict(status="complete", keeps=verdict["keeps"], decision=result["decision"].split(":")[0]), allow_nan=False), flush=True)


def line(name, row):
    seeds = ", ".join(f"{r['seed']}:{100 * r['difference']:.2f}" for r in row["per_seed"])
    return (f"| {name} | {row['comparison']} | {100 * row['difference']:.2f} | [{100 * row['ci_low']:.2f},{100 * row['ci_high']:.2f}] "
            f"| {seeds} |")


def report(result, start):
    delta = result["keeps"]
    lines = ["# B8W: B8 warm start from M1-L4 (oracle_random, r = 128, Qwen3-0.6B-Base)", "",
             f"Test [{start}, {start + 2000}), eval seed 4242, 2000 episodes, L = 4, seeds 17/29/43; 95% paired episode-bootstrap "
             "CIs, 2000 draws (97.5% in endpoints.json); N = 2000 continuation steps.", "",
             f"**Decision (docs/prereg/B8W.md): {result['decision']}**", "",
             f"Delta = untied@4 - shared@4 on hops 2 and 3: {100 * delta['difference']:.2f} pp "
             f"[{100 * delta['ci_low']:.2f}, {100 * delta['ci_high']:.2f}]; every seed within 2 pp: "
             f"{'yes' if delta['every_seed_within_2pp'] else 'no'}; interval excludes a loss worse than 2 pp: "
             f"{'yes' if delta['interval_excludes_worse_than_2pp'] else 'no'}.", ""]
    if result["protocol_check_shared_below_start_by_more_than_2pp"]:
        lines += ["Protocol check: the shared arm is more than 2 pp below the starting M1-L4 on hops 2 and 3; the continuation "
                  "recipe itself perturbs the solution and Delta is read with that caveat (the decision rule is unchanged).", ""]
    lines += ["| Contrast | Comparison | Difference(pp) | 95% CI(pp) | Per-seed differences(pp) |", "| --- | --- | ---: | --- | --- |"]
    lines.extend(line(name, row) for name, row in result["contrasts"].items())
    lines += ["", "Descriptive: accuracy by hop, `accuracy.csv`; train loss, `train-loss.csv`; drift of the continued tensors, `drift.csv`."]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    finalize(args)


if __name__ == "__main__":
    main()
