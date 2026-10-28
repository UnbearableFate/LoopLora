#!/usr/bin/env python3
"""Audit the P5L confirmation artifacts and produce the registered endpoints (docs/prereg/P5L.md).

Reuses P1-L's endpoint, B7 and B6 computations unchanged (finalize_p1l_main); the audit, the window (test [104000,
106000)), the backbone checks, the decision line and the cross-scale table are P5L's. No figure, no readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_p4s_stopping import require_compute
from looplora.llm.config import CONFIRMATION_WINDOWS
from looplora.llm.evaluate import aggregate_samples
from looplora.sym.stats import load_samples

from evaluate_p5l import BACKBONE, CONDITION, WINDOW, check_training
from finalize_p1l_main import (endpoints, meets_g1_criterion, memory_use_check, memory_use_rows,
                               write_csv)
from run_p1l_main import ROOT, digest, save
from run_p5l import LABELS

P1L_ENDPOINTS = "reports/p1l/r128/endpoints.json"  # the 0.6B r = 128 result, for the descriptive cross-scale table
CROSS_SCALE = ("M1_vs_B4", "G1_reach", "G2_fixed_query", "G2_no_memory", "B8_vs_M1", "M1_depth_reach")


def decision(memory_use_passed: bool, b6: bool) -> str:
    """The registered decision use (docs/prereg/P5L.md, "Endpoints and decision rules")."""
    if not memory_use_passed:
        return "B7_failed: the readers did not learn to use the oracle memory at 1.7B (no other endpoint is interpreted)"
    if b6:
        return "holds_at_second_scale: the equal-parameter M1-L4 result holds on Qwen3-1.7B-Base (oracle condition, seen names, trained hops; B7 rule 3 limits)"
    return "LP4_narrowed_to_0.6B: B7 passes and B6 fails"


def finalize(args):
    manifest = json.loads(args.manifest.read_text())
    evaluations = json.loads(args.evaluations.read_text())["runs"]
    authorization = json.loads(args.authorization.read_text())
    check_training(manifest, authorization)
    start = CONFIRMATION_WINDOWS[WINDOW]
    if set(evaluations) != set(manifest["runs"]):
        raise ValueError("confirmation manifest does not match the complete training matrix")
    args.out.mkdir(parents=True, exist_ok=True)
    primary, typed = {label: {} for label in LABELS}, {label: {} for label in LABELS}
    counts, artifact_rows, swaps, inference = [], [], [], []
    oracle_hashes, reference_memory = set(), None
    for name, directory in evaluations.items():
        out, train = Path(directory), Path(manifest["runs"][name])
        record = json.loads((out / "record.json").read_text())
        if (record["status"] != "complete" or record["checkpoint_hash"] != digest(train / "checkpoint.pt")
                or record["authorization_hash"] != digest(args.authorization)):
            raise ValueError(f"invalid confirmation provenance: {name}")
        label, seed = name.rsplit("-s", 1)
        seed = int(seed)
        measured = json.loads((out / "inference.json").read_text())
        for row in measured["timings"]:
            inference.append(dict(rank=128, reader=label, seed=seed,
                adapter_parameters=measured["adapter_parameters"],
                trainable_adapter_parameters=measured["trainable_adapter_parameters"],
                **{k: v for k, v in row.items() if k not in {"batch_seconds", "question_shape"}}))
        artifact_rows.append(dict(name=name, run=str(train), evaluation=str(out),
            config_hash=digest(train / "config.json"), checkpoint_hash=digest(train / "checkpoint.pt"),
            record_hash=digest(train / "record.json"), evaluation_record_hash=digest(out / "record.json"),
            inference_hash=digest(out / "inference.json")))
        for setting in ("primary", "typed"):
            directory = out / setting
            samples = load_samples(directory / "samples.npz")
            metrics = json.loads((directory / "metrics.json").read_text())
            cfg = json.loads((directory / "config.json").read_text())
            if (set(samples["split"]) != {"confirm"} or not cfg["eval"]["allow_confirmation"]
                    or cfg["eval"]["episodes"] != 2000 or cfg["eval"]["seed"] != 4242
                    or cfg["eval"].get("confirmation") != WINDOW or cfg.get("backbone") != BACKBONE
                    or metrics["episode_ranges"] != {"confirm": dict(split="test", start=start, count=2000)}
                    or not metrics["gate0_passed"]):
                raise ValueError(f"confirmation range/protocol/backbone mismatch: {name}/{setting}")
            if not np.array_equal(samples["correct"], samples["prediction"] == samples["answer"]):
                raise ValueError("full-vocabulary correctness labels were changed")
            if metrics["table"] != aggregate_samples(samples):
                raise ValueError("metrics cannot be reproduced from sample arrays")
            if label in LABELS:
                (primary if setting == "primary" else typed)[label][seed] = samples
            for row in metrics["table"]:
                keep = ((samples["condition"] == row["condition"]) & (samples["hop"] == row["hop"])
                        & (samples["L"] == row["L"]))
                if int(keep.sum()) != 2000:
                    raise ValueError("each condition/depth/hop requires 2000 paired queries")
                counts.append(dict(rank=128, reader=label, seed=seed, setting=setting, protocol=row["protocol"],
                    hop=row["hop"], L=row["L"], n=row["n"], correct=int(samples["correct"][keep].sum()),
                    accuracy=row["accuracy"], candidate_correct=int(samples["candidate_correct"][keep].sum()),
                    candidate_accuracy=row["candidate_accuracy"]))
            for key, file in (("samples", "samples.npz"), ("metrics", "metrics.json"), ("config", "config.json")):
                artifact_rows[-1][f"{setting}_{key}_hash"] = digest(directory / file)
            if setting == "primary" and label in LABELS:
                training = json.loads((train / "record.json").read_text())
                snapshot = {(item["split"], item["protocol"], item["condition"], item["batch_start"]):
                            (item["memory"], item["write_count"]) for item in metrics["memory_hashes"]}
                oracle_hashes.add(training["writer_hash"])
                if reference_memory is None:
                    reference_memory = snapshot
                elif reference_memory != snapshot:
                    raise ValueError("oracle support snapshots/write counts differ across readers/seeds")
        if label in {"M1-L4", "B8"}:
            swap = json.loads((out / "swap-successors.json").read_text())
            with np.load(out / "swap-successors-samples.npz", allow_pickle=False) as source:
                samples = {key: source[key] for key in source.files}
            for row in swap:
                keep = (samples["hop"] == row["hop"]) & (samples["L"] == row["L"])
                if (int(keep.sum()) != row["n"] or row["n"] != 2000
                        or float((samples["prediction"][keep] == samples["answer"][keep]).mean()) != row["follows_new"]
                        or float((samples["original_prediction"][keep] == samples["answer"][keep]).mean()) != row["original_predicts_new"]):
                    raise ValueError("swap_successors table cannot be reproduced from samples")
                swaps.append(dict(rank=128, reader=label, seed=seed, **row))
            artifact_rows[-1]["swap_samples_hash"] = digest(out / "swap-successors-samples.npz")
    if len(oracle_hashes) != 1:
        raise ValueError("oracle code hash differs across primary runs")
    confidence = 1 - .05 / authorization["gated_conditions"]
    result = endpoints(primary, typed, confidence, "oracle_random")
    result["memory_use"] = memory_use_check(primary, confidence)
    result["memory_use_passed"] = result["memory_use"]["passed"]
    result["B6_equal_parameter"] = meets_g1_criterion(result["contrasts"]["M1_vs_B4"])
    result["decision"] = decision(result["memory_use_passed"], result["B6_equal_parameter"])
    result.update(study="p5l", rank=128, backbone=BACKBONE, confirmation_window=WINDOW, authorization=authorization)
    save(args.out / "endpoints.json", result)
    write_csv(args.out / "accuracy.csv", counts)
    write_csv(args.out / "swap-successors.csv", swaps)
    write_csv(args.out / "inference.csv", inference)
    write_csv(args.out / "memory-use.csv", memory_use_rows(128, result["memory_use"]))
    save(args.out / "run-manifest.json", dict(status="audited", rank=128, condition=CONDITION, S=manifest["S"],
        lrs=manifest["lrs"], backbone=BACKBONE, oracle_hash=next(iter(oracle_hashes)), runs=artifact_rows))
    (args.out / "results.md").write_text("\n".join(report(result, manifest, start, confidence)) + "\n")
    print(json.dumps(dict(status="complete", B7=result["memory_use_passed"], B6=result["B6_equal_parameter"],
                          G1=result["G1_L"], G2=result["G2_L"], decision=result["decision"].split(":")[0])), flush=True)


def row_line(name, row):
    seeds = ", ".join(f"{r['seed']}:{100 * r['difference']:.2f}" for r in row["per_seed"])
    return (f"| {name} | {100 * row['difference']:.2f} | [{100 * row['ci_low']:.2f},{100 * row['ci_high']:.2f}] | {seeds} |")


def report(result, manifest, start, confidence):
    memory = result["memory_use"]
    lines = ["# P5L oracle_random r128 confirmation on Qwen3-1.7B-Base", "",
             f"S={manifest['S']}; seeds 17/29/43; {confidence * 100:g}% paired episode-bootstrap CIs, 2000 draws; test "
             f"[{start}, {start + 2000}), eval seed 4242, 2000 episodes; B0 evaluated (see accuracy.csv).", "",
             f"**Decision (docs/prereg/P5L.md): {result['decision']}**", "",
             f"B7: **{'pass' if memory['passed'] else 'fail'}**. Amendment B6 (M1-L4@4 - B4@1, hops 2 and 3, >= 3 pp, CI > 0, "
             f"every seed): **{'met' if result['B6_equal_parameter'] else 'not met'}**. "
             f"G1-L: **{'pass' if result['G1_L'] else 'fail'}**. G2-L: **{'pass' if result['G2_L'] else 'fail'}**.", "",
             "| Contrast | Difference(pp) | CI(pp) | Per-seed differences(pp) |", "| --- | ---: | --- | --- |"]
    lines.extend(row_line(name, row) for name, row in result["contrasts"].items())
    lines += ["", "## Amendment B7: memory-use check", "",
              "Hop-1 full-vocabulary EM; each reader minus B3-L4@4 (no memory).", "",
              "| Contrast | Reader EM(%) | B3-L4@4 EM(%) | Difference(pp) | CI(pp) | Per-seed differences(pp) |",
              "| --- | ---: | ---: | ---: | --- | --- |"]
    for name, row in memory["contrasts"].items():
        seeds = ", ".join(f"{r['seed']}:{100 * r['difference']:.2f}" for r in row["per_seed"])
        lines.append(f"| {name} | {100 * row['accuracy_a']:.2f} | {100 * row['accuracy_b']:.2f} | "
                     f"{100 * row['difference']:.2f} | [{100 * row['ci_low']:.2f},{100 * row['ci_high']:.2f}] | {seeds} |")
    lines.append("")
    if not memory["passed"]:
        lines.append("The readers did not learn to use the oracle memory at 1.7B (rule 1): the other outcomes are "
                     "reported, not interpreted.")
    else:
        lines.append("Rule 3: a pass shows that the reader learns the value-to-key transition and composes reads inside "
                     "the LM; it does not establish equivalence with the symbolic oracle condition, and no "
                     "unseen-name claim is made.")
    reference = json.loads((ROOT / P1L_ENDPOINTS).read_text())["contrasts"]
    lines += ["", "## Cross-scale, descriptive (no test): Qwen3-0.6B-Base (P1-L r = 128) and Qwen3-1.7B-Base", "",
              "| Contrast | 0.6B pp [CI] | 1.7B pp [CI] |", "| --- | --- | --- |"]
    for name in CROSS_SCALE:
        old, new = reference[name], result["contrasts"][name]
        lines.append(f"| {name} | {100 * old['difference']:.2f} [{100 * old['ci_low']:.2f},{100 * old['ci_high']:.2f}] | "
                     f"{100 * new['difference']:.2f} [{100 * new['ci_low']:.2f},{100 * new['ci_high']:.2f}] |")
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
