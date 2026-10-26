#!/usr/bin/env python3
"""Audit all confirmation artifacts and produce registered P1-L endpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_p4s_stopping import require_compute
from looplora.llm.evaluate import aggregate_samples
from looplora.sym.stats import load_samples, pooled_compare_runs, pooled_walk_specificity

from evaluate_p1l_main import check_training
from run_p1l_main import LABELS, SEEDS, digest, labels_for_rank, save


def endpoint(samples, left, right, *, depth_left=4, depth_right=4, hops=(2, 3),
             confidence=.95):
    result = pooled_compare_runs(samples[left], samples[right], depth_left, depth_right,
        hops=list(hops), split="confirm", condition="memory_only/original",
        resamples=2000, seed=2718, confidence=confidence)
    result["comparison"] = f"{left}@{depth_left} - {right}@{depth_right}"
    return result


def endpoints(primary, typed, confidence):
    contrasts = {}
    for name, left, right, da, db, hops, dataset in (
        ("G1_reach", "B8", "B4", 4, 1, (2, 3), primary),
        ("G1_hop1", "B8", "B4", 4, 1, (1,), primary),
        ("G1_typed", "B8", "B4", 4, 1, (2, 3), typed),
        ("G2_fixed_query", "B8", "B5-L4", 4, 4, (2, 3), primary),
        ("G2_no_memory", "B8", "B3-L4", 4, 4, (2, 3), primary),
        ("M1_vs_B4", "M1-L4", "B4", 4, 1, (2, 3), primary),
        ("M1_vs_B5", "M1-L4", "B5-L4", 4, 4, (2, 3), primary),
        ("M1_vs_B3", "M1-L4", "B3-L4", 4, 4, (2, 3), primary),
        ("B8_vs_M1", "B8", "M1-L4", 4, 4, (2, 3), primary),
        ("M1_depth_reach", "M1-L4", "M1-L4", 4, 1, (2, 3), primary),
        ("M1_depth_far", "M1-L4", "M1-L4", 4, 1, (6, 7, 8), primary),
    ):
        contrasts[name] = endpoint(dataset, left, right, depth_left=da, depth_right=db,
                                  hops=hops, confidence=confidence)
    contrasts["M1_walk_specificity"] = pooled_walk_specificity(primary["M1-L4"],
        split="confirm", condition="memory_only/original", confidence=confidence,
        resamples=2000, seed=2718)
    reach, hop1, replication = (contrasts[name] for name in ("G1_reach", "G1_hop1", "G1_typed"))
    g1 = (reach["difference"] >= .03 and reach["ci_low"] > 0
          and all(row["difference"] > 0 for row in reach["per_seed"])
          and hop1["difference"] >= -.01 and replication["ci_low"] > 0)
    g2 = all(contrasts[name]["ci_low"] > 0 for name in ("G2_fixed_query", "G2_no_memory"))
    return dict(writer="oracle_random", confidence=confidence, G1_L=bool(g1), G2_L=bool(g2),
                passes=bool(g1 and g2), contrasts=contrasts)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finalize(args):
    manifest = json.loads(args.manifest.read_text())
    evaluations = json.loads(args.evaluations.read_text())["runs"]
    authorization = json.loads(args.authorization.read_text())
    check_training(manifest, authorization)
    rank = manifest.get("rank", 32)
    labels = labels_for_rank(rank)
    if evaluations.keys() != manifest["runs"].keys():
        raise ValueError("confirmation manifest does not match the complete training matrix")
    args.out.mkdir(parents=True, exist_ok=True)
    primary, typed = {label: {} for label in labels}, {label: {} for label in labels}
    counts, artifact_rows, readouts, faithfulness, swaps = [], [], [], [], []
    oracle_hashes = set()
    reference_memory = None
    for name, directory in evaluations.items():
        out, train = Path(directory), Path(manifest["runs"][name])
        record = json.loads((out / "record.json").read_text())
        if (record["status"] != "complete" or record["checkpoint_hash"] != digest(train / "checkpoint.pt")
                or record["authorization_hash"] != digest(args.authorization)):
            raise ValueError(f"invalid confirmation provenance: {name}")
        label, seed = name.rsplit("-s", 1)
        seed = int(seed)
        artifact_rows.append(dict(name=name, run=str(train), evaluation=str(out),
            config_hash=digest(train / "config.json"), checkpoint_hash=digest(train / "checkpoint.pt"),
            record_hash=digest(train / "record.json"), evaluation_record_hash=digest(out / "record.json")))
        for setting in ("primary", "typed"):
            directory = out / setting
            samples = load_samples(directory / "samples.npz")
            metrics = json.loads((directory / "metrics.json").read_text())
            cfg = json.loads((directory / "config.json").read_text())
            if (set(samples["split"]) != {"confirm"} or not cfg["eval"]["allow_confirmation"]
                    or cfg["eval"]["episodes"] != 2000 or cfg["eval"]["seed"] != 4242
                    or metrics["episode_ranges"] != {"confirm": dict(split="test", start=100000, count=2000)}
                    or not metrics["gate0_passed"]):
                raise ValueError(f"confirmation range/protocol mismatch: {name}/{setting}")
            if not np.array_equal(samples["correct"], samples["prediction"] == samples["answer"]):
                raise ValueError("full-vocabulary correctness labels were changed")
            if metrics["table"] != aggregate_samples(samples):
                raise ValueError("metrics cannot be reproduced from sample arrays")
            if label in labels:
                (primary if setting == "primary" else typed)[label][seed] = samples
            for row in metrics["table"]:
                keep = ((samples["condition"] == row["condition"]) & (samples["hop"] == row["hop"])
                        & (samples["L"] == row["L"]))
                if int(keep.sum()) != 2000:
                    raise ValueError("each condition/depth/hop requires2000 paired queries")
                counts.append(dict(writer="oracle_random" if label not in {"B0", "B1", "B2"} else "none", rank=rank,
                    reader=label, seed=seed, setting=setting, protocol=row["protocol"], hop=row["hop"],
                    L=row["L"], n=row["n"], correct=int(samples["correct"][keep].sum()),
                    accuracy=row["accuracy"], candidate_correct=int(samples["candidate_correct"][keep].sum()),
                    candidate_accuracy=row["candidate_accuracy"]))
            artifact_rows[-1][f"{setting}_samples_hash"] = digest(directory / "samples.npz")
            artifact_rows[-1][f"{setting}_metrics_hash"] = digest(directory / "metrics.json")
            artifact_rows[-1][f"{setting}_config_hash"] = digest(directory / "config.json")
            if setting == "primary" and label in {"B8", "M1-L4", "B4", "B5-L4", "B3-L4"}:
                training = json.loads((train / "record.json").read_text())
                oracle_hashes.add(training["writer_hash"])
                snapshot = {(item["split"], item["protocol"], item["condition"], item["batch_start"]):
                            (item["memory"], item["write_count"])
                            for item in metrics["memory_hashes"]}
                if reference_memory is None:
                    reference_memory = snapshot
                elif reference_memory != snapshot:
                    raise ValueError("oracle support snapshots/write counts differ across readers/seeds")
        if label in {"M1-L4", "B8"}:
            data = json.loads((out / "readout.json").read_text())
            readouts.extend(dict(rank=rank, reader=label, seed=seed, **row) for row in data["table"])
            faithfulness.extend(dict(rank=rank, reader=label, seed=seed, **row) for row in data["faithfulness"])
            swap = json.loads((out / "swap-successors.json").read_text())
            with np.load(out / "swap-successors-samples.npz", allow_pickle=False) as source:
                samples = {key: source[key] for key in source.files}
            for row in swap:
                keep = (samples["hop"] == row["hop"]) & (samples["L"] == row["L"])
                if (int(keep.sum()) != row["n"] or row["n"] != 2000
                        or float((samples["prediction"][keep] == samples["answer"][keep]).mean()) != row["follows_new"]
                        or float((samples["original_prediction"][keep] == samples["answer"][keep]).mean()) != row["original_predicts_new"]):
                    raise ValueError("swap_successors table cannot be reproduced from samples")
                swaps.append(dict(rank=rank, reader=label, seed=seed, **row))
            artifact_rows[-1]["swap_samples_hash"] = digest(out / "swap-successors-samples.npz")
            artifact_rows[-1]["readout_hash"] = digest(out / "readout.json")
    if len(oracle_hashes) != 1:
        raise ValueError("oracle code hash differs across primary runs")
    confidence = 1 - .05 / authorization["gated_conditions"]
    result = endpoints(primary, typed, confidence)
    result["rank"] = rank
    result["authorization"] = authorization
    save(args.out / "endpoints.json", result)
    write_csv(args.out / "accuracy.csv", counts)
    write_csv(args.out / "per-step-readout.csv", readouts)
    write_csv(args.out / "readout-faithfulness.csv", faithfulness)
    write_csv(args.out / "swap-successors.csv", swaps)
    save(args.out / "run-manifest.json", dict(status="audited", rank=rank, condition=manifest["condition"],
        S=manifest["S"], lrs=manifest["lrs"],
        oracle_hash=next(iter(oracle_hashes)), runs=artifact_rows))
    plot(args.out, counts)
    lines = [f"# P1-L oracle-random r{rank} confirmation", "",
        f"S={manifest['S']}; seeds17/29/43; {confidence*100:g}% paired episode-bootstrap CIs,2000 draws.",
        f"G1-L: **{'pass' if result['G1_L'] else 'fail'}**. G2-L: **{'pass' if result['G2_L'] else 'fail'}**.", "",
        "| Contrast | Difference(pp) | CI(pp) | Per-seed differences(pp) |",
        "| --- | ---: | --- | --- |"]
    for name, row in result["contrasts"].items():
        seed_values = ", ".join(f"{r['seed']}:{100*r['difference']:.2f}" for r in row["per_seed"])
        lines.append(f"| {name} | {100*row['difference']:.2f} | "
                     f"[{100*row['ci_low']:.2f},{100*row['ci_high']:.2f}] | {seed_values} |")
    (args.out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status="complete", G1=result["G1_L"], G2=result["G2_L"])), flush=True)


def plot(out, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = counts[0]["rank"]
    labels = [label for label in ["B8", "M1-L4", "B4", "B5-L4", "B3-L4", "B0", "B1", "B2"]
              if any(row["reader"] == label for row in counts)]
    values = np.zeros((len(labels), 8))
    for i, label in enumerate(labels):
        depth = 1 if label in {"B0", "B1", "B2", "B4"} else 4
        for hop in range(1, 9):
            selected = [r["accuracy"] for r in counts if r["reader"] == label and r["hop"] == hop
                        and r["L"] == depth and r["protocol"] == "memory_only" and r["setting"] == "primary"]
            values[i, hop-1] = np.mean(selected)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(100*values, vmin=0, vmax=100, cmap="viridis")
    ax.set(xticks=range(8), xticklabels=range(1, 9), yticks=range(len(labels)), yticklabels=labels,
           xlabel="True query hop", title=f"P1-L r{rank} oracle memory: full-vocabulary EM (%)")
    for i in range(len(labels)):
        for j in range(8):
            ax.text(j, i, f"{100*values[i,j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=180)
    fig.savefig(out / "accuracy.pdf")
    plt.close(fig)
    values = np.asarray([[np.mean([r["accuracy"] for r in counts if r["reader"] == "M1-L4"
        and r["L"] == depth and r["hop"] == hop and r["setting"] == "primary"
        and r["protocol"] == "memory_only"]) for hop in range(1, 9)] for depth in range(1, 9)])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(100*values, vmin=0, vmax=100, cmap="viridis", origin="lower")
    ax.set(xticks=range(8), xticklabels=range(1, 9), yticks=range(8), yticklabels=range(1, 9),
           xlabel="True query hop", ylabel="Read depth", title=f"M1-L4 r{rank}: same checkpoint and memory")
    fig.colorbar(image, ax=ax, label="Full-vocabulary EM (%)")
    fig.tight_layout()
    fig.savefig(out / "M1-depth.png", dpi=180)
    fig.savefig(out / "M1-depth.pdf")
    plt.close(fig)


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
