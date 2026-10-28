#!/usr/bin/env python3
"""Audit all confirmation artifacts and produce registered P1-L endpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_p4s_stopping import require_compute
from looplora.llm.config import CONFIRMATION_WINDOWS
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


def endpoints(primary, typed, confidence, writer="oracle_random"):
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
    return dict(writer=writer, confidence=confidence, G1_L=bool(g1), G2_L=bool(g2),
                passes=bool(g1 and g2), contrasts=contrasts)


def meets_g1_criterion(row):
    """Amendment B6's rule on one pooled contrast: at least 3 pp, CI above zero, positive in every seed."""
    return bool(row["difference"] >= .03 and row["ci_low"] > 0
                and all(item["difference"] > 0 for item in row["per_seed"]))


# Amendment B7, rule 1: every memory reader must beat the no-memory loop B3-L4@4 at hop 1.
MEMORY_USE_CONTRASTS = (("B4_vs_B3_hop1", "B4", 1), ("B8_vs_B3_hop1", "B8", 4),
                        ("M1_vs_B3_hop1", "M1-L4", 4))


def memory_use_check(primary, confidence):
    """Hop-1 full-vocabulary EM of B4@1, B8@4 and M1-L4@4 minus B3-L4@4 on the confirmation set.

    Pooled paired episode bootstrap (2000 draws) at the gate confidence; passed only if every
    interval lies above zero. It reads the same samples as endpoints() and gates nothing there.
    """
    contrasts = {name: endpoint(primary, reader, "B3-L4", depth_left=depth, depth_right=4,
                                hops=(1,), confidence=confidence)
                 for name, reader, depth in MEMORY_USE_CONTRASTS}
    return dict(rule="B7.1", hop=1, comparator="B3-L4@4", confidence=confidence,
                passed=bool(all(row["ci_low"] > 0 for row in contrasts.values())),
                contrasts=contrasts)


def memory_use_rows(rank, memory):
    rows = []
    for name, row in memory["contrasts"].items():
        base = dict(rank=rank, contrast=name, comparison=row["comparison"], hop=memory["hop"])
        rows.append(dict(base, seed="pooled", reader_accuracy=row["accuracy_a"],
                         comparator_accuracy=row["accuracy_b"], difference=row["difference"],
                         ci_low=row["ci_low"], ci_high=row["ci_high"], confidence=row["confidence"],
                         ci_above_zero=bool(row["ci_low"] > 0)))
        rows.extend(dict(base, seed=item["seed"], reader_accuracy=item["accuracy_a"],
                         comparator_accuracy=item["accuracy_b"], difference=item["difference"])
                    for item in row["per_seed"])
    return rows


def memory_use_lines(rank, memory, study="p1l", gates=None):
    passed = memory["passed"]
    lines = ["", "## Amendment B7: memory-use check", "",
             f"Hop-1 full-vocabulary EM on the confirmation set; each reader minus B3-L4@4 (no memory); "
             f"{memory['confidence']*100:g}% paired episode-bootstrap CIs, 2000 draws.",
             f"Memory-use check: **{'pass' if passed else 'fail'}**.", "",
             "| Contrast | Reader EM(%) | B3-L4@4 EM(%) | Difference(pp) | CI(pp) | Per-seed differences(pp) |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for name, row in memory["contrasts"].items():
        seed_values = ", ".join(f"{r['seed']}:{100*r['difference']:.2f}" for r in row["per_seed"])
        lines.append(f"| {name} | {100*row['accuracy_a']:.2f} | {100*row['accuracy_b']:.2f} | "
                     f"{100*row['difference']:.2f} | [{100*row['ci_low']:.2f},{100*row['ci_high']:.2f}] | "
                     f"{seed_values} |")
    lines.append("")
    if study != "p1l":
        # P1L-S supersedes P1-L's rule 2 at r = 32: its decision text (docs/prereg/P1L-S.md), for the selected code only.
        # `gates` is (G1-L, G2-L): a B7 pass supports a multi-read claim only through them.
        if passed:
            if gates is None:
                raise ValueError("the P1L-S memory-use text needs the (G1-L, G2-L) outcomes")
            g1, g2 = gates
            if g1 and g2:
                reading = ("G1-L and G2-L pass: with this oracle code the multi-read reader composes inside the LM at "
                           "r = 32 and beats the single read of the same rank; the equal-parameter statement follows "
                           "the Amendment-B6 line above.")
            elif not g1:
                reading = "G1-L fails: a negative LLM result at r = 32 for this oracle code."
            else:
                reading = ("G1-L passes and G2-L fails: report which comparator matches B8; no evolving-query claim.")
            lines.append(f"B7 passes, so the outcomes above are read as docs/prereg/P1L-S.md states. {reading} This "
                         "holds for this oracle code only: it is not equivalence with the symbolic oracle and says "
                         "nothing about unseen names.")
        else:
            lines.append("The readers did not learn to use the oracle memory at r = 32 with this code (rule 1): the "
                         "G1-L, G2-L and Amendment-B6 outcomes above are reported but not interpreted as evidence "
                         "about multiple reads, and no multi-read claim at r = 32 follows from this study.")
        return lines
    if not passed:
        lines.append("The readers did not learn to use the oracle memory (rule 1): the G1-L, G2-L and "
                     "Amendment-B6 outcomes above are reported but not interpreted as evidence for or "
                     "against multiple reads.")
    if rank == 32:
        lines.append("Rule 2: no claim about multiple reads in the LM at r = 32, whatever the gate outcomes.")
    elif passed:
        lines.append("Rule 3: a G1-L or Amendment-B6 pass shows that the reader learns the value-to-key "
                     "transition and composes reads inside the LM; it does not establish equivalence with "
                     "the symbolic oracle condition, and no unseen-name claim is made.")
    return lines


def audit_fixed_code(writer, rows):
    """A fixed oracle code (oracle_random, oracle_rot): one writer hash and identical support snapshots per group."""
    hashes = {row["training"]["writer_hash"] for row in rows}
    if len(hashes) != 1 or any(row["snapshot"] != rows[0]["snapshot"] for row in rows):
        raise ValueError(f"{writer} support snapshots or code hashes differ across runs")
    return dict(runs=[row["name"] for row in rows], writer_hash=next(iter(hashes)))


def audit_shared_code(rows):
    """oracle_shared trains its table C per reader, so memories differ by design: the fixed parts must still be one.

    global_codes and rotations (from each checkpoint's writer state, whose file hash finalize verifies) must equal a
    writer rebuilt from oracle_seed in every run; each run's trained table is hashed and recorded, not compared.
    """
    import torch
    from looplora.llm.oracle import OracleSharedWriter
    from looplora.sym.evaluate import tensor_hash

    reference, tables = None, {}
    for row in rows:
        cfg = row["training"]["config"]
        state = torch.load(row["directory"] / "checkpoint.pt", map_location="cpu", weights_only=True)["writer"]
        if reference is None:
            rebuilt = OracleSharedWriter(cfg["task"]["V"], cfg["task"]["n_rel"], cfg["oracle_seed"],
                                         r_mem=cfg["model"]["r"])
            reference = {key: getattr(rebuilt, key) for key in ("global_codes", "rotations")}
        if any(not torch.equal(state[key], value) for key, value in reference.items()):
            raise ValueError(f"oracle_shared global codes or rotations differ from the seeded writer: {row['name']}")
        if row["training"].get("writer_trainable_parameters") != state["C"].numel():
            raise ValueError(f"oracle_shared trained-table size differs from the training record: {row['name']}")
        tables[row["name"]] = tensor_hash(state["C"])
    return dict(runs=list(tables), global_codes_hash=tensor_hash(reference["global_codes"]),
                rotations_hash=tensor_hash(reference["rotations"]), trained_table_hashes=tables,
                writer_trainable_parameters=rows[0]["training"]["writer_trainable_parameters"],
                code_drift={row["name"]: row["training"].get("code_drift") for row in rows})


def audit_writer_groups(audited, manifest):
    """P1L-S: audit the oracle code per writer group; the records go to run-manifest.json.

    The retrained runs share the selected code W and the reused B3-L4 runs use oracle_random, so they are audited as
    two groups (a mixed set that is not exactly the manifest's retrained/reused split fails). Across all runs only the
    write counts are compared: the episode/query keys, hops and answers are compared by pooled_compare_runs, which
    requires them identical on every contrasted pair.
    """
    groups = {}
    for row in audited:
        groups.setdefault(row["training"]["config"]["writer"], []).append(row)
    expected = {manifest["writer"]: sorted(name for name in manifest["runs"] if name not in manifest["reused"]),
                "oracle_random": sorted(manifest["reused"])}
    if {writer: sorted(row["name"] for row in rows) for writer, rows in groups.items()} != expected:
        raise ValueError("training writers do not match the manifest's retrained and reused runs")
    writes = [{key: value[1] for key, value in row["snapshot"].items()} for row in audited]
    if any(item != writes[0] for item in writes):
        raise ValueError("oracle write counts differ across runs")
    return {writer: audit_shared_code(rows) if writer == "oracle_shared" else audit_fixed_code(writer, rows)
            for writer, rows in groups.items()}


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
    study = manifest.get("study", "p1l")
    rank = manifest.get("rank", 32)
    labels = labels_for_rank(rank, study)
    window = authorization.get("confirmation_window")  # absent: P1-L's registered range
    start = CONFIRMATION_WINDOWS[window or "p1l"]
    headline = "oracle_random" if study == "p1l" else manifest["writer"]
    if evaluations.keys() != manifest["runs"].keys():
        raise ValueError("confirmation manifest does not match the complete training matrix")
    args.out.mkdir(parents=True, exist_ok=True)
    primary, typed = {label: {} for label in labels}, {label: {} for label in labels}
    counts, artifact_rows, readouts, faithfulness, swaps, inference = [], [], [], [], [], []
    oracle_hashes, audited, writers = set(), [], {}
    reference_memory = None
    for name, directory in evaluations.items():
        out, train = Path(directory), Path(manifest["runs"][name])
        record = json.loads((out / "record.json").read_text())
        if (record["status"] != "complete" or record["checkpoint_hash"] != digest(train / "checkpoint.pt")
                or record["authorization_hash"] != digest(args.authorization)):
            raise ValueError(f"invalid confirmation provenance: {name}")
        label, seed = name.rsplit("-s", 1)
        seed = int(seed)
        writer = "oracle_random" if label not in {"B0", "B1", "B2"} else "none"
        if study != "p1l":
            writer = writers[label] = json.loads((train / "record.json").read_text())["config"]["writer"]
        measured = json.loads((out / "inference.json").read_text())
        for row in measured["timings"]:
            inference.append(dict(rank=rank, reader=label, seed=seed,
                adapter_parameters=measured["adapter_parameters"],
                trainable_adapter_parameters=measured["trainable_adapter_parameters"],
                **({} if study == "p1l" else {"writer_parameters": measured.get("writer_parameters", 0)}),
                **{k: v for k, v in row.items() if k not in {"batch_seconds", "question_shape"}}))
        artifact_rows.append(dict(name=name, run=str(train), evaluation=str(out),
            config_hash=digest(train / "config.json"), checkpoint_hash=digest(train / "checkpoint.pt"),
            record_hash=digest(train / "record.json"), evaluation_record_hash=digest(out / "record.json")))
        artifact_rows[-1]["inference_hash"] = digest(out / "inference.json")
        for setting in ("primary", "typed"):
            directory = out / setting
            samples = load_samples(directory / "samples.npz")
            metrics = json.loads((directory / "metrics.json").read_text())
            cfg = json.loads((directory / "config.json").read_text())
            if (set(samples["split"]) != {"confirm"} or not cfg["eval"]["allow_confirmation"]
                    or cfg["eval"]["episodes"] != 2000 or cfg["eval"]["seed"] != 4242
                    or cfg["eval"].get("confirmation") != window
                    or metrics["episode_ranges"] != {"confirm": dict(split="test", start=start, count=2000)}
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
                counts.append(dict(writer=writer, rank=rank,
                    reader=label, seed=seed, setting=setting, protocol=row["protocol"], hop=row["hop"],
                    L=row["L"], n=row["n"], correct=int(samples["correct"][keep].sum()),
                    accuracy=row["accuracy"], candidate_correct=int(samples["candidate_correct"][keep].sum()),
                    candidate_accuracy=row["candidate_accuracy"]))
            artifact_rows[-1][f"{setting}_samples_hash"] = digest(directory / "samples.npz")
            artifact_rows[-1][f"{setting}_metrics_hash"] = digest(directory / "metrics.json")
            artifact_rows[-1][f"{setting}_config_hash"] = digest(directory / "config.json")
            if setting == "primary" and label in {"B8", "M1-L4", "B4", "B5-L4", "B3-L4"}:
                training = json.loads((train / "record.json").read_text())
                snapshot = {(item["split"], item["protocol"], item["condition"], item["batch_start"]):
                            (item["memory"], item["write_count"])
                            for item in metrics["memory_hashes"]}
                if study != "p1l":
                    audited.append(dict(name=name, training=training, snapshot=snapshot, directory=train))
                else:
                    oracle_hashes.add(training["writer_hash"])
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
    if study == "p1l" and len(oracle_hashes) != 1:
        raise ValueError("oracle code hash differs across primary runs")
    groups = audit_writer_groups(audited, manifest) if study != "p1l" else None
    confidence = 1 - .05 / authorization["gated_conditions"]
    result = endpoints(primary, typed, confidence, headline)
    result["memory_use"] = memory_use_check(primary, confidence)
    result["memory_use_passed"] = result["memory_use"]["passed"]
    result["rank"] = rank
    result["authorization"] = authorization
    if study != "p1l":
        result.update(study=study, confirmation_window=window, writer_by_reader=writers,
                      B6_equal_parameter=meets_g1_criterion(result["contrasts"]["M1_vs_B4"]))
    save(args.out / "endpoints.json", result)
    write_csv(args.out / "accuracy.csv", counts)
    write_csv(args.out / "per-step-readout.csv", readouts)
    write_csv(args.out / "readout-faithfulness.csv", faithfulness)
    write_csv(args.out / "swap-successors.csv", swaps)
    write_csv(args.out / "inference.csv", inference)
    write_csv(args.out / "memory-use.csv", memory_use_rows(rank, result["memory_use"]))
    audit = (dict(oracle_hash=next(iter(oracle_hashes))) if study == "p1l" else
             dict(study=study, writer=manifest["writer"], reused=manifest["reused"], groups=groups))
    save(args.out / "run-manifest.json", dict(status="audited", rank=rank, condition=manifest["condition"],
        S=manifest["S"], lrs=manifest["lrs"], runs=artifact_rows, **audit))
    plot(args.out, counts, study, headline)
    lines = [f"# P1-L oracle-random r{rank} confirmation" if study == "p1l"
             else f"# P1L-S {headline} r{rank} confirmation", "",
        f"S={manifest['S']}; seeds17/29/43; {confidence*100:g}% paired episode-bootstrap CIs,2000 draws.",
        *(() if study == "p1l" else (f"Confirmation range: test [{start}, {start + 2000}), eval seed 4242, 2000 "
                                     f"episodes; B3-L4 is reused from P1-L (oracle_random).",)),
        f"G1-L: **{'pass' if result['G1_L'] else 'fail'}**. G2-L: **{'pass' if result['G2_L'] else 'fail'}**.", "",
        "| Contrast | Difference(pp) | CI(pp) | Per-seed differences(pp) |",
        "| --- | ---: | --- | --- |"]
    for name, row in result["contrasts"].items():
        seed_values = ", ".join(f"{r['seed']}:{100*r['difference']:.2f}" for r in row["per_seed"])
        lines.append(f"| {name} | {100*row['difference']:.2f} | "
                     f"[{100*row['ci_low']:.2f},{100*row['ci_high']:.2f}] | {seed_values} |")
    if study != "p1l":
        lines.extend(["", "Amendment B6 (equal reader parameters): M1-L4@4 - B4@1 meets the G1-L criterion: "
                      f"**{'yes' if result['B6_equal_parameter'] else 'no'}**."])
    lines.extend(memory_use_lines(rank, result["memory_use"], study, (result["G1_L"], result["G2_L"])))
    (args.out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status="complete", G1=result["G1_L"], G2=result["G2_L"],
                          memory_use=result["memory_use_passed"])), flush=True)


def plot(out, counts, study="p1l", writer="oracle_random"):
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
           xlabel="True query hop", title=f"P1-L r{rank} oracle memory: full-vocabulary EM (%)" if study == "p1l"
           else f"P1L-S r{rank} {writer} memory (B3-L4: oracle_random): full-vocabulary EM (%)")
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
