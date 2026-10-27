#!/usr/bin/env python3
"""Audit the P1L-C evaluation artifacts and produce the registered results (docs/prereg/P1L-C.md)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_p4s_stopping import require_compute
from looplora.infra.seeds import derive_seed
from looplora.llm.p1lc import PATCH_CONDITIONS, PATCH_EPISODES, READ_HOPS, REGISTERED_RANGE
from looplora.sym.stats import load_samples, pooled_compare_runs

from evaluate_p1l_causal import PATCH_READERS, READERS, reader_depth, require_memory_use
from evaluate_p1l_main import check_training
from finalize_p1l_main import write_csv
from run_p1l_main import SEEDS, digest, save

RESAMPLES, BOOTSTRAP_SEED = 2000, 2718
SHIFT_THRESHOLD = .10
PAIRED_LABEL = "memory_only/p1lc"      # common condition label for the paired read-position contrast
POOLED_HOPS = f"hops{min(READ_HOPS)}-{max(READ_HOPS)}"
BRIDGE_KEYS = ("episode_id", "episode_index", "start", "target_a", "target_b", "bridge_b", "unrelated",
               "token_t_A", "token_t_B", "conditions")


def _condition(samples, condition, label=None):
    keep = samples["condition"] == condition
    chosen = {key: value[keep] for key, value in samples.items()}
    if label is not None:
        chosen["condition"] = np.full(int(keep.sum()), label)
    return chosen


def read_position(reader, samples, *, confidence=.95):
    """Analysis A: full EM with memory only at the final question position minus deployed reads.

    samples maps training seed to the reader's flat read-position samples. Descriptive; hops 1-3.
    """
    depth = reader_depth(reader)
    final = {seed: _condition(s, "memory_only/final_only", PAIRED_LABEL) for seed, s in samples.items()}
    deployed = {seed: _condition(s, "memory_only/deployed", PAIRED_LABEL) for seed, s in samples.items()}
    selections = {f"hop{hop}": [hop] for hop in READ_HOPS}
    selections[POOLED_HOPS] = list(READ_HOPS)
    result = {}
    for name, hops in selections.items():
        row = pooled_compare_runs(final, deployed, depth, depth, hops=hops, split="p1lc",
                                  condition=PAIRED_LABEL, resamples=RESAMPLES, seed=BOOTSTRAP_SEED,
                                  confidence=confidence)
        row["comparison"] = f"{reader}@{depth}: final-position memory - deployed reads"
        result[name] = row
    return result


def replication(samples, confidence):
    """Descriptive: M1-L4@4 and B8@4 minus B4@1 on hops 2 and 3 of this range (deployed reads)."""
    result = {}
    for reader in ("M1-L4", "B8"):
        row = pooled_compare_runs(samples[reader], samples["B4"], reader_depth(reader), reader_depth("B4"),
                                  hops=[2, 3], split="p1lc", condition="memory_only/deployed",
                                  resamples=RESAMPLES, seed=BOOTSTRAP_SEED, confidence=confidence)
        row["comparison"] = f"{reader}@{reader_depth(reader)} - B4@{reader_depth('B4')}"
        result[f"{reader}_vs_B4"] = row
    return result


def bridge_summary(reader, patches, *, confidence=.95, resamples=RESAMPLES, seed=BOOTSTRAP_SEED):
    """Analysis B: rates of answering t_B or t_A under each patch, and the registered shift.

    patches maps training seed to the reader's bridge-patch arrays. shift = rate(t_B | B-patch) -
    max(rate(t_B | C-patch), rate(t_B | random patch)) per seed; the pooled shift is the mean over
    seeds, its CI a percentile bootstrap over episodes shared by all seeds and conditions (the max
    is recomputed in every draw). The primary decision uses the final-position patch; the
    all-position patch is secondary.
    """
    seeds = sorted(patches)
    first = patches[seeds[0]]
    conditions = [str(name) for name in first["conditions"]]
    if conditions != list(PATCH_CONDITIONS):
        raise ValueError("bridge-patch conditions differ from the registered list")
    for training_seed in seeds[1:]:
        for key in BRIDGE_KEYS:
            if not np.array_equal(patches[training_seed][key], first[key]):
                raise ValueError(f"bridge-patch episodes or worlds differ between seeds ({key})")
    n = len(first["episode_id"])
    index = {name: i for i, name in enumerate(conditions)}
    hit = {target: {s: patches[s]["prediction"] == first[f"token_t_{target}"][None, :] for s in seeds}
           for target in ("B", "A")}

    def shift(hits, where, draws=None):
        values = hits.mean(axis=-1) if draws is None else hits[:, draws].mean(axis=-1)
        return values[index[f"B_{where}"]] - np.maximum(values[index[f"C_{where}"]], values[index[f"R_{where}"]])

    alpha = (1 - confidence) / 2
    bootstrap_seed = derive_seed(seed, "p1lc", "bridge-shift", reader, seeds)
    draws = np.random.default_rng(bootstrap_seed).integers(0, n, size=(resamples, n))
    result = dict(reader=reader, n_episodes=n, training_seeds=seeds, confidence=confidence,
                  resamples=resamples, bootstrap_seed=bootstrap_seed, threshold=SHIFT_THRESHOLD)
    for where in ("final", "all"):
        per_seed = []
        for s in seeds:
            rates = hit["B"][s].mean(axis=-1)
            per_seed.append(dict(seed=s, shift=float(shift(hit["B"][s], where)),
                                 rate_B=float(rates[index[f"B_{where}"]]), rate_C=float(rates[index[f"C_{where}"]]),
                                 rate_R=float(rates[index[f"R_{where}"]])))
        boot = np.mean([shift(hit["B"][s], where, draws) for s in seeds], axis=0)
        low, high = np.quantile(boot, [alpha, 1 - alpha])
        pooled = float(np.mean([row["shift"] for row in per_seed]))
        result[where] = dict(shift=pooled, ci_low=float(low), ci_high=float(high), per_seed=per_seed)
    primary = result["final"]
    primary["claim"] = bool(primary["shift"] >= SHIFT_THRESHOLD and primary["ci_low"] > 0
                            and all(row["shift"] > 0 for row in primary["per_seed"]))
    result["rates"] = [dict(reader=reader, seed=s, condition=name, n=n,
                            rate_t_B=float(hit["B"][s][i].mean()), rate_t_A=float(hit["A"][s][i].mean()))
                       for s in seeds for name, i in index.items()]
    result["rates"].extend(dict(reader=reader, seed="pooled", condition=name, n=n,
                                rate_t_B=float(np.mean([hit["B"][s][i].mean() for s in seeds])),
                                rate_t_A=float(np.mean([hit["A"][s][i].mean() for s in seeds])))
                           for name, i in index.items())
    result["state_norms"] = [dict(reader=reader, seed=s, **{
        name: float(patches[s][name].mean()) for name in ("norm_a", "norm_b", "norm_c")}) for s in seeds]
    return result


def analyze(read_samples, bridge_patches):
    """read_samples[reader][seed] and bridge_patches[reader][seed] -> the registered results."""
    return dict(
        read_position={reader: read_position(reader, read_samples[reader]) for reader in READERS},
        replication={f"{confidence*100:g}": replication(read_samples, confidence) for confidence in (.95, .975)},
        bridge={reader: bridge_summary(reader, bridge_patches[reader]) for reader in PATCH_READERS})


def _load_read_position(path, depth):
    samples = load_samples(path)
    if (set(samples["split"]) != {"p1lc"} or set(samples["L"].tolist()) != {depth}
            or set(samples["condition"]) != {"memory_only/deployed", "memory_only/final_only"}):
        raise ValueError(f"read-position samples have unexpected labels: {path}")
    if not np.array_equal(samples["correct"], samples["prediction"] == samples["answer"]):
        raise ValueError(f"full-vocabulary correctness labels were changed: {path}")
    for condition in ("memory_only/deployed", "memory_only/final_only"):
        for hop in range(1, 9):
            if int(((samples["condition"] == condition) & (samples["hop"] == hop)).sum()) != REGISTERED_RANGE["count"]:
                raise ValueError(f"each condition and hop needs {REGISTERED_RANGE['count']} queries: {path}")
    return samples


def _load_bridge_patch(path):
    with np.load(path, allow_pickle=False) as archive:
        patch = {key: archive[key] for key in archive.files}
    if (patch["prediction"].shape != (len(PATCH_CONDITIONS), PATCH_EPISODES)
            or [str(name) for name in patch["conditions"]] != list(PATCH_CONDITIONS)):
        raise ValueError(f"unexpected bridge-patch layout: {path}")
    return patch


def audit(root, manifest, paths):
    """Check every run's provenance and load its samples; returns (read_samples, patches, artifacts)."""
    read_samples = {reader: {} for reader in READERS}
    patches = {reader: {} for reader in PATCH_READERS}
    artifacts, reference = [], {}
    for reader in READERS:
        for seed in SEEDS:
            name = f"{reader}-s{seed}"
            out, train = Path(root) / name, Path(manifest["runs"][name])
            record = json.loads((out / "record.json").read_text())
            expected = dict(status="complete", name=name, smoke=False, label=reader, depth=reader_depth(reader),
                            window=REGISTERED_RANGE, checkpoint_hash=digest(train / "checkpoint.pt"),
                            manifest_hash=digest(paths["manifest"]), authorization_hash=digest(paths["authorization"]),
                            endpoints_hash=digest(paths["endpoints"]),
                            read_position_hash=digest(out / "read-position.npz"))
            if reader in PATCH_READERS:
                expected.update(bridge_patch_hash=digest(out / "bridge-patch.npz"), patch_episodes=PATCH_EPISODES)
            wrong = sorted(key for key, value in expected.items() if record.get(key) != value)
            if wrong:
                raise ValueError(f"invalid P1L-C provenance for {name}: {wrong}")
            shared = ["spec_hash", "episodes_hash", "read_position_memory_hashes"]
            shared += ["worlds_hash", "bridge_memory_hashes"] if reader in PATCH_READERS else []
            for key in shared:
                if reference.setdefault(key, record[key]) != record[key]:
                    raise ValueError(f"{key} differs across P1L-C runs ({name})")
            read_samples[reader][seed] = _load_read_position(out / "read-position.npz", reader_depth(reader))
            if reader in PATCH_READERS:
                patches[reader][seed] = _load_bridge_patch(out / "bridge-patch.npz")
            artifacts.append(dict(name=name, evaluation=str(out), checkpoint_hash=record["checkpoint_hash"],
                                  record_hash=digest(out / "record.json"),
                                  read_position_hash=record["read_position_hash"],
                                  bridge_patch_hash=record.get("bridge_patch_hash"), seconds=record["seconds"],
                                  git_commit=record["git_commit"], source_hash=record["source_hash"],
                                  job_id=record["job_id"]))
    return read_samples, patches, artifacts


def _pp(value):
    return f"{100*value:.2f}"


def tables(result):
    """Flat rows for the CSV files."""
    read_rows = []
    for reader, hops in result["read_position"].items():
        for name, row in hops.items():
            base = dict(reader=reader, depth=row["depth_a"], hops=name)
            read_rows.append(dict(base, seed="pooled", deployed=row["accuracy_b"], final_position_only=row["accuracy_a"],
                                  difference=row["difference"], ci_low=row["ci_low"], ci_high=row["ci_high"],
                                  confidence=row["confidence"]))
            read_rows.extend(dict(base, seed=item["seed"], deployed=item["accuracy_b"],
                                  final_position_only=item["accuracy_a"], difference=item["difference"])
                             for item in row["per_seed"])
    replication_rows = []
    for level, contrasts in result["replication"].items():
        for name, row in contrasts.items():
            replication_rows.append(dict(contrast=name, confidence=level, seed="pooled", difference=row["difference"],
                                         ci_low=row["ci_low"], ci_high=row["ci_high"], accuracy_a=row["accuracy_a"],
                                         accuracy_b=row["accuracy_b"]))
            replication_rows.extend(dict(contrast=name, confidence=level, seed=item["seed"],
                                         difference=item["difference"], accuracy_a=item["accuracy_a"],
                                         accuracy_b=item["accuracy_b"]) for item in row["per_seed"])
    rate_rows = [row for summary in result["bridge"].values() for row in summary["rates"]]
    shift_rows = []
    for reader, summary in result["bridge"].items():
        for where in ("final", "all"):
            row = summary[where]
            shift_rows.append(dict(reader=reader, patch=where, seed="pooled", shift=row["shift"], ci_low=row["ci_low"],
                                   ci_high=row["ci_high"], confidence=summary["confidence"],
                                   claim=row.get("claim", "")))
            shift_rows.extend(dict(reader=reader, patch=where, seed=item["seed"], shift=item["shift"],
                                   rate_B=item["rate_B"], rate_C=item["rate_C"], rate_R=item["rate_R"])
                              for item in row["per_seed"])
    return read_rows, replication_rows, rate_rows, shift_rows


def report_lines(result):
    lines = ["# P1L-C: causal and read-position analysis of the r = 128 oracle readers", "",
             "Test [102000, 104000), eval seed 4242; checkpoints M1-L4, B8, B4 x seeds 17/29/43 (nothing retrained); "
             "2000 bootstrap draws.", "", "## A. Read position (descriptive; gates nothing)", "",
             "Full-vocabulary EM with memory only at the final question position, minus deployed reads; 95% paired "
             "episode-bootstrap CIs. A drop is a distribution shift, not evidence of leakage.", "",
             "| Reader | Hops | Deployed EM(%) | Final-position-only EM(%) | Difference(pp) | CI(pp) | Per-seed(pp) |",
             "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for reader, hops in result["read_position"].items():
        for name, row in hops.items():
            seed_values = ", ".join(f"{r['seed']}:{_pp(r['difference'])}" for r in row["per_seed"])
            lines.append(f"| {reader}@{row['depth_a']} | {name} | {_pp(row['accuracy_b'])} | {_pp(row['accuracy_a'])} | "
                         f"{_pp(row['difference'])} | [{_pp(row['ci_low'])},{_pp(row['ci_high'])}] | {seed_values} |")
    lines += ["", "## Descriptive replication on this range (hops 2 and 3, deployed reads)", "",
              "| Contrast | Difference(pp) | 95% CI(pp) | 97.5% CI(pp) | Per-seed(pp) |", "| --- | ---: | --- | --- | --- |"]
    for name, row in result["replication"]["95"].items():
        wide = result["replication"]["97.5"][name]
        seed_values = ", ".join(f"{r['seed']}:{_pp(r['difference'])}" for r in row["per_seed"])
        lines.append(f"| {row['comparison']} | {_pp(row['difference'])} | [{_pp(row['ci_low'])},{_pp(row['ci_high'])}] | "
                     f"[{_pp(wide['ci_low'])},{_pp(wide['ci_high'])}] | {seed_values} |")
    lines += ["", "## B. Bridge patching (hop-2 questions on the long chain, L = 4)", "",
              "shift = rate(t_B | B-patch) - max(rate(t_B | C-patch), rate(t_B | random patch)); pooled over seeds, "
              "95% paired episode-bootstrap CI. Causal bridge claim: pooled shift >= 10 pp, CI > 0 and a positive shift "
              "in every seed.", "",
              "| Reader | Patch | Shift(pp) | CI(pp) | Per-seed shift(pp) | Claim |", "| --- | --- | ---: | --- | --- | --- |"]
    for reader, summary in result["bridge"].items():
        for where in ("final", "all"):
            row = summary[where]
            seed_values = ", ".join(f"{r['seed']}:{_pp(r['shift'])}" for r in row["per_seed"])
            claim = ("yes" if row["claim"] else "no") if where == "final" else "secondary"
            lines.append(f"| {reader} | {where} position | {_pp(row['shift'])} | "
                         f"[{_pp(row['ci_low'])},{_pp(row['ci_high'])}] | {seed_values} | {claim} |")
    lines += ["", "Rates of answering t_B and t_A (pooled over seeds; world_X = deployed answer on memory X):", "",
              "| Reader | Condition | t_B (%) | t_A (%) |", "| --- | --- | ---: | ---: |"]
    for summary in result["bridge"].values():
        lines.extend(f"| {row['reader']} | {row['condition']} | {_pp(row['rate_t_B'])} | {_pp(row['rate_t_A'])} |"
                     for row in summary["rates"] if row["seed"] == "pooled")
    lines += ["", "Final-position state norms (mean over episodes and seeds). The C-patch is the registered "
              "unrelated-entity control and is not norm-matched to the B-patch; the random patch is.", "",
              "| Reader | norm z1^A | norm (z1^B - z1^A) | norm (z1^C - z1^A) |", "| --- | ---: | ---: | ---: |"]
    for reader, summary in result["bridge"].items():
        mean = {name: np.mean([row[name] for row in summary["state_norms"]]) for name in ("norm_a", "norm_b", "norm_c")}
        lines.append(f"| {reader} | {mean['norm_a']:.3f} | {mean['norm_b']:.3f} | {mean['norm_c']:.3f} |")
    return lines


def finalize(args):
    manifest = json.loads(args.manifest.read_text())
    authorization = json.loads(args.authorization.read_text())
    if manifest.get("rank") != 128:
        raise ValueError("P1L-C analyzes the r = 128 runs")
    require_memory_use(json.loads(args.endpoints.read_text()), authorization)
    check_training(manifest, authorization, check_all=False)
    paths = dict(manifest=args.manifest, authorization=args.authorization, endpoints=args.endpoints)
    read_samples, patches, artifacts = audit(args.root, manifest, paths)
    result = analyze(read_samples, patches)
    result.update(registered_range=REGISTERED_RANGE, patch_episodes=PATCH_EPISODES)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "results.json", result)
    for stem, rows in zip(("read-position", "replication", "bridge-rates", "bridge-shift"), tables(result)):
        write_csv(args.out / f"{stem}.csv", rows)
    save(args.out / "run-manifest.json", dict(status="audited", runs=artifacts,
                                              seconds=sum(row["seconds"] for row in artifacts)))
    (args.out / "results.md").write_text("\n".join(report_lines(result)) + "\n")
    print(json.dumps(dict(status="complete", claim={reader: summary["final"]["claim"]
                                                   for reader, summary in result["bridge"].items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory holding the nine run directories")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    finalize(args)


if __name__ == "__main__":
    main()
