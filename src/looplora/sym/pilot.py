"""Validation-only P1-S pilot manifests, locked selection rules, and reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import resolve_config

LRS = (3e-4, 1e-3, 3e-3)
VARIANTS = ("M1", "M1-lin", "B5", "B5-lin", "B3", "B4")
WRITERS = ("oracle_random", "oracle_orth", "t1_frozen", "joint")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def base_config(name: str, writer: str, variant: str, r: int, E: int, lr: float,
                task: str = "t1", chains: list[int] | None = None) -> dict:
    return resolve_config({
        "name": name, "seed": 17, "backbone_seed": 31415, "device": "cuda:0",
        "writer": writer, "variant": variant,
        "model": {"d_e": 64, "r": r, "r_mem": r, "reference_r": r},
        "runtime": {"threads": 1},
        "task": {"name": task, "E": E, "Q": 4, "n_rel": 4,
                 "head_sharing": .3 if task == "t1" else 0.,
                 "hops": [1] if task == "t1" else [1, 2, 3], "chain_lengths": chains},
        "train": {"steps": 5000, "batch_size": 64, "lr": lr, "log_every": 10},
        "pretrain": {"steps": 5000, "batch_size": 64, "lr": lr,
                     "task": {"head_sharing": .3, "n_rel": 4}},
        "eval": {"episodes": 1000, "batch_size": 64, "splits": ["val"],
                 "seed": 4242, "counterfactuals": False},
    })


def pilot1_configs() -> list[dict]:
    configs = []
    for writer in ("joint", "oracle_random", "oracle_orth"):
        for r in (16, 32):
            for E in (8, 12, 16, 24):
                # With positive probability no heads are shared: all 2E entities
                # must fit. Do not condition the training distribution on fit.
                if writer == "oracle_orth" and 2 * E > r:
                    continue
                for lr in LRS:
                    name = f"p1-{writer}-r{r}-E{E}-lr{lr:g}"
                    configs.append(base_config(name, writer, "B4", r, E, lr))
    return configs


def chains_for(E: int) -> list[int]:
    return {12: [8, 4], 16: [8, 4, 2, 2], 24: [8, 8, 4, 2, 2]}[E]


def pilot2_configs(selection: dict) -> list[dict]:
    E = selection["E_star"]
    lr_pretrain = selection["selected"]["joint/r32"]["lr"]
    checkpoint = selection["selected"]["joint/r32"]["writer_checkpoint"]
    configs = []
    for writer in WRITERS:
        if writer == "oracle_orth" and E + len(chains_for(E)) > 32:
            continue
        for variant in VARIANTS:
            for lr in LRS:
                name = f"p2-{writer}-{variant}-lr{lr:g}"
                config = base_config(name, writer, variant, 32, E, lr, "t2_chain_end", chains_for(E))
                config["pretrain"]["lr"] = lr_pretrain
                if writer == "t1_frozen":
                    config["pretrain"]["writer_checkpoint"] = checkpoint
                configs.append(config)
    return configs


def replication_configs(selection1: dict, selection2: dict) -> list[dict]:
    E = selection1["E_star"]
    configs = []
    for writer in ("t1_frozen", "joint", "oracle_random"):
        for variant in VARIANTS:
            lr = selection2["selected"][writer + "/" + variant]["lr"]
            if writer == "oracle_random":
                load = 3 * E // 2
                chains = {18: [8, 8, 2], 24: [8, 8, 4, 2, 2],
                          36: [8, 8, 8, 4, 4, 2, 2]}[load]
                config = base_config(f"rep-{writer}-{variant}", writer, variant, 32, load, lr,
                                     "t2_chain_end", chains)
            else:
                distractors = E // 4
                signal = E - distractors
                # One length-eight chain and fixed short chains cover the rest.
                remainder = signal - 8
                chains = [8] + [2] * (remainder // 2) + ([1] if remainder % 2 else [])
                config = base_config(f"rep-{writer}-{variant}", writer, variant, 32, signal, lr,
                                     "t2_chain_end", chains)
                config["task"].update(name="t2_typed", E=E, n_distract=distractors)
                config = resolve_config(config)
            config["pretrain"]["lr"] = selection1["selected"]["joint/r32"]["lr"]
            config["pretrain"]["task"]["E"] = E
            if writer == "t1_frozen":
                config["pretrain"]["writer_checkpoint"] = selection1["selected"]["joint/r32"]["writer_checkpoint"]
            configs.append(config)
    return configs


def write_plan(configs: list[dict], out: Path, shards: int = 3) -> dict:
    if shards < 1 or shards > len(configs):
        raise ValueError("invalid shard count")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for config in configs:
        if config["seed"] != 17 or config["eval"]["splits"] != ["val"] or config["eval"]["seed"] != 4242:
            raise ValueError("pilot manifests are validation-only with fixed seeds")
        name = config["name"]
        write_json(out / (name + ".json"), config)
        entry = {"name": name, "config": name + ".json"}
        checkpoint = config["pretrain"]["writer_checkpoint"]
        if config["writer"] == "t1_frozen" and checkpoint:
            entry["writer_from"] = {"checkpoint": checkpoint}
        entries.append(entry)
    for shard in range(shards):
        write_json(out / f"shard-{shard:02d}.yaml", entries[shard::shards])
    manifest = {"runs": len(configs), "shards": shards, "names": [c["name"] for c in configs],
                "selection_rule": "best LR per E; largest qualifying E; LR at selected E per rank",
                "lr_tiebreak": "smaller LR", "plateau_rule": "absolute relative loss change <=2% between steps4001..4500 and4501..5000; diagnostic only"}
    write_json(out / "manifest.json", manifest)
    return manifest


def loss_diagnostic(log: Path) -> dict:
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    rows = [row for row in rows if row["stage"] == "train"]
    if not rows:
        raise ValueError("missing training loss log")
    earlier = [row["loss"] for row in rows if 4000 < row["step"] <= 4500]
    latest = [row["loss"] for row in rows if 4500 < row["step"] <= 5000]
    if not earlier or not latest:
        raise ValueError("pilot loss trace does not cover 5000 steps")
    a, b = sum(earlier) / len(earlier), sum(latest) / len(latest)
    change = (a - b) / max(abs(a), 1e-12)
    return {"loss_tail_early": a, "loss_tail_late": b, "relative_loss_improvement": change,
            "plateau_2pct": abs(change) <= .02, "final_step": rows[-1]["step"]}


def collect(runs: Path, manifest: dict) -> list[dict]:
    rows = []
    for path in sorted(runs.glob("shard-*/*/record.json")):
        record = json.loads(path.read_text())
        if record["status"] != "complete":
            raise ValueError(f"pilot run not complete: {path}: {record['failure_reason']}")
        config = json.loads(path.with_name("config.json").read_text())["config"]
        if config["seed"] != 17 or config["eval"]["splits"] != ["val"] or config["eval"]["seed"] != 4242:
            raise ValueError("pilot contains forbidden seed/split")
        metrics = json.loads(path.with_name("metrics.json").read_text())
        if not metrics["gate0_passed"]:
            raise AssertionError("Gate0 failed")
        depth = 1 if config["variant"] == "B4" else 4
        chosen = [row for row in metrics["table"] if row["split"] == "val"
                  and row["condition"] == "original" and row["hop"] in (1, 2, 3) and row["L"] == depth]
        if not chosen:
            raise ValueError("missing prespecified selection cells")
        count = sum(row["n"] for row in chosen)
        accuracy = sum(row["accuracy"] * row["n"] for row in chosen) / count
        rows.append({"name": config["name"], "writer": config["writer"], "variant": config["variant"],
                     "r": config["model"]["r"], "E": config["task"]["E"], "lr": config["train"]["lr"],
                     "accuracy": accuracy, "depth": depth, "n": count,
                     "train_seconds": record["train_seconds"], "run_dir": str(path.parent.resolve()),
                     "writer_checkpoint": str(path.with_name("writer.pt").resolve()),
                     **loss_diagnostic(path.with_name("train_log.jsonl"))})
    names = [row["name"] for row in rows]
    if len(set(names)) != len(names) or set(names) != set(manifest["names"]):
        raise ValueError("pilot run set differs from manifest")
    return rows


def best(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty LR comparison")
    return max(rows, key=lambda row: (row["accuracy"], -row["lr"]))


def select_pilot1(rows: list[dict]) -> dict:
    joint32 = [row for row in rows if row["writer"] == "joint" and row["r"] == 32]
    best_by_E = {E: best([row for row in joint32 if row["E"] == E]) for E in (8, 12, 16, 24)}
    eligible = [E for E in (16, 24) if best_by_E[E]["accuracy"] >= .95]
    E = max(eligible) if eligible else 12
    selected = {}
    for writer, r in sorted({(row["writer"], row["r"]) for row in rows}):
        candidates = [row for row in rows if row["writer"] == writer and row["r"] == r and row["E"] == E]
        selected[f"{writer}/r{r}"] = best(candidates) if candidates else None
    return {"E_star": E, "fallback_to_12": not bool(eligible), "selected": selected,
            "joint_r32_best_by_E": best_by_E,
            "writer_diagnosis_required": all(row["accuracy"] < .95 for row in best_by_E.values()),
            "status": "proposed; orchestrator confirms prereg", "threshold": .95,
            "rule": "best LR per E, then E*, then LR at E*; ties choose smaller LR"}


def select_pilot2(rows: list[dict]) -> dict:
    return {"selected": {writer + "/" + variant: best([row for row in rows
                if row["writer"] == writer and row["variant"] == variant])
                for writer, variant in sorted({(row["writer"], row["variant"]) for row in rows})},
            "status": "proposed; orchestrator confirms prereg",
            "rule": "maximum mean val accuracy on hops1..3 at L4 (B4:L1); smaller LR breaks ties"}


def write_report_data(rows: list[dict], selection: dict, out: Path, *, stage: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "selection.json", selection)
    write_json(out / "full_table.json", rows)
    with (out / "full_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Curves include all LRs, not only selected checkpoints.
    groups = sorted({(row["writer"], row["r"]) for row in rows})
    for writer, r in groups:
        subset = [row for row in rows if row["writer"] == writer and row["r"] == r]
        labels = sorted({str(row["E"]) if stage == "pilot1" else row["variant"] for row in subset})
        fig, axes = plt.subplots(len(labels), 1, figsize=(8, 2.6 * len(labels)), squeeze=False)
        for ax, label in zip(axes[:, 0], labels):
            for row in subset:
                if (str(row["E"]) if stage == "pilot1" else row["variant"]) != label:
                    continue
                trace = [json.loads(line) for line in (Path(row["run_dir"]) / "train_log.jsonl").read_text().splitlines()]
                trace = [point for point in trace if point["stage"] == "train"]
                ax.plot([p["step"] for p in trace], [p["loss"] for p in trace], label=f"lr={row['lr']:g}", alpha=.8)
            ax.set(title=f"{writer}, r={r}, {'E=' if stage == 'pilot1' else ''}{label}", xlabel="step", ylabel="training CE")
            ax.legend()
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / f"loss-{writer}-r{r}.png", dpi=140)
        plt.close(fig)
    if stage != "pilot1":
        tables = {}
        for label, row in selection["selected"].items():
            tables[label] = json.loads((Path(row["run_dir"]) / "metrics.json").read_text())["table"]
        write_json(out / "selected_hop_depth_tables.json", tables)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan1", "plan2", "plan-replication", "summarize1", "summarize2", "summarize-replication"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--selection1", type=Path)
    parser.add_argument("--selection2", type=Path)
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args(argv)
    if args.action.startswith("plan"):
        if args.action == "plan1":
            configs = pilot1_configs()
        else:
            selection1 = json.loads(args.selection1.read_text())
            configs = pilot2_configs(selection1) if args.action == "plan2" else replication_configs(
                selection1, json.loads(args.selection2.read_text()))
        print(json.dumps(write_plan(configs, args.out, args.shards)))
    else:
        rows = collect(args.runs, json.loads(args.manifest.read_text()))
        selection = select_pilot1(rows) if args.action == "summarize1" else select_pilot2(rows)
        write_report_data(rows, selection, args.out, stage="pilot1" if args.action == "summarize1" else "pilot2")
        print(json.dumps({"runs": len(rows), "report": str(args.out), "E_star": selection.get("E_star")}))


if __name__ == "__main__":
    main()
