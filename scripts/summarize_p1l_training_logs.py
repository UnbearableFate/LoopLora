"""Committed summary of the P1-L training logs and Pilot-2 validation behind Amendment B7 (stdlib only).

The run directories are not tracked; this records, per main run, the last five logged training batches
(batch accuracy and loss over hops 1-3 at the trained L) and, per Pilot-2 run, the validation accuracy by hop.
No test or confirmation data are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MANIFESTS = {32: "runs/p1l/main/training-manifest.json", 128: "runs/p1l/main/r128/training-manifest.json"}
PILOTS = {32: "runs/p1l/main/pilot", 128: "runs/p1l/main/r128/pilot"}


def main_runs(root: Path) -> list[dict]:
    rows = []
    for rank, manifest in MANIFESTS.items():
        runs = json.loads((root / manifest).read_text())["runs"]
        for name, directory in sorted(runs.items()):
            log = Path(directory) / "train_log.jsonl"
            row = {"rank": rank, "run": name}
            if not log.exists():
                rows.append({**row, "status": "no log (not run yet)"})
                continue
            entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            entries = [e for e in entries if "accuracy" in e]
            if not entries:
                rows.append({**row, "status": "no training rows (untrained baseline)"})
                continue
            last = entries[-5:]
            rows.append({**row, "status": "trained", "final_step": entries[-1]["step"], "lr": entries[-1]["lr"],
                         "last5_steps": [e["step"] for e in last],
                         "last5_batch_accuracy": sum(e["accuracy"] for e in last) / len(last),
                         "last5_loss": sum(e["loss"] for e in last) / len(last),
                         "first5_batch_accuracy": sum(e["accuracy"] for e in entries[:5]) / len(entries[:5])})
    return rows


def pilot_runs(root: Path) -> list[dict]:
    rows = []
    for rank, pilot in PILOTS.items():
        for metrics in sorted((root / pilot).glob("*/*/validation/metrics.json")):
            table = json.loads(metrics.read_text())["table"]
            for entry in table:
                if entry["split"] == "val" and entry["condition"] == "memory_only/original" and entry["hop"] <= 3:
                    rows.append({"rank": rank, "run": metrics.parts[-3], "L": entry["L"], "hop": entry["hop"],
                                 "n": entry["n"], "full_accuracy": entry["full_accuracy"],
                                 "candidate_accuracy": entry["candidate_accuracy"],
                                 "chance_candidate_accuracy": entry["chance_accuracy"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("reports/p1l-c1/training-logs.json"))
    args = parser.parse_args()
    result = {"source": "P1-L train_log.jsonl (main runs) and Pilot-2 validation metrics; no test data",
              "main_runs": main_runs(args.root), "pilot2_validation": pilot_runs(args.root)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    trained = [r for r in result["main_runs"] if r["status"] == "trained"]
    print(f"{len(trained)} trained main runs, {len(result['pilot2_validation'])} Pilot-2 rows -> {args.out}")


if __name__ == "__main__":
    main()
