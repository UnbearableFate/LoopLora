#!/usr/bin/env python3
"""Render retained validation pilot report JSON and logs as readable Markdown.

Usage: python scripts/report_p1s_pilot_tables.py --reports reports/p1s-pilot
Uses only the standard library; writes <reports>/tables.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

LRS = (3e-4, 1e-3, 3e-3)
VARIANTS = ("M1", "M1-lin", "B5", "B5-lin", "B3", "B4")
DEPTHS = (0, 1, 2, 3, 4, 6, 8, 12, 16)
WRITERS = ("oracle_random", "oracle_orth", "t1_frozen", "joint")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _text(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _number(value: Any) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), f"expected finite number: {value!r}")
    return float(value)


def _percent(value: Any, *, signed: bool = False) -> str:
    value = 100 * _number(value)
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(_text(cell) for cell in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(_text(cell) for cell in row) + " |" for row in rows), ""]


def _links(root: Path, stage: str) -> list[str]:
    folder = root / stage
    csv = folder / "full_table.csv"
    _require(csv.is_file(), f"missing full CSV: {csv}")
    pngs = sorted(folder.glob("loss-*.png"))
    _require(bool(pngs), f"missing loss PNGs: {folder}")
    items = [("Full CSV", csv), *((path.stem, path) for path in pngs)]
    return [" · ".join(f"[{_text(label)}]({quote(path.relative_to(root).as_posix())})"
                       for label, path in items), ""]


def _stage_data(root: Path, stage: str) -> tuple[list[dict], dict]:
    rows = _read(root / stage / "full_table.json")
    selection = _read(root / stage / "selection.json")
    _require(isinstance(rows, list) and bool(rows) and isinstance(selection, dict),
             f"invalid report data: {stage}")
    by_name = {row["name"]: row for row in rows}
    _require(len(by_name) == len(rows), f"duplicate full-table run names: {stage}")
    for label, row in selection["selected"].items():
        if row is not None:
            _require(row == by_name.get(row["name"]), f"selected row differs from full table: {stage}/{label}")
    return rows, selection


def _pilot1(root: Path) -> list[str]:
    rows, selection = _stage_data(root, "pilot1")
    cells: dict[tuple[str, int, int], dict[float, float]] = {}
    for row in rows:
        key = row["writer"], row["r"], row["E"]
        lr = _number(row["lr"])
        by_lr = cells.setdefault(key, {})
        _require(lr in LRS and lr not in by_lr, f"unexpected or duplicate Pilot-1 LR cell: {key}/{lr}")
        by_lr[lr] = _number(row["accuracy"])
    lines = ["## Pilot-1: complete single-hop accuracy table", "",
             "Candidate-set validation accuracy (%), seed 17; B4 at L=1. "
             "Rows include every reported writer/rank/fact-load group.", ""]
    full = []
    for (writer, rank, facts), by_lr in sorted(cells.items()):
        _require(set(by_lr) == set(LRS), f"missing Pilot-1 LR cells: {writer}/r{rank}/E{facts}")
        full.append([writer, rank, facts, *(_percent(by_lr[lr]) for lr in LRS)])
    lines.extend(_table(["Writer", "r", "E", *(f"LR {lr:g}" for lr in LRS)], full))
    lines.extend([f"Recorded E*: {_text(selection['E_star'])}. "
                  f"Selection status: {_text(selection['status'])}.", "",
                  "Recorded selections at E*:", ""])
    chosen = []
    for label, row in sorted(selection["selected"].items()):
        chosen.append([label, row["E"] if row else "—", f"{row['lr']:g}" if row else "—",
                       _percent(row["accuracy"]) if row else "—"])
    lines.extend(_table(["Writer / rank", "E", "LR", "Val accuracy (%)"], chosen))
    lines.extend(_links(root, "pilot1"))
    return lines


def _final_loss(row: dict) -> float:
    path = Path(row["run_dir"]) / "train_log.jsonl"
    final = None
    with path.open() as stream:
        for line in stream:
            if line.strip():
                point = json.loads(line)
                if point["stage"] == "train":
                    final = point
    _require(final is not None and final["step"] == 5000, f"missing step-5000 train loss: {path}")
    return _number(final["loss"])


def _selected_stage(root: Path, stage: str) -> list[str]:
    _, selection = _stage_data(root, stage)
    selected = selection["selected"]
    expected_writers = WRITERS if stage == "pilot2" else ("oracle_random", "t1_frozen", "joint")
    expected = {writer + "/" + variant for writer in expected_writers for variant in VARIANTS}
    _require(set(selected) == expected, f"selected {stage} rows must contain all six A1 variants per condition")
    heading = "Pilot-2: selected LR results" if stage == "pilot2" else "Replication: retained LR results"
    lines = [f"## {heading}", "",
             "Candidate-set validation accuracy (%), seed 17, mean over hops 1–3; "
             "L=4 for the loop variants and L=1 for B4. "
             "Loss is step-5000 minibatch cross-entropy (CE).", ""]
    table_rows = []
    ordered = [(writer, variant, selected[writer + "/" + variant])
               for writer in expected_writers for variant in VARIANTS]
    for writer, variant, row in ordered:
        _require(row is not None and row["writer"] == writer and row["variant"] == variant,
                 f"misidentified selected row: {stage}/{writer}/{variant}")
        _require(row["depth"] == (1 if variant == "B4" else 4),
                 f"unexpected selected scoring depth: {stage}/{writer}/{variant}")
        plateau = row["plateau_2pct"]
        _require(isinstance(plateau, bool), f"invalid plateau flag: {stage}/{writer}/{variant}")
        table_rows.append([
            writer, variant, row["E"], f"{row['lr']:g}", row["depth"], _percent(row["accuracy"]),
            f"{_final_loss(row):.4f}",
            f"{_number(row['loss_tail_early']):.4f} → {_number(row['loss_tail_late']):.4f}",
            _percent(row["relative_loss_improvement"], signed=True),
            "yes" if plateau else "no",
        ])
    lines.extend(_table(["Writer", "Variant", "E", "LR", "L", "Mean hops 1–3 (%)",
                         "Step-5000 minibatch CE", "Tail mean CE (earlier → later)",
                         "Tail loss decrease (%)", "2% plateau"], table_rows))
    lines.extend(_links(root, stage))
    tables = _read(root / stage / "selected_hop_depth_tables.json")
    _require(set(tables) == expected, f"selected hop/depth tables differ from selected runs: {stage}")
    lines.extend(["### Every selected hop × depth table", "",
                  "All cells are candidate-set validation accuracy (%), seed 17, "
                  "with 1000 episodes per hop. Each table uses its recorded selected LR.", ""])
    for writer, variant, selected_row in ordered:
        label = writer + "/" + variant
        cells = {}
        depths = (1,) if variant == "B4" else DEPTHS
        for row in tables[label]:
            _require(row["split"] == "val" and row["condition"] == "original" and row["n"] == 1000,
                     f"unexpected hop/depth evaluation scope: {stage}/{label}")
            key = row["hop"], row["L"]
            _require(key not in cells, f"duplicate selected hop/depth cell: {stage}/{label}/{key}")
            cells[key] = row["accuracy"]
        _require(set(cells) == {(hop, depth) for hop in range(1, 9) for depth in depths},
                 f"incomplete selected hop/depth table: {stage}/{label}")
        lines.extend([f"#### {_text(writer)} / {_text(variant)} (LR={selected_row['lr']:g})", ""])
        lines.extend(_table(["Hop", *(f"L={depth}" for depth in depths)],
                            [[hop, *(_percent(cells[hop, depth]) for depth in depths)] for hop in range(1, 9)]))
    return lines


def render(reports: Path) -> Path:
    reports = reports.resolve()
    lines = ["# P1-S validation pilot tables", "",
             "Validation split only; training seed 17 and evaluation generator seed 4242. "
             "Percentages below are candidate-set accuracies. These tables reproduce retained report data "
             "and recorded selections; they add no scientific contrasts or gate conclusions.", "",
             "The plateau flag is a diagnostic: the absolute relative change between mean logged training CE "
             "at steps 4001–4500 and 4501–5000 is at most 2% (50 logged points per window). "
             "Tail loss decrease (%) = 100 × (earlier mean − later mean) / max(|earlier mean|, 1e-12). "
             "A negative value denotes increasing loss. Neither the plateau flag nor step-5000 minibatch CE "
             "is proof of convergence.", ""]
    lines.extend(_pilot1(reports))
    lines.extend(_selected_stage(reports, "pilot2"))
    lines.extend(_selected_stage(reports, "replication"))
    output = reports / "tables.md"
    output.write_text("\n".join(lines).rstrip() + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, type=Path)
    args = parser.parse_args(argv)
    print(render(args.reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
