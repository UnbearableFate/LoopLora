#!/usr/bin/env python3
"""Regenerate and check the tables of the WP-S9 study reports (IVD, CVH, TSM, UNS, NCR). Standard library only.

Every table of ``docs/reports/{IVD,CVH,TSM,UNS,NCR}.md`` sits between the markers ``<!-- table:ID -->`` and
``<!-- /table:ID -->`` and is produced from the committed outputs under ``reports/<study>/`` (a few tables read
``runs/``, which is git-ignored; they are marked in the reports and skipped when the file is absent).

* ``--check`` (default) rebuilds each table and exits 1 if a block of a report differs, a block or an ID is missing,
  a number does not appear in the study's own ``summary.md`` where it must (evidence), or a quoted passage
  (``<!-- quote:PATH -->`` ... ``<!-- /quote -->``) is not a substring of the quoted file (whitespace collapsed).
* ``--write`` replaces the blocks in the reports.
* ``--print ID`` prints one table.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "reports"
STATE = REPORT_DIR / "WP-S9-state.md"
STUDIES = ("IVD", "CVH", "TSM", "UNS", "NCR")


class SourceMissing(Exception):
    """A source file under ``runs/`` is not present in this checkout."""


# --------------------------------------------------------------------------------------------- reading helpers
def rows(path: str) -> list[dict[str, str]]:
    with open(ROOT / path, newline="") as handle:
        return list(csv.DictReader(handle))


def jload(path: str):
    target = ROOT / path
    if not target.exists():
        raise SourceMissing(path)
    return json.loads(target.read_text())


def f(value: str) -> float:
    return float(value)


def flag(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"not a boolean: {value!r}")
    return value == "True"


def pp(x: float) -> str:
    """Signed percentage points, two decimals."""
    return f"{100 * x:+.2f}"


def pc(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def cipp(diff: float, low: float, high: float) -> str:
    return f"{pp(diff)} [{pp(low)}, {pp(high)}]"


def yes(value: bool) -> str:
    return "yes" if value else "no"


def md(headers: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(out)


def section(text: str, key: str) -> str:
    """The part of a summary.md under the '## ' heading that contains ``key``."""
    parts = text.split("\n## ")
    for exact in (True, False):
        for part in parts:
            heading = part.split("\n", 1)[0].lstrip("# ").strip()
            if (heading == key) if exact else (key in heading):
                return part
    raise KeyError(f"no section {key!r}")


EVIDENCE: list[tuple[str, str, str, str]] = []  # (table id, summary path, section key or '', needle)


def evidence(table_id: str, summary: str, needle: str, key: str = "") -> None:
    EVIDENCE.append((table_id, summary, key, needle))


# --------------------------------------------------------------------------------------------- registry
BUILDERS: dict[str, tuple[str, Callable[[], str]]] = {}


def table(study: str, table_id: str):
    def register(function: Callable[[], str]) -> Callable[[], str]:
        BUILDERS[table_id] = (study, function)
        return function
    return register


# --------------------------------------------------------------------------------------------- jobs (state doc)
JOBS = {
    "IVD": ["ivd-lock", "ivd-smoke", "ivd-run", "ivd-analyze"],
    "CVH": ["cvh-lock", "cvh-smoke", "cvh-train", "cvh-confirm", "cvh-analyze"],
    "TSM": ["tsm-lock", "tsm-smoke", "tsm-train-17", "tsm-train-29", "tsm-train-43", "tsm-eval", "tsm-analyze"],
    "UNS": ["uns-lock", "uns-a", "uns-time", "uns-b", "uns-b-audit"],
    "NCR": ["ncr"],
}
SHARED_JOB = "s9-tests"


def state_jobs() -> dict[str, tuple[str, str, str]]:
    """Job name -> (ID, node-hours text, outcome) from the table of section 11 of the state report.

    A job that appears twice (a submission row and a result row) keeps its last row.
    """
    text = STATE.read_text()
    block = text[text.index("## 11. Step 2 on compute"):]
    found: dict[str, tuple[str, str, str]] = {}
    for line in block.splitlines():
        match = re.match(r"^\| (.+?) \| (\d+\.opbs) \| ([0-9.]*) \| (.*) \|$", line)
        if match:
            name = match.group(1).split(" (")[0].split(" ")[0]
            found[name] = (match.group(2), match.group(3), match.group(4))
    return found


def jobs_table(study: str) -> str:
    found = state_jobs()
    body, total = [], 0.0
    for name in JOBS[study]:
        job_id, hours, outcome = found[name]
        if not hours:
            raise ValueError(f"{name}: no node-hours in the state report")
        total += float(hours)
        body.append([name, job_id, hours, outcome])
    job_id, hours, outcome = found[SHARED_JOB]
    body.append([f"{SHARED_JOB} (shared by the five studies; not in the sum)", job_id, hours, outcome])
    body.append(["sum of the study rows above", "", f"{total:.2f}", ""])
    return md(["job", "ID", "node-h", "outcome (state report section 11)"], body)


for _study in STUDIES:
    table(_study, f"{_study.lower()}-jobs")(lambda s=_study: jobs_table(s))


# --------------------------------------------------------------------------------------------- IVD
IVD = "reports/ivd/results/"
IVD_CELLS = ["base", "L6-low", "L6-nat", "L6-high", "L24-low", "L24-nat", "L24-high", "L6-decay", "L24-decay"]
IVD_GROUPS = [("hop1", "1"), ("reach", "2,3"), ("mid", "4,5"), ("far", "6,7,8")]
WRITERS2 = ("oracle_random", "joint_fact")


@table("IVD", "ivd-decision")
def ivd_decision() -> str:
    summary = jload(IVD + "summary.json")
    inter = rows(IVD + "interference.csv")
    minimum = summary["bootstrap"]["minimum"]
    body = []
    for writer in WRITERS2:
        row = [x for x in inter if x["writer"] == writer and x["role"] == "decision"]
        assert len(row) == 1 and row[0]["load"] == "L24" and row[0]["hop_group"] == "reach", writer
        row = row[0]
        diff, low, high = f(row["difference"]), f(row["ci_low"]), f(row["ci_high"])
        passes = diff >= minimum and low > 0
        assert passes == flag(row["passes"]) == summary["decisions"][writer]["passes"], writer
        gap = summary["decisions"][writer]["overlap_gap"]
        label = summary["decisions"][writer]["label"]
        text = cipp(diff, low, high)
        body.append([writer, f"{100 * float(row['confidence']):.1f}%", text, yes(diff >= minimum), yes(low > 0),
                     yes(passes), f"{gap:.3f}", yes(gap >= summary["min_overlap_gap"]), label])
        evidence("ivd-decision", IVD + "summary.md",
                 f"| {writer} | {text} | {yes(passes)} | {gap:.3f} | {label} |", "Registered decision")
    return md(["writer", "interval level", "IC at L24, hops 2-3 [interval], pp", "IC >= 5 pp", "lower bound > 0",
               "pattern passes", "overlap gap (high - low)", "gap >= 0.2", "analysis label"], body)


@table("IVD", "ivd-ic")
def ivd_ic() -> str:
    inter = rows(IVD + "interference.csv")
    order = {name: i for i, (name, _) in enumerate(IVD_GROUPS)}
    body = []
    for writer in WRITERS2:
        picked = [x for x in inter if x["writer"] == writer]
        picked.sort(key=lambda x: (order[x["hop_group"]], x["load"]))
        for x in picked:
            text = cipp(f(x["difference"]), f(x["ci_low"]), f(x["ci_high"]))
            body.append([writer, x["load"], x["hops"], x["role"], text, yes(flag(x["passes"])),
                         pp(f(x["difference_seed_17"])), pp(f(x["difference_seed_29"])),
                         pp(f(x["difference_seed_43"])), x["n_episodes"]])
            evidence("ivd-ic", IVD + "summary.md",
                     f"| {writer} | {x['load']} | {x['hop_group']} | {text} | {yes(flag(x['passes']))} |",
                     "Interference contrast by load")
    return md(["writer", "load", "hops", "role", "IC [97.5% interval], pp", "pattern passes", "seed 17", "seed 29",
               "seed 43", "episodes per seed"], body)


@table("IVD", "ivd-gaps")
def ivd_gaps() -> str:
    summary = jload(IVD + "summary.json")
    body = []
    for writer in WRITERS2:
        for load in ("L6", "L24"):
            gap = summary["overlap_gaps"][writer][load]
            body.append([writer, load, f"{gap:.3f}", yes(gap >= summary["min_overlap_gap"])])
    return md(["writer", "load", "achieved overlap gap (high - low, hops 2-3)", "gap >= 0.2"], body)


def ivd_cell_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    return {(x["writer"], x["condition"], x["hop_group"]): x for x in rows(IVD + "cells_m1.csv")}


@table("IVD", "ivd-cells")
def ivd_cells() -> str:
    cells = ivd_cell_rows()
    b4 = {(x["writer"], x["condition"], x["hop_group"]): x for x in rows(IVD + "accuracy_b4.csv")}
    body = []
    for writer in WRITERS2:
        overlap = {}
        for cell in IVD_CELLS:
            if (writer, cell, "reach") not in cells:
                continue
            x, y = cells[writer, cell, "reach"], b4[writer, cell, "reach"]
            gain = pp(f(x["gain"]))
            low, high = pp(f(x["gain_ci_low"])), pp(f(x["gain_ci_high"]))
            l1, l4 = pc(f(x["accuracy_l1"])), pc(f(x["accuracy_l4"]))
            ov = f"{f(x['overlap_mean']):.3f}" if x["overlap_mean"] else "none"
            if x["overlap_mean"]:
                overlap[cell] = f(x["overlap_mean"])
            body.append([writer, cell, gain, f"[{low}, {high}]", l1, l4, ov, f"{f(x['read_residual_l1']):.3f}",
                         pc(f(y["accuracy"])), f"[{pc(f(y['ci_low']))}, {pc(f(y['ci_high']))}]"])
            evidence("ivd-cells", IVD + "summary.md",
                     f"| {cell} | {gain} | [{low}, {high}] | {l1} | {l4} | ", f"{writer}: M1 depth gain")
            evidence("ivd-cells", IVD + "summary.md",
                     f"| {cell} | {pc(f(y['accuracy']))} | [{pc(f(y['ci_low']))}, {pc(f(y['ci_high']))}] |",
                     f"{writer}: M1 depth gain")
        for load in ("L6", "L24"):  # manipulation check: the achieved overlap is ordered low < nat < high
            assert overlap[f"{load}-low"] < overlap[f"{load}-nat"] < overlap[f"{load}-high"], (writer, load)
    return md(["writer", "cell", "G = L4 - L1, hops 2-3, pp", "95% interval", "M1 L1 accuracy %", "M1 L4 accuracy %",
               "mean max abs cosine", "read residual", "B4 L1 accuracy %", "B4 95% interval"], body)


@table("IVD", "ivd-groups")
def ivd_groups() -> str:
    cells = ivd_cell_rows()
    body = []
    for writer in WRITERS2:
        for cell in IVD_CELLS:
            if (writer, cell, "reach") not in cells:
                continue
            body.append([writer, cell] + [
                cipp(f(cells[writer, cell, g]["gain"]), f(cells[writer, cell, g]["gain_ci_low"]),
                     f(cells[writer, cell, g]["gain_ci_high"])) for g, _ in IVD_GROUPS])
    return md(["writer", "cell"] + [f"G, hops {h} [95%]" for _, h in IVD_GROUPS], body)


@table("IVD", "ivd-groups-acc")
def ivd_groups_acc() -> str:
    cells = ivd_cell_rows()
    body = []
    for writer in WRITERS2:
        for cell in IVD_CELLS:
            if (writer, cell, "reach") not in cells:
                continue
            body.append([writer, cell] + [
                f"{pc(f(cells[writer, cell, g]['accuracy_l1']))} / {pc(f(cells[writer, cell, g]['accuracy_l4']))}"
                for g, _ in IVD_GROUPS])
    return md(["writer", "cell"] + [f"L1 / L4 accuracy %, hops {h}" for _, h in IVD_GROUPS], body)


@table("IVD", "ivd-fullvocab")
def ivd_fullvocab() -> str:
    cells = {(x["writer"], x["condition"], x["hop_group"]): x for x in rows(IVD + "cells_m1_full_vocabulary.csv")}
    b4 = {(x["writer"], x["condition"], x["hop_group"]): x for x in rows(IVD + "accuracy_b4_full_vocabulary.csv")}
    body = []
    for writer in WRITERS2:
        for cell in IVD_CELLS:
            if (writer, cell, "reach") not in cells:
                continue
            x, y = cells[writer, cell, "reach"], b4[writer, cell, "reach"]
            body.append([writer, cell, cipp(f(x["gain"]), f(x["gain_ci_low"]), f(x["gain_ci_high"])),
                         pc(f(x["accuracy_l1"])), pc(f(x["accuracy_l4"])), pc(f(y["accuracy"]))])
    return md(["writer", "cell", "full-vocabulary G, hops 2-3 [95%], pp", "M1 L1 exact match %",
               "M1 L4 exact match %", "B4 L1 exact match %"], body)


@table("IVD", "ivd-run")
def ivd_run() -> str:
    manifest = jload("runs/ivd/main/manifest.json")
    summary = jload(IVD + "summary.json")
    evaluation = manifest["evaluation"]
    body = [["status", manifest["status"]],
            ["checkpoints unchanged after the run", yes(manifest["checkpoints_unchanged"])],
            ["checkpoints (runs listed)", str(len(manifest["runs"]))],
            ["design lock commit (manifest = analysis summary)",
             f"{manifest['design_lock_commit']} = {summary['design_lock_commit']}"],
            ["design hash (manifest = analysis summary)", f"{manifest['design_hash']} = {summary['design_hash']}"],
            ["run node-h (manifest)", f"{manifest['node_hours']:.4f}"],
            ["run node-h (analysis summary)", f"{summary['run_node_hours']:.4f}"],
            ["evaluation record", json.dumps(evaluation, sort_keys=True)]]
    assert manifest["design_lock_commit"] == summary["design_lock_commit"]
    assert manifest["design_hash"] == summary["design_hash"]
    assert abs(manifest["node_hours"] - summary["run_node_hours"]) < 1e-9
    return md(["item (runs/ivd/main/manifest.json, uncommitted)", "value"], body)


# --------------------------------------------------------------------------------------------- CVH
CVH = "reports/cvh/results/"
CVH_FAMILIES = ["oracle_random/r32/M1", "joint_fact/r32/M1", "t1_frozen_fact/r32/M1"]


def cvh_endpoint_rows() -> list[dict[str, str]]:
    data = rows(CVH + "endpoints.csv")
    return sorted(data, key=lambda x: (CVH_FAMILIES.index(x["family"]), x["arm"]))


def cvh_latency_rows() -> dict[tuple[str, str], dict[str, str]]:
    return {(x["family"], x["arm"]): x for x in rows(CVH + "latency.csv")}


@table("CVH", "cvh-endpoints")
def cvh_endpoints() -> str:
    summary = jload(CVH + "summary.json")
    bootstrap = summary["bootstrap"]
    body = []
    for x in cvh_endpoint_rows():
        gain, low, high = f(x["reach_gain_difference"]), f(x["reach_gain_ci_low"]), f(x["reach_gain_ci_high"])
        hop1, executed = f(x["hop1_delta_difference"]), f(x["mean_executed"])
        ok_gain = gain >= bootstrap["gain_minimum"] and low > 0
        ok_hop1 = hop1 >= -bootstrap["hop1_loss"]
        ok_depth = executed <= bootstrap["mean_depth_limit"]
        assert (ok_gain, ok_hop1, ok_depth) == (flag(x["passes_gain"]), flag(x["passes_hop1"]),
                                                flag(x["passes_depth"])), (x["family"], x["arm"])
        passes = ok_gain and ok_hop1 and ok_depth
        assert passes == flag(x["passes"])
        assert (x["arm"] != "A0") == flag(x["gating"])
        assert passes == (f"{x['family']}/{x['arm']}" in summary["passing"])
        text, theta = cipp(gain, low, high), f"{f(x['theta']):.2f}"
        wall = flag(cvh_latency_rows()[x["family"], x["arm"]]["claim"])
        body.append([x["family"], x["arm"] + (" (reference)" if x["arm"] == "A0" else ""), theta, text, pp(hop1),
                     f"{executed:.2f}", yes(ok_gain), yes(ok_hop1), yes(ok_depth), yes(passes)])
        evidence("cvh-endpoints", CVH + "summary.md",
                 f"| {x['family']} | {x['arm']} | {theta} | {text} | {pp(hop1)} | {executed:.2f} | {yes(passes)} | "
                 f"{yes(wall)} |", "Registered pattern")
    return md(["family", "arm", "theta", "hops 2-3 minus L1 [99.17% interval], pp", "hop 1 minus L1, pp",
               "mean executed updates", "gain >= 3 pp, interval > 0", "hop 1 >= -1 pp", "executed <= 8",
               "pattern holds"], body)


@table("CVH", "cvh-reference")
def cvh_reference() -> str:
    body = []
    for x in cvh_endpoint_rows():
        body.append([x["family"], x["arm"], cipp(f(x["reach_vs_l4_difference"]), f(x["reach_vs_l4_ci_low"]),
                                                  f(x["reach_vs_l4_ci_high"])),
                     cipp(f(x["reach_vs_l16_difference"]), f(x["reach_vs_l16_ci_low"]), f(x["reach_vs_l16_ci_high"])),
                     f"{f(x['mean_selected']):.2f}", f"{f(x['mean_executed']):.2f}"])
    return md(["family", "arm", "hops 2-3 minus L4 [99.17%], pp", "hops 2-3 minus L16 [99.17%], pp",
               "mean selected step", "mean executed updates"], body)


@table("CVH", "cvh-latency")
def cvh_latency() -> str:
    lat = cvh_latency_rows()
    endpoints = {(x["family"], x["arm"]): x for x in cvh_endpoint_rows()}
    body = []
    for family in CVH_FAMILIES:
        for arm in ("A0", "A1", "A2"):
            x = lat[family, arm]
            assert flag(x["measured"])
            wall, fixed = f(x["wall_ms"]), f(x["wall_ms_fixed_16"])
            faster = wall < fixed
            assert faster == flag(x["claim"])
            cuda, cuda16 = f(x["cuda_ms"]), f(x["cuda_ms_fixed_16"])
            body.append([family, arm, f"{wall:.3f}", f"{fixed:.3f}", f"{cuda:.3f}", f"{cuda16:.3f}",
                         f"{f(x['mean_executed']):.2f}", yes(faster), yes(flag(endpoints[family, arm]["passes"]))])
            evidence("cvh-latency", CVH + "summary.md",
                     f"| {family} | {arm} | {wall:.3f} | {fixed:.3f} | {cuda:.3f} | {cuda16:.3f} | {yes(faster)} |",
                     "Latency")
    return md(["family", "arm", "wall ms", "wall ms, fixed L16", "CUDA ms", "CUDA ms, fixed L16",
               "mean executed updates in the timed batches", "wall time below fixed L16", "pattern holds"], body)


@table("CVH", "cvh-theta")
def cvh_theta() -> str:
    data = rows(CVH + "theta.csv")
    data.sort(key=lambda x: (CVH_FAMILIES.index(x["family"]), x["arm"]))
    lock = jload("reports/cvh/theta.json")
    for x in data:  # the results copy equals the committed theta lock
        match = [s for s in lock["selected"] if s["family"] == x["family"] and s["arm"] == x["arm"]]
        assert len(match) == 1 and match[0]["theta"] == f(x["theta"]) and match[0]["feasible"] == flag(x["feasible"])
        assert match[0]["validation_accuracy"] == f(x["validation_accuracy"])
    body = [[x["family"], x["arm"], yes(flag(x["feasible"])), f"{f(x['theta']):.2f}", x["source"] or "validation",
             pp(f(x["hop1_delta"])), pc(f(x["validation_accuracy"]), 2),
             f"{f(x['validation_mean_executed_steps']):.2f}", f"{f(x['validation_mean_steps']):.2f}"] for x in data]
    return md(["family", "arm", "feasible", "theta", "source", "validation hop 1 minus L1, pp",
               "validation accuracy % (hops 1-8)", "validation mean executed", "validation mean selected"], body)


@table("CVH", "cvh-a0")
def cvh_a0() -> str:
    lock = jload("reports/cvh/theta.json")
    body = []
    for x in sorted(lock["a0_reproduction"], key=lambda y: CVH_FAMILIES.index(y["family"])):
        deviation = abs(x["recomputed_validation_accuracy"] - x["p4s_l_validation_accuracy"])
        body.append([x["family"], f"{x['p4s_l_validation_accuracy']:.6f}", f"{x['recomputed_validation_accuracy']:.6f}",
                     f"{deviation:.1e}", yes(x["reproduced"])])
    return md(["family", "P4-S-L record, validation accuracy", "recomputed", "absolute deviation", "reproduced"], body)


@table("CVH", "cvh-accuracy")
def cvh_accuracy() -> str:
    data = rows(CVH + "accuracy.csv")
    index = {(x["family"], x["method"], x["hop_group"]): x for x in data}
    groups = ["hop1", "reach", "mid", "far", "all"]
    body = []
    for family in CVH_FAMILIES:
        for method in ("L1", "L4", "L16", "A0", "A1", "A2"):
            cells = [index[family, method, g] for g in groups]
            body.append([family, method] + [f"{pc(f(c['accuracy']))} [{pc(f(c['ci_low']))}, {pc(f(c['ci_high']))}]"
                                            for c in cells])
            evidence("cvh-accuracy", CVH + "summary.md",
                     f"| {family} | {method} | " + " | ".join(pc(f(c["accuracy"])) for c in cells) + " |",
                     "Accuracy by hop group")
    return md(["family", "method", "hop 1 %", "hops 2-3 %", "hops 4-5 %", "hops 6-8 %", "all hops %"], body)


@table("CVH", "cvh-labels")
def cvh_labels() -> str:
    records = jload("runs/cvh/main/head_records.json")
    body = []
    for family, prefix in (("oracle_random", "primary-oracle_random-M1-r32"), ("joint_fact", "f2-main-joint_fact-M1"),
                           ("t1_frozen_fact", "f2-main-t1_frozen_fact-M1")):
        names = [f"{prefix}-s{seed}" for seed in (17, 29, 43)]
        curves = [records[name]["label_rate"] for name in names]
        assert all(len(c) == 16 for c in curves)
        mean = [sum(c[s] for c in curves) / 3 for s in range(16)]
        body.append([family] + [pc(m) for m in mean])
    return md(["A2 writer family"] + [f"s{s}" for s in range(1, 17)], body)


@table("CVH", "cvh-run")
def cvh_run() -> str:
    manifest = jload("runs/cvh/main/manifest.json")
    train = jload("runs/cvh/main/train_manifest.json")
    summary = jload(CVH + "summary.json")
    assert manifest["design_lock_commit"] == summary["design_lock_commit"]
    assert manifest["theta_lock_commit"] == summary["theta_lock_commit"]
    body = [["training manifest status", train["status"]],
            ["confirmation manifest status", manifest["status"]],
            ["checkpoints unchanged (training, confirmation)",
             f"{yes(train['checkpoints_unchanged'])}, {yes(manifest['checkpoints_unchanged'])}"],
            ["readers", str(len(manifest["readers"]))],
            ["families traced on the confirmation set", ", ".join(manifest["active_families"])],
            ["latency measured", yes(manifest["latency_measured"])],
            ["design lock commit (= analysis summary)", manifest["design_lock_commit"]],
            ["theta lock commit (= analysis summary)", manifest["theta_lock_commit"]],
            ["confirmation node-h (analysis summary)", f"{summary['confirmation_node_hours']:.4f}"]]
    return md(["item (runs/cvh/main, uncommitted)", "value"], body)


# --------------------------------------------------------------------------------------------- TSM
TSM = "reports/tsm/results/"
TSM_CONTRASTS = {"hop1_l8_vs_l1": "hop 1: arm at L8 minus arm at L1",
                 "hop1_l16_vs_l1": "hop 1: arm at L16 minus arm at L1",
                 "reach_l4_vs_t0_l4": "hops 2-3 at L4: arm minus T0",
                 "far_l16_vs_t0_l16": "hops 4-8 at L16: arm minus T0"}


def tsm_endpoint_rows() -> list[dict[str, str]]:
    data = rows(TSM + "endpoints.csv")
    return sorted(data, key=lambda x: (WRITERS2.index(x["writer"]), x["arm"]))


@table("TSM", "tsm-conditions")
def tsm_conditions() -> str:
    summary = jload(TSM + "summary.json")
    bootstrap = summary["bootstrap"]
    body = []
    for x in tsm_endpoint_rows():
        stay, far, gain = f(x["stay_difference"]), f(x["far_difference"]), f(x["gain_difference"])
        ok = (abs(stay) <= bootstrap["stay_tolerance"], far >= -bootstrap["far_tolerance"],
              gain >= bootstrap["gain_minimum"] and f(x["gain_ci_low"]) > 0)
        assert ok == (flag(x["stay_passes"]), flag(x["far_passes"]), flag(x["gain_passes"])), (x["writer"], x["arm"])
        assert not flag(x["training_failed"]) and x["failed_seeds"] == ""
        assert all(ok) == flag(x["passes"]) == summary["writer_passes"][f"{x['writer']}/{x['arm']}"]
        s = cipp(stay, f(x["stay_ci_low"]), f(x["stay_ci_high"]))
        fa = cipp(far, f(x["far_ci_low"]), f(x["far_ci_high"]))
        g = cipp(gain, f(x["gain_ci_low"]), f(x["gain_ci_high"]))
        body.append([x["writer"], x["arm"], f"{100 * f(x['confidence']):.2f}%", s, yes(ok[0]), fa, yes(ok[1]), g,
                     yes(ok[2]), yes(all(ok))])
        evidence("tsm-conditions", TSM + "summary.md",
                 f"| {x['writer']} | {x['arm']} | {s} | {yes(ok[0])} | {fa} | {yes(ok[1])} | {g} | {yes(ok[2])} | "
                 f"{yes(all(ok))} |", "Registered conditions")
    return md(["writer", "arm", "interval level", "stay: hops 1-3, L16 minus L4 [interval], pp", "|stay| <= 1 pp",
               "far: hops 4-8, arm at L8 minus T0 at L8 [interval], pp", "far >= -1 pp",
               "gain: hops 2-3, arm at L4 minus B4 at L1 [interval], pp", "gain >= 3 pp, interval > 0",
               "writer passes"], body)


@table("TSM", "tsm-arms")
def tsm_arms() -> str:
    summary = jload(TSM + "summary.json")
    per = summary["writer_passes"]
    body = []
    for arm in ("T1", "T2"):
        both = all(per[f"{w}/{arm}"] for w in WRITERS2)
        assert both == summary["arms_pass"][arm]
        body.append([arm, yes(per[f"oracle_random/{arm}"]), yes(per[f"joint_fact/{arm}"]), yes(both),
                     summary["arm_outcome"][arm]])
    wording = [[w, "no arm passes" if not any(per[f"{w}/{a}"] for a in ("T1", "T2")) else "see the table"]
               for w in WRITERS2]
    return (md(["arm", "oracle_random passes", "joint_fact passes", "arm passes (both writers)",
                "analysis arm_outcome"], body) + "\n\n" + md(["writer", "per-writer statement"], wording))


@table("TSM", "tsm-descriptive")
def tsm_descriptive() -> str:
    data = rows(TSM + "descriptive_contrasts.csv")
    order = list(TSM_CONTRASTS)
    data.sort(key=lambda x: (WRITERS2.index(x["writer"]), x["arm"], order.index(x["contrast"])))
    body = [[x["writer"], x["arm"], TSM_CONTRASTS[x["contrast"]],
             cipp(f(x["difference"]), f(x["ci_low"]), f(x["ci_high"]))] for x in data]
    return md(["writer", "arm", "contrast", "difference [98.75% interval], pp"], body)


@table("TSM", "tsm-accuracy")
def tsm_accuracy() -> str:
    data = rows(TSM + "accuracy.csv")
    index = {(x["writer"], x["method"], x["L"], x["hop_group"]): x for x in data}
    groups = ["hop1", "reach", "mid", "far", "all"]
    body = []
    for writer in WRITERS2:
        for method, depths in (("B4", ("1",)), ("T0", ("1", "4", "8", "16")), ("T1", ("1", "4", "8", "16")),
                               ("T2", ("1", "4", "8", "16"))):
            for depth in depths:
                cells = [index[writer, method, depth, g] for g in groups]
                body.append([writer, method, depth] + [
                    f"{pc(f(c['accuracy']))} [{pc(f(c['ci_low']))}, {pc(f(c['ci_high']))}]" for c in cells])
    for writer in sorted(WRITERS2):  # summary.md lists the writers in alphabetical order, three columns
        for method, depth in (("B4", "1"), ("T0", "1"), ("T0", "4"), ("T0", "8"), ("T0", "16"), ("T1", "1"),
                              ("T1", "4"), ("T1", "8"), ("T1", "16"), ("T2", "1"), ("T2", "4"), ("T2", "8"),
                              ("T2", "16")):
            reach, far, hop1 = (pc(f(index[writer, method, depth, g]["accuracy"])) for g in ("reach", "far", "hop1"))
            need = f"| {writer} | {method} | {depth} | 17;29;43 | {reach} | {far} | {hop1} |"
            evidence("tsm-accuracy", TSM + "summary.md", need, "Accuracy by depth")
    return md(["writer", "method", "L", "hop 1 % [95%]", "hops 2-3 % [95%]", "hops 4-5 % [95%]", "hops 6-8 % [95%]",
               "all hops % [95%]"], body)


@table("TSM", "tsm-training")
def tsm_training() -> str:
    data = rows(TSM + "training.csv")
    data.sort(key=lambda x: (WRITERS2.index(x["writer"]), x["arm"], int(x["seed"])))
    body = [[x["writer"], x["arm"], x["seed"], x["status"], x["attempt"], x["steps"], f"{f(x['final_loss']):.4f}",
             f"{f(x['train_seconds']):.0f}", f"{f(x['step_seconds']):.4f}", x["trainable_parameters"],
             f"{f(x['peak_memory_bytes']) / 2 ** 20:.1f}"] for x in data]
    return md(["writer", "arm", "seed", "status", "attempt", "steps", "final loss", "train seconds", "seconds per step",
               "trainable parameters", "peak memory MiB"], body)


@table("TSM", "tsm-gate")
def tsm_gate() -> str:
    data = rows(TSM + "gate_profiles.csv")
    steps = [f"step_{s}" for s in range(1, 17)]
    out = []
    for writer in WRITERS2:
        body = []
        for hop in range(1, 9):
            picked = [x for x in data if x["name"].startswith(f"tsm-{writer}-T2-") and int(x["hop"]) == hop]
            assert len(picked) == 3, (writer, hop)
            body.append([str(hop)] + [f"{sum(f(x[s]) for x in picked) / 3:.3f}" for s in steps])
        out.append(f"{writer} (T2; mean of seeds 17, 29, 43 of the mean gate over the queries of the hop):\n\n" +
                   md(["hop"] + [f"step {s}" for s in range(1, 17)], body))
    return "\n\n".join(out)


@table("TSM", "tsm-run")
def tsm_run() -> str:
    summary = jload(TSM + "summary.json")
    manifest = jload("runs/tsm/main/evaluation/manifest.json")
    trains = [jload(f"runs/tsm/main/train_manifest-s{seed}.json") for seed in (17, 29, 43)]
    assert manifest["status"] == "complete"
    body = [["training manifests (seeds 17, 29, 43): status", ", ".join(t["status"] for t in trains)],
            ["training source hash (analysis summary)", summary["training_source_hash"]],
            ["evaluation source hash (analysis summary)", summary["evaluation_source_hash"]],
            ["evaluation attempt, decision, restarts",
             f"{summary['evaluation_attempt']}, {summary['evaluation_decision'] or 'none'}, "
             f"{len(summary['evaluation_restarts'])}"],
            ["training failures, failed cells", f"{len(summary['training_failures'])}, {len(summary['failed_cells'])}"],
            ["evaluation manifest status", manifest["status"]],
            ["design lock commit (analysis summary)", summary["design_lock_commit"]],
            ["evaluation node-h (analysis summary)", f"{summary['evaluation_node_hours']:.4f}"]]
    assert summary["training_source_hash"] == summary["evaluation_source_hash"]
    return md(["item (runs/tsm/main, uncommitted, and the analysis summary)", "value"], body)


# --------------------------------------------------------------------------------------------- UNS
UNS = {"A": "reports/uns/part-a/", "B": "reports/uns/part-b/"}
UNS_WRITERS = ["joint", "joint_fact", "oracle_random"]
UNS_ENDPOINTS = ["m1_depth_gain", "m1_l4_minus_b4"]
UNS_GROUPS = [("hop1", "1"), ("reach", "2,3"), ("far", "6,7,8")]


@table("UNS", "uns-audit")
def uns_audit() -> str:
    body = []
    for part in ("A", "B"):
        audit = jload(UNS[part] + "audit.json")
        assert audit["status"] == "passed" and audit["failures"] == []
        repro = rows(UNS[part] + "reproduction.csv") if part == "A" else []
        for writer in UNS_WRITERS:
            runs = [r for r in audit["runs"] if r["writer"] == writer]
            failures = sum(len(r["failures"]) for r in runs)
            if part == "A":
                mine = [r for r in repro if r["writer"] == writer]
                reproduced = f"{sum(flag(r['reproduced']) for r in mine)} of {len(mine)}"
            else:
                reproduced = "not applicable"
            body.append([f"Part {part}", writer, str(len(runs)), str(failures), reproduced])
    return md(["part", "writer", "runs audited", "audit failures", "committed rows reproduced"], body)


@table("UNS", "uns-structure")
def uns_structure() -> str:
    body = []
    for part in ("A", "B"):
        audit = jload(UNS[part] + "audit.json")
        for pool in ("seen", "unseen"):
            kinds = {json.dumps(r["pools"][pool], sort_keys=True) for r in audit["runs"]}
            assert len(kinds) == 1, (part, pool)
            x = json.loads(kinds.pop())
            body.append([f"Part {part}", pool, x["split"], str(x["episodes"]), ",".join(map(str, x["hops"])),
                         ",".join(map(str, x["depths"])), ",".join(map(str, x["candidate_sizes"])), str(x["rows"]),
                         str(len(audit["runs"]))])
    return md(["part", "pool", "split", "episodes per run", "hops", "depths", "candidate sizes", "rows per run",
               "runs (all identical)"], body)


@table("UNS", "uns-reproduction")
def uns_reproduction() -> str:
    data = rows(UNS["A"] + "reproduction.csv")
    body = [[x["writer"], x["endpoint"], x["confidence"], f"{f(x['recomputed']):.6f}", f"{f(x['committed']):.6f}",
             f"{f(x['max_abs_deviation']):.1e}", yes(flag(x["reproduced"]))] for x in data]
    return md(["writer", "endpoint (M1 L4 - L1, unseen pool)", "confidence", "recomputed", "committed row",
               "max abs deviation (difference and both interval ends)", "reproduced"], body)


def uns_endpoint_table(part: str, writers: list[str]) -> str:
    data = rows(UNS[part] + "endpoints.csv")
    index = {(x["writer"], x["endpoint"], x["group"], x["pool"]): x for x in data}
    body = []
    for writer in writers:
        for endpoint in UNS_ENDPOINTS:
            for group, hops in UNS_GROUPS:
                cells = []
                for pool in ("seen", "unseen", "unseen_minus_seen"):
                    x = index[writer, endpoint, group, pool]
                    assert x["confidence"] == "0.95" and x["hops"] == hops
                    cells.append(cipp(f(x["difference"]), f(x["ci_low"]), f(x["ci_high"])))
                body.append([writer, endpoint, hops] + cells)
                evidence(f"uns-{part.lower()}-endpoints", UNS[part] + "summary.md",
                         f"| {endpoint} | {group} | " + " | ".join(cells) + " |", writer)
    return md(["writer", "endpoint", "hops", "seen pool [95%], pp", "unseen pool [95%], pp",
               "unseen minus seen [95%], pp"], body)


@table("UNS", "uns-a-endpoints")
def uns_a_endpoints() -> str:
    return uns_endpoint_table("A", UNS_WRITERS)


@table("UNS", "uns-b-endpoints")
def uns_b_endpoints() -> str:
    return uns_endpoint_table("B", UNS_WRITERS)


@table("UNS", "uns-wording")
def uns_wording() -> str:
    body = []
    for part, reader in (("A", "M1 (released, mixed depth)"), ("B", "M1 trained at L = 4 (P2-S-F1)")):
        index = {(x["writer"], x["endpoint"], x["group"], x["pool"]): x for x in rows(UNS[part] + "endpoints.csv")}
        for writer in UNS_WRITERS:
            x = index[writer, "m1_depth_gain", "reach", "unseen"]
            d = index[writer, "m1_depth_gain", "reach", "unseen_minus_seen"]
            diff, low = f(x["difference"]), f(x["ci_low"])
            holds = diff >= 0.03 and low > 0
            excludes = f(d["ci_low"]) > 0 or f(d["ci_high"]) < 0
            body.append([f"Part {part}", writer, reader, cipp(diff, low, f(x["ci_high"])), yes(holds),
                         "the depth gain is also present on unseen-pool entities" if holds
                         else "not shown on unseen-pool entities",
                         cipp(f(d["difference"]), f(d["ci_low"]), f(d["ci_high"])), yes(excludes)])
    return md(["part", "writer", "reader", "unseen-pool M1 depth gain, hops 2-3 [95%], pp",
               "at least 3 pp, interval above 0", "wording of the prereg",
               "unseen minus seen, same endpoint [95%], pp", "that interval excludes 0"], body)


def uns_accuracy_table(part: str) -> str:
    data = rows(UNS[part] + "accuracy.csv")
    index = {(x["writer"], x["reader"], x["L"], x["hop"], x["pool"]): x for x in data}
    body = []
    for writer in UNS_WRITERS:
        for reader, depth in (("M1", "1"), ("M1", "4"), ("B4", "1")):
            for pool in ("seen", "unseen", "unseen_minus_seen"):
                cells = []
                for hop in range(1, 9):
                    x = index[writer, reader, depth, str(hop), pool]
                    cells.append(f"{100 * f(x['accuracy']):+.2f}" if pool == "unseen_minus_seen"
                                 else f"{100 * f(x['accuracy']):.2f}")
                body.append([writer, f"{reader} at L{depth}", pool.replace("_", " ")] + cells)
        for hop in range(1, 9):
            seen, unseen, diff = (index[writer, "M1", "4", str(hop), p]
                                  for p in ("seen", "unseen", "unseen_minus_seen"))
            evidence(f"uns-{part.lower()}-accuracy", UNS[part] + "summary.md",
                     f"| {hop} | {100 * f(seen['accuracy']):.2f} | {100 * f(unseen['accuracy']):.2f} | "
                     f"{100 * f(diff['accuracy']):+.2f} |", writer)
    return md(["writer", "cell", "pool"] + [f"hop {h} %" for h in range(1, 9)], body)


@table("UNS", "uns-a-accuracy")
def uns_a_accuracy() -> str:
    return uns_accuracy_table("A")


@table("UNS", "uns-b-accuracy")
def uns_b_accuracy() -> str:
    return uns_accuracy_table("B")


@table("UNS", "uns-run")
def uns_run() -> str:
    manifest = jload("runs/uns/main/manifest.json")
    design = jload("reports/uns/design.json")
    summary = jload(UNS["B"] + "summary.json")
    assert manifest["design_lock_commit"] == summary["design_lock_commit"]
    body = [["evaluation seed (design lock, committed)", str(design["extension"]["seed"])],
            ["splits, episodes, hops, depths", f"{', '.join(design['extension']['splits'])}; "
             f"{design['extension']['episodes']}; {design['extension']['hops'][0]}-{design['extension']['hops'][-1]}; "
             f"{','.join(map(str, design['extension']['depths']))}"],
            ["checkpoints in the design lock", str(len(design["checkpoints"]))],
            ["manifest status, partial, checkpoints unchanged",
             f"{manifest['status']}, {yes(manifest['partial'])}, {yes(manifest['checkpoints_unchanged'])}"],
            ["runs in the manifest", str(len(manifest["runs"]))],
            ["design lock commit (manifest = analysis summary)", manifest["design_lock_commit"]],
            ["evaluation node-h (manifest)", f"{manifest['node_hours']:.4f}"],
            ["evaluation node-h (Part B summary)", f"{summary['evaluation_node_hours']:.4f}"]]
    return md(["item (runs/uns/main/manifest.json, uncommitted, and the committed design lock)", "value"], body)


# --------------------------------------------------------------------------------------------- NCR
NCR = "reports/ncr/"
NCR_WRITERS = ["t1_frozen_fact", "joint_fact", "joint", "t1_frozen", "oracle_random", "oracle_orth"]
NCR_READERS = ["M1", "B4"]
NCR_SETS = ["screening", "confirmation"]


def ncr_index():
    cat = {(x["writer"], x["variant"], x["dataset"], x["hop"], x["category"]): x for x in rows(NCR + "categories.csv")}
    acc = {(x["writer"], x["variant"], x["dataset"], x["hop"]): x for x in rows(NCR + "accuracy.csv")}
    return cat, acc


def ncr_hop_table(hop: int) -> str:
    cat, acc = ncr_index()
    others = ["start", "terminal_other", "other"] + (["one_short"] if hop == 3 else [])
    body = []
    for writer in NCR_WRITERS:
        for reader in NCR_READERS:
            for dataset in NCR_SETS:
                key = (writer, reader, dataset, str(hop))
                a = acc[key]
                first = cat[key + ("first_hop",)]
                below = flag(a["below_chance"])
                cells = [writer, reader, dataset, pc(f(a["accuracy"])), pc(f(a["chance_accuracy"])), yes(below),
                         pc(f(first["rate"])), pc(f(first["uniform_rate"])),
                         f"{pc(f(first['share']))} [{pc(f(first['share_ci_low']))}, {pc(f(first['share_ci_high']))}]",
                         pc(f(first["uniform_share"]))]
                for name in others:
                    x = cat[key + (name,)]
                    cells.append(f"{pc(f(x['share']))} ({pc(f(x['uniform_share']))})")
                body.append(cells)
                evidence(f"ncr-hop{hop}", NCR + "summary.md", "| " + " | ".join(cells) + " |", f"Hop {hop}")
    headers = ["writer", "reader", "evaluation", "accuracy %", "chance %", "below chance", "first-hop rate %",
               "uniform rate %", "first-hop share of errors % [95%]", "uniform share %", "start share (uniform)",
               "terminal_other share (uniform)", "other share (uniform)"] + (["one_short share (uniform)"]
                                                                            if hop == 3 else [])
    return md(headers, body)


@table("NCR", "ncr-hop2")
def ncr_hop2() -> str:
    return ncr_hop_table(2)


@table("NCR", "ncr-hop3")
def ncr_hop3() -> str:
    return ncr_hop_table(3)


@table("NCR", "ncr-endpoint")
def ncr_endpoint() -> str:
    cat, acc = ncr_index()
    body = []
    for writer in NCR_WRITERS:
        for reader in NCR_READERS:
            for dataset in NCR_SETS:
                key = (writer, reader, dataset, "2")
                a = acc[key]
                if not flag(a["below_chance"]):
                    continue
                x = cat[key + ("first_hop",)]
                uniform = f(x["uniform_share"])
                body.append([writer, reader, dataset, pc(f(a["accuracy"])), pc(f(a["chance_accuracy"])),
                             f"{pc(f(x['rate']))} ({pc(f(x['uniform_rate']))})",
                             f"{pc(f(x['share']))} [{pc(f(x['share_ci_low']))}, {pc(f(x['share_ci_high']))}]",
                             pc(uniform), yes(f(x["share"]) > uniform), yes(f(x["share_ci_low"]) > uniform)])
    return md(["writer", "reader", "evaluation", "hop-2 accuracy %", "chance %", "first-hop rate % (uniform)",
               "first-hop share of errors % [95%]", "uniform share %", "share above uniform",
               "interval above uniform"], body)


@table("NCR", "ncr-seeds")
def ncr_seeds() -> str:
    cat, acc = ncr_index()
    body = []
    for hop in (2, 3):
        for writer in NCR_WRITERS:
            for reader in NCR_READERS:
                for dataset in NCR_SETS:
                    x = cat[writer, reader, dataset, str(hop), "first_hop"]
                    errors = acc[writer, reader, dataset, str(hop)]
                    body.append([str(hop), writer, reader, dataset] +
                                [pc(f(x[f"share_seed_{s}"])) for s in (17, 29, 43)] +
                                [errors[f"errors_seed_{s}"] for s in (17, 29, 43)])
    return md(["hop", "writer", "reader", "evaluation", "first-hop share seed 17 %", "seed 29 %", "seed 43 %",
               "errors seed 17", "errors seed 29", "errors seed 43"], body)


@table("NCR", "ncr-counts")
def ncr_counts() -> str:
    s = jload(NCR + "summary.json")
    body = [["groups", str(s["groups"])], ["verified directories", str(s["verified_directories"])],
            ["verified rows", str(s["verified_rows"])], ["missing directories", str(len(s["missing"]))],
            ["undefined groups", str(len(s["undefined"]))], ["bootstrap resamples", str(s["resamples"])]]
    return md(["item (reports/ncr/summary.json)", "value"], body)


# --------------------------------------------------------------------------------------------- observation checks
def observation_checks() -> list[str]:
    """Assertions behind the sentences of the 'Observations (not registered)' sections; returns failed checks."""
    failed: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failed.append(f"observation not supported by the sources: {name}")

    # IVD
    inter = rows(IVD + "interference.csv")
    for writer in WRITERS2:
        x = [r for r in inter if r["writer"] == writer and r["load"] == "L24" and r["hop_group"] == "hop1"][0]
        check(f"IVD hop-1 IC at L24 negative with interval below 0 ({writer})",
              f(x["difference"]) < 0 and f(x["ci_high"]) < 0)
    cells = ivd_cell_rows()
    b4 = {(x["writer"], x["condition"], x["hop_group"]): x for x in rows(IVD + "accuracy_b4.csv")}
    for load in ("L6", "L24"):
        base = cells["joint_fact", "base", "reach"]
        decay = cells["joint_fact", f"{load}-decay", "reach"]
        check(f"IVD decay-only {load} equals base",
              pp(f(decay["gain"])) == pp(f(base["gain"])) and pc(f(decay["accuracy_l1"])) == pc(f(base["accuracy_l1"]))
              and pc(f(decay["accuracy_l4"])) == pc(f(base["accuracy_l4"])))
    for writer in WRITERS2:
        base = cells[writer, "base", "reach"]
        for arm in ("low", "nat", "high"):
            check(f"IVD G at L24-{arm} below base ({writer})",
                  f(cells[writer, f"L24-{arm}", "reach"]["gain"]) < f(base["gain"]))
            check(f"IVD B4 L1 at L24-{arm} above base ({writer})",
                  f(b4[writer, f"L24-{arm}", "reach"]["accuracy"]) > f(b4[writer, "base", "reach"]["accuracy"]))
    # CVH
    ep = {(x["family"], x["arm"]): x for x in rows(CVH + "endpoints.csv")}
    acc = {(x["family"], x["method"], x["hop_group"]): x for x in rows(CVH + "accuracy.csv")}
    for family in CVH_FAMILIES[1:]:
        for arm in ("A1", "A2"):
            x = ep[family, arm]
            check(f"CVH {family} {arm} stops at step 1",
                  f(x["theta"]) == 0.05 and f(x["mean_executed"]) == 1.0 and f(x["reach_gain_difference"]) == 0.0
                  and f(x["reach_gain_ci_low"]) == 0.0 and f(x["reach_gain_ci_high"]) == 0.0)
            for group in ("hop1", "reach", "mid", "far", "all"):
                check(f"CVH {family} {arm} accuracy equals L1 ({group})",
                      pc(f(acc[family, arm, group]["accuracy"])) == pc(f(acc[family, "L1", group]["accuracy"])))
    x = ep["oracle_random/r32/M1", "A2"]
    check("CVH oracle_random A2 executed 1.05 and gain below 1 pp",
          f"{f(x['mean_executed']):.2f}" == "1.05" and f(x["reach_gain_difference"]) < 0.01)
    x = ep["oracle_random/r32/M1", "A1"]
    check("CVH passing arm below L4 and L16 with intervals below 0",
          f(x["reach_vs_l4_ci_high"]) < 0 and f(x["reach_vs_l16_ci_high"]) < 0)
    for family in CVH_FAMILIES:
        gains = {arm: f(ep[family, arm]["reach_gain_difference"]) for arm in ("A0", "A1", "A2")}
        check(f"CVH A0 largest gain, hop 1 lost ({family})",
              gains["A0"] == max(gains.values()) and f(ep[family, "A0"]["hop1_delta_difference"]) < 0)
    # TSM
    end = {(x["writer"], x["arm"]): x for x in rows(TSM + "endpoints.csv")}
    check("TSM gain holds in all four cells", all(flag(x["gain_passes"]) for x in end.values()))
    check("TSM stay holds only for joint_fact T1",
          [k for k, x in end.items() if flag(x["stay_passes"])] == [("joint_fact", "T1")]
          and f"{100 * f(end['joint_fact', 'T1']['stay_difference']):.2f}" == "-0.79")
    check("TSM far fails everywhere with upper ends below -1 pp",
          all(not flag(x["far_passes"]) and f(x["far_ci_high"]) < -0.01 for x in end.values()))
    for key in ("stay", "far", "gain"):
        check(f"TSM oracle_random T1 and T2 within 0.3 pp ({key})",
              abs(f(end["oracle_random", "T1"][f"{key}_difference"])
                  - f(end["oracle_random", "T2"][f"{key}_difference"])) < 0.003)
    desc = rows(TSM + "descriptive_contrasts.csv")
    for x in desc:
        if x["contrast"].startswith("hop1"):
            check(f"TSM hop-1 contrast negative with interval below 0 ({x['writer']} {x['arm']} {x['contrast']})",
                  f(x["ci_high"]) < 0)
        if x["contrast"] == "reach_l4_vs_t0_l4":
            check(f"TSM reach vs T0 negative ({x['writer']} {x['arm']})", f(x["difference"]) < 0)
        if x["contrast"] == "far_l16_vs_t0_l16":
            check(f"TSM far at L16 vs T0 sign ({x['writer']} {x['arm']})",
                  (f(x["difference"]) > 0) == (x["writer"] == "oracle_random"))
    gates = rows(TSM + "gate_profiles.csv")
    steps = [f"step_{k}" for k in range(1, 17)]

    def gate(writer: str, hop: int, step: str) -> float:
        picked = [x for x in gates if x["name"].startswith(f"tsm-{writer}-T2-") and int(x["hop"]) == hop]
        return sum(f(x[step]) for x in picked) / 3

    check("TSM oracle_random mean T2 gate at least 0.9999 everywhere",
          all(gate("oracle_random", h, k) >= 0.9999 for h in range(1, 9) for k in steps))
    check("TSM joint_fact step-1 gate 0.599 to 0.601 at every hop",
          all(f"{gate('joint_fact', h, 'step_1'):.3f}" in ("0.599", "0.600", "0.601") for h in range(1, 9)))
    check("TSM joint_fact step-16 gate between 0.28 and 0.33",
          all(0.275 <= gate("joint_fact", h, "step_16") < 0.335 for h in range(1, 9)))
    # UNS
    for part, excluding in (("A", {"joint_fact"}), ("B", {"joint", "joint_fact"})):
        index = {(x["writer"], x["endpoint"], x["group"], x["pool"]): x for x in rows(UNS[part] + "endpoints.csv")}
        for writer in UNS_WRITERS:
            d = index[writer, "m1_depth_gain", "reach", "unseen_minus_seen"]
            excludes = f(d["ci_low"]) > 0 or f(d["ci_high"]) < 0
            check(f"UNS Part {part} {writer} difference excludes 0 = {writer in excluding}",
                  excludes == (writer in excluding))
            if excludes:
                check(f"UNS Part {part} {writer} unseen gain smaller", f(d["difference"]) < 0)
    index = {(x["writer"], x["endpoint"], x["group"], x["pool"]): x for x in rows(UNS["A"] + "endpoints.csv")}
    for writer, sign in (("joint", 1), ("joint_fact", -1), ("oracle_random", -1)):
        for pool in ("seen", "unseen"):
            check(f"UNS Part A hop-1 gain sign ({writer}, {pool})",
                  f(index[writer, "m1_depth_gain", "hop1", pool]["difference"]) * sign > 0)
    # NCR
    cat, acc2 = ncr_index()
    for hop in ("2", "3"):
        for reader in NCR_READERS:
            for dataset in NCR_SETS:
                x = cat["oracle_orth", reader, dataset, hop, "first_hop"]
                if reader == "B4":
                    check(f"NCR oracle_orth B4 first-hop share 100 ({hop}, {dataset})",
                          pc(f(x["share"])) == "100.0" and pc(f(x["share_ci_low"])) == "100.0")
                    check(f"NCR oracle_orth B4 start 0.0 ({hop}, {dataset})",
                          pc(f(cat["oracle_orth", reader, dataset, hop, "start"]["share"])) == "0.0")
                else:
                    check(f"NCR oracle_orth M1 first-hop about 59 and start about 41 ({hop}, {dataset})",
                          57 <= 100 * f(x["share"]) <= 60 and 40 <= 100 * f(cat["oracle_orth", reader, dataset, hop,
                                                                                "start"]["share"]) <= 43)
                    other = 100 * f(cat["oracle_orth", reader, dataset, hop, "other"]["share"])
                    check(f"NCR oracle_orth M1 other at most 0.1 ({hop}, {dataset})",
                          other <= 0.15 and (hop == "3" or round(other, 1) == 0.0))
                for writer in ("joint", "t1_frozen"):
                    x = cat[writer, reader, dataset, hop, "first_hop"]
                    above = f(x["share"]) > f(x["uniform_share"])
                    check(f"NCR {writer} {reader} first-hop vs uniform ({hop}, {dataset})",
                          above == (reader == "M1" or writer == "t1_frozen"))
                for writer in ("t1_frozen_fact", "joint_fact", "oracle_random"):
                    other = 100 * f(cat[writer, reader, dataset, hop, "other"]["share"])
                    check(f"NCR {writer} other share far below uniform ({hop}, {dataset}, {reader})", other < 15)
                for reader_hop_writer in ("joint",):
                    shares = {c: f(cat[reader_hop_writer, reader, dataset, hop, c]["share"])
                              for c in ("first_hop", "start", "terminal_other", "other")}
                    check(f"NCR joint terminal_other largest ({reader}, {hop}, {dataset})",
                          max(shares, key=shares.get) == "terminal_other")
    for dataset in NCR_SETS:
        for reader in NCR_READERS:
            for writer in ("joint", "t1_frozen"):
                check(f"NCR {writer} above chance at hop 2",
                      not flag(acc2[writer, reader, dataset, "2"]["below_chance"]))
    # numbers and statements of the prose of the reports
    train = rows(TSM + "training.csv")
    params = {(x["writer"], x["arm"]): x["trainable_parameters"] for x in train}
    check("TSM parameters 4096/4098 and 13314/13316",
          [params["oracle_random", "T1"], params["oracle_random", "T2"], params["joint_fact", "T1"],
           params["joint_fact", "T2"]] == ["4096", "4098", "13314", "13316"])
    check("TSM 18 runs complete at attempt 1 with 15000 steps",
          len(train) == 18 and all(x["status"] == "complete" and x["attempt"] == "1" and x["steps"] == "15000"
                                   for x in train))
    summary = jload(NCR + "summary.json")
    check("NCR counts", summary["verified_directories"] == 72 == 6 * 2 * 3 * 2
          and summary["verified_rows"] == 72 * 2 * 2000 and summary["missing"] == []
          and summary["undefined"] == [] and summary["groups"] == 48)
    cvh = jload(CVH + "summary.json")
    check("CVH no failures, one passing arm, four latency-condition arms",
          cvh["failures"] == [] and cvh["passing"] == ["oracle_random/r32/M1/A1"] and len(cvh["latency_claims"]) == 4)
    check("CVH all validation choices feasible", all(flag(x["feasible"]) for x in rows(CVH + "theta.csv")))
    check("CVH A0 reproduced", all(x["reproduced"] for x in jload("reports/cvh/theta.json")["a0_reproduction"])
          and jload("reports/cvh/theta.json")["confirmation_allowed"] is True)
    check("UNS Part A all 18 reproduced", len(rows(UNS["A"] + "reproduction.csv")) == 18
          and all(flag(x["reproduced"]) for x in rows(UNS["A"] + "reproduction.csv")))
    check("UNS Part B seed 999006", jload("reports/uns/design.json")["extension"]["seed"] == 999006)
    check("IVD design lock commit", jload(IVD + "summary.json")["design_lock_commit"]
          == "2a8d063dbd29af4018c8d32e44686515971ee2ef")
    check("TSM design lock commit", jload(TSM + "summary.json")["design_lock_commit"]
          == "1bf689be3589c52da21dc309cff0f8d7668357a9")
    check("CVH lock commits", cvh["design_lock_commit"] == "8a13c45d2207f0c6d78091d18645c6e5cbdc9d77"
          and cvh["theta_lock_commit"] == "19c56137c122d1277103070d390dfe1e16fa624e")
    check("UNS Part B lock commit", jload(UNS["B"] + "summary.json")["design_lock_commit"]
          == "a8369284f11f8aea556437ebadf51e2f9076ef60")
    return failed


# --------------------------------------------------------------------------------------------- blocks and quotes
BLOCK = re.compile(r"(<!-- table:(?P<id>[a-z0-9-]+) -->\n)(?P<body>.*?)(\n<!-- /table:(?P=id) -->)", re.S)
QUOTE = re.compile(r"<!-- quote:(?P<path>[^ ]+) -->\n(?P<body>.*?)\n<!-- /quote -->", re.S)


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def report_path(study: str) -> Path:
    return REPORT_DIR / f"{study}.md"


def process(write: bool, only: str | None) -> list[str]:
    problems: list[str] = []
    skipped: list[str] = []
    built: dict[str, str] = {}
    for table_id, (study, builder) in BUILDERS.items():
        if only and table_id != only:
            continue
        try:
            built[table_id] = builder()
        except SourceMissing as error:
            skipped.append(f"{table_id}: source {error} is not present; table not checked")
    if only:
        print(built[only])
        return problems
    for study in STUDIES:
        path = report_path(study)
        if not path.exists():
            problems.append(f"{path.name}: missing")
            continue
        text = path.read_text()
        present = {m.group("id") for m in BLOCK.finditer(text)}
        wanted = {i for i, (s, _) in BUILDERS.items() if s == study}
        for table_id in sorted(wanted - present):
            problems.append(f"{path.name}: no block for {table_id}")
        for table_id in sorted(present - wanted):
            problems.append(f"{path.name}: block {table_id} has no builder")

        def replace(match: re.Match) -> str:
            table_id = match.group("id")
            if table_id not in built:
                return match.group(0)
            if match.group("body") != built[table_id]:
                if write:
                    return match.group(1) + built[table_id] + match.group(4)
                problems.append(f"{path.name}: block {table_id} differs from the regenerated table")
            return match.group(0)

        new = BLOCK.sub(replace, text)
        if write and new != text:
            path.write_text(new)
        for match in QUOTE.finditer(new):
            quoted = collapse("\n".join(re.sub(r"^> ?", "", line) for line in match.group("body").splitlines()))
            source = collapse((ROOT / match.group("path")).read_text())
            if not quoted or quoted not in source:
                problems.append(f"{path.name}: quote not found in {match.group('path')}: {quoted[:70]!r}")
    summaries: dict[str, str] = {}
    for table_id, summary, key, needle in EVIDENCE:
        if table_id not in built:
            continue
        text = summaries.setdefault(summary, (ROOT / summary).read_text())
        scope = section(text, key) if key else text
        if needle not in scope:
            problems.append(f"{table_id}: not in {summary}" + (f" ({key})" if key else "") + f": {needle[:90]}")
    problems.extend(observation_checks())
    for line in skipped:
        print("note:", line)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="replace the table blocks in the reports")
    parser.add_argument("--check", action="store_true", help="check the reports (default)")
    parser.add_argument("--print", dest="only", metavar="ID", help="print one table")
    args = parser.parse_args()
    problems = process(args.write, args.only)
    for problem in problems:
        print("problem:", problem, file=sys.stderr)
    if not problems and not args.only:
        count = len(EVIDENCE)
        print(f"ok: {len(BUILDERS)} tables, {count} evidence strings")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
