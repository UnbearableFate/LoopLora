#!/usr/bin/env python3
"""Render WP-T5 figures from immutable, committed summary artifacts only.

Run on Miyabi-G through scripts/submit.py. This script never loads checkpoints,
run directories, sample traces, or uncommitted scientific input, and performs no
bootstrap or new experiment. See generated README.md and provenance.json.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P1_ACCURACY = "reports/p1s/accuracy_tables.csv"
F2_ACCURACY = "reports/p1s-f2/accuracy_summary.csv"
P1_GATES = "reports/p1s/gates.json"
F2_ENDPOINTS = "reports/p1s-f2/gate_endpoints.csv"
P2_COST = "reports/p2s/cost.csv"
P4_HEADLINE = "reports/p4s/headline.csv"
P4_LATENCY = "reports/p4s/latency.csv"
DEPTHS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
SEEDS = [17, 29, 43]
F2_LIMIT = (
    "F2 commits only three-seed mean hop/depth cells; per-checkpoint cells are "
    "not available in committed artifacts. Its panels explicitly use the "
    "three-seed mean, not a seed-17 checkpoint."
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


class CommittedData:
    def __init__(self, revision: str) -> None:
        self.revision = git("rev-parse", "--verify", f"{revision}^{{commit}}").strip()
        self.sources: dict[str, dict] = {}
        self.cache: dict[str, object] = {}

    def load(self, path: str):
        if path not in self.cache:
            # git show rejects missing/untracked inputs and ignores workspace edits.
            content = git("show", f"{self.revision}:{path}")
            if path.endswith(".csv"):
                value = list(csv.DictReader(io.StringIO(content)))
            elif path.endswith(".json"):
                value = json.loads(content)
            else:
                value = content
            additions = git("log", "--diff-filter=A", "--format=%H", self.revision,
                            "--", path).splitlines()
            self.sources[path] = {
                "revision": self.revision,
                "introducing_commit": additions[-1] if additions else None,
                "last_change_commit": git("log", "-1", "--format=%H", self.revision,
                                          "--", path).strip(),
                "git_blob": git("rev-parse", f"{self.revision}:{path}").strip(),
            }
            self.cache[path] = value
        return self.cache[path]


def select(rows: list[dict], **selector) -> list[dict]:
    return [row for row in rows if all(str(row.get(k)) == str(v) for k, v in selector.items())]


def one(rows: list[dict], **selector) -> dict:
    selected = select(rows, **selector)
    if len(selected) != 1:
        raise ValueError(f"Expected one committed row, found {len(selected)}: {selector}")
    return selected[0]


class Figures:
    def __init__(self, data: CommittedData, destination: Path, plt) -> None:
        self.data, self.destination, self.plt = data, destination, plt
        self.entries: list[dict] = []
        self.destination.mkdir(parents=True, exist_ok=True)

    def save(self, figure, name: str, *, claims: list[str], scope: str,
             selectors: list[dict], notes: list[str], points: list[dict]) -> None:
        evidence = self.data.load("docs/paper/evidence_symbolic.md")
        statuses = {}
        for claim in claims:
            rows = [line for line in evidence.splitlines() if line.startswith(f"| {claim}. ")]
            if len(rows) != 1:
                raise ValueError(f"Missing or duplicate evidence-ledger claim: {claim}")
            statuses[claim] = rows[0].split("|")[2].strip()
        for suffix in ("pdf", "png"):
            metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
            figure.savefig(self.destination / f"{name}.{suffix}", dpi=240,
                           bbox_inches="tight", facecolor="white", metadata=metadata)
        self.plt.close(figure)
        self.entries.append({"name": name, "files": [f"{name}.pdf", f"{name}.png"],
                             "claim_ids": claims, "claim_statuses": statuses, "scope": scope,
                             "selectors": selectors, "notes": notes, "points": points})

    def depth_data(self):
        p1_selector = dict(group="primary", writer="oracle_orth", variant="M1-lin",
                           seed=17, r=32, dataset="confirmation", split="test", condition="original")
        f2_selector = dict(group="primary", writer="joint_fact", variant="M1",
                           dataset="confirmation", split="test", condition="original")
        p1 = select(self.data.load(P1_ACCURACY), **p1_selector)
        f2 = select(self.data.load(F2_ACCURACY), **f2_selector)
        panels = []
        for path, selector, rows, title in [
            (P1_ACCURACY, p1_selector, p1, "P1-S oracle orthogonal / M1-lin\nOne checkpoint: seed 17, rank 32"),
            (F2_ACCURACY, f2_selector, f2, "F2 factorized joint / M1\nThree-seed mean, rank 32"),
        ]:
            matrix, points = [], []
            for hop in range(1, 9):
                cells = [one(rows, hop=hop, L=depth) for depth in DEPTHS]
                required_n = 2000 if path == P1_ACCURACY else 6000
                if any(int(row["n"]) != required_n for row in cells):
                    raise ValueError(f"Unexpected episode/seed scope in {path}")
                matrix.append([100 * float(row["accuracy"]) for row in cells])
                points.extend(dict(source=path, **row) for row in cells)
            panels.append((path, selector, title, matrix, points))
        return panels

    def depth_figures(self) -> None:
        panels = self.depth_data()
        colors = self.plt.get_cmap("tab10").colors
        fig, axes = self.plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True, sharey=True)
        for ax, (_, _, title, matrix, _) in zip(axes, panels):
            for index, values in enumerate(matrix):
                ax.plot(DEPTHS, values, color=colors[index], marker="o", markersize=3,
                        linewidth=1.55, label=f"Hop {index + 1}")
            ax.axvspan(1, 4, color="0.85", alpha=.35, zorder=-1)
            ax.set(title=title, xlabel="Reader updates L", ylim=(-2, 103),
                   xticks=[0, 1, 2, 4, 8, 12, 16])
            ax.grid(axis="y", color="0.9", linewidth=.6)
        axes[0].set_ylabel("Candidate accuracy (%)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=8,
                   bbox_to_anchor=(.5, -.025), frameon=False)
        fig.suptitle("Depth profiles on the fixed confirmation set C777", fontsize=12, y=1.02)
        fig.text(.5, -.105, "Shaded: training-depth range; sampled L=1, 2, 4. Training hops 1–3. "
                 "Absolute accuracy: no confidence interval. F2 is not one checkpoint.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, 1))
        selectors = [dict(source=p[0], selector=p[1]) for p in panels]
        notes = [F2_LIMIT, "P1-S is the exact seed-17 checkpoint; no seed averaging.",
                 "C777: test/eval seed777001; 2000 episodes; 15000 training steps; mixed L1/2/4.",
                 "Curves are descriptive absolute accuracies with no supplied CI."]
        points = [row for panel in panels for row in panel[4]]
        self.save(fig, "quality_depth", claims=["S7", "S9", "S12"], scope="C777 descriptive",
                  selectors=selectors, notes=notes, points=points)

        fig, axes = self.plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
        for ax, (_, _, title, matrix, _) in zip(axes, panels):
            image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
            ax.set(title=title, xlabel="Reader updates L", xticks=range(len(DEPTHS)),
                   xticklabels=DEPTHS, yticks=range(8), yticklabels=range(1, 9))
            for row in range(8):
                for col in range(len(DEPTHS)):
                    value = matrix[row][col]
                    ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=6.8,
                            color="white" if value < 48 else "black")
        axes[0].set_ylabel("True graph distance (hop)")
        fig.subplots_adjust(wspace=.12, right=.90, bottom=.18, top=.80)
        color_ax = fig.add_axes([.92, .20, .015, .58])
        fig.colorbar(image, cax=color_ax, label="Candidate accuracy (%)")
        fig.text(.5, .025, "Column spacing is categorical. P1: seed 17 only; "
                 "F2: three-seed mean. No cell-level confidence intervals.", ha="center", fontsize=8)
        self.save(fig, "hop_depth_heatmap", claims=["S7", "S9", "S12"],
                  scope="C777 descriptive", selectors=selectors, notes=notes, points=points)

    def paired_controls(self) -> None:
        p1 = self.data.load(P1_GATES)
        f2 = self.data.load(F2_ENDPOINTS)
        systems = [("oracle_orth", "M1", "Oracle orthogonal · M1"),
                   ("joint", "M1", "Joint · M1"),
                   ("joint", "M1-lin", "Joint · M1-lin"),
                   ("joint_fact", "M1", "Factorized joint · M1"),
                   ("t1_frozen_fact", "M1", "Factorized frozen · M1")]
        fig, ax = self.plt.subplots(figsize=(9.4, 4.3))
        points, selectors = [], []
        for index, (writer, reader, label) in enumerate(systems):
            path = F2_ENDPOINTS if writer.endswith("_fact") else P1_GATES
            for endpoint, color, marker, offset, comparator in [
                ("vs_fixed", "#0072B2", "o", -.13, "B5 (matched activation)"),
                ("vs_loop", "#D55E00", "s", .13, "B3 (memory-free)"),
            ]:
                selector = dict(writer=writer, reader=reader)
                if path == F2_ENDPOINTS:
                    selector.update(endpoint=endpoint, confidence="0.975")
                    row = one(f2, **selector)
                    value = {k: float(row[k]) for k in ("difference", "ci_low", "ci_high")}
                else:
                    gate = one(p1, **selector)
                    value = gate["statistics"][endpoint]["gate"]
                    if value["confidence"] != .975:
                        raise ValueError("The control plot requires the committed 97.5% interval")
                middle, low, high = (100 * value[k] for k in ("difference", "ci_low", "ci_high"))
                ax.errorbar(middle, index + offset, xerr=[[middle - low], [high - middle]],
                            fmt=marker, color=color, capsize=3, markersize=5,
                            label=comparator if index == 0 else None)
                selectors.append(dict(source=path, selector=selector,
                                      json_endpoint=endpoint if path == P1_GATES else None))
                points.append(dict(writer=writer, reader=reader, endpoint=endpoint,
                                   difference_pp=middle, ci_low_pp=low, ci_high_pp=high))
        ax.axvline(0, color="0.5", linewidth=.8)
        ax.set(yticks=range(len(systems)), yticklabels=[s[2] for s in systems],
               xlabel="Evolving-query minus control accuracy (percentage points)",
               title="Paired system contrasts at L=4, hops 2–3", xlim=(-2, 102))
        ax.invert_yaxis()
        ax.grid(axis="x", color="0.91", linewidth=.6)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout(rect=(0, .09, 1, 1))
        fig.text(.5, .015, "C777; 97.5% paired episode-bootstrap intervals, conditional on seeds 17/29/43.\n"
                 "Joint writer tensors differ across variants. Positive control contrasts do not imply a full gate pass.",
                 ha="center", fontsize=8)
        self.save(fig, "paired_controls", claims=["S2", "S3", "S4", "S9"],
                  scope="C777; existing 97.5% paired intervals only", selectors=selectors,
                  notes=["No bootstrap rerun. Registered intervals condition on the three training seeds.",
                         "Training15000 steps, mixed L1/2/4; comparison at L4 on reachable hops2/3.",
                         "joint and joint_fact comparisons are separately trained systems, not fixed-writer interventions."],
                  points=points)

    def budget_latency(self) -> None:
        rows = self.data.load(P2_COST)
        fig, axes = self.plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
        points = []
        for ax, writer in zip(axes, ("joint", "oracle_random")):
            eligible = select(rows, writer=writer, batch_size=64, steps=15000)
            iterative = [one(eligible, source="wph", variant="M1", r=32, r_mem=32, L=L)
                         for L in (1, 2, 4, 8)]
            ax.plot([float(r["cuda_ms_mean"]) for r in iterative],
                    [100 * float(r["accuracy_hops1to3"]) for r in iterative],
                    "o-", color="#0072B2", linewidth=1.5, markersize=5, label="M1 r32, multiple reads")
            for row in iterative:
                ax.annotate(f"L{row['L']}", (float(row["cuda_ms_mean"]), 100 * float(row["accuracy_hops1to3"])),
                            xytext=(4, 5), textcoords="offset points", fontsize=7)
            chosen = list(iterative)
            specs = [("B6", "^", "#D55E00", "Larger-rank single read"),
                     ("B7", "s", "#009E73", "Larger-state single read"),
                     ("B8", "D", "#CC79A7", "Unshared L4")]
            for variant, marker, color, label in specs:
                candidates = select(eligible, source="p2s", variant=variant,
                                    L=4 if variant == "B8" else 1)
                for index, row in enumerate(candidates):
                    x, y = float(row["cuda_ms_mean"]), 100 * float(row["accuracy_hops1to3"])
                    ax.scatter(x, y, s=52, marker=marker, color=color, edgecolor="white", linewidth=.4,
                               label=label if index == 0 else None, zorder=3)
                    detail = f"B6 r{row['r']}" if variant == "B6" else (
                        f"B7 m{row['r_mem']}" if variant == "B7" else "B8 L4")
                    ax.annotate(detail, (x, y), xytext=(4, -12 if index else 6),
                                textcoords="offset points", fontsize=7,
                                fontweight="bold" if variant == "B6" else "normal")
                    chosen.append(row)
            if {r["r"] for r in chosen if r["variant"] == "B6"} != {"64", "128"}:
                raise ValueError("Both registered larger-rank single-read controls must be present")
            points.extend(dict(row) for row in chosen)
            ax.set(xscale="log", ylim=(-3, 103), xlabel="CUDA mean latency (ms / prepared batch 64)",
                   title="Joint writing" if writer == "joint" else "Random oracle writing")
            ax.grid(color="0.92", linewidth=.6)
        axes[0].set_ylabel("Candidate accuracy, hops 1–3 (%)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
                   bbox_to_anchor=(.5, -.10), frameon=False)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.text(.5, -.15, "C888; three-seed means, no CIs. All models: 15,000 steps. "
                 "Prepared queries exclude writer/backbone costs. Static budgets differ.", ha="center", fontsize=8)
        self.save(fig, "budget_latency", claims=["B1", "B2", "B7"], scope="C888 descriptive cost points; no CI",
                  selectors=[dict(source=P2_COST, selector=dict(writer=["joint", "oracle_random"],
                              batch_size=64, steps=15000, source=["wph", "p2s"],
                              variants={"wph": ["M1"], "p2s": ["B6", "B7", "B8"]},
                              evaluation_L={"M1": [1, 2, 4, 8], "B6": [1], "B7": [1], "B8": [4]}))],
                  notes=["Quality is hops1–3, not the separate hops2–3 capability endpoint.",
                         "Excludes all F1/20000-step descriptive references.",
                         "Both B6 ranks64/128 are marked; joint B7 has memory64/128. No oracle B7 point was exported.",
                         "Joint Gate3 practical/capability failure remains a failure; this is not an end-to-end efficiency measurement."],
                  points=points)

    def stopping_latency(self) -> None:
        accuracies, times = self.data.load(P4_HEADLINE), self.data.load(P4_LATENCY)
        fig, axes = self.plt.subplots(2, 2, figsize=(10.2, 7.1), sharey=True)
        points, selectors = [], []
        styles = [("relative_change", "Relative change", "*", "#0072B2"),
                  ("read_norm", "Read norm", "^", "#009E73"),
                  ("output_stability", "Output stability", "s", "#D55E00")]
        for index, variant in enumerate(("M1", "M1-lin")):
            family_selector = dict(writer="oracle_random", r=128, variant=variant)
            acc = select(accuracies, **family_selector)
            timing = select(times, **family_selector, batch_size=64)
            selectors.extend([dict(source=P4_HEADLINE, selector=family_selector),
                              dict(source=P4_LATENCY, selector=dict(**family_selector, batch_size=64))])
            for column, (metric, title) in enumerate([
                ("cuda_ms_mean", "Mean CUDA latency"),
                ("cuda_ms_p95", "Mean of per-run p95 CUDA latencies"),
            ]):
                ax = axes[index, column]
                fixed = []
                for depth in (1, 2, 4, 8, 16):
                    method = f"fixed_{depth}"
                    quality, time = one(acc, method=method), one(timing, method=method)
                    x, y = float(time[metric]), 100 * float(quality["overall_accuracy"])
                    fixed.append((x, y))
                    ax.annotate(str(depth), (x, y), xytext=(3, 4), textcoords="offset points", fontsize=7)
                    if column == 0:
                        points.append(dict(family=family_selector, method=method, quality=quality, latency=time))
                ax.plot([p[0] for p in fixed], [p[1] for p in fixed], "o-", color="0.3",
                        markersize=4, linewidth=1.25, label="Fixed L (labels = updates)")
                for method, label, marker, color in styles:
                    quality, time = one(acc, method=method), one(timing, method=method)
                    ax.scatter(float(time[metric]), 100 * float(quality["overall_accuracy"]),
                               marker=marker, color=color, s=85 if marker == "*" else 45,
                               edgecolor="white", linewidth=.4, zorder=3, label=label)
                    if column == 0:
                        points.append(dict(family=family_selector, method=method, quality=quality, latency=time))
                ax.set(xscale="log", ylim=(0, 103), title=f"{variant}: {title}",
                       xlabel="ms / prepared batch 64")
                ax.grid(color="0.92", linewidth=.6)
                if column == 0:
                    ax.set_ylabel("Candidate accuracy, hops 1–8 (%)")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2,
                   bbox_to_anchor=(.5, -.015), frameon=False)
        fig.suptitle("Stopping pilot: random oracle, rank 128, three-seed means", y=1.01, fontsize=12)
        fig.tight_layout(rect=(0, .07, 1, 1))
        fig.text(.5, -.065, "S4242 screening; thresholds selected on V4242. No CIs. "
                 "Tail columns average per-run/query-hop p95s, not a pooled percentile.\n"
                 "Timing includes stopping bookkeeping and candidate prediction; excludes writer/backbone preparation.",
                 ha="center", fontsize=8)
        self.save(fig, "stopping_latency", claims=["H1", "H2"], scope="S4242 descriptive stopping pilot; no CI",
                  selectors=selectors,
                  notes=["Frozen F1 checkpoints trained20000 steps with mixed L1/2/4; no retraining.",
                         "Methods use validation-selected frozen thresholds; no screening selection or formal gate inference.",
                         "P95 is the committed mean of per-run/query-hop percentiles, not a pooled-query p95.",
                         "Five warmups,20 repeats,FP32 GH200. Read-norm costs one more read than updates.",
                         "Absolute quality is overall hops1–8, distinct from H1's long-hop summary."], points=points)

    def write_provenance(self) -> None:
        manifest = {"source_revision": self.data.revision,
                    "scientific_input_policy": "git show resolved-commit:path; no workspace scientific inputs",
                    "training_seeds": SEEDS, "sources": self.data.sources,
                    "new_experiments": False, "new_bootstrap": False,
                    "limitations": [F2_LIMIT, "Absolute accuracy and latency have no supplied confidence intervals.",
                                    "P4 p95 columns are means of percentiles, not pooled tail estimates."],
                    "figures": self.entries,
                    "placeholders": {"plan9.3_item4": "TODO: memory load × L; study IS pending",
                                     "plan9.3_item6": "TODO: symbolic/end-to-end correspondence; P1-L pending"}}
        (self.destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
        lines = ["# Symbolic paper figures", "", f"Committed source revision: `{self.data.revision}`.", "",
                 "Generated by `scripts/paper_figures.py`; all scientific input is read using `git show`.",
                 "No new experiment, bootstrap, confidence interval, or checkpoint evaluation is performed.", "",
                 "## Checkpoint limitation", "", F2_LIMIT, "",
                 "## Figures", ""]
        for entry in self.entries:
            lines.extend([f"### {entry['name']}", "",
                          f"[PDF]({entry['name']}.pdf) · [PNG]({entry['name']}.png)", "",
                          "Evidence claims: " + ", ".join(
                              f"{claim} ({entry['claim_statuses'][claim]})" for claim in entry['claim_ids'])
                          + f". Scope: {entry['scope']}.", ""])
            lines.extend(f"- {note}" for note in entry["notes"])
            lines.extend(["", "Selectors:", "", "```json", json.dumps(entry["selectors"], indent=2), "```", ""])
        lines.extend(["## Source provenance", "", "| Source | Introducing commit | Last change |", "|---|---|---|"])
        for path, source in sorted(self.data.sources.items()):
            lines.append(f"| `{path}` | `{source['introducing_commit']}` | `{source['last_change_commit']}` |")
        lines.extend(["", "The JSON manifest records exact plotted points, selectors, source Git objects, and claim scopes.",
                      "Git object IDs are repository provenance, not an independently computed data digest.", "",
                      "## Required placeholders", "", "- Plan9.3 item4: **TODO — memory load × L; study IS pending.**",
                      "- Plan9.3 item6: **TODO — symbolic/end-to-end correspondence; P1-L pending.**", ""])
        (self.destination / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD", help="Committed scientific-input revision (default HEAD)")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/paper/figures")
    args = parser.parse_args()
    data = CommittedData(args.revision)
    data.load("docs/paper/evidence_symbolic.md")
    # Plotting is deliberately imported only at execution, on the compute node.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"})
    figures = Figures(data, args.output.resolve(), plt)
    figures.depth_figures()
    figures.paired_controls()
    figures.budget_latency()
    figures.stopping_latency()
    figures.write_provenance()
    print(json.dumps({"status": "complete", "source_revision": data.revision,
                      "output": str(figures.destination), "figures": [entry["name"] for entry in figures.entries],
                      "limitations": [F2_LIMIT]}, sort_keys=True))


if __name__ == "__main__":
    main()
