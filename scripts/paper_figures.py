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
IS_DESIGN = "reports/is/design.json"
IS_ACCURACY = "reports/is/accuracy.csv"
IS_ENDPOINTS = "reports/is/endpoints.csv"
JS_DESIGN = "reports/js/design.json"
JS_ACCURACY = "reports/js/accuracy.csv"
JS_ENDPOINTS = "reports/js/endpoints.csv"
JS_TRAINING = "reports/js/training_manifest.csv"
JS_CHECKPOINTS = "reports/js/checkpoint_manifest.csv"
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

    def memory_load_depth(self) -> None:
        design = self.data.load(IS_DESIGN)
        accuracy = self.data.load(IS_ACCURACY)
        endpoints = self.data.load(IS_ENDPOINTS)
        if (design["confirmation"] != {"after_all_36_training": True, "episodes": 2000,
                                      "seed": 999001, "split": "test"}
                or design["r"] != 32 or design["r_mem"] != 32
                or design["seeds"] != SEEDS):
            raise ValueError("IS figure expects the registered rank32 C999 three-seed design")
        layouts = {int(row["E"]): row["chain_lengths"] for row in design["layouts"]}
        loads, depths = [6, 12, 18, 24, 36], list(range(1, 9))
        if set(layouts) != set(loads):
            raise ValueError("IS load/layout coverage differs from the registered five loads")
        if design["same_checkpoint_depths"]["M1_mixed"] != depths:
            raise ValueError("IS mixed-depth checkpoint sweep must contain L1 through L8")
        fig, axes = self.plt.subplots(2, 2, figsize=(10.2, 7.0), sharex="row", sharey="row")
        colors = self.plt.get_cmap("tab10").colors
        points, selectors, checkpoint_ids = [], [], []
        for column, writer in enumerate(("oracle_random", "joint_fact")):
            selector = dict(writer=writer, reader="M1-mixed", variant="M1",
                            train_E=12, update="False")
            rows = select(accuracy, **selector)
            references = [row for row in design["checkpoints"]
                          if row["writer"] == writer and row["reader"] == "M1-mixed"
                          and row["train_E"] == 12]
            if sorted(row["seed"] for row in references) != SEEDS:
                raise ValueError(f"IS has no unique fixed E12 checkpoint per seed for {writer}")
            checkpoint_ids.extend({key: row[key] for key in (
                "name", "writer", "reader", "seed", "train_E", "checkpoint_hash", "config_hash")}
                for row in references)
            selectors.append(dict(source=IS_ACCURACY, selector=dict(
                **selector, seed=SEEDS, E=loads, base_E="equal to E", hop=[2, 3], L=depths)))
            d_points = []
            for index, load in enumerate(loads):
                means = []
                for depth in depths:
                    seed_points = []
                    for seed in SEEDS:
                        cells = [one(rows, seed=seed, E=load, base_E=load, hop=hop, L=depth)
                                 for hop in (2, 3)]
                        if any(int(row["n"]) != 2000 for row in cells):
                            raise ValueError("IS curve cells must contain2000 episodes per seed and hop")
                        mean = sum(float(row["accuracy"]) for row in cells) / 2
                        seed_points.append(dict(seed=seed, accuracy=mean, source_cells=cells))
                    mean = sum(row["accuracy"] for row in seed_points) / len(SEEDS)
                    means.append(100 * mean)
                    points.append(dict(kind="absolute_reach_curve", source=IS_ACCURACY,
                        writer=writer, reader="M1-mixed", train_E=12, E=load,
                        chain_lengths=layouts[load], L=depth, hops=[2, 3],
                        training_seeds=SEEDS, accuracy=mean, n_queries=12000,
                        ci_available=False, per_seed=seed_points))
                chain_label = ",".join(str(length) for length in layouts[load])
                axes[0, column].plot(depths, means, color=colors[index], marker="o",
                    markersize=3.5, linewidth=1.6, label=f"E={load}: ({chain_label})")
                endpoint_selector = dict(endpoint="IS-E1-D", writer=writer, reader="M1-mixed",
                                         train_E=12, E=load, subset="reach", confidence="0.975")
                endpoint = one(endpoints, **endpoint_selector)
                if (endpoint["descriptive"] != "False" or int(endpoint["n_seeds"]) != 3
                        or int(endpoint["n_episodes_per_load"]) != 2000):
                    raise ValueError("IS depth-effect interval has unexpected inference scope")
                middle, low, high = (100 * float(endpoint[key])
                                    for key in ("difference", "ci_low", "ci_high"))
                axes[1, column].errorbar(load, middle, yerr=[[middle - low], [high - middle]],
                    fmt="o", color=colors[index], capsize=3, markersize=5)
                d_points.append((load, middle))
                selectors.append(dict(source=IS_ENDPOINTS, selector=endpoint_selector))
                points.append(dict(kind="registered_depth_effect", source=IS_ENDPOINTS,
                                   source_row=endpoint, difference_pp=middle,
                                   ci_low_pp=low, ci_high_pp=high))
            axes[0, column].axvspan(1, 4, color="0.85", alpha=.3, zorder=-1)
            axes[0, column].set(title="Random oracle · M1-mixed" if writer == "oracle_random"
                                else "Factorized joint · M1-mixed",
                                xlabel="Reader updates L", xticks=depths, ylim=(-2, 102))
            axes[1, column].plot([point[0] for point in d_points], [point[1] for point in d_points],
                                color="0.6", linewidth=.8, zorder=0)
            axes[1, column].axhline(0, color="0.5", linewidth=.8)
            axes[1, column].set(title="Registered depth contrast: L4 minus L1",
                                xlabel="Stored facts E (chain layout also changes)", xticks=loads)
            for row in (0, 1):
                axes[row, column].grid(axis="y", color="0.91", linewidth=.6)
        axes[0, 0].set_ylabel("Candidate accuracy, hops 2–3 (%)")
        axes[1, 0].set_ylabel("Depth effect on hops 2–3 (pp)")
        fig.suptitle("Memory load and reader depth: fixed E12-trained checkpoints, rank 32", y=1.01, fontsize=12)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3,
                   bbox_to_anchor=(.5, -.015), frameon=False, title="Evaluation load and chain lengths")
        fig.tight_layout(rect=(0, .13, 1, 1))
        fig.text(.5, -.08, "C999 (evaluation seed 999001); equal means over seeds 17/29/43 and hops 2–3. "
                 "The same checkpoint per seed is used at every load.\n"
                 "Top: absolute means, no supplied CIs; shaded training-depth range (sampled L=1, 2, 4). "
                 "Bottom: committed 97.5% paired contrast CIs.\n"
                 "Load, chain layout and candidate count covary; this is not an isolated interference intervention.",
                 ha="center", fontsize=8)
        selectors.append(dict(source=IS_DESIGN, selector=dict(
            fields=["confirmation", "r", "r_mem", "seeds", "layouts", "same_checkpoint_depths"],
            checkpoints=checkpoint_ids)))
        self.save(fig, "memory_load_depth", claims=["IS1"],
                  scope="C999; fixed E12-trained checkpoint per seed across loads; absolute means and existing 97.5% contrasts",
                  selectors=selectors,
                  notes=["Plan9.3 item4 complete; no new evaluation, bootstrap or confidence interval.",
                         "Rank32 M1-mixed, training15000 steps on E12, training hops1–3, sampled L1/2/4.",
                         "Absolute means equally weight seeds17/29/43 and hops2/3; full per-seed source cells retained.",
                         "Top curves have no supplied CIs. Bottom bars reuse IS-E1-D,confidence.975 exactly.",
                         "The paired intervals condition on the three training seeds; load comparisons are not fresh-seed replications.",
                         "Evaluation E6/12/18/24/36 also changes chain lengths and candidate count; do not attribute changes solely to key interference.",
                         "No E24/E36-trained models, fixed-L4 retraining, updated-fact episodes, or descriptive joint-writer rows are included."],
                  points=points)

    def js_overshoot(self) -> None:
        design = self.data.load(JS_DESIGN)
        rows = self.data.load(JS_ACCURACY)
        endpoints = self.data.load(JS_ENDPOINTS)
        training = self.data.load(JS_TRAINING)
        checkpoints = self.data.load(JS_CHECKPOINTS)
        depths = list(range(1, 17))
        confirmation = design["confirmation"]
        if (confirmation["seed"] != 999003 or confirmation["episodes"] != 2000
                or confirmation["hops"] != list(range(1, 9))
                or confirmation["depths"] != depths or design["seeds"] != SEEDS
                or design["model"]["r"] != 32 or design["model"]["r_mem"] != 32):
            raise ValueError("JS figure requires the committed rank 32 C999003 design with three training seeds")
        if (design["training"]["steps"] != 15000
                or design["training"]["loop_sampling"] != "one independent Uniform{4,5,6,7,8} draw per batch"
                or design["penalty"]["gamma"] != .9 or design["penalty"]["lambda_J"] != 1
                or design["penalty"]["method"]["method"] != "exact_eigvals"):
            raise ValueError("JS random-depth/exact-Jacobian penalty protocol differs from the figure")
        styles = [("RS", "RS: sampled L = 4–8", "#D55E00", "-"),
                  ("RS+J", "RS+J: sampled L = 4–8 + penalty", "#0072B2", "-"),
                  ("M1-L4", "M1-L4 reference", "#009E73", "--"),
                  ("M1-mixed", "M1-mixed reference", "0.4", ":")]
        subsets = [("1–3", list(range(1, 4))), ("4–8", list(range(4, 9)))]
        fig, axes = self.plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True, sharey="col")
        points, selectors, checkpoint_ids = [], [], []
        column_maxima = [0., 0.]
        for index, writer in enumerate(("oracle_random", "joint_fact")):
            writer_label = "Random oracle" if writer == "oracle_random" else "Factorized joint"
            for condition, label, color, linestyle in styles:
                for seed in SEEDS:
                    checkpoint = one(checkpoints, writer=writer, condition=condition, seed=seed)
                    checkpoint_ids.append({key: checkpoint[key] for key in
                                           ("name", "writer", "condition", "seed", "checkpoint_hash")})
                    if condition in ("RS", "RS+J"):
                        entry = one(training, writer=writer, condition=condition, seed=seed)
                        if (entry["steps"] != "15000" or entry["batch_size"] != "64"
                                or entry["lr"] != "0.003"):
                            raise ValueError("JS new conditions require 15,000 steps, batch size 64 and learning rate 0.003")
                for column, (hop_label, hops) in enumerate(subsets):
                    means = []
                    selectors.append(dict(source=JS_ACCURACY, selector=dict(
                        writer=writer, condition=condition, seed=SEEDS, hop=hops, L=depths)))
                    for depth in depths:
                        per_seed = []
                        for seed in SEEDS:
                            cells = [one(rows, writer=writer, condition=condition,
                                         seed=seed, hop=hop, L=depth) for hop in hops]
                            if any(cell["n"] != "2000" for cell in cells):
                                raise ValueError("JS accuracy cells must contain 2,000 episodes per seed and hop")
                            per_seed.append(dict(seed=seed, source_cells=cells,
                                accuracy=sum(float(cell["accuracy"]) for cell in cells) / len(hops)))
                        mean = sum(point["accuracy"] for point in per_seed) / len(SEEDS)
                        means.append(100 * mean)
                        points.append(dict(kind="absolute_depth_profile", source=JS_ACCURACY,
                            writer=writer, condition=condition, L=depth, hops=hops,
                            accuracy=mean, n_queries=2000 * len(SEEDS) * len(hops),
                            ci_available=False, training_seeds=SEEDS, per_seed=per_seed))
                    column_maxima[column] = max(column_maxima[column], max(means))
                    axes[index, column].plot(depths, means, label=label, color=color,
                        linestyle=linestyle, linewidth=1.9 if condition == "RS+J" else 1.5,
                        marker="o" if condition in ("RS", "RS+J") else None,
                        markersize=2.5, markevery=[0, 3, 7, 11, 15])
                    suffix = " (zoomed)" if column == 1 else ""
                    axes[index, column].set(title=f"{writer_label}: hops {hop_label}{suffix}")
            short_summary = ["S = A(16) − A(4), hops 1–3 (pp)"]
            for condition in ("RS", "RS+J"):
                selector = dict(endpoint="JS-E1-S", writer=writer, condition=condition,
                                comparator=condition, L_a=16, L_b=4, hops="1-3", confidence="0.975")
                endpoint = one(endpoints, **selector)
                value, low, high = (100 * float(endpoint[key])
                                    for key in ("difference", "ci_low", "ci_high"))
                short_summary.append(f"{condition}: {value:+.2f} [{low:+.2f}, {high:+.2f}]")
                points.append(dict(kind="registered_overshoot", source=JS_ENDPOINTS, source_row=endpoint))
                selectors.append(dict(source=JS_ENDPOINTS, selector=selector))
            axes[index, 0].text(.98, .96, "\n".join(short_summary), transform=axes[index, 0].transAxes,
                va="top", ha="right", fontsize=7.3,
                bbox=dict(boxstyle="round,pad=.4", facecolor="white", alpha=.9, edgecolor="0.85"))
            selector = dict(endpoint="JS-E3", writer=writer, condition="RS+J", comparator="RS",
                            L_a=16, L_b=16, hops="4-8", confidence="0.975")
            endpoint = one(endpoints, **selector)
            value, low, high = (100 * float(endpoint[key])
                                for key in ("difference", "ci_low", "ci_high"))
            axes[index, 1].text(.98, .96,
                f"L = 16, RS+J − RS (pp)\n{value:+.2f} [{low:+.2f}, {high:+.2f}]",
                transform=axes[index, 1].transAxes, va="top", ha="right", fontsize=7.3,
                bbox=dict(boxstyle="round,pad=.4", facecolor="white", alpha=.9, edgecolor="0.85"))
            points.append(dict(kind="registered_long_hop_contrast", source=JS_ENDPOINTS, source_row=endpoint))
            selectors.append(dict(source=JS_ENDPOINTS, selector=selector))
            for ax in axes[index]:
                ax.axvspan(4, 8, color="0.85", alpha=.3, zorder=-1)
                ax.set(xlim=(.7, 16.3), xticks=[1, 4, 8, 12, 16],
                       ylabel="Candidate accuracy (%)")
                ax.tick_params(axis="y", labelleft=True)
                ax.grid(axis="y", color="0.91", linewidth=.6)
        # Leave room for the callouts while sharing each hop subset's scale across writers.
        long_limit = min(102, max(10, 5 * (int(1.2 * column_maxima[1] / 5) + 1)))
        for index in range(2):
            axes[index, 0].set_ylim(0, 102)
            axes[index, 1].set_ylim(0, long_limit)
        for ax in axes[1]:
            ax.set_xlabel("Reader updates L")
        fig.suptitle("JS: depth profiles with a final-state Jacobian penalty", y=1.01, fontsize=12)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2,
                   bbox_to_anchor=(.5, -.015), frameon=False)
        fig.tight_layout(rect=(0, .09, 1, 1))
        fig.text(.5, -.14, "C999003: test seed 999003; equal means over training seeds 17/29/43 and the stated hops. "
                 "Curves: no supplied absolute CIs; callouts: existing 97.5% paired intervals.\n"
                 "RS/RS+J: 15,000 steps, sampled L = 4–8 (shaded). References: fixed L = 4 or mixed L = 1/2/4. "
                 "All models: rank 32, shared-step M1.\n"
                 "RS+J penalizes the exact Jacobian spectral radius at reached z_L (gamma = 0.9, weight = 1); "
                 "this is not a contraction or answer-correctness certificate.\n"
                 "Y scales differ by column, start at zero, and are shared across writers; long-hop panels are zoomed.",
                 ha="center", fontsize=8)
        selectors.extend([
            dict(source=JS_DESIGN, selector=dict(fields=["confirmation", "seeds", "model", "training", "penalty"])),
            dict(source=JS_TRAINING, selector=dict(writer=["oracle_random", "joint_fact"],
                 condition=["RS", "RS+J"], seed=SEEDS, steps=15000, batch_size=64)),
            dict(source=JS_CHECKPOINTS, selector=dict(checkpoints=checkpoint_ids))])
        self.save(fig, "js_overshoot", claims=["JS1", "JS3", "JS5"],
            scope="C999003; three fixed training seeds; absolute profiles and existing 97.5% contrasts",
            selectors=selectors,
            notes=["All scientific input comes from committed reports/js summaries; checkpoint paths are never opened.",
                   "Two writers × short hops 1–3/long hops 4–8; each curve equally averages seeds 17/29/43 and its hop subset.",
                   "RS/RS+J use 15,000 steps, batch size 64, learning rate 0.003 and one uniform L = 4–8 draw per batch, with paired data/depth RNG streams.",
                   "M1-L4 and M1-mixed are unchanged registered references with distinct training-depth protocols.",
                   "Curves have no supplied absolute CIs; callouts reuse JS-E1-S and JS-E3 at confidence 0.975 exactly. No bootstrap rerun.",
                   "S is A(16) − A(4) on hops 1–3; a negative S denotes retained depth damage. Do not equate reduced overshoot with removal.",
                   "RS+J uses exact eigenvalues for the penalty relu(radius(J(z_L)) − 0.9)^2 with weight 1, not a fixed-point/contraction/correctness certificate.",
                   f"Y scales start at zero and are shared within each column: short hops 0–102%, long hops 0–{long_limit:g}%; the long-hop column is explicitly labeled zoomed.",
                   "Stopped-policy rows are excluded; no sentinel depth is treated as an extra read. No per-query radii or models are recomputed."],
            points=points)

    def write_provenance(self) -> None:
        manifest = {"source_revision": self.data.revision,
                    "scientific_input_policy": "git show resolved-commit:path; no workspace scientific inputs",
                    "training_seeds": SEEDS, "sources": self.data.sources,
                    "new_experiments": False, "new_bootstrap": False,
                    "limitations": [F2_LIMIT, "Absolute accuracy and latency have no supplied confidence intervals.",
                                    "P4 p95 columns are means of percentiles, not pooled tail estimates."],
                    "figures": self.entries,
                    "completed_plan_items": {"plan9.3_item4": "memory_load_depth"},
                    "placeholders": {"plan9.3_item6": "TODO: symbolic/end-to-end correspondence; P1-L pending"}}
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
                      "## Plan9.3 coverage and remaining placeholder", "",
                      "- Plan9.3 item4: **Complete — [memory load × L](memory_load_depth.png), [PDF](memory_load_depth.pdf).**",
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
    figures.memory_load_depth()
    figures.js_overshoot()
    figures.write_provenance()
    print(json.dumps({"status": "complete", "source_revision": data.revision,
                      "output": str(figures.destination), "figures": [entry["name"] for entry in figures.entries],
                      "limitations": [F2_LIMIT]}, sort_keys=True))


if __name__ == "__main__":
    main()
