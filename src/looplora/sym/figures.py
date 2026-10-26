"""Render symbolic accuracy curves, heatmaps and paired bootstrap contrasts.

Run on a compute node with ``PYTHONPATH=src``::

    python -m looplora.sym.figures --samples runs/m1/samples.npz
    python -m looplora.sym.figures --samples runs/m1/samples.npz \
        --reference runs/b5/samples.npz --depth-a 4 --depth-b 4 \
        --label-a M1 --label-b B5 --out reports/figures/p1s/m1-v-b5

Defaults to ``reports/figures/p1s``. Comparisons are explicit and never selected
from observed accuracy. Each split and intervention condition has its own plots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from looplora.sym.stats import SampleSource, compare_runs, depth_comparisons, load_samples


def create_figures(
    samples: SampleSource,
    out: str | Path = "reports/figures/p1s",
    *,
    reference: SampleSource | None = None,
    depth_a: int | None = None,
    depth_b: int | None = None,
    label_a: str = "A",
    label_b: str = "B",
    resamples: int = 2000,
    seed: int = 2718,
) -> list[Path]:
    """Save PNGs and exact paired-difference data; return all artifact paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = load_samples(samples)
    comparisons = depth_comparisons(arrays, resamples, seed)
    if reference is not None:
        if depth_a is None or depth_b is None:
            raise ValueError("A reference comparison requires depth_a and depth_b")
        comparisons.extend(compare_runs(arrays, reference, depth_a, depth_b, resamples, seed,
                                        label_a=label_a, label_b=label_b))
    elif depth_a is not None or depth_b is not None:
        raise ValueError("Comparison depths require a reference")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    def save(fig: Any, filename: str) -> None:
        path = output / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(path)

    groups = sorted(set(zip(arrays["split"].tolist(), arrays["condition"].tolist())))
    for group_index, (split, condition) in enumerate(groups):
        keep = (arrays["split"] == split) & (arrays["condition"] == condition)
        hops = sorted(int(h) for h in np.unique(arrays["hop"][keep]))
        depths = sorted(int(d) for d in np.unique(arrays["L"][keep]))
        table = np.full((len(hops), len(depths)), np.nan)
        for i, hop in enumerate(hops):
            for j, depth in enumerate(depths):
                rows = keep & (arrays["hop"] == hop) & (arrays["L"] == depth)
                if np.any(rows):
                    table[i, j] = arrays["correct"][rows].mean()
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{split}_{condition}")
        slug = f"{group_index:02d}_{slug}"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for hop, accuracy in zip(hops, table):
            ax.plot(depths, accuracy, marker="o", label=f"hop {hop}")
        ax.set(xlabel="Reader depth L", ylabel="Candidate-set accuracy", ylim=(0, 1),
               title=f"{split} / {condition}")
        ax.set_xticks(depths)
        ax.legend(ncols=2, fontsize="small")
        ax.grid(alpha=0.2)
        save(fig, f"accuracy_L_{slug}.png")

        fig, ax = plt.subplots(figsize=(7, max(3.5, 0.45 * len(hops))))
        image = ax.imshow(np.ma.masked_invalid(table), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(depths)), labels=depths)
        ax.set_yticks(range(len(hops)), labels=hops)
        ax.set(xlabel="Reader depth L", ylabel="Query hop", title=f"{split} / {condition}")
        fig.colorbar(image, ax=ax, label="Candidate-set accuracy")
        save(fig, f"hop_L_{slug}.png")

    if comparisons:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(comparisons))))
        for i, result in enumerate(comparisons):
            point = result["difference"]
            low, high = result["ci_low"], result["ci_high"]
            ax.hlines(i, low, high, color="tab:blue", linewidth=1.5)
            ax.plot(point, i, "o", color="tab:blue", markersize=4)
        labels = [f'{r["comparison"]} | {r["split"]}/{r["condition"]} | '
                  f'hop {r["hop"] if r["hop"] is not None else "all"}' for r in comparisons]
        ax.set_yticks(range(len(comparisons)), labels=labels, fontsize="small")
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Paired accuracy difference (95% episode bootstrap CI)",
               title="Prespecified paired comparisons")
        ax.grid(axis="x", alpha=0.2)
        save(fig, "paired_differences.png")
    path = output / "paired_differences.json"
    path.write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    artifacts.append(path)
    return artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/figures/p1s"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--depth-a", type=int)
    parser.add_argument("--depth-b", type=int)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args(argv)
    if args.reference is not None and (args.depth_a is None or args.depth_b is None):
        parser.error("--reference requires both --depth-a and --depth-b")
    if args.reference is None and (args.depth_a is not None or args.depth_b is not None):
        parser.error("Depth arguments require --reference")
    create_figures(args.samples, args.out, reference=args.reference,
                   depth_a=args.depth_a, depth_b=args.depth_b,
                   label_a=args.label_a, label_b=args.label_b,
                   resamples=args.resamples, seed=args.seed)


if __name__ == "__main__":
    main()
