"""Single-process generator throughput; run through scripts/submit.py on compute."""
from __future__ import annotations

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np

from looplora.data.relations import TaskSpec, generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.count < 1 or args.repeats < 1:
        parser.error("count and repeats must be positive")
    specs = [
        TaskSpec("t1"),
        TaskSpec("t2_chain_end", chain_lengths=(8, 8, 8)),
        TaskSpec("t2_walk", chain_lengths=(8, 8, 8)),
        TaskSpec("t2_typed", chain_lengths=(8, 8, 4), n_distract=4),
        TaskSpec("t3_update", base="t1", n_updates=2),
        TaskSpec("t3_update", base="t2_chain_end", chain_lengths=(8, 8, 6), n_updates=2),
        TaskSpec("t3_missing", base="t1"),
        TaskSpec("t3_missing", base="t2_walk", chain_lengths=(8, 8, 8)),
    ]
    print(json.dumps({"host": platform.node(), "python": platform.python_version(),
                      "numpy": np.__version__, "processes": 1, "E": 24, "Q": 4,
                      "count": args.count, "repeats": args.repeats}), flush=True)
    for spec in specs:
        generate(spec, "train", 0, 100, 2026)
        elapsed = []
        for _ in range(args.repeats):
            begin = perf_counter()
            arrays = generate(spec, "train", 0, args.count, 2026)
            elapsed.append(perf_counter() - begin)
            assert arrays.facts.shape == (args.count, spec.E, 3)
        print(json.dumps({"task": spec.name, "base": spec.base, "seconds": elapsed,
                          "episodes_per_second": args.count / median(elapsed)}), flush=True)


if __name__ == "__main__":
    main()
