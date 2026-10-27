#!/usr/bin/env python3
"""Write configs/p1l-s/main/*: P1-L's r = 32 reader configs under the oracle code WP-L6 selected (stdlib, login-safe).

docs/prereg/P1L-S.md changes one thing relative to P1-L: the oracle writer W. Each file is P1-L's config with only
`name` and `writer` replaced; W is `selected` in the committed reports/wp-l6/decision.json (status must be `selected`).
B3-L4 is not written: it is not retrained. Existing files are never rewritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_p1l_main import ROOT, SEEDS, STUDIES, config, trained_labels

CODES = ("oracle_rot", "oracle_shared")


def make(decision: dict) -> dict[str, dict]:
    """The twelve configs, keyed by run name, for a WP-L6 decision (status selected)."""
    if decision.get("status") != "selected" or decision.get("selected") not in CODES:
        raise ValueError(f"P1L-S needs a WP-L6 decision with status selected and one of {CODES}")
    configs = {}
    for label in trained_labels(32, "p1l-s"):
        for seed in SEEDS:
            base = config(label, seed, 32, "p1l")
            if (base["writer"], base["oracle_seed"], base["model"]["r"]) != ("oracle_random", 17, 32):
                raise ValueError(f"P1-L config {label}-s{seed} is not an r = 32 oracle_random run with oracle seed 17")
            configs[f"{label}-s{seed}"] = {**base, "name": f"p1l-s-{label}-s{seed}", "writer": decision["selected"]}
    return configs


def write(configs: dict[str, dict]) -> list[Path]:
    directory = ROOT / STUDIES["p1l-s"]["configs"] / "main"
    texts = {directory / f"{name}.json": json.dumps(value, indent=2, allow_nan=False) + "\n"
             for name, value in configs.items()}
    for path, text in texts.items():
        if path.exists() and path.read_text() != text:
            raise FileExistsError(f"{path} exists and differs; configs are never rewritten once committed")
    directory.mkdir(parents=True, exist_ok=True)
    for path, text in texts.items():
        path.write_text(text)
    return list(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, default=ROOT / STUDIES["p1l-s"]["decision"])
    args = parser.parse_args()
    paths = write(make(json.loads(args.decision.read_text())))
    print(json.dumps(dict(written=len(paths), directory=str(paths[0].parent))))


if __name__ == "__main__":
    main()
