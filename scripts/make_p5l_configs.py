#!/usr/bin/env python3
"""Write configs/p5l/main-r128/*: P1-L's r = 128 reader configs on Qwen3-1.7B-Base (stdlib, login-safe).

docs/prereg/P5L.md changes one thing relative to P1-L's r = 128 matrix: the backbone. Each file is P1-L's config with only
`name` replaced and `backbone` added; lr (0.003; 0.001 for B3-L4), S = 17,000 and every other field are carried unchanged.
Existing files are never rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = {"id": "Qwen/Qwen3-1.7B-Base", "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"}
LABELS = ("B8", "M1-L4", "B4", "B5-L4", "B3-L4")
SEEDS = (17, 29, 43)
SOURCE = "configs/p1l/main-r128"
TARGET = "configs/p5l/main-r128"


def make() -> dict[str, dict]:
    """The fifteen configs, keyed by run name."""
    configs = {}
    for label in LABELS:
        for seed in SEEDS:
            name = f"{label}-s{seed}"
            base = json.loads((ROOT / SOURCE / f"{name}.json").read_text())
            if (base["writer"], base["oracle_seed"], base["model"]["r"], base["train"]["steps"]) != ("oracle_random", 17, 128, 17000):
                raise ValueError(f"P1-L config {name} is not an r = 128 oracle_random run of 17,000 steps")
            configs[name] = {**base, "name": f"p5l-r128-{name}", "backbone": dict(BACKBONE)}
    return configs


def write(configs: dict[str, dict]) -> list[Path]:
    directory = ROOT / TARGET
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
    paths = write(make())
    print(json.dumps(dict(written=len(paths), directory=str(paths[0].parent))))


if __name__ == "__main__":
    main()
