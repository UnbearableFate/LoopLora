#!/usr/bin/env python3
"""Registered secondary cross-rank confirmation contrasts from existing samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_p4s_stopping import require_compute
from evaluate_p1l_main import check_training
from finalize_p1l_main import endpoint
from run_p1l_main import SEEDS, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r32-manifest", type=Path, required=True)
    parser.add_argument("--r128-manifest", type=Path, required=True)
    parser.add_argument("--r32-evaluations", type=Path, required=True)
    parser.add_argument("--r128-evaluations", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    authorization = json.loads(args.authorization.read_text())
    manifests, evaluations = {}, {}
    for rank in (32, 128):
        manifests[rank] = json.loads(getattr(args, f"r{rank}_manifest").read_text())
        check_training(manifests[rank], authorization)
        evaluations[rank] = json.loads(getattr(args, f"r{rank}_evaluations").read_text())["runs"]
    sources = {}
    for rank, label in ((32, "B8"), (32, "M1-L4"), (128, "B4"), (128, "B8")):
        sources[f"{label}-r{rank}"] = {seed: Path(evaluations[rank][f"{label}-s{seed}"])
                                             / "primary/samples.npz" for seed in SEEDS}
    confidence = 1 - .05 / authorization["gated_conditions"]
    results = {}
    for name, a, b, da, db in (
        ("B8r32_vs_B4r128", "B8-r32", "B4-r128", 4, 1),
        ("M1r32_vs_B4r128", "M1-L4-r32", "B4-r128", 4, 1),
        ("B8r128_vs_B8r32", "B8-r128", "B8-r32", 4, 4)):
        results[name] = endpoint(sources, a, b, depth_left=da, depth_right=db,
                                 hops=(2, 3), confidence=confidence)
    save(args.out, dict(secondary_only=True, confidence=confidence,
        steps={str(rank): manifests[rank]["S"] for rank in (32, 128)},
        unequal_steps_flag=manifests[32]["S"] != manifests[128]["S"], contrasts=results))


if __name__ == "__main__":
    main()
