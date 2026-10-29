#!/usr/bin/env python3
"""SLOT-LOOP-v1 metadata on the login node: stage configs, the pending manifest and, after a passed preflight, the lock.

numpy and the standard library only (no torch, no model, no run). `looplora.llm.slot_data` and `slot_rules` re-derive every
registered integer (a mismatch is an error), the episode ids and their digests, and the disjointness of all registered sets and
train streams from the old protected ones; `inventory()` hashes the real parent files.

    /work/xg24i002/x10041/LoopLora/.venv/bin/python scripts/prepare_slot_manifest.py --write-configs configs/slot_loop_v1
    /work/xg24i002/x10041/LoopLora/.venv/bin/python scripts/prepare_slot_manifest.py --configs configs/slot_loop_v1 --out reports/slot-loop-v1/manifest.pending.json
    /work/xg24i002/x10041/LoopLora/.venv/bin/python scripts/prepare_slot_manifest.py --configs configs/slot_loop_v1 --preflight-report PREFLIGHT.json \\
        --out reports/slot-loop-v1/manifest.lock.json --inventory-out reports/slot-loop-v1/checkpoint_inventory.json

The lock is what `looplora.llm.slot_run` requires before any science mode runs: it refuses unless the preflight passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from looplora.infra.seeds import stable_hash  # noqa: E402
from looplora.llm import slot_data, slot_rules  # noqa: E402

STAGES = {"S0": "s0.json", "S1A": "s1a.json"}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


def write_configs(directory: Path) -> None:
    for stage, name in STAGES.items():
        dump(directory / name, slot_rules.stage_config(stage))


def check_configs(directory: Path) -> dict:
    """Each stage file equals the registered rendering; returns {stage: stable hash}."""
    hashes = {}
    for stage, name in STAGES.items():
        found, expected = json.loads((directory / name).read_text()), slot_rules.stage_config(stage)
        if found != expected:
            raise SystemExit(f"{directory / name} differs from the registered {stage} configuration")
        hashes[stage] = stable_hash(expected)
    return hashes


def pending(configs: Path) -> dict:
    if slot_data.file_digest(ROOT / slot_data.PREREG) != slot_data.PREREG_BLAKE2B_128:
        raise SystemExit("docs/prereg/SLOT-LOOP-v1.md differs from the registered file")
    return dict(version=1, status="pending", study=slot_data.STUDY, prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128,
                config_hashes=check_configs(configs), manifest=json.loads(json.dumps(slot_data.manifest())))


def lock(configs: Path, report_path: Path) -> tuple[dict, dict]:
    """The locked manifest and the checkpoint inventory, from a passed preflight report."""
    base = pending(configs)
    report = json.loads(report_path.read_text())
    inventory = slot_data.inventory()
    required = set(report.get("required", []))
    if report.get("status") != "complete" or report.get("passed") is not True or not required:
        raise SystemExit("the preflight did not pass: no lock")
    if any(report["checks"].get(name, {}).get("passed") is not True for name in required):
        raise SystemExit("a required preflight check did not pass: no lock")
    if report.get("inventory") != inventory or report.get("config_hash") != base["config_hashes"]["S0"]:
        raise SystemExit("the preflight ran on other parent files or another configuration than the ones now on disk")
    if report.get("source_blake2b_128") != slot_data.source_digest():
        raise SystemExit("the source differs from the source that passed the preflight: rerun preflight")
    locked = dict(base, status="locked", inventory=inventory,
                  preflight=dict(passed=True, source_blake2b_128=report["source_blake2b_128"], report_blake2b_128=slot_data.file_digest(report_path),
                                 job_id=report.get("job_id"), git_commit=report.get("git_commit"),
                                 checks={name: report["checks"][name]["passed"] for name in sorted(required)},
                                 matrix_max_abs_difference=report["checks"]["matrix_equals_native"]["max_abs_difference"]))
    replay = {name: report["checks"][key]["replayed"] for name, key in (("S-RE", "native_replay"), ("D128-3", "d128_3_replay"))}
    table = {
        "S-RE": dict(loader="looplora.llm.sq2.load_inference", format="read_fix_version 1 inference export (adapter, signature, config)",
                     files=inventory["S-RE"], registered="no: first digests recorded by this study (job SQ2 3724486.opbs wrote them)",
                     old_range_replay=replay["S-RE"], role="S0 parent; both S1A arms start from its writer.pt"),
        "D128-3": dict(loader="looplora.llm.dpfp_run.load_inference", format="read_fix_version 1 inference export",
                       files=inventory["D128-3"], registered=dict(slot_data.D128_3["digests"]), old_range_replay=replay["D128-3"],
                       role="S0 delta-store recheck"),
    }
    return locked, dict(version=1, study=slot_data.STUDY, parents=table)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-configs", type=Path, metavar="DIR")
    parser.add_argument("--configs", type=Path, metavar="DIR")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inventory-out", type=Path)
    args = parser.parse_args(argv)
    if args.write_configs:
        write_configs(args.write_configs)
        print(f"wrote {', '.join(STAGES.values())} to {args.write_configs}")
        return 0
    if not (args.configs and args.out):
        parser.error("--configs and --out are required")
    if args.preflight_report:
        record, table = lock(args.configs, args.preflight_report)
        if args.inventory_out:
            dump(args.inventory_out, table)
    else:
        record = pending(args.configs)
    dump(args.out, record)
    print(json.dumps(dict(status=record["status"], sets=len(record["manifest"]["sets"]), out=str(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
