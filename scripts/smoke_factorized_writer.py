#!/usr/bin/env python3
"""Bounded WP-C2 integration smokes; run only through scripts/submit.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from looplora.infra.seeds import derive_seed
from looplora.sym.run import run_one
from looplora.sym.train import load_checkpoint


def smoke(out: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("factorized writer smokes require a compute-node GPU")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for writer in ("t1_frozen_fact", "joint_fact"):
        frozen = writer == "t1_frozen_fact"
        config = {
            "name": f"wpc2-{writer}", "seed": 17, "writer": writer,
            "device": "cuda", "variant": "M1",
            "model": {"d_e": 64, "r": 32, "r_mem": 32},
            "task": {"name": "t2_chain_end", "E": 12, "Q": 4,
                     "n_rel": 4, "chain_lengths": [8, 4], "hops": [1, 2, 3]},
            "train": {"steps": 100 if frozen else 200, "batch_size": 64,
                      "log_every": 50},
            "pretrain": {"steps": 100, "batch_size": 64,
                         "task": {"E": 12, "n_rel": 4, "head_sharing": .3}},
            "eval": {"episodes": 8, "batch_size": 8, "depths": [0, 1, 2, 4, 8],
                     "hops": [1, 2, 3], "splits": ["val", "test", "test_unseen"],
                     "seed": derive_seed(17, "wpc2", "smoke-evaluation"),
                     "counterfactuals": True, "swap_successors": True},
        }
        run_dir = out / writer
        record = run_one(config, run_dir)
        assert record["status"] == "complete"
        assert record["runtime"]["device"] == "cuda"
        assert record["peak_memory_bytes"] > 0
        model = load_checkpoint(run_dir / "checkpoint.pt", device="cpu")
        assert all(p.requires_grad != frozen for p in model.writer.parameters())
        exported = torch.load(run_dir / "writer.pt", map_location="cpu", weights_only=True)
        assert exported["signature"]["writer_type"] == "factorized"
        for name, value in model.writer.state_dict().items():
            assert torch.equal(value, exported["writer"][name]), name
        if frozen:
            assert exported["pretrain_config"]["writer"] == "joint_fact"
            assert exported["pretrain"]["steps"] == 100
        with np.load(run_dir / "samples.npz", allow_pickle=False) as samples:
            assert set(samples["condition"]) == {
                "original", "replace_key_fact", "remove_required_fact", "swap_memory", "swap_successors",
            }
            assert set(samples["split"]) == {"val", "test", "test_unseen"}
            assert set(samples["L"]) == {0, 1, 2, 4, 8}
            sample_count = len(samples["answer"])
        rows.append({"writer": writer, "status": record["status"], "run_dir": str(run_dir),
                     "reader_steps": config["train"]["steps"], "pretrain_steps": 100 if frozen else 0,
                     "train_seconds": record["train_seconds"],
                     "pretrain_seconds": record["pretrain_seconds"],
                     "total_seconds": record["latency"]["total_seconds"],
                     "eval_samples": sample_count, "runtime": record["runtime"]})
    result = {"scope": "WP-C2 code acceptance only; no F2 pilot/main or selection", "runs": rows}
    (out / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = smoke(args.out.resolve())
    print(json.dumps({"status": "complete", "runs": len(result["runs"]),
                      "audit": str(args.out.resolve() / "audit.json")}))


if __name__ == "__main__":
    main()
