#!/usr/bin/env python3
"""P1L-C evaluation: read-position and bridge-state patching of the P1-L r = 128 oracle readers.

Evaluation only (docs/prereg/P1L-C.md). The registered run needs the r = 128 confirmation to have
passed the Amendment-B7 memory-use check and reads test [102000, 104000), never the P1-L confirmation
range. ``--smoke`` is a validation-only timing and audit run on val [0, 64) that needs no authorization.
Invoke only through scripts/submit.py.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_p4s_stopping import require_compute
from looplora.data.relations import generate
from looplora.infra.seeds import stable_hash
from looplora.llm.config import recursive_merge, resolve_config
from looplora.llm.inject import load_model, load_tokenizer
from looplora.llm.p1lc import (PATCH_EPISODES, REGISTERED_RANGE, SMOKE_PATCH_EPISODES, SMOKE_RANGE,
                               bridge_patch_samples, read_position_samples, select_bridge_worlds)
from looplora.llm.train import load_checkpoint, make_task_spec
from looplora.sym.evaluate import stratify_queries
from looplora.sym.run import source_metadata

from evaluate_p1l_main import check_training
from run_p1l_main import ROOT, digest, runtime, save

READERS = ("M1-L4", "B8", "B4")
PATCH_READERS = ("M1-L4", "B8")


def reader_depth(label: str) -> int:
    """The depth each reader was trained and confirmed at."""
    return 1 if label == "B4" else 4


def evaluation_config(base: dict) -> tuple[dict, object]:
    """Resolved config and task spec of the P1-L confirmation, with confirmation data not allowed.

    The spec (hence every episode seed) must equal the one the confirmation evaluates; only the
    episode range and the eval splits differ.
    """
    template = json.loads((ROOT / "configs/p1l/main/eval-confirmation.json").read_text())
    confirmation = resolve_config(recursive_merge(base, template))
    cfg = resolve_config(recursive_merge(confirmation, dict(eval=dict(splits=["val"], allow_confirmation=False))))
    spec = make_task_spec(cfg, evaluation=True)
    if asdict(spec) != asdict(make_task_spec(confirmation, evaluation=True)):
        raise ValueError("P1L-C episodes must use the P1-L confirmation task spec")
    return cfg, spec


def evaluate_run(experiment, label: str, out: Path, *, window: dict, patch_episodes: int) -> dict:
    """Analysis A for every reader and analysis B for M1-L4 and B8; returns the audit/timing record."""
    cfg, spec = evaluation_config(copy.deepcopy(experiment.config))
    depth, batch_size, seed = reader_depth(label), cfg["eval"]["batch_size"], window["seed"]
    if experiment.config["train"]["depths"] != [depth]:
        raise ValueError(f"{label} was not trained at L={depth}")
    if cfg["eval"]["seed"] != seed:
        raise ValueError("the P1L-C evaluation seed must equal the P1-L confirmation seed")
    started = time.perf_counter()
    arrays = stratify_queries(generate(spec, window["split"], window["start"], window["count"], seed),
                              spec, cfg["eval"]["hops"], seed)
    record = dict(label=label, depth=depth, batch_size=batch_size, window=window,
                  spec_hash=stable_hash(asdict(spec)), episodes_hash=stable_hash(arrays.episode_seed.tolist()))
    samples, audit = read_position_samples(experiment, arrays, depth, batch_size=batch_size)
    np.savez_compressed(out / "read-position.npz", **samples)
    record.update(read_position_memory_hashes=audit["memory_hashes"],
                  read_position_seconds=time.perf_counter() - started)
    if label in PATCH_READERS:
        started = time.perf_counter()
        worlds = select_bridge_worlds(arrays, spec, first_index=window["start"], base_seed=seed,
                                      limit=patch_episodes)
        patch, audit = bridge_patch_samples(experiment, arrays, worlds, depth, batch_size=batch_size,
                                            base_seed=seed)
        np.savez_compressed(out / "bridge-patch.npz", **patch)
        record.update(patch_episodes=len(worlds), worlds_hash=audit["worlds_hash"],
                      bridge_memory_hashes=audit["memory_hashes"],
                      bridge_patch_seconds=time.perf_counter() - started)
    return record


def require_memory_use(endpoints: dict, authorization: dict) -> None:
    """The registered run exists only if the r = 128 confirmation, under this authorization, passed B7."""
    if (endpoints.get("rank") != 128 or endpoints.get("memory_use_passed") is not True
            or endpoints.get("authorization") != authorization):
        raise ValueError("P1L-C runs only if the r = 128 confirmation passed the Amendment-B7 memory-use check")


def check_runs(manifest: dict, names: list[str]) -> None:
    """Light provenance check for the smoke: the named checkpoints are complete and unchanged."""
    for name in names:
        run = Path(manifest["runs"][name])
        record = json.loads((run / "record.json").read_text())
        if record["status"] != "complete" or record["checkpoint_hash"] != digest(run / "checkpoint.pt"):
            raise ValueError(f"incomplete or changed main checkpoint: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="r = 128 training-manifest.json")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--endpoints", type=Path, help="reports/p1l/r128/endpoints.json of the confirmation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="e.g. M1-L4-s17 B8-s17 B4-s17")
    parser.add_argument("--smoke", action="store_true", help="validation-only timing and audit run")
    args = parser.parse_args()
    require_compute()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("rank") != 128:
        raise ValueError("P1L-C evaluates the r = 128 runs")
    for name in args.runs:
        if name not in manifest["runs"] or name.rsplit("-s", 1)[0] not in READERS:
            raise ValueError(f"not a P1L-C run: {name}")
    if args.smoke:
        if args.authorization or args.endpoints:
            parser.error("--smoke is validation-only and takes neither --authorization nor --endpoints")
        check_runs(manifest, args.runs)
        window, patch_episodes, provenance = SMOKE_RANGE, SMOKE_PATCH_EPISODES, {}
    else:
        if not (args.authorization and args.endpoints):
            parser.error("the registered run requires --authorization and --endpoints")
        authorization = json.loads(args.authorization.read_text())
        require_memory_use(json.loads(args.endpoints.read_text()), authorization)
        check_training(manifest, authorization, check_all=False)  # before any test episode is generated
        window, patch_episodes = REGISTERED_RANGE, PATCH_EPISODES
        provenance = dict(authorization_hash=digest(args.authorization), endpoints_hash=digest(args.endpoints))
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(), load_tokenizer()
    for name in args.runs:
        label, seed = name.rsplit("-s", 1)
        run = Path(manifest["runs"][name])
        out = args.out / name
        out.mkdir()
        started = time.perf_counter()
        experiment = load_checkpoint(run / "checkpoint.pt", model=model, tokenizer=tokenizer)
        try:
            record = evaluate_run(experiment, label, out, window=window, patch_episodes=patch_episodes)
        finally:
            experiment.close()
        hashes = {f"{key}_hash": digest(out / f"{file}.npz")
                  for key, file in (("read_position", "read-position"), ("bridge_patch", "bridge-patch"))
                  if (out / f"{file}.npz").exists()}
        save(out / "record.json", dict(status="complete", name=name, seed=int(seed), smoke=args.smoke,
            checkpoint_hash=digest(run / "checkpoint.pt"), manifest_hash=digest(args.manifest),
            seconds=time.perf_counter() - started, **provenance, **hashes, **record,
            **source_metadata(), **runtime()))
        print(json.dumps(dict(run=name, seconds=time.perf_counter() - started)), flush=True)


if __name__ == "__main__":
    main()
