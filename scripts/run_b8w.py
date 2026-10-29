#!/usr/bin/env python3
"""B8W continuation (docs/prereg/B8W.md); invoke on Miyabi-G only through scripts/submit.py.

One job per seed: the equality gate (the M1-L4 checkpoint and its B8 conversion give bitwise-identical validation
logits), then the two arms, each N continuation steps from the same starting function with the same data stream, lr and
schedule and a fresh optimizer: `shared` (the M1-L4 checkpoint continues) and `untied` (the converted B8 continues).
`smoke` is the validation-only timing run (gate for one seed and a few steps per arm; no test episode is generated);
`main` additionally requires the committed, locked reports/b8w/checkpoint-lock.json to agree with the checkpoint files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_p1l_main import ROOT, digest, runtime, save

SEEDS = (17, 29, 43)
ARMS = ("shared", "untied")
LOCK = "reports/b8w/checkpoint-lock.json"
PREREG = "docs/prereg/B8W.md"
P1L_MANIFEST = "reports/p1l/r128/run-manifest.json"
STEPS, DATA_START, LR = 2000, 17000, 0.003     # N, the P1-L stream length S, the M1-L4 lr of reports/p1l/r128/lr-S-lock.json


class GateFailed(RuntimeError):
    pass


def checkpoint_of(seed: int, label: str = "M1-L4", manifest: dict | None = None) -> tuple[Path, str]:
    """(checkpoint path, recorded hash) of a registered P1-L r = 128 checkpoint, from the P1-L run manifest."""
    manifest = manifest or json.loads((ROOT / P1L_MANIFEST).read_text())
    row = next(item for item in manifest["runs"] if item["name"] == f"{label}-s{seed}")
    return Path(row["run"]) / "checkpoint.pt", row["checkpoint_hash"]


def check_lock(lock: dict, seed: int) -> None:
    """The committed lock agrees with the registered prereg, this runner's constants and the checkpoint files."""
    if lock.get("status") != "locked" or lock.get("study") != "b8w":
        raise ValueError("main requires the committed, locked B8W checkpoint lock")
    if lock["prereg_hash"] != digest(ROOT / PREREG):
        raise ValueError("the lock names another prereg text than the one in the source tree")
    if (lock["steps"], lock["lr"], lock["data_start_step"]) != (STEPS, LR, DATA_START):
        raise ValueError("the lock disagrees with the registered steps, lr or stream start")
    for label in ("M1-L4", "B8", "B4"):
        entry = lock["checkpoints"][f"{label}-s{seed}"]
        if digest(Path(entry["path"])) != entry["hash"]:
            raise ValueError(f"{label}-s{seed}: the checkpoint file does not match the lock")


def gate(seed: int, start: Path, out: Path, model, tokenizer) -> dict:
    """Equality gate: M1-L4 (twice, as the control) against its B8 conversion on validation episodes; tolerance 0."""
    import torch
    from looplora.llm.b8w import compare_signatures, convert_m1_to_b8, gate_signature
    from looplora.llm.train import load_checkpoint

    record = torch.load(start, map_location="cpu", weights_only=True)
    converted = convert_m1_to_b8(record, name=f"b8w-converted-s{seed}")
    torch.save(converted, out / "converted.pt")
    signatures = []
    for path in (start, start, out / "converted.pt"):
        experiment = load_checkpoint(path, model=model, tokenizer=tokenizer)
        try:
            signatures.append(gate_signature(experiment))
        finally:
            experiment.close()
    result = compare_signatures(*signatures)
    result.update(seed=seed, start_checkpoint_hash=digest(start), converted_checkpoint_hash=digest(out / "converted.pt"))
    save(out / "gate.json", result)
    if not result["passed"]:
        raise GateFailed(f"equality gate failed for seed {seed}: max |dlogit| = {result['max_abs_difference']}")
    return result


def run_arm(arm: str, seed: int, start: Path, out: Path, model, tokenizer, *, steps: int, log_every: int,
            gate_result: dict) -> dict:
    import torch
    from looplora.infra.seeds import stable_hash
    from looplora.llm.train import continue_training, load_checkpoint
    from looplora.sym.evaluate import tensor_hash
    from looplora.sym.run import source_metadata

    target = out / arm
    experiment = load_checkpoint(start, model=model, tokenizer=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    try:
        started = digest(start)
        options = dict(steps=steps, lr=LR, data_start_step=DATA_START, log_every=log_every)
        summary = continue_training(experiment, target, options, name=f"b8w-{arm}-s{seed}",
                                    continued_from=dict(arm=arm, checkpoint=str(start), checkpoint_hash=started))
        with (target / "train_log.jsonl").open() as stream:
            durations = [row["step_seconds"] for line in stream if (row := json.loads(line))["step"] > 5]
        steady = sum(durations) / len(durations) if durations else None
        record = dict(status="complete", arm=arm, seed=seed, config=experiment.config,
                      config_hash=stable_hash(experiment.config), steps=steps, lr=LR, data_start_step=DATA_START,
                      start_checkpoint_hash=started, checkpoint_hash=digest(target / "checkpoint.pt"),
                      writer_hash=stable_hash({k: tensor_hash(v) for k, v in experiment.writer_state().items()}),
                      trainable_parameters=summary["trainable_parameters"], train_seconds=summary["train_seconds"],
                      steady_step_seconds=steady, peak_memory_bytes=torch.cuda.max_memory_allocated(),
                      seconds=time.perf_counter() - begun, gate=gate_result, **runtime(), **source_metadata())
        save(target / "record.json", record)
        return record
    except BaseException as error:
        target.mkdir(parents=True, exist_ok=True)
        save(target / "failure.json", dict(error=f"{type(error).__name__}: {error}", **runtime()))
        raise
    finally:
        experiment.close()


def stage(args) -> None:
    import torch
    from audit_p4s_stopping import require_compute
    from looplora.llm.inject import load_model, load_tokenizer

    require_compute()
    torch.set_num_threads(1)
    if args.stage == "main":
        if args.steps is not None:
            raise ValueError("main runs the registered N; --steps is for the smoke")
        check_lock(json.loads((ROOT / LOCK).read_text()), args.seed)
    manifest = json.loads((ROOT / P1L_MANIFEST).read_text())
    start, recorded = checkpoint_of(args.seed, "M1-L4", manifest)
    if digest(start) != recorded:
        raise ValueError("the M1-L4 checkpoint does not match the P1-L run manifest")
    steps = STEPS if args.stage == "main" else args.steps
    args.out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(), load_tokenizer()
    gated = gate(args.seed, start, args.out, model, tokenizer)
    print(json.dumps(dict(gate=gated["passed"], max_abs_difference=gated["max_abs_difference"])), flush=True)
    records = []
    for arm in ARMS:
        path = start if arm == "shared" else args.out / "converted.pt"
        record = run_arm(arm, args.seed, path, args.out, model, tokenizer, steps=steps,
                         log_every=1 if args.stage == "smoke" else 100, gate_result=gated)
        records.append(record)
        print(json.dumps(dict(arm=arm, **{k: record[k] for k in ("train_seconds", "steady_step_seconds", "peak_memory_bytes")})),
              flush=True)
        save(args.out / "stage.json", dict(stage=args.stage, status="running", seed=args.seed, arms=[r["arm"] for r in records],
                                           **runtime()))
    save(args.out / "stage.json", dict(stage=args.stage, status="complete", seed=args.seed, arms=list(ARMS), **runtime()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "main"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=17)
    args = parser.parse_args()
    if args.stage == "smoke" and (args.steps is None or args.steps < 10):
        parser.error("smoke requires --steps >= 10")
    stage(args)


if __name__ == "__main__":
    main()
