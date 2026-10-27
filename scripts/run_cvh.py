#!/usr/bin/env python3
"""CVH (F9): continuation-value halting heads on frozen r = 32 readers (docs/prereg/CVH.md).

Four stages, each its own job (scripts/submit.py in a Miyabi-G allocation):

  lock     write the design: settings, the nine sealed M1 readers, the existing P4-S-L heads (arm A0
           and the head of arm A1) and the locked P4-S-L thetas;
  smoke    the whole pipeline on small validation data for the seed-17 readers (timings, label rates,
           latency path); no confirmation data;
  train    fit the continuation heads (arm A2) on training traces, trace the validation episodes, choose the
           thetas of arms A1 and A2 under the hop-1 constraint and write theta.json (the validation lock);
  confirm  trace the new confirmation episodes (split test, seed 999005), with the three arms' heads, and
           measure the end-to-end latency; requires both Git locks and refuses to run after a registered failure.

The readers are frozen: their state and file hashes are verified after use. Nothing here changes a reader.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from s9_common import absolute_path, digest_checkpoint, now, require, require_compute, verify_git_lock, write_json
from s9_torch import load_frozen, state_hash, synchronize
from looplora.core.learned_halting import HaltingHead
from looplora.infra.seeds import stable_hash
from looplora.sym import cvh
from looplora.sym.run import source_metadata

STUDY = "CVH"
HOPS = list(range(1, 9))
TRAINING = dict(split="train", seed=31337, episodes=20000, steps=2000, lr=0.001, batch_episodes=256,
                label="cvh", margin=cvh.MARGIN)
VALIDATION = dict(split="val", seed=4242, episodes=2000)
CONFIRMATION = dict(split="test", seed=999005, episodes=2000)
SMOKE = dict(training_episodes=2000, steps=200, validation_episodes=200, training_seed=17)
LATENCY = dict(batch_size=64, warmup=5, repeats=20)  # the P4-S-L protocol
P4SL = dict(design="reports/p4s-l/design.json", heads="reports/p4s-l/head_manifest.json",
            theta="reports/p4s-l/theta.json", exported_heads="reports/p4s-l/heads")
REPRODUCTION_TOLERANCE = 1e-3  # recomputed A0 validation accuracy versus the P4-S-L record
BOOTSTRAP = dict(resamples=2000, seed=2718)
BOOTSTRAP_KWARGS = dict(resamples=BOOTSTRAP["resamples"])


def settings() -> dict:
    return dict(
        study=STUDY, prereg="docs/prereg/CVH.md", hops=list(HOPS), training=dict(TRAINING),
        validation=dict(VALIDATION), confirmation=dict(CONFIRMATION), smoke=dict(SMOKE), latency=dict(LATENCY),
        thetas=list(cvh.THETAS), hop1_tolerance=cvh.HOP1_TOLERANCE, arms=list(cvh.ARMS),
        families=list(cvh.FAMILIES), seeds=list(cvh.SEEDS),
        decision=dict(gain_minimum=cvh.GAIN_MINIMUM, hop1_loss=cvh.HOP1_LOSS, mean_depth_limit=cvh.MEAN_DEPTH_LIMIT,
                      comparisons=cvh.COMPARISONS, alpha=0.05, confidence=cvh.CONFIDENCE, **BOOTSTRAP))


def build_design(repo: str | Path) -> dict:
    """Settings plus the sealed readers and heads, read from the committed P4-S-L files of ``repo``."""
    repo = Path(repo).resolve()
    paths = {key: repo / value for key, value in P4SL.items()}
    reference = json.loads(paths["design"].read_text())
    manifest = json.loads(paths["heads"].read_text())
    theta = json.loads(paths["theta"].read_text())
    require(manifest["status"] == "passed" and manifest["all_readers_unchanged"] is True,
            "the P4-S-L head manifest is not a passed, reader-unchanged record")
    heads = {row["name"]: row for row in manifest["runs"]}
    readers = []
    for run in reference["runs"]:
        if run["family"] not in cvh.FAMILIES:
            continue
        require(digest_checkpoint(Path(run["checkpoint"])) == run["checkpoint_blake2b"],
                f"reader checkpoint changed: {run['name']}")
        exported = paths["exported_heads"] / f"{run['name']}.pt"
        require(digest_checkpoint(exported) == heads[run["name"]]["head_blake2b"],
                f"exported P4-S-L head differs from its record: {run['name']}")
        readers.append(dict(name=run["name"], writer=run["writer"], variant=run["variant"], r=run["r"],
                            seed=run["seed"], family=run["family"], path=run["checkpoint"],
                            blake2b=run["checkpoint_blake2b"], head_path=str(exported),
                            head_blake2b=heads[run["name"]]["head_blake2b"]))
    readers.sort(key=lambda row: row["name"])
    a0_theta = {row["family"]: dict(theta=row["theta"], validation_accuracy=row["validation_accuracy"])
                for row in theta["selected"] if row["family"] in cvh.FAMILIES}
    sources = {key: dict(path=str(path), blake2b=digest_checkpoint(path)) for key, path in paths.items()
               if path.is_file()}
    design = {**settings(), "readers": readers, "a0_theta": a0_theta, "sources": sources}
    verify_design(design)
    return design


def verify_design(design: dict) -> list[dict]:
    keys = ("readers", "a0_theta", "sources")
    require({key: value for key, value in design.items() if key not in keys} == settings(),
            "the design differs from the registered settings")
    readers = design["readers"]
    found = sorted((row["family"], row["seed"]) for row in readers)
    require(found == sorted((family, seed) for family in cvh.FAMILIES for seed in cvh.SEEDS),
            "the design must list exactly the nine registered readers")
    require(set(design["a0_theta"]) == set(cvh.FAMILIES), "the design lacks a locked P4-S-L theta for a family")
    for row in readers:
        require(row["variant"] == "M1" and row["r"] == 32, f"unexpected reader {row['name']}")
        require(digest_checkpoint(Path(row["path"])) == row["blake2b"], f"reader checkpoint changed: {row['name']}")
        require(digest_checkpoint(Path(row["head_path"])) == row["head_blake2b"], f"A0 head changed: {row['name']}")
    return readers


def load_head(path: str | Path, device: str, *, reader_blake2b: str) -> tuple[HaltingHead, dict]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    require(record["reader_blake2b"] == reader_blake2b, f"head {Path(path).name} was trained on another reader")
    head = HaltingHead().to(device)
    head.load_state_dict(record["head"], strict=True)
    return head.eval().requires_grad_(False), record


def sizes(design: dict, smoke: bool) -> tuple[int, int, int]:
    if smoke:
        return design["smoke"]["training_episodes"], design["smoke"]["steps"], design["smoke"]["validation_episodes"]
    return design["training"]["episodes"], design["training"]["steps"], design["validation"]["episodes"]


def save_traces(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: trace[key] for key in ("predictions", "losses", "answer", "hop", "episode_id")}
    arrays.update({f"p_{name}": value for name, value in trace["probabilities"].items()})
    np.savez_compressed(path, **arrays)


def train_reader(entry: dict, design: dict, *, device: str, smoke: bool) -> dict:
    """Continuation head (A2) of one reader, and its validation traces together with the existing head's."""
    episodes, steps, validation_episodes = sizes(design, smoke)
    training = design["training"]
    model = load_frozen(entry, device)
    before = state_hash(model)
    a0, _ = load_head(entry["head_path"], device, reader_blake2b=entry["blake2b"])
    started = time.perf_counter()
    cache = cvh.collect_traces(model, design["hops"], training["split"], training["seed"], episodes,
                               training_margin=training["margin"], batch_episodes=64)
    synchronize(device)
    trace_seconds = time.perf_counter() - started
    require(not cache["features"].requires_grad and not cache["targets"].requires_grad,
            "the frozen-reader cache unexpectedly tracks gradients")
    label_rate = cache["targets"].float().mean(dim=(0, 1)).cpu().tolist()
    started = time.perf_counter()
    head, fit = cvh.fit_head(cache["features"], cache["targets"], seed=entry["seed"], family=entry["family"],
                             label=training["label"], steps=steps, lr=training["lr"],
                             batch_episodes=training["batch_episodes"])
    synchronize(device)
    fit["seconds"] = time.perf_counter() - started
    del cache
    started = time.perf_counter()
    validation = cvh.collect_traces(model, design["hops"], design["validation"]["split"],
                                    design["validation"]["seed"], validation_episodes,
                                    heads={"A0": a0, "A2": head}, batch_episodes=64)
    synchronize(device)
    validation_seconds = time.perf_counter() - started
    require(state_hash(model) == before, f"reader parameters changed: {entry['name']}")
    return dict(head=head, fit=fit, validation=validation, label_rate=label_rate, reader_state_hash=before,
                timing=dict(name=entry["name"], training_trace_seconds=trace_seconds,
                            head_training_seconds=fit["seconds"], validation_seconds=validation_seconds,
                            training_episodes=episodes, steps=steps, validation_episodes=validation_episodes))


def family_traces(design: dict, validation: dict[str, dict]) -> dict[str, dict[int, dict]]:
    """Validation traces by family and training seed, with the head probabilities as ``p_<arm>`` keys."""
    out: dict[str, dict[int, dict]] = {}
    for entry in design["readers"]:
        if entry["name"] not in validation:
            continue
        trace = validation[entry["name"]]
        out.setdefault(entry["family"], {})[entry["seed"]] = dict(
            predictions=trace["predictions"], answer=trace["answer"], hop=trace["hop"],
            **{f"p_{name}": value for name, value in trace["probabilities"].items()})
    return out


def select_thetas(design: dict, validation: dict[str, dict], *, smoke: bool = False) -> dict:
    """Per family: A0 at its P4-S-L theta; A1 and A2 under the hop-1 constraint (None if infeasible)."""
    selected, failures, reproduction = [], [], []
    for family, traces in sorted(family_traces(design, validation).items()):
        tables = {arm: cvh.threshold_table(traces, f"p_{head}") for arm, head in (("A0", "A0"), ("A1", "A0"), ("A2", "A2"))}
        locked = design["a0_theta"][family]
        row = min(tables["A0"], key=lambda r: abs(r["theta"] - locked["theta"]))
        require(abs(row["theta"] - locked["theta"]) < 1e-9, f"{family}: the locked theta is not on the grid")
        reproduction.append(dict(family=family, p4s_l_validation_accuracy=locked["validation_accuracy"],
                                 recomputed_validation_accuracy=row["validation_accuracy"],
                                 reproduced=abs(row["validation_accuracy"] - locked["validation_accuracy"])
                                 <= REPRODUCTION_TOLERANCE))
        choices = {"A0": dict(row, source="p4s-l-lock"),
                   "A1": cvh.select_theta(tables["A1"], tolerance=design["hop1_tolerance"]),
                   "A2": cvh.select_theta(tables["A2"], tolerance=design["hop1_tolerance"])}
        for arm in cvh.ARMS:
            choice = choices[arm]
            if choice is None:
                failures.append(dict(family=family, arm=arm,
                                     reason="no validation threshold keeps hop 1 within the registered tolerance"))
            selected.append(dict(family=family, arm=arm, feasible=choice is not None,
                                 theta=None if choice is None else choice["theta"],
                                 **({} if choice is None else {key: value for key, value in choice.items()
                                                               if key != "theta"})))
    candidates = [row for row in selected if row["arm"] in ("A1", "A2")]
    return dict(design_hash=stable_hash(design), locked_at=now(), grid=list(design["thetas"]),
                tolerance=design["hop1_tolerance"], margin=design["training"]["margin"], selected=selected,
                failures=failures, a0_reproduction=reproduction, smoke=smoke,
                confirmation_allowed=any(row["feasible"] for row in candidates))


def active_families(theta: dict) -> list[str]:
    """Families with a feasible hop-1-constrained arm (A1 or A2); only these are looked at on the confirmation set."""
    return sorted({row["family"] for row in theta["selected"] if row["arm"] in ("A1", "A2") and row["feasible"]})


def runtime_metadata(device: str) -> dict:
    cuda = torch.device(device).type == "cuda"
    return dict(host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(device) if cuda else None, device=device, threads=1)


def stage_lock(args: argparse.Namespace) -> int:
    out = Path(args.out)
    require(out.is_absolute() and not out.exists(), "the lock file must be a new absolute path")
    design = build_design(args.repo)
    write_json(out, design)
    print(json.dumps(dict(stage="lock", readers=len(design["readers"]), design_hash=stable_hash(design),
                          out=str(out))), flush=True)
    return 0


def read_design(args: argparse.Namespace, *, locked: bool) -> tuple[Path, dict, list[dict]]:
    path = Path(args.design).resolve()
    design = json.loads(path.read_text())
    if locked:
        verify_git_lock(path, args.design_commit, "the design")
    return path, design, verify_design(design)


def stage_train(args: argparse.Namespace) -> int:
    root = absolute_path(args.root, "--root")
    design_path, design, readers = read_design(args, locked=True)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = dict(status="running", smoke=False, design=design, design_hash=stable_hash(design),
                    design_path=str(design_path), design_lock_commit=args.design_commit,
                    readers=copy.deepcopy(readers), started_at=now(), source=source_metadata(),
                    runtime=runtime_metadata(args.device))
    write_json(root / "train_manifest.json", manifest)
    by_name = {row["name"]: row for row in manifest["readers"]}
    validation, timings, records = {}, [], {}
    for entry in readers:
        result = train_reader(entry, design, device=args.device, smoke=False)
        path = root / "heads" / f"{entry['name']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(head={k: v.detach().cpu() for k, v in result["head"].state_dict().items()},
                        reader_blake2b=entry["blake2b"], design_hash=stable_hash(design), training=result["fit"],
                        reader_state_hash=result["reader_state_hash"], label=design["training"]["label"],
                        margin=design["training"]["margin"]), path)
        by_name[entry["name"]].update(a2_head_path=str(path), a2_head_blake2b=digest_checkpoint(path))
        save_traces(root / "traces" / entry["name"] / "val.npz", result["validation"])
        validation[entry["name"]] = result["validation"]
        timings.append(result["timing"])
        records[entry["name"]] = dict(label_rate=result["label_rate"], fit=result["fit"])
        print(json.dumps(dict(stage="train", name=entry["name"],
                              seconds=sum(v for k, v in result["timing"].items() if k.endswith("seconds")))),
              flush=True)
        del result
    for entry in readers:
        require(digest_checkpoint(Path(entry["path"])) == entry["blake2b"], f"reader checkpoint changed: {entry['name']}")
    theta = select_thetas(design, validation)
    seconds = time.perf_counter() - started
    manifest.update(status="complete", finished_at=now(), total_seconds=seconds, checkpoints_unchanged=True)
    write_json(root / "train_manifest.json", manifest)
    write_json(root / "timing.json", timings)
    write_json(root / "head_records.json", records)
    write_json(root / "theta.json", theta)
    print(json.dumps(dict(stage="train", status="complete", seconds=seconds,
                          confirmation_allowed=theta["confirmation_allowed"], failures=theta["failures"],
                          a0_reproduced=all(row["reproduced"] for row in theta["a0_reproduction"]))), flush=True)
    return 0


def smoke_theta(theta: dict) -> dict:
    """Smoke only: give every infeasible arm a placeholder theta so that each code path is exercised."""
    rows = [row if row["feasible"] else dict(row, feasible=True, theta=0.5) for row in theta["selected"]]
    return dict(theta, selected=rows)


def stage_smoke(args: argparse.Namespace) -> int:
    """The whole pipeline on small validation data; nothing from it is used, only timings and sanity checks."""
    root = absolute_path(args.root, "--root")
    _, design, readers = read_design(args, locked=False)
    readers = [row for row in readers if row["seed"] == design["smoke"]["training_seed"]]
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    validation, heads, timings, records = {}, {}, [], {}
    for entry in readers:
        result = train_reader(entry, design, device=args.device, smoke=True)
        validation[entry["name"]], heads[entry["name"]] = result["validation"], result["head"]
        timings.append(result["timing"])
        records[entry["name"]] = dict(label_rate=result["label_rate"], final_loss=result["fit"]["logs"][-1]["loss"])
        print(json.dumps(dict(stage="smoke-train", name=entry["name"])), flush=True)
    theta = select_thetas(design, validation, smoke=True)
    placeholder = smoke_theta(theta)
    cuda = torch.device(args.device).type == "cuda"
    episodes = design["smoke"]["validation_episodes"]
    traces, confirm_timings, latency = {}, [], []
    for entry in readers:  # the confirmation code path, on validation episodes only
        trace, rows, timing = confirm_reader(entry, design, placeholder, heads[entry["name"]], device=args.device,
                                             split=design["validation"]["split"], seed=design["validation"]["seed"],
                                             episodes=episodes, latency=cuda)
        traces[entry["family"]] = {entry["seed"]: trace}
        confirm_timings.append(timing)
        latency.extend(dict(row, name=entry["name"], family=entry["family"]) for row in rows)
    endpoints = {}
    for family, by_seed in traces.items():
        for row in placeholder["selected"]:
            if row["family"] != family:
                continue
            probabilities = {seed: by_seed[seed]["probabilities"]["A2" if row["arm"] == "A2" else "A0"] for seed in by_seed}
            steps = {seed: cvh.policy_steps(value, row["theta"]) for seed, value in probabilities.items()}
            result = cvh.arm_endpoints(by_seed, steps, **BOOTSTRAP_KWARGS)
            endpoints[f"{family}/{row['arm']}"] = dict(mean_executed=result["mean_executed"],
                                                      reach_gain=result["reach_gain"]["difference"],
                                                      hop1_delta=result["hop1_delta"]["difference"])
    scale = dict(trace=design["training"]["episodes"] / design["smoke"]["training_episodes"],
                 fit=design["training"]["steps"] / design["smoke"]["steps"],
                 validation=design["validation"]["episodes"] / design["smoke"]["validation_episodes"])
    per_seed = sum(scale["trace"] * t["training_trace_seconds"] + scale["fit"] * t["head_training_seconds"]
                   + scale["validation"] * t["validation_seconds"] for t in timings)
    train_estimate = per_seed * len(cvh.SEEDS)
    latency_seconds = sum(row.get("wall_ms_mean", 0.0) * (LATENCY["warmup"] + LATENCY["repeats"]) / 1000.0
                          for row in latency)
    confirm_estimate = len(cvh.SEEDS) * (design["confirmation"]["episodes"] / episodes
                                         * sum(t["trace_seconds"] for t in confirm_timings) + latency_seconds)
    margin = 1.5
    report = dict(status="passed", created=now(), design_hash=stable_hash(design), timings=timings,
                  confirm_timings=confirm_timings, label_rates=records, theta=theta, endpoints=endpoints,
                  latency_rows=len(latency), train_seconds_estimate=train_estimate,
                  confirm_seconds_estimate=confirm_estimate,
                  suggested_train_walltime_seconds=int(-(-margin * train_estimate // 60) * 60),
                  suggested_confirm_walltime_seconds=int(-(-margin * confirm_estimate // 60) * 60),
                  total_seconds=time.perf_counter() - started, runtime=runtime_metadata(args.device),
                  note="smoke thetas come from one seed and small samples and are never used; the estimates "
                       "scale the smoke timings by the episode and step ratios and are conservative")
    write_json(root / "smoke.json", report)
    print(json.dumps(dict(stage="smoke", status="passed",
                          suggested_train_walltime_seconds=report["suggested_train_walltime_seconds"],
                          suggested_confirm_walltime_seconds=report["suggested_confirm_walltime_seconds"])), flush=True)
    return 0


def confirm_reader(entry: dict, design: dict, theta: dict, a2: HaltingHead | None, *, device: str, split: str,
                   seed: int, episodes: int, latency: bool) -> tuple[dict, list[dict], dict]:
    """Trace one reader with the A0 and A2 heads; returns (trace, latency rows, timing)."""
    model = load_frozen(entry, device)
    before = state_hash(model)
    a0, _ = load_head(entry["head_path"], device, reader_blake2b=entry["blake2b"])
    heads = {"A0": a0, **({} if a2 is None else {"A2": a2})}
    started = time.perf_counter()
    trace = cvh.collect_traces(model, design["hops"], split, seed, episodes, heads=heads, batch_episodes=64)
    synchronize(device)
    seconds = time.perf_counter() - started
    rows = []
    if latency:
        from run_p4s_learned import benchmark  # CUDA-event timing of the learned read and the fixed references
        for arm in cvh.ARMS:
            chosen = next(row for row in theta["selected"] if row["family"] == entry["family"] and row["arm"] == arm)
            if not chosen["feasible"]:
                continue
            head = a2 if arm == "A2" else a0
            for row in benchmark(model, trace["first"], head, chosen["theta"], entry, design):
                if row["method"] == "learned" or arm == "A0":  # the fixed references are arm-independent
                    rows.append(dict(row, arm=arm if row["method"] == "learned" else "fixed", theta=chosen["theta"]))
    require(state_hash(model) == before, f"reader parameters changed: {entry['name']}")
    return trace, rows, dict(name=entry["name"], split=split, generator_seed=seed, episodes=episodes,
                             trace_seconds=seconds)


def stage_confirm(args: argparse.Namespace) -> int:
    design_path, theta_path = Path(args.design).resolve(), Path(args.theta).resolve()
    root = absolute_path(args.root, "--root")
    trained = json.loads((root / "train_manifest.json").read_text())
    require(trained["status"] == "complete" and trained["smoke"] is False, "the training stage is not complete")
    design = json.loads(design_path.read_text())
    verify_git_lock(design_path, args.design_commit, "the design")
    verify_git_lock(theta_path, args.theta_commit, "theta.json")
    require(trained["design_hash"] == stable_hash(design), "the design changed after training")
    theta = json.loads(theta_path.read_text())
    require(theta == json.loads((root / "theta.json").read_text()), "the committed theta differs from the training record")
    require(theta["design_hash"] == stable_hash(design) and theta["smoke"] is False, "theta.json belongs to another design")
    require(all(row["reproduced"] for row in theta["a0_reproduction"]),
            "the P4-S-L validation accuracy was not reproduced; the pipeline is not the registered one")
    require(theta["confirmation_allowed"], "registered failure: no validation theta keeps hop 1; no confirmation look")
    require({row["name"] for row in verify_design(design)} == {row["name"] for row in trained["readers"]},
            "the trained readers differ from the design")
    active = active_families(theta)  # a family whose new arms both failed on validation is never looked at
    readers = [row for row in trained["readers"] if row["family"] in active]
    require(not (root / "manifest.json").exists(), "the confirmation has already started")
    manifest = dict(trained, status="running", theta_path=str(theta_path), theta_lock_commit=args.theta_commit,
                    theta_blake2b=digest_checkpoint(theta_path), confirmation_started_at=now(),
                    confirmation_source=source_metadata(), runtime=runtime_metadata(args.device),
                    latency_measured=not args.skip_latency, active_families=active)
    write_json(root / "manifest.json", manifest)
    confirmation, started = design["confirmation"], time.perf_counter()
    timings, latency = [], []
    for entry in readers:
        a2, record = load_head(entry["a2_head_path"], args.device, reader_blake2b=entry["blake2b"])
        require(digest_checkpoint(Path(entry["a2_head_path"])) == entry["a2_head_blake2b"]
                and record["design_hash"] == stable_hash(design), f"A2 head provenance differs: {entry['name']}")
        trace, rows, timing = confirm_reader(entry, design, theta, a2, device=args.device,
                                             split=confirmation["split"], seed=confirmation["seed"],
                                             episodes=confirmation["episodes"], latency=not args.skip_latency)
        path = root / "traces" / entry["name"] / "test.npz"
        save_traces(path, trace)
        timing["traces_blake2b"] = digest_checkpoint(path)
        timings.append(timing)
        latency.extend(dict(row, name=entry["name"], family=entry["family"], seed=entry["seed"]) for row in rows)
        print(json.dumps(dict(stage="confirm", name=entry["name"], seconds=timing["trace_seconds"])), flush=True)
        del trace
    for entry in readers:
        require(digest_checkpoint(Path(entry["path"])) == entry["blake2b"], f"reader checkpoint changed: {entry['name']}")
    seconds = time.perf_counter() - started
    write_json(root / "latency.json", latency)
    write_json(root / "confirmation_timing.json", timings)
    manifest.update(status="complete", confirmation_finished_at=now(), confirmation_seconds=seconds,
                    checkpoints_unchanged=True)
    write_json(root / "manifest.json", manifest)
    print(json.dumps(dict(stage="confirm", status="complete", seconds=seconds)), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    lock = sub.add_parser("lock", help="write the design and seals")
    lock.add_argument("--repo", default=".", help="repository holding the committed reports/p4s-l files")
    lock.add_argument("--out", required=True, help="new design JSON (absolute path)")
    for name in ("smoke", "train"):
        stage = sub.add_parser(name)
        stage.add_argument("--design", required=True)
        stage.add_argument("--root", required=True, help="new run directory (absolute path)")
        if name == "train":
            stage.add_argument("--design-commit", required=True, help="full commit hash that locked the design")
    confirm = sub.add_parser("confirm")
    confirm.add_argument("--design", required=True)
    confirm.add_argument("--design-commit", required=True)
    confirm.add_argument("--theta", required=True, help="the committed theta.json")
    confirm.add_argument("--theta-commit", required=True, help="full commit hash that locked theta.json")
    confirm.add_argument("--root", required=True, help="the training run directory")
    confirm.add_argument("--skip-latency", action="store_true", help="tests only: no CUDA latency measurement")
    for stage in sub.choices.values():
        if stage is not lock:
            stage.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    require_compute()
    torch.set_num_threads(1)
    if args.stage != "lock" and torch.device(args.device).type == "cuda":
        torch.cuda.set_device(torch.device(args.device))
    return {"lock": stage_lock, "smoke": stage_smoke, "train": stage_train, "confirm": stage_confirm}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
