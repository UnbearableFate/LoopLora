#!/usr/bin/env python3
"""Fingerprints of every P1-L default that the P1L-S pipeline work touched; one run per source tree.

    PYTHONPATH=<tree>/src python tests/assets/p1ls_defaults_probe.py --tree <tree> --fixtures fixtures.json \
        --out probe.json --artifacts finalize-out

<tree> supplies src/, scripts/ and configs/: the current checkout, or a `git archive` of the baseline commit.
tests/test_p1ls_defaults.py runs this on both and requires every part to be identical. Parts: resolved config hashes
(training and the two confirmation templates), run_p1l_main's labels and configs, the launcher (command lines,
receipts, manifests; subprocess replaced), analyze_p1l_main size/select, check_training on one accepted and a battery of
damaged manifests, authorizations and records, endpoints and the memory-use text, finalize on the synthetic r = 32
matrix, and evaluate()'s episode windows on a tiny model. Compute node only. No test episode is ever generated.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE = "0123456789abcdef0123456789abcdef01234567"
FINALIZE_OUTPUTS = ("endpoints.json", "accuracy.csv", "inference.csv", "memory-use.csv", "per-step-readout.csv",
                    "readout-faithfulness.csv", "swap-successors.csv", "run-manifest.json", "results.md")


def attempt(function, *args, **kwargs):
    """The value `function` returns, or the exception it raises as text (both trees must raise the same text)."""
    try:
        return function(*args, **kwargs)
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def outcome(function, *args, **kwargs) -> str:
    return "ok" if (result := attempt(function, *args, **kwargs)) is None else result


@contextlib.contextmanager
def edited(path, edit):
    """Rewrite a JSON file with `edit` applied, and restore its text afterwards."""
    path = Path(path)
    original = path.read_text()
    value = json.loads(original)
    edit(value)
    path.write_text(json.dumps(value))
    try:
        yield
    finally:
        path.write_text(original)


# ---- configuration ------------------------------------------------------------------------------------------------

def configs_part(tree: Path) -> dict:
    from looplora.infra.seeds import stable_hash
    from looplora.llm.config import recursive_merge, resolve_config

    root = tree / "configs/p1l"
    templates = {name: json.loads((root / "main" / f"eval-{name}.json").read_text())
                 for name in ("confirmation", "typed-replication")}
    result = {}
    for path in sorted(root.rglob("*.json")):
        key = str(path.relative_to(root))
        try:
            resolved = resolve_config(json.loads(path.read_text()))
        except Exception as error:  # both trees must reject the same files with the same message
            result[key] = f"{type(error).__name__}: {error}"
            continue
        result[key] = dict(resolved=stable_hash(resolved))
        if path.stem.rsplit("-s", 1)[-1].isdigit():  # a registered run: also its two confirmation configs
            for name, template in templates.items():
                result[key][name] = attempt(lambda: stable_hash(resolve_config(recursive_merge(resolved, template))))
    return result


def run_part() -> dict:
    import run_p1l_main as run
    from looplora.infra.seeds import stable_hash

    configs = {f"r{rank}/{label}-s{seed}": stable_hash(run.config(label, seed, rank))
               for rank in (32, 128) for label in (*run.labels_for_rank(rank), *(("B0",) if rank == 32 else ()))
               for seed in ((17,) if label == "B0" else run.SEEDS)}
    return dict(labels=list(run.LABELS), seeds=list(run.SEEDS), r32=list(run.labels_for_rank(32)),
                r128=list(run.labels_for_rank(128)), default_config=stable_hash(run.config("B8")), configs=configs)


# ---- launcher -----------------------------------------------------------------------------------------------------

class FakeSubprocess:
    """Stands in for `subprocess` inside launch_p1l_main: git answers from a stand-in repository root and
    scripts/submit.py returns a receipt. Every submit command is kept."""

    def __init__(self, root: Path) -> None:
        self.root, self.commands = root, []

    def check_output(self, command, text=True):
        assert command[:2] == ["git", "-C"] and text
        if command[3] == "rev-parse":
            return SOURCE + "\n"
        revision, relative = command[4].split(":", 1)
        assert command[3] == "show" and revision == SOURCE
        return (self.root / relative).read_text()

    def run(self, command, check, capture_output, text):
        assert check and capture_output and text
        self.commands.append(command)
        return SimpleNamespace(stdout=json.dumps(dict(job_id=f"{len(self.commands)}.opbs",
                                                      log=f"log-{len(self.commands)}")) + "\n")


STEP = {32: dict(B8=.193, **{"M1-L4": .197}, B4=.133, **{"B5-L4": .152, "B3-L4": .135}, B1=.11, B2=.12),
        128: dict(B8=.25, **{"M1-L4": .25}, B4=.21, **{"B5-L4": .22, "B3-L4": .21})}
LRS = {"B8": .001, "M1-L4": .003, "B4": .001, "B5-L4": .001, "B3-L4": .001, "B1": .001, "B2": .001}


def launcher_part(scratch: Path) -> dict:
    import launch_p1l_main as launch

    root = scratch / "repository"
    for relative in ("reports/p1l/lr-S-lock.json", "reports/p1l/r128/lr-S-lock.json"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(json.dumps(dict(status="locked", S=17000, lrs=LRS), indent=2) + "\n")
    launch.ROOT = root
    result = {"walltime": {str(seconds): attempt(launch.walltime, seconds)
                           for seconds in (1, 299, 300, 1200, 1201, 3600, 4800, 4801, 7200)}}
    for stage, rank in (("pilot", None), ("pilot", 32), ("main", 32), ("pilot", 128), ("main", 128)):
        name = f"{stage}-r{rank}"
        effective = rank or 32
        sizing = scratch / f"sizing-{name}.json"
        sizing.write_text(json.dumps(dict(rank=effective, S=17000, pilot_steps=5666, timing_rows=[
            dict(label=label, step_seconds=step, overhead_seconds=20.) for label, step in STEP[effective].items()])))
        launch.subprocess = FakeSubprocess(root)
        argv = [stage, "--root", str(scratch / name), "--sizing", str(sizing), *(["--rank", str(rank)] if rank else [])]
        stdout = io.StringIO()
        sys.argv = ["launch_p1l_main.py", *argv]
        with contextlib.redirect_stdout(stdout):
            launch.main()
        files = {path.name: json.loads(path.read_text()) for path in sorted((scratch / name).glob("*.json"))}
        payload = dict(commands=launch.subprocess.commands, stdout=stdout.getvalue().splitlines(), files=files)
        result[name] = json.loads(json.dumps(payload, sort_keys=True).replace(str(scratch), "<S>"))
    return result


# ---- analysis -----------------------------------------------------------------------------------------------------

def analyze_part(scratch: Path) -> dict:
    import analyze_p1l_main as analyze
    import run_p1l_main as run

    result = {}
    for rank in (32, 128):
        root = scratch / f"analyze-r{rank}"
        labels = run.labels_for_rank(rank)
        for index, label in enumerate(labels):
            directory = root / f"{label}-s17"
            directory.mkdir(parents=True)
            step = .1 + .01 * index
            (directory / "record.json").write_text(json.dumps(dict(
                status="complete", config=dict(model=dict(r=rank), smoke=True, train=dict(steps=100),
                                               eval=dict(splits=["val"], allow_confirmation=False)),
                train_seconds=20 + 100 * step, peak_memory_bytes=1000 + index, checkpoint_hash=f"c{index}",
                job_id=f"{index}.opbs", host="mg0001")))
            (directory / "train_log.jsonl").write_text("".join(
                json.dumps(dict(step=number, step_seconds=step)) + "\n" for number in range(1, 41)))
        sizing_path = root / "sizing.json"
        with contextlib.redirect_stdout(io.StringIO()):
            analyze.size(SimpleNamespace(root=root, rank=rank, out=sizing_path, study="p1l"))
        sizing = json.loads(sizing_path.read_text())
        sizing.pop("time")  # the lock records the sizing file's hash, so the file must not carry a timestamp or a path
        sizing["source"] = "<root>"
        sizing_path.write_text(json.dumps(sizing, sort_keys=True))
        pilot = scratch / f"pilot-r{rank}"
        for index, label in enumerate(labels):
            for rate in (.001, .003):
                directory = pilot / label / f"{label}-lr{rate:g}"
                directory.mkdir(parents=True)
                tie = index == 2  # equal scores: the lower learning rate must win
                accuracy = .30 + .01 * index + (0. if tie or rate == .001 else .02)
                (directory / "record.json").write_text(json.dumps(dict(
                    status="complete", checkpoint_hash=f"p{index}", config=dict(
                        seed=17, train=dict(steps=sizing["pilot_steps"]), model=dict(r=rank), protocol="memory_only",
                        eval=dict(splits=["val"], hops=[1, 2, 3])),
                    validation=[dict(condition="memory_only/original", hop=hop, n=1000, accuracy=accuracy)
                                for hop in (1, 2, 3)])))
        lock_path = root / "lock.json"
        with contextlib.redirect_stdout(io.StringIO()):
            analyze.select(SimpleNamespace(root=pilot, sizing=sizing_path, rank=rank, out=lock_path, study="p1l"))
        lock = json.loads(lock_path.read_text())
        lock.pop("time")
        result[f"r{rank}"] = json.loads(json.dumps(dict(sizing=sizing, lock=lock)).replace(str(scratch), "<S>"))
    return result


# ---- check_training -----------------------------------------------------------------------------------------------

def check_part(fix: dict) -> dict:
    from evaluate_p1l_main import check_training

    manifest = json.loads(Path(fix["manifest32"]).read_text())
    authorization = json.loads(Path(fix["authorization"]).read_text())
    record = {name: Path(path) / "record.json" for name, path in manifest["runs"].items()}
    cases = dict(accepted=outcome(check_training, manifest, authorization),
                 accepted_without_lock_check=outcome(check_training, manifest, authorization, check_all=False))
    for name, edit in dict(
            held=lambda a: a.update(status="held"), no_decision=lambda a: a.pop("orchestrator_decision"),
            one_condition=lambda a: a.update(gated_conditions=1), bool_count=lambda a: a.update(gated_conditions=True),
            missing_condition=lambda a: a["condition_manifests"].pop("oracle-r128")).items():
        damaged = copy.deepcopy(authorization)
        edit(damaged)
        cases[f"authorization_{name}"] = outcome(check_training, manifest, damaged)
    for name, edit in dict(
            missing_run=lambda m: m["runs"].pop("B2-s43"), no_B0=lambda m: m["runs"].pop("B0-s17"),
            extra_run=lambda m: m["runs"].update({"B9-s17": "/nowhere"}),
            changed_S=lambda m: m.update(S=1)).items():
        damaged = copy.deepcopy(manifest)
        edit(damaged)
        cases[f"manifest_{name}"] = outcome(check_training, damaged, authorization)
    for name, (run, edit) in dict(
            incomplete=("B8-s17", lambda r: r.update(status="running")),
            changed_checkpoint=("B4-s17", lambda r: r.update(checkpoint_hash="0" * 32)),
            smoke=("M1-L4-s29", lambda r: r["config"].update(smoke=True)),
            confirmation=("B4-s43", lambda r: r["config"]["eval"].update(allow_confirmation=True)),
            test_split=("B5-L4-s17", lambda r: r["config"]["eval"].update(splits=["test"])),
            rank=("B3-L4-s17", lambda r: r["config"]["model"].update(r=128)),
            steps=("B1-s17", lambda r: r["config"]["train"].update(steps=1)),
            lr=("B2-s29", lambda r: r["config"]["train"].update(lr=.5)),
            B0_steps_unchecked=("B0-s17", lambda r: r["config"]["train"].update(steps=5))).items():
        with edited(record[run], edit):
            cases[f"record_{name}"] = outcome(check_training, manifest, authorization)
    for name, edit in dict(
            condition_label=lambda m: m.update(condition="oracle-r32"), rank_label=lambda m: m.update(rank=32),
            incomplete_r128=lambda m: m["runs"].pop("B8-s17")).items():
        with edited(fix["manifest128"], edit):
            cases[f"r128_{name}"] = outcome(check_training, manifest, authorization)
    return cases


# ---- endpoints ----------------------------------------------------------------------------------------------------

def sample(depths, *, good=False, fail_seed=False, hop1_failure=False):
    rows = []
    for depth in depths:
        for episode in range(8):
            for hop in range(1, 9):
                correct = good and not fail_seed
                if hop1_failure and hop == 1:
                    correct = False
                rows.append(("confirm", "memory_only/original", episode, hop - 1, hop, depth, 7, correct))
    names = ("split", "condition", "episode_id", "query_id", "hop", "L", "answer", "correct")
    return {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(names)}


def cohorts():
    import run_p1l_main as run

    result = {}
    for label in run.LABELS:
        depths = range(1, 9) if label == "M1-L4" else [1] if label in {"B4", "B1", "B2"} else [4]
        result[label] = {seed: sample(depths, good=label in {"B8", "M1-L4"}) for seed in run.SEEDS}
    return result


def endpoints_part() -> dict:
    import finalize_p1l_main as fin
    import run_p1l_main as run

    primary, typed = cohorts(), cohorts()
    variants = {"good": (primary, typed)}
    weak_seed = cohorts()
    weak_seed["B8"][43] = sample([4], good=False)
    variants["weak_seed"] = (weak_seed, typed)
    weak_typed = cohorts()
    weak_typed["B8"] = {seed: sample([4], good=False) for seed in run.SEEDS}
    variants["weak_typed"] = (primary, weak_typed)
    hop1 = cohorts()
    hop1["B8"] = {seed: sample([4], good=True, hop1_failure=True) for seed in run.SEEDS}
    variants["hop1_failure"] = (hop1, typed)
    fixed_query = cohorts()
    fixed_query["B5-L4"] = {seed: sample([4], good=True) for seed in run.SEEDS}
    variants["fixed_query_matches"] = (fixed_query, typed)
    unused = cohorts()
    unused["B4"] = {seed: sample([1], good=False) for seed in run.SEEDS}
    unused["B3-L4"] = {seed: sample([4], good=True) for seed in run.SEEDS}
    variants["memory_not_used"] = (unused, typed)
    result = {}
    for name, (first, second) in variants.items():
        for confidence in (.95, .975):
            memory = fin.memory_use_check(first, confidence)
            result[f"{name}@{confidence}"] = dict(
                endpoints=json.loads(json.dumps(fin.endpoints(first, second, confidence), sort_keys=True)),
                memory_use=json.loads(json.dumps(memory, sort_keys=True)),
                lines={str(rank): fin.memory_use_lines(rank, memory) for rank in (32, 128)})
    return result


# ---- finalize and the evaluator -----------------------------------------------------------------------------------

def finalize_part(fix: dict, artifacts: Path) -> dict:
    import finalize_p1l_main as fin

    with contextlib.redirect_stdout(io.StringIO()):
        fin.finalize(SimpleNamespace(manifest=Path(fix["manifest32"]), evaluations=Path(fix["evaluations"]),
                                     authorization=Path(fix["authorization"]), out=artifacts))
    return {name: (artifacts / name).read_text() for name in FINALIZE_OUTPUTS}


def tiny_model():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from looplora.infra.seeds import derive_seed

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "oracle-tiny-qwen"))
        return Qwen3ForCausalLM(Qwen3Config(
            vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()


def evaluate_part(scratch: Path) -> dict:
    """Which (split, start) evaluate() asks the generator for, and the ranges it records. The generator is replaced by
    a recorder that always returns validation episodes, so no test or confirmation episode exists."""
    import looplora.llm.evaluate as ev
    from looplora.llm.config import resolve_config
    from looplora.llm.inject import load_tokenizer
    from looplora.llm.train import LLMExperiment

    model, tokenizer, real = tiny_model(), load_tokenizer(), ev.generate
    result = {}
    for name, splits in dict(val=["val"], test=["test"], confirm=["confirm"], val_and_confirm=["val", "confirm"]).items():
        calls = []

        def recorder(spec, split, start, count, seed, calls=calls):
            calls.append([split, start, count, seed])
            return real(spec, "val", 0, count, seed)

        cfg = resolve_config(dict(
            name="p1ls-defaults", writer="oracle_random", stage="reader", variant="M1", model=dict(r=32, layer=1),
            task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
            train=dict(steps=1, batch_size=2, depths=[4]),
            eval=dict(episodes=2, batch_size=2, depths=[1, 4], hops=[1, 2, 3, 4], counterfactuals=False,
                      splits=splits, allow_confirmation="confirm" in splits)))
        experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
        ev.generate = recorder
        try:
            metrics, _ = ev.evaluate(experiment, cfg, scratch / f"evaluate-{name}")
        finally:
            ev.generate = real
            experiment.close()
        result[name] = dict(calls=calls, episode_ranges=metrics["episode_ranges"],
                            cells=[[row["split"], row["condition"], row["hop"], row["L"], row["n"]]
                                   for row in metrics["table"]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tree / "scripts"))
    import looplora

    fix = json.loads(args.fixtures.read_text())
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        result = dict(source=str(Path(looplora.__file__).resolve().parents[1]), configs=configs_part(args.tree),
                      run=run_part(), launcher=launcher_part(scratch), analyze=analyze_part(scratch),
                      check_training=check_part(fix), endpoints=endpoints_part(),
                      finalize=finalize_part(fix, args.artifacts), evaluate=evaluate_part(scratch))
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
