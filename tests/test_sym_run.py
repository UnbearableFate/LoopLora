"""Runner contracts; execute only in a Miyabi-G compute allocation."""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from looplora.sym.config import VARIANTS, load_config, recursive_merge, resolve_config
from looplora.sym import run

ROOT = Path(__file__).resolve().parents[1]


def small_config(**updates):
    base = {
        "name": "tiny", "device": "cpu", "writer": "oracle_orth",
        "model": {"d_e": 8, "r": 16, "r_mem": 16},
        "task": {"name": "t1", "E": 4, "Q": 2, "V": 64,
                 "seen": [0, 48], "unseen": [48, 64], "hops": [1]},
        "train": {"steps": 2, "batch_size": 2, "warmup_steps": 0, "log_every": 1},
        "pretrain": {"steps": 2, "batch_size": 2, "warmup_steps": 0},
        "eval": {"episodes": 2, "batch_size": 2, "depths": [0, 1, 4, 8],
                 "splits": ["test"], "hops": [1], "counterfactuals": False},
    }
    return resolve_config(recursive_merge(base, updates))


def test_resolution_is_idempotent_and_variant_controls_are_explicit():
    for variant in VARIANTS:
        config = resolve_config({"variant": variant, "writer": "joint"})
        assert resolve_config(config) == config
        if variant in {"B4", "B6", "B7"}:
            assert config["train"]["depths"] == config["eval"]["depths"] == [1]
        if variant == "B8":
            assert config["train"]["depths"] == config["eval"]["depths"] == [4]
    assert resolve_config({"variant": "M1-lin"})["activation"] == "identity"
    assert resolve_config({"variant": "B6", "writer": "joint"})["model"]["r"] == 96
    assert resolve_config({"variant": "B7", "writer": "joint"})["model"]["r_mem"] == 96
    assert resolve_config({"task": {"name": "t1"}})["eval"]["hops"] == [1]


@pytest.mark.parametrize("updates", [
    {"train": {"steps": 0}}, {"train": {"batch_size": True}},
    {"train": {"depths": []}}, {"eval": {"depths": [-1]}},
    {"model": {"rho": float("nan")}}, {"model": {"beta_init": 1}},
    {"variant": "B7", "writer": "oracle_random"},
    {"stats": {"resamples": 1999}}, {"train": {"typo": 1}}, {"typo": 1},
])
def test_invalid_experiment_configs_fail_before_execution(updates):
    with pytest.raises((TypeError, ValueError)):
        resolve_config(updates)


def test_smoke_matrix_has_all_variants_and_one_shared_writer(tmp_path):
    defaults = load_config(ROOT / "configs/p1s/default.yaml")
    jobs = run.load_suite(ROOT / "configs/p1s/smoke.yaml", tmp_path)
    assert len(jobs) == 22
    for task in ("t1", "t2_chain_end"):
        main = [job for job in jobs if job["config"]["task"]["name"] == task
                and job["config"]["writer"] != "t1_frozen"]
        assert {job["config"]["variant"] for job in main} == set(VARIANTS)
    assert {job["config"]["seed"] for job in jobs} == {17}
    assert all(job["config"]["train"]["steps"] == defaults["train"]["steps"] // 10 for job in jobs)
    assert all(job["config"]["pretrain"]["steps"] == defaults["pretrain"]["steps"] // 10 for job in jobs)
    consumers = [job for job in jobs if job["dependencies"]]
    assert len(consumers) == 5
    assert {job["config"]["pretrain"]["writer_checkpoint"] for job in consumers} == {
        str(tmp_path / "t1-shared-writer/writer.pt")}


def _suite(tmp_path, entries):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"device": "cpu"}))
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(entries))
    return path


@pytest.mark.parametrize("entries", [
    [{"name": "same", "config": "base.yaml"}] * 2,
    [{"name": "../escape", "config": "base.yaml"}],
    [{"name": "a", "config": "base.yaml", "depends_on": ["absent"]}],
    [{"name": "a", "config": "base.yaml", "depends_on": ["b"]},
     {"name": "b", "config": "base.yaml", "depends_on": ["a"]}],
])
def test_suite_rejects_collisions_and_invalid_dependencies(tmp_path, entries):
    path = _suite(tmp_path, entries)
    with pytest.raises(ValueError):
        run.load_suite(path, tmp_path / "out")


def test_suite_bounds_real_processes_shares_gpu_and_propagates_failure(tmp_path, monkeypatch):
    path = _suite(tmp_path, [
        {"name": "a", "config": "base.yaml"},
        {"name": "b", "config": "base.yaml"},
        {"name": "fail", "config": "base.yaml"},
        {"name": "dependent", "config": "base.yaml", "depends_on": ["fail"]},
        {"name": "after", "config": "base.yaml", "depends_on": ["a"]},
    ])
    # Substitute a tiny child executable while retaining real process scheduling.
    child = tmp_path / "child.py"
    child.write_text('''import json, os, pathlib, sys, time
config_path, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
config = json.loads(config_path.read_text())
output.mkdir()
info = {"start": time.monotonic(), "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}
time.sleep(.25)
info["end"] = time.monotonic()
(output / "trace.json").write_text(json.dumps(info))
sys.exit(7 if config["name"] == "fail" else 0)
''')
    popen = subprocess.Popen
    def launch(command, **kwargs):
        config = command[command.index("--config") + 1]
        out = command[command.index("--out") + 1]
        return popen([sys.executable, str(child), config, out], **kwargs)
    monkeypatch.setattr(run.subprocess, "Popen", launch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    out = tmp_path / "out"
    assert run.run_suite(path, out, procs=2) == 1
    summary = json.loads((out / "suite.json").read_text())
    assert summary["runs"]["fail"]["returncode"] == 7
    assert summary["runs"]["dependent"]["status"] == "skipped"
    assert not (out / "dependent").exists()
    traces = {name: json.loads((out / name / "trace.json").read_text())
              for name in ("a", "b", "fail", "after")}
    assert {trace["gpu"] for trace in traces.values()} == {"3"}
    events = sorted([(t["start"], 1) for t in traces.values()] +
                    [(t["end"], -1) for t in traces.values()])
    active, peak = 0, 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert peak == 2
    assert traces["after"]["start"] >= traces["a"]["end"]
    with pytest.raises(FileExistsError):
        run.run_suite(path, out, procs=2)


def test_source_provenance_uses_explicit_root_and_tracks_dirty_code(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/source.py").write_text("VALUE = 1\n")
    git("add", "src/source.py")
    git("-c", "user.name=Runner test", "-c", "user.email=runner@example.invalid",
        "commit", "-m", "fixture")
    monkeypatch.setenv("LL_SOURCE_ROOT", str(tmp_path))
    first = run.source_metadata()
    assert first["git_commit"] == git("rev-parse", "HEAD")
    assert first["source_dirty"] is False
    (tmp_path / "src/source.py").write_text("VALUE = 2\n")
    changed = run.source_metadata()
    assert changed["git_commit"] == first["git_commit"]
    assert changed["source_dirty"] is True
    assert changed["source_hash"] != first["source_hash"]


def test_snapshot_provenance_uses_submitter_commit_without_git(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    (archive / "src").mkdir(parents=True)
    (archive / "src/module.py").write_text("VALUE = 1\n")
    spec = tmp_path / "job-spec.json"
    commit = "a1" * 20
    spec.write_text(json.dumps({"source": {"kind": "commit", "ref": commit}}))
    monkeypatch.setenv("LL_SOURCE_ROOT", str(archive))
    monkeypatch.setenv("LL_SPEC", str(spec))
    first = run.source_metadata()
    assert first["git_commit"] == commit
    assert first["source_dirty"] is False
    assert len(first["source_hash"]) == 32
    (archive / "src/module.py").write_text("VALUE = 2\n")
    assert run.source_metadata()["source_hash"] != first["source_hash"]
    spec.write_text(json.dumps({"source": {"kind": "worktree", "ref": str(archive)}}))
    with pytest.raises(ValueError, match="full commit"):
        run.source_metadata()


def test_run_writes_recomputable_artifacts_and_does_not_overwrite(tmp_path):
    out = tmp_path / "tiny"
    record = run.run_one(small_config(), out)
    assert record["status"] == "complete"
    for name in ("config.json", "checkpoint.pt", "train_log.jsonl", "metrics.json",
                 "samples.npz", "timing.json", "record.json"):
        assert (out / name).is_file()
    metadata = json.loads((out / "config.json").read_text())
    assert metadata["config_hash"] == record["config_hash"]
    assert len(metadata["config_hash"]) == len(metadata["source_hash"]) == 32
    with np.load(out / "samples.npz", allow_pickle=False) as samples:
        assert record["eval_samples"] == len(samples["answer"])
        for row in record["scores"]:
            selection = ((samples["split"] == row["split"]) &
                         (samples["condition"] == row["condition"]) &
                         (samples["hop"] == row["hop"]) & (samples["L"] == row["L"]))
            assert selection.sum() == row["n"]
            assert samples["correct"][selection].mean() == pytest.approx(row["accuracy"])
    assert record["parameters"]["trainable"] > 0
    assert record["train_tokens"] > 0
    assert record["state_bytes"] == 2 * 16 * 16 * 4
    with pytest.raises(FileExistsError):
        run.run_one(small_config(), out)


def test_run_failure_retains_provenance_and_reason(tmp_path, monkeypatch):
    from looplora.sym import train
    def fail(*args):
        raise RuntimeError("intentional training failure")
    monkeypatch.setattr(train, "train", fail)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="intentional"):
        run.run_one(small_config(), out)
    record = json.loads((out / "record.json").read_text())
    assert record["status"] == "failed"
    assert record["failure_reason"] == "RuntimeError: intentional training failure"
    assert record["config_hash"]
    assert (out / "timing.json").is_file()


def test_frozen_writer_checkpoint_is_shared_across_readers(tmp_path):
    from looplora.data.relations import generate
    from looplora.sym.model import make_task_spec
    from looplora.sym.train import train

    source_config = small_config(writer="t1_frozen", variant="B4")
    source, _ = train(source_config, tmp_path / "source")
    target_config = small_config(
        writer="t1_frozen", variant="B5",
        pretrain={"writer_checkpoint": str(tmp_path / "source/writer.pt")},
        task={"name": "t2_chain_end", "chain_lengths": [4], "hops": [1, 2]},
        eval={"hops": [1, 2]})
    target, summary = train(target_config, tmp_path / "target")
    assert summary["pretrain_seconds"] == 0
    assert all(not parameter.requires_grad for parameter in target.writer.parameters())
    for name, parameter in source.writer.state_dict().items():
        assert torch.equal(parameter, target.writer.state_dict()[name])
    arrays = generate(make_task_spec(target_config), "test", 0, 2, 934)
    assert torch.equal(source.prepare(arrays).memory, target.prepare(arrays).memory)


def test_registry_is_deterministic_includes_failures_and_preserves_nested_fields(tmp_path):
    spec = importlib.util.spec_from_file_location("collect_registry", ROOT / "scripts/collect_registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, status in (("z", "failed"), ("a", "complete")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "record.json").write_text(json.dumps({
            "experiment_id": name, "status": status, "scores": [{"accuracy": .5}],
            "failure_reason": "training,failed\nline 2" if status == "failed" else None,
            "extra": {"z": 2, "a": 1},
        }))
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    assert module.write_registry([tmp_path / "z", tmp_path / "a"], one) == 2
    module.write_registry([tmp_path / "a", tmp_path / "z", tmp_path / "a"], two)
    assert one.read_bytes() == two.read_bytes()
    with one.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["experiment_id"] for row in rows] == ["a", "z"]
    assert rows[1]["failure_reason"] == "training,failed\nline 2"
    assert json.loads(rows[0]["scores"]) == [{"accuracy": .5}]
    assert rows[0]["extra"] == '{"a":1,"z":2}'
