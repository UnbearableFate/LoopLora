"""P1-L configuration, shared-writer comparisons and scheduling contracts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from looplora.infra.seeds import derive_seed
from looplora.llm.config import load_config, resolve_config
from looplora.llm import run

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_matrix_matches_stage_budget_and_keeps_formal_data_separate(tmp_path):
    default = load_config(ROOT / "configs/p1l/default.yaml")
    jobs = run.load_suite(ROOT / "configs/p1l/smoke.yaml", tmp_path)
    assert len(jobs) == 14
    assert default["eval"]["seed"] == 4242
    for job in jobs:
        config = job["config"]
        assert resolve_config(config) == config
        assert config["smoke"] and config["seed"] == 20261027
        assert config["train"]["steps"] == default["train"]["steps"] // 10
        assert config["train"]["batch_size"] == 8
        assert config["model"]["r"] == 32 and config["model"]["layer"] == 14
        assert config["task"]["E"] == 24 and config["task"]["Q"] == 4
        assert config["eval"]["episodes"] == 200 and config["eval"]["splits"] == ["val"]
        assert config["eval"]["seed"] == derive_seed(20261027, "p1l-smoke-eval")
        assert not config["eval"]["allow_confirmation"]
        if config["stage"] in ("writer", "baseline", "lora_ctx") or config["variant"] == "B4":
            assert config["eval"]["depths"] == [1]
        else:
            assert config["eval"]["depths"] == [0, 1, 2, 3, 4, 6, 8]
    readers = [j for j in jobs if j["config"]["stage"] == "reader"]
    joint = [j for j in jobs if j["config"]["stage"] == "joint"]
    assert {j["config"]["variant"] for j in readers} == {"M1", "M1-lin", "B5", "B3", "B4"}
    assert {j["config"]["variant"] for j in joint} == {"M1", "M1-lin", "B5", "B3", "B4"}
    assert {j["config"]["writer_checkpoint"] for j in readers} == {str(tmp_path / "writer/writer.pt")}
    assert all(j["dependencies"] == ["writer"] for j in readers)
    writer = next(j["config"] for j in jobs if j["name"] == "writer")
    assert writer["task"]["name"] == "t1" and writer["task"]["head_sharing"] == .3
    assert writer["task"]["n_rel"] == 4 and writer["eval"]["hops"] == [1]
    for job in jobs:
        config = job["config"]
        secondary = config["variant"] in ("B0", "B1", "B2") or (
            config["variant"] == "M1" and config["stage"] in ("reader", "joint"))
        assert ("context_retained" in config["eval"]["protocols"]) == secondary
    context = {j["config"]["variant"]: j["config"] for j in jobs if j["config"]["stage"] == "lora_ctx"}
    assert set(context) == {"B1", "B2"}
    for key in ("model", "task", "train", "eval", "seed", "protocol"):
        assert context["B1"][key] == context["B2"][key]
    assert context["B2"]["protocol"] == "context_retained"


@pytest.mark.parametrize("updates", [
    {"smoke": True, "eval": {"splits": ["confirm"], "allow_confirmation": True}},
    {"smoke": True, "eval": {"splits": ["test"]}},
    {"smoke": True, "eval": {"allow_confirmation": True}},
    {"eval": {"splits": ["confirm"]}}, {"eval": {"episodes": 2001}},
    {"eval": {"episode_start": 100000}}, {"eval": {"start": 99999}},
    {"stage": "reader"}, {"stage": "writer", "variant": "B3"},
    {"stage": "lora_ctx", "variant": "B1"},
    {"stage": "lora_ctx", "variant": "B2"},
    {"stage": "joint", "variant": "B2"},
    {"train": {"steps": 0}}, {"train": {"batch_size": True}},
    {"train": {"depths": []}}, {"eval": {"depths": [-1]}},
    {"eval": {"hops": [9]}}, {"model": {"rho": float("nan")}},
    {"stats": {"resamples": 1999}}, {"train": {"typo": 1}}, {"typo": 1},
])
def test_invalid_config_fails_before_loading_a_model(updates):
    with pytest.raises((ValueError, TypeError)):
        resolve_config(updates)


def test_confirmation_requires_explicit_non_smoke_authorization():
    config = resolve_config({"eval": {"splits": ["confirm"], "allow_confirmation": True}})
    assert not config["smoke"] and config["eval"]["splits"] == ["confirm"]


@pytest.mark.parametrize("stage,variant", [("writer", "M1"), ("reader", "M1"), ("lora_ctx", "B2")])
def test_record_preserves_training_seed_and_external_writer_cost_lineage(tmp_path, monkeypatch, stage, variant):
    import torch
    from looplora.llm import evaluate as evaluation
    from looplora.llm import train as training

    mapping = {"name": "record", "device": "cpu", "stage": stage, "variant": variant,
               "seed": 20261027, "eval": {"episodes": 2, "depths": [1], "counterfactuals": False}}
    if stage == "reader":
        mapping["writer_checkpoint"] = str(tmp_path / "writer/writer.pt")
    if stage == "lora_ctx":
        mapping["protocol"] = "context_retained"
    config = resolve_config(mapping)
    model, tokenizer = torch.nn.Linear(2, 2), object()
    closed = []
    expected_seed = derive_seed(config["seed"], "llm", "writer-data" if stage == "writer" else "train-data")
    def train(config, out, **kwargs):
        assert kwargs == {"model": model, "tokenizer": tokenizer}
        experiment = SimpleNamespace(model=model, adapter=None, writer=None, close=lambda: closed.append(True))
        return experiment, {"data_seed": expected_seed, "train_seconds": 1., "train_tokens": 16}
    def evaluate(experiment, config, out):
        np.savez(out / "samples.npz", split=np.array(["val"]), condition=np.array(["memory_only/original"]),
                 episode_id=np.array([0]), query_id=np.array([0]), hop=np.array([1]), L=np.array([1]),
                 answer=np.array([1]), correct=np.array([True]))
        return {"table": []}, {}
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setattr(run, "source_metadata", lambda: {"git_commit": "fixture", "source_hash": "fixture", "source_dirty": False})
    record = run.run_one(config, tmp_path / "run", model=model, tokenizer=tokenizer)
    assert record["status"] == "complete" and closed == [True]
    assert record["data_seed"] == expected_seed
    assert record["pretrain_tokens"] == record["pretrain_seconds"] == 0
    if stage == "reader":
        assert record["consumed_writer_checkpoint"] == str(tmp_path / "writer/writer.pt")
        assert record["writer_source_record"] == str(tmp_path / "writer/record.json")
        assert "no cost allocated to this reader" in record["pretrain_cost_accounting"]
    else:
        assert record["consumed_writer_checkpoint"] is None and record["writer_source_record"] is None
    if variant == "B2":
        assert record["state_bytes"] == 0


def test_parameter_accounting_includes_only_active_external_writer_parameters():
    import torch

    model = torch.nn.Linear(2, 2)
    model.requires_grad_(False)
    adapter = torch.nn.Module()
    for name in ("A", "B", "R"):
        adapter.register_parameter(name, torch.nn.Parameter(torch.zeros(2, 2)))
    model.add_module("adapter", adapter)
    writer = torch.nn.Module()
    writer.A = adapter.A
    for name in ("K", "V"):
        writer.register_parameter(name, torch.nn.Parameter(torch.zeros(3, 2)))
    for name in ("a_lambda", "a_beta"):
        writer.register_parameter(name, torch.nn.Parameter(torch.zeros(())))
    for name in ("R", "U", "Q", "B"):
        writer.register_parameter(name, torch.nn.Parameter(torch.zeros(40, 40)))
    writer.requires_grad_(False)
    counts = run._parameter_counts(SimpleNamespace(model=model, adapter=adapter, writer=writer))
    assert counts == {"total": 32, "trainable": 8, "model_total": 18,
                      "backbone": 6, "adapter": 12, "external_writer": 14, "active_adapter": 26}


def _suite(tmp_path, entries):
    (tmp_path / "base.yaml").write_text("device: cpu\n")
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(entries))
    return path


@pytest.mark.parametrize("entries", [
    [{"name": "same", "config": "base.yaml"}] * 2,
    [{"name": "../escape", "config": "base.yaml"}],
    [{"name": "a", "config": "base.yaml", "depends_on": ["absent"]}],
    [{"name": "a", "config": "base.yaml", "depends_on": ["b"]},
     {"name": "b", "config": "base.yaml", "depends_on": ["a"]}],
    [{"name": "a", "config": "base.yaml"},
     {"name": "b", "config": "base.yaml", "writer_from": "a", "overrides": {"stage": "reader"}}],
])
def test_suite_rejects_collisions_and_invalid_artifact_dependencies(tmp_path, entries):
    with pytest.raises(ValueError):
        run.load_suite(_suite(tmp_path, entries), tmp_path / "out")


def test_suite_rejects_duplicate_comparison_variant_instead_of_overwriting_lr_run(tmp_path):
    entries = [{"name": "low-lr", "config": "base.yaml", "overrides": {"train": {"lr": .0001}}},
               {"name": "high-lr", "config": "base.yaml", "overrides": {"train": {"lr": .001}}}]
    with pytest.raises(ValueError, match="duplicate variant M1.*low-lr and high-lr"):
        run.load_suite(_suite(tmp_path, entries), tmp_path / "out")
    jobs = [{"name": entry["name"], "config": resolve_config({"name": entry["name"], **entry["overrides"]})}
            for entry in entries]
    with pytest.raises(ValueError, match="duplicate variant M1.*low-lr and high-lr"):
        run.compare_suite(jobs, tmp_path / "no-artifacts")


def test_sequential_execution_reuses_one_backbone_and_skips_failed_dependents(tmp_path, monkeypatch):
    model, tokenizer = object(), object()
    loads, seen = [], []
    def load(device):
        loads.append(device)
        return model, tokenizer, .2
    def one(config, out, **kwargs):
        assert kwargs == {"model": model, "tokenizer": tokenizer}
        seen.append(config["name"])
        if config["name"] == "fail":
            raise RuntimeError("deliberate training failure")
        return {"status": "complete"}
    monkeypatch.setattr(run, "_load_backbone", load)
    monkeypatch.setattr(run, "run_one", one)
    configs = [resolve_config({"name": name, "device": "cpu"}) for name in ("first", "fail", "dependent", "last")]
    jobs = [{"name": c["name"], "config": c, "dependencies": ["fail"] if c["name"] == "dependent" else []}
            for c in configs]
    summary = run._execute_jobs(jobs, tmp_path / "runs", 1)
    assert loads == ["cpu"] and seen == ["first", "fail", "last"]
    assert summary["runs"]["dependent"]["status"] == "skipped"
    assert summary["runs"]["fail"]["status"] == "failed"
    assert len(summary["model_load_seconds_by_pid"]) == 1
    assert summary["node_hours"] == summary["seconds"] / 3600
    assert run._WORKER_MODEL is None and run._WORKER_TOKENIZER is None


def test_scheduling_measurement_is_bounded_and_selects_successful_faster_mode(tmp_path, monkeypatch):
    calls = []
    def execute(jobs, out, procs):
        calls.append(procs)
        assert len(jobs) == 2
        for job in jobs:
            config = job["config"]
            assert config["stage"] == "baseline" and config["variant"] == "B0"
            assert config["smoke"] and config["eval"]["splits"] == ["val"]
            assert config["eval"]["episodes"] == 32 and config["eval"]["depths"] == [1]
            assert not config["eval"]["counterfactuals"]
        seconds = 8. if procs == 1 else 5.
        return {"seconds": seconds, "node_hours": seconds / 3600,
                "runs": {j["name"]: {"status": "complete"} for j in jobs}}
    monkeypatch.setattr(run, "_execute_jobs", execute)
    measured = run.measure_scheduling(tmp_path / "schedule", device="cpu")
    assert calls == [1, 2] and measured["selected_procs"] == 2
    assert measured["total_node_hours"] == pytest.approx(13 / 3600)
    assert json.loads((tmp_path / "schedule/scheduling.json").read_text()) == measured


def _comparison_artifacts(out, name, variant, *, stage="reader", bad_hash=False):
    config = resolve_config({"name": name, "stage": stage, "variant": variant, "device": "cpu",
                             **({"writer_checkpoint": "/shared/writer.pt"} if stage == "reader" else {}),
                             "eval": {"episodes": 2, "hops": [1], "counterfactuals": False}})
    folder = out / name
    folder.mkdir(parents=True)
    protocols = config["eval"]["protocols"]
    rows = [(p, L, ep) for p in protocols for L in config["eval"]["depths"] for ep in range(2)]
    np.savez(folder / "samples.npz", split=np.array(["val"] * len(rows)),
             condition=np.array([f"{p}/original" for p, _, _ in rows]),
             protocol=np.array([p for p, _, _ in rows]),
             episode_id=np.array([ep for _, _, ep in rows]), query_id=np.zeros(len(rows), dtype=int),
             hop=np.ones(len(rows), dtype=int), L=np.array([L for _, L, _ in rows]),
             answer=np.ones(len(rows), dtype=int), correct=np.ones(len(rows), dtype=bool))
    hashes = [{"split": "val", "protocol": p, "condition": f"{p}/original", "episode_start": 0,
               "x": "different" if bad_hash else "x", "u": "u", "M": "M", "write_count": "count"}
              for p in protocols]
    (folder / "metrics.json").write_text(json.dumps({"invariant_hashes": hashes}))
    return {"name": name, "config": config, "dependencies": []}


def test_paired_comparison_filters_only_protocol_and_asserts_shared_writer_hashes(tmp_path):
    jobs = [_comparison_artifacts(tmp_path, "M1", "M1"),
            _comparison_artifacts(tmp_path, "B4", "B4"),
            _comparison_artifacts(tmp_path, "B3", "B3")]
    compared = run.compare_suite(jobs, tmp_path)
    assert compared["frozen_writer_hashes_verified"] == [["M1", "B4", "B3"]]
    assert {(c["b"], c["depth_b"]) for c in compared["comparisons"]} == {("B4", 1), ("B3", 4)}
    path = tmp_path / "B3/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["invariant_hashes"][0]["u"] = "changed"
    path.write_text(json.dumps(metrics))
    with pytest.raises(AssertionError, match="hashes differ"):
        run.compare_suite(jobs, tmp_path)


def test_joint_variants_do_not_assert_shared_writer(tmp_path):
    jobs = [_comparison_artifacts(tmp_path, "M1", "M1", stage="joint"),
            _comparison_artifacts(tmp_path, "B3", "B3", stage="joint", bad_hash=True)]
    assert run.compare_suite(jobs, tmp_path)["frozen_writer_hashes_verified"] == []


def test_paired_comparison_rejects_missing_episode_instead_of_intersecting(tmp_path):
    jobs = [_comparison_artifacts(tmp_path, "M1", "M1"),
            _comparison_artifacts(tmp_path, "B3", "B3")]
    path = tmp_path / "B3/samples.npz"
    with np.load(path, allow_pickle=False) as archive:
        keep = ~((archive["L"] == 4) & (archive["episode_id"] == 1))
        samples = {key: archive[key][keep] for key in archive.files}
    np.savez(path, **samples)
    with pytest.raises(ValueError, match="identical"):
        run.compare_suite(jobs, tmp_path)
