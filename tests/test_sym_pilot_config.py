"""Pilot capacity references, T1 pretraining task, and checkpoint provenance."""
from __future__ import annotations

import copy

import pytest
import torch

from looplora.data.relations import TaskSpec
from looplora.sym.config import resolve_config
from looplora.sym.model import SymbolicModel
from looplora.sym.train import _pretrain_config, load_checkpoint, train


def _config(**updates):
    return resolve_config({
        "device": "cpu", "writer": "t1_frozen",
        "model": {"d_e": 8, "r": 16, "r_mem": 16},
        "task": {"name": "t2_chain_end", "E": 4, "Q": 2, "V": 64,
                 "n_rel": 4, "seen": [0, 48], "unseen": [48, 64],
                 "chain_lengths": [4], "hops": [1, 2]},
        "train": {"steps": 2, "batch_size": 2, "log_every": 1},
        "pretrain": {"steps": 2, "batch_size": 2,
                     "task": {"n_rel": 4, "head_sharing": .3}},
        "eval": {"hops": [1, 2]},
        **updates,
    })


@pytest.mark.parametrize("rank", [16, 32, 80])
def test_reference_rank_defaults_to_declared_main_comparison_rank(rank):
    config = resolve_config({"model": {"r": rank, "r_mem": rank}})
    assert config["model"]["reference_r"] == rank
    assert resolve_config(config) == config


@pytest.mark.parametrize("reference,larger", [(16, 24), (32, 48), (80, 96)])
def test_capacity_controls_use_explicit_reference_instead_of_fixed64(reference, larger):
    b6 = resolve_config({"variant": "B6", "writer": "joint", "model": {
        "r": larger, "r_mem": larger, "reference_r": reference}})
    b7 = resolve_config({"variant": "B7", "writer": "joint", "model": {
        "r": reference, "r_mem": larger, "reference_r": reference}})
    assert b6["model"]["r"] == larger
    assert b7["model"]["r_mem"] == larger
    assert resolve_config(b6) == b6
    assert resolve_config(b7) == b7


@pytest.mark.parametrize("variant,model", [
    ("B6", {}), ("B6", {"reference_r": 16}),
    ("B6", {"r": 32, "r_mem": 32}),
    ("B6", {"r": 32, "r_mem": 32, "reference_r": 32}),
    ("B6", {"r": 16, "reference_r": 32}),
    ("B7", {"r": 16, "reference_r": 16}),
    ("B7", {"r": 16, "r_mem": 24, "reference_r": 32}),
    ("B7", {"r": 32, "r_mem": 32, "reference_r": 16}),
])
def test_capacity_controls_reject_missing_or_nonlarger_dimensions(variant, model):
    with pytest.raises(ValueError, match=variant):
        resolve_config({"variant": variant, "writer": "joint", "model": model})


@pytest.mark.parametrize("reference", [None, 0, -1, True, 16.5])
def test_explicit_reference_rank_requires_positive_integer(reference):
    with pytest.raises(ValueError, match="reference_r"):
        resolve_config({"model": {"reference_r": reference}})


def test_pretrain_task_merges_t1_overrides_and_preserves_input_and_signature():
    config = _config()
    before = copy.deepcopy(config)
    teacher_config = _pretrain_config(config)
    spec = TaskSpec(**teacher_config["task"])
    assert spec.name == "t1"
    assert spec.hops == (1,)
    assert spec.n_rel == 4
    assert spec.head_sharing == .3
    assert spec.E == config["task"]["E"]
    assert spec.V == config["task"]["V"]
    assert teacher_config["variant"] == "B4"
    assert teacher_config["writer"] == "joint"
    assert teacher_config["train"]["depths"] == [1]
    assert SymbolicModel(config).writer_signature() == SymbolicModel(teacher_config).writer_signature()
    assert config == before
    assert resolve_config(config) == config


@pytest.mark.parametrize("task,pattern", [
    (None, "mapping"), ([], "mapping"),
    ({"name": "t2_chain_end"}, "remain t1"),
    ({"n_rel": 2}, "n_rel.*backbone"),
    ({"V": 128}, "V.*backbone"),
    ({"head_sharing": 1.1}, "head_sharing"),
])
def test_pretrain_task_rejects_invalid_specs_and_backbone_mismatches(task, pattern):
    with pytest.raises(ValueError, match=pattern):
        _config(pretrain={"task": task})


def test_pretrain_builder_rejects_backbone_mismatch_for_direct_callers():
    config = _config()
    config["pretrain"]["task"]["n_rel"] = 2
    with pytest.raises(ValueError, match="n_rel.*backbone"):
        _pretrain_config(config)


def test_trained_writer_retains_resolved_pretrain_task_and_reload_provenance(tmp_path):
    config = _config()
    model, summary = train(config, tmp_path / "trained")
    writer = torch.load(tmp_path / "trained/writer.pt", map_location="cpu", weights_only=True)
    assert writer["pretrain_config"]["task"]["n_rel"] == 4
    assert writer["pretrain_config"]["task"]["head_sharing"] == .3
    assert writer["signature"] == model.writer_signature()
    assert summary["pretrain"]["train_tokens"] > 0
    assert all(not parameter.requires_grad for parameter in model.writer.parameters())

    checkpoint = torch.load(tmp_path / "trained/checkpoint.pt", map_location="cpu", weights_only=True)
    checkpoint["config"]["device"] = "cuda:7"
    path = tmp_path / "recorded-cuda.pt"
    torch.save(checkpoint, path)
    restored = load_checkpoint(path, device="cpu")
    assert restored.config["device"] == "cpu"
    assert restored.checkpoint_config == checkpoint["config"]
    restored.config["pretrain"]["task"]["head_sharing"] = .8
    assert restored.checkpoint_config["pretrain"]["task"]["head_sharing"] == .3
    assert restored.checkpoint_config["device"] == "cuda:7"


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_runtime_threads_requires_positive_integer(threads):
    with pytest.raises(ValueError, match="runtime.threads"):
        resolve_config({"runtime": {"threads": threads}})


def test_runtime_threads_is_resolved_and_roundtrips():
    assert resolve_config({})["runtime"]["threads"] == 1
    config = resolve_config({"runtime": {"threads": 3}})
    assert resolve_config(config)["runtime"]["threads"] == 3
    with pytest.raises(ValueError, match="unknown runtime"):
        resolve_config({"runtime": {"threadz": 1}})


def test_joint_t1_export_is_reused_exactly_by_frozen_t2_reader(tmp_path):
    from looplora.data.relations import generate
    from looplora.sym.model import make_task_spec

    source_config = _config(writer="joint", variant="B4")
    source_config["task"].update(name="t1", chain_lengths=None, hops=[1], head_sharing=.3)
    source_config = resolve_config(source_config)
    source, source_summary = train(source_config, tmp_path / "source")
    writer_path = tmp_path / "source/writer.pt"
    assert source_summary["writer_checkpoint"] == str(writer_path)
    exported = torch.load(writer_path, map_location="cpu", weights_only=True)
    assert exported["training_config"] == source_config
    assert exported["training"]["train_tokens"] == source_summary["train_tokens"]
    assert exported["signature"] == source.writer_signature()
    for name, tensor in source.writer.state_dict().items():
        assert torch.equal(exported["writer"][name], tensor.cpu())

    target_config = _config(variant="B5")
    target_config["pretrain"]["writer_checkpoint"] = str(writer_path)
    target, summary = train(target_config, tmp_path / "target")
    assert summary["pretrain"] is None
    assert summary["pretrain_seconds"] == 0
    assert all(not parameter.requires_grad for parameter in target.writer.parameters())
    for name, tensor in source.writer.state_dict().items():
        assert torch.equal(tensor, target.writer.state_dict()[name])
    arrays = generate(make_task_spec(target_config), "val", 0, 2, 219)
    assert torch.equal(source.prepare(arrays).memory, target.prepare(arrays).memory)
