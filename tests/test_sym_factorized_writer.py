"""F2 factorization, shared writer lifecycle, and symbolic pipeline contracts.

Run only inside a Miyabi-G compute allocation, including the CPU cases.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from looplora.core.memory import delta_update
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.model import SymbolicModel, make_task_spec
from looplora.sym.run import run_one
from looplora.sym.train import load_checkpoint, train
from looplora.sym.writers import FactorizedWriter, LearnedWriter


VARIANTS = ("M1", "M1-lin", "B5", "B5-lin", "B3", "B4")


def _random(shape, label):
    generator = torch.Generator().manual_seed(derive_seed(17, "factorized_test", label))
    return torch.randn(shape, generator=generator, dtype=torch.float64)


def _config(**updates):
    base = {
        "name": "factorized-tiny", "device": "cpu", "seed": 17,
        "writer": "joint_fact", "variant": "M1",
        "model": {"d_e": 4, "r": 8, "r_mem": 8, "max_hop": 8},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 64,
                 "n_rel": 4, "seen": [0, 32], "unseen": [32, 64],
                 "chain_lengths": [4, 4], "hops": [1, 2, 3, 4]},
        "train": {"steps": 3, "batch_size": 2, "warmup_steps": 0,
                  "depths": [1, 4], "log_every": 1},
        "pretrain": {"steps": 3, "batch_size": 2, "warmup_steps": 0,
                     "task": {"E": 12, "head_sharing": .3}},
        "eval": {"episodes": 2, "batch_size": 2, "splits": ["test", "test_unseen"],
                 "depths": [1, 4, 8], "hops": [1, 2, 3, 4],
                 "counterfactuals": True, "swap_successors": True},
    }
    return resolve_config(recursive_merge(base, updates))


def _parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_initial_blocks_gates_and_parameter_budget_match_learned_writer(seed):
    learned = LearnedWriter(15, 7, 5, seed, lambda_init=.8, beta_init=.3)
    factorized = FactorizedWriter(15, 7, 5, seed, lambda_init=.8, beta_init=.3)
    assert factorized.A_k.shape == (7, 10)
    assert factorized.A_v.shape == (7, 5)
    assert torch.equal(factorized.A_k, learned.A[:, :10])
    assert torch.equal(factorized.A_v, learned.A[:, 10:])
    for name in ("K", "V", "a_lambda", "a_beta"):
        assert torch.equal(getattr(factorized, name), getattr(learned, name))
    assert _parameter_count(factorized) == _parameter_count(learned) == 7 * 15 + 2 * 5 * 7 + 2
    assert not hasattr(factorized, "A")


def test_initial_query_projection_equals_zero_tail_unfactorized_projection():
    factorized = FactorizedWriter(15, 7, 5, 17).double()
    learned = LearnedWriter(15, 7, 5, 17).double()
    query = _random((2, 4, 15), "query")
    query[..., 10:] = 0
    torch.testing.assert_close(factorized.u(query), learned.u(query), rtol=1e-12, atol=1e-12)


def _single_write_key_and_value(writer, features):
    # A single write into zero memory gives M = beta * value * key^T.
    # Recover its actual key from public outputs, retaining input gradients.
    memory, values = writer(features[:, None, :], torch.ones(len(features), 1, dtype=torch.bool))
    value = values[:, 0]
    denominator = writer.a_beta.sigmoid() * value.square().sum(-1, keepdim=True)
    assert (denominator.detach() > 1e-10).all()
    key = (memory * value[..., None]).sum(-2) / denominator
    return key, value


def _assert_factorization(writer):
    features = _random((3, 15), "invariance_features").requires_grad_()
    key, value = _single_write_key_and_value(writer, features)
    changed_tail = torch.cat((features[:, :10], _random((3, 5), "new_tail")), -1)
    changed_head_relation = torch.cat((_random((3, 10), "new_head_relation"), features[:, 10:]), -1)
    same_key, _ = _single_write_key_and_value(writer, changed_tail)
    _, same_value = _single_write_key_and_value(writer, changed_head_relation)
    torch.testing.assert_close(key, same_key, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(value, same_value, rtol=0, atol=0)
    torch.testing.assert_close(key.norm(dim=-1), torch.ones(3, dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)
    key_gradient = torch.autograd.grad((key * _random(key.shape, "key_probe")).sum(),
                                       features, retain_graph=True)[0]
    value_gradient = torch.autograd.grad((value * _random(value.shape, "value_probe")).sum(), features)[0]
    torch.testing.assert_close(key_gradient[:, 10:], torch.zeros_like(key_gradient[:, 10:]),
                               rtol=0, atol=1e-11)
    assert torch.count_nonzero(value_gradient[:, :10]) == 0
    assert key_gradient[:, :10].norm() > 0
    assert value_gradient[:, 10:].norm() > 0


def test_outputs_and_input_gradients_stay_factorized_after_parameter_updates():
    writer = FactorizedWriter(15, 7, 5, 17).double()
    initial = {name: value.detach().clone() for name, value in writer.named_parameters()}
    _assert_factorization(writer)
    optimizer = torch.optim.SGD(writer.parameters(), lr=.1)
    features = _random((3, 4, 15), "training_features")
    valid = torch.ones(3, 4, dtype=torch.bool)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        memory, values = writer(features, valid)
        loss = (memory - _random(memory.shape, "memory_target")).square().mean()
        loss = loss + (values - _random(values.shape, "value_target")).square().mean()
        loss.backward()
        optimizer.step()
    for name in ("A_k", "A_v", "K", "V", "a_lambda", "a_beta"):
        assert not torch.equal(getattr(writer, name), initial[name]), name
    _assert_factorization(writer)


def test_factored_codes_follow_masked_decay_first_updates_with_finite_gradients():
    writer = FactorizedWriter(15, 7, 5, 17).double()
    features = _random((3, 4, 15), "masked_features").requires_grad_()
    valid = torch.tensor([[True, False, True, True], [False, False, False, False],
                          [True, True, False, False]])
    state, values = writer(features, valid)
    keys = F.normalize(F.linear(F.linear(features[..., :10], writer.A_k), writer.K),
                       dim=-1, eps=1e-6)
    expected_values = F.linear(F.linear(features[..., 10:], writer.A_v), writer.V)
    expected = torch.zeros_like(state)
    for token in range(features.shape[1]):
        expected = delta_update(expected, keys[:, token], expected_values[:, token], valid[:, token],
                                writer.a_lambda.sigmoid(), writer.a_beta.sigmoid())
    torch.testing.assert_close(state, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(values, expected_values, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(state[1]) == 0
    gradient = torch.autograd.grad(state.square().sum(), features)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[~valid]) == 0
    assert gradient[valid].norm() > 0


def _assert_shared_parameters(model):
    assert isinstance(model.writer, FactorizedWriter)
    assert model.adapter.A is model.writer.A_k
    assert model.adapter.cfg.d_in == 2 * model.backbone.d_e
    assert not hasattr(model.writer, "A")
    for name in ("K", "V", "a_lambda", "a_beta"):
        if hasattr(model.adapter, name):
            assert getattr(model.adapter, name) is getattr(model.writer, name)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("condition,legacy", [("joint_fact", "joint"), ("t1_frozen_fact", "t1_frozen")])
def test_models_share_only_factored_writer_parameters_and_keep_parameter_count(variant, condition, legacy):
    config = _config(writer=condition, variant=variant)
    model = SymbolicModel(config)
    original = SymbolicModel(_config(writer=legacy, variant=variant))
    _assert_shared_parameters(model)
    assert _parameter_count(model) == _parameter_count(original)
    assert model.writer_signature() == {**original.writer_signature(), "writer_type": "factorized"}
    assert "writer_type" not in original.writer_signature()
    arrays = generate(make_task_spec(config), "val", 0, 2, derive_seed(17, "factorized_model_queries"))
    prepared = model.prepare(arrays)
    torch.testing.assert_close(prepared.u, original.writer.u(prepared.x), rtol=1e-5, atol=1e-7)
    model.freeze_writer()
    assert all(not parameter.requires_grad for parameter in model.writer.parameters())
    assert model.adapter.B.requires_grad


@pytest.fixture(scope="module")
def shared_writer(tmp_path_factory):
    config = _config(writer="t1_frozen_fact", variant="B4")
    output = tmp_path_factory.mktemp("factorized-shared")
    model, summary = train(config, output)
    return model, summary, output / "writer.pt"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("external", [False, True], ids=["internal-t1", "shared-checkpoint"])
def test_frozen_t1_writer_and_full_checkpoint_roundtrip_all_readers(tmp_path, shared_writer, variant, external):
    source, source_summary, writer_path = shared_writer
    config = _config(writer="t1_frozen_fact", variant=variant)
    if external:
        config["pretrain"]["writer_checkpoint"] = str(writer_path)
    output = tmp_path / "reader"
    model, summary = train(config, output)
    assert source_summary["pretrain_seconds"] > 0
    assert summary["writer_frozen"] is True
    assert summary["writer_checkpoint"] == str(output / "writer.pt")
    if external:
        assert summary["pretrain_seconds"] == 0
    else:
        assert summary["pretrain_seconds"] > 0
    assert (summary["pretrain"] is None) == external
    record = torch.load(output / "writer.pt", map_location="cpu", weights_only=True)
    assert record["pretrain_config"]["writer"] == "joint_fact"
    assert record["pretrain_config"]["task"]["name"] == "t1"
    assert record["pretrain_config"]["task"]["E"] == 12
    assert record["pretrain_config"]["task"]["n_rel"] == 4
    assert record["pretrain_config"]["task"]["head_sharing"] == .3
    assert record["signature"] == model.writer_signature()
    for name, value in source.writer.state_dict().items():
        assert torch.equal(model.writer.state_dict()[name], value)
        assert torch.equal(record["writer"][name], value)
    assert all(not parameter.requires_grad for parameter in model.writer.parameters())
    _assert_shared_parameters(model)
    restored = load_checkpoint(output / "checkpoint.pt", device="cpu")
    _assert_shared_parameters(restored)
    assert all(not parameter.requires_grad for parameter in restored.writer.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name
    arrays = generate(make_task_spec(config), "val", 0, 2, derive_seed(17, "factorized_roundtrip"))
    batch, source_batch = model.prepare(arrays), source.prepare(arrays)
    for name in ("x", "u", "memory", "write_count"):
        assert torch.equal(getattr(batch, name), getattr(source_batch, name)), name
    for depth in (1, 4):
        actual = restored.read(restored.prepare(arrays), depth)
        expected = model.read(batch, depth)
        for name in ("logits", "z", "first_read_cosine"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, equal_nan=True)


def test_joint_training_updates_both_factorized_blocks_and_exports_reloadable_writer(tmp_path):
    config = _config(variant="B4")
    initial = SymbolicModel(config)
    model, summary = train(config, tmp_path)
    assert summary["writer_frozen"] is False
    assert summary["pretrain_seconds"] == 0
    assert summary["writer_checkpoint"] == str(tmp_path / "writer.pt")
    assert all(parameter.requires_grad for parameter in model.writer.parameters())
    for name in ("A_k", "A_v", "K", "V", "a_lambda", "a_beta"):
        assert not torch.equal(getattr(model.writer, name), getattr(initial.writer, name)), name
    record = torch.load(tmp_path / "writer.pt", map_location="cpu", weights_only=True)
    assert record["signature"] == model.writer_signature()
    assert record["training_config"] == config
    for name, value in model.writer.state_dict().items():
        assert torch.equal(record["writer"][name], value)
    restored = load_checkpoint(tmp_path / "checkpoint.pt", device="cpu")
    _assert_shared_parameters(restored)
    assert all(parameter.requires_grad for parameter in restored.writer.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name


@pytest.mark.parametrize("condition", ["joint_fact", "t1_frozen_fact"])
def test_run_one_evaluates_counterfactuals_and_records_writer_provenance(tmp_path, shared_writer, condition):
    _, _, writer_path = shared_writer
    config = _config(writer=condition)
    if condition == "t1_frozen_fact":
        config["pretrain"]["writer_checkpoint"] = str(writer_path)
    output = tmp_path / "pipeline"
    record = run_one(config, output)
    assert record["status"] == "complete"
    assert record["writer"] == condition
    assert record["writer_checkpoint"] == str(output / "writer.pt")
    assert record["consumed_writer_checkpoint"] == (str(writer_path) if condition == "t1_frozen_fact" else None)
    assert record["writer_source_record"] == (str(writer_path.parent / "record.json")
                                               if condition == "t1_frozen_fact" else None)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["gate0_passed"] is True
    with np.load(output / "samples.npz", allow_pickle=False) as samples:
        assert record["eval_samples"] == len(samples["answer"])
        assert {"original", "remove_required_fact", "replace_key_fact", "swap_memory", "swap_successors"} <= set(samples["condition"])
        assert set(samples["split"]) == {"test", "test_unseen"}
        assert set(samples["L"]) == {1, 4, 8}
        for row in record["scores"]:
            selected = ((samples["split"] == row["split"]) & (samples["condition"] == row["condition"])
                        & (samples["hop"] == row["hop"]) & (samples["L"] == row["L"]))
            assert int(selected.sum()) == row["n"]
            assert samples["correct"][selected].mean() == pytest.approx(row["accuracy"])
