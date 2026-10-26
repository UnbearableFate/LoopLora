"""Symbolic features, controlled writer state, and memory-QA behavior."""
import copy

import numpy as np
import pytest
import torch

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.sym.backbone import SymbolicBackbone
from looplora.sym.config import resolve_config
from looplora.sym.model import SymbolicModel, make_task_spec
from looplora.sym.train import load_checkpoint, train


def small_config(**updates):
    return resolve_config({"device": "cpu", "writer": "oracle_orth", "variant": "M1",
        "model": {"r": 16, "r_mem": 16, "d_e": 8},
        "task": {"name": "t1", "E": 4, "Q": 3, "V": 32,
                 "seen": [0, 16], "unseen": [16, 32]},
        "train": {"steps": 3, "batch_size": 4},
        "pretrain": {"steps": 3, "batch_size": 4}, **updates})


def test_backbone_is_fixed_seeded_and_has_exact_fact_query_layout():
    torch.random.default_generator.manual_seed(derive_seed(1, "test_backbone"))
    before = torch.random.get_rng_state().clone()
    backbone = SymbolicBackbone(V=32, n_rel=2, d_e=8, max_hop=8, backbone_seed=7)
    assert torch.equal(before, torch.random.get_rng_state())
    again = SymbolicBackbone(V=32, n_rel=2, d_e=8, max_hop=8, backbone_seed=7)
    assert not list(backbone.parameters())
    assert torch.equal(backbone.entities, again.entities)
    assert not torch.equal(backbone.entities, SymbolicBackbone(V=32, n_rel=2, d_e=8, backbone_seed=8).entities)
    facts = torch.tensor([[[2, 1, 5]]])
    torch.testing.assert_close(backbone.facts(facts)[0, 0],
                               torch.cat((backbone.entities[2], backbone.relations[1], backbone.entities[5])))
    x = backbone.queries(torch.tensor([[2]]), torch.tensor([[1]]), torch.tensor([[-1]]))
    torch.testing.assert_close(x[0, 0, 8:16], backbone.relations[1] + backbone.hops[0])
    assert torch.count_nonzero(x[..., 16:]) == 0


def test_orthogonal_writer_retrieves_every_stored_value_exactly():
    config = small_config()
    model = SymbolicModel(config)
    arrays = generate(make_task_spec(config), "test", 0, 4, derive_seed(9, "test_oracle"))
    batch = model.prepare(arrays)
    for i, facts in enumerate(arrays.facts):
        for head, _, tail in facts:
            torch.testing.assert_close(batch.memory[i] @ batch.codes[i, head], batch.codes[i, tail],
                                       atol=5e-7, rtol=5e-7)
    assert not list(model.writer.parameters())
    assert torch.count_nonzero(model.read(batch, 4)["logits"]) == 0
    assert batch.write_count.tolist() == [4] * 4


def test_orthogonal_writer_rejects_overloaded_episode():
    config = small_config(model={"r": 4, "r_mem": 4, "d_e": 8})
    model = SymbolicModel(config)
    arrays = generate(make_task_spec(config), "val", 0, 1, 19)
    with pytest.raises(ValueError, match="entities <= r_mem"):
        model.prepare(arrays)


def test_b5_linear_control_keeps_signed_fixed_query_read_at_every_step():
    config = small_config(variant="B5-lin")
    model = SymbolicModel(config)
    arrays = generate(make_task_spec(config), "val", 0, 2, 19)
    batch = model.prepare(arrays)
    with torch.no_grad():
        model.adapter.R.copy_(.2 * torch.eye(16))
        model.adapter.Q.copy_(torch.eye(16))
        model.adapter.U.copy_(torch.eye(16))
    fixed_read = torch.einsum("bij,bqj->bqi", batch.memory, batch.u)
    expected = batch.u
    for _ in range(4):
        expected = batch.u + .25 * (.2 * expected + fixed_read)
    actual = model.read(batch, 4)["z"]
    torch.testing.assert_close(actual, expected)
    assert (actual < 0).any()


@pytest.mark.parametrize("variant", ["B3", "B5", "B5-lin", "M1", "M1-lin"])
@pytest.mark.parametrize("writer", ["oracle_random", "joint"])
def test_reading_never_changes_context_or_query_and_zero_depth_is_input(variant, writer):
    config = small_config(variant=variant, writer=writer)
    model = SymbolicModel(config)
    arrays = generate(make_task_spec(config), "val", 0, 2, 19)
    prepared = model.prepare(arrays)
    snapshots = {name: getattr(prepared, name).clone() for name in ("x", "u", "memory", "write_count")}
    for L in (0, 1, 2, 4, 8):
        result = model.read(prepared, L)
        if L == 0:
            torch.testing.assert_close(result["z"], prepared.u, rtol=0, atol=0)
        for name, snapshot in snapshots.items():
            torch.testing.assert_close(getattr(prepared, name), snapshot, rtol=0, atol=0)
    if writer == "joint":
        assert model.adapter.A is model.writer.A
        if variant != "B3":
            assert model.adapter.K is model.writer.K


def test_shared_pretrained_writer_is_identical_across_variants_and_checkpoint_reload(tmp_path):
    config = small_config(writer="t1_frozen", variant="B4")
    source, summary = train(config, tmp_path / "source")
    assert summary["pretrain_seconds"] > 0
    original = {name: value.clone() for name, value in source.writer.state_dict().items()}
    arrays = generate(make_task_spec(config), "val", 0, 2, 19)
    source_batch = source.prepare(arrays)
    for variant in ("B3", "B5", "B5-lin", "M1", "M1-lin"):
        reader_config = copy.deepcopy(config)
        reader_config.update(variant=variant, activation="identity" if variant in {"M1-lin", "B5-lin"} else "relu")
        reader_config["pretrain"]["writer_checkpoint"] = str(tmp_path / "source" / "writer.pt")
        reader, result = train(reader_config, tmp_path / variant)
        assert result["pretrain_seconds"] == 0
        assert all(not p.requires_grad for p in reader.writer.parameters())
        for name, expected in original.items():
            torch.testing.assert_close(reader.writer.state_dict()[name], expected, rtol=0, atol=0)
        actual = reader.prepare(arrays)
        for name in ("x", "u", "memory", "write_count"):
            torch.testing.assert_close(getattr(actual, name), getattr(source_batch, name), rtol=0, atol=0)
        restored = load_checkpoint(tmp_path / variant / "checkpoint.pt", device="cpu")
        torch.testing.assert_close(restored.read(restored.prepare(arrays), 1)["logits"],
                                   reader.read(actual, 1)["logits"], rtol=0, atol=0)


def test_joint_writer_receives_gradients_and_changes_after_reader_starts(tmp_path):
    config = small_config(writer="joint", variant="B4")
    initial = SymbolicModel(config)
    trained, _ = train(config, tmp_path)
    assert any(not torch.equal(value, initial.writer.state_dict()[name])
               for name, value in trained.writer.state_dict().items())


def test_300_step_t1_gpu_smoke_is_clearly_above_candidate_chance(tmp_path):
    assert torch.cuda.is_available(), "WP-D acceptance requires a Miyabi-G GPU allocation"
    config = resolve_config({"name": "t1-acceptance", "writer": "oracle_random", "variant": "B4",
                            "task": {"name": "t1"},
                            "train": {"steps": 300, "batch_size": 32, "depths": [1]}})
    model, summary = train(config, tmp_path)
    arrays = generate(make_task_spec(config), "val", 0, 256, derive_seed(4242, "acceptance_val"))
    model.eval()
    with torch.no_grad():
        batch = model.prepare(arrays)
        logits = model.read(batch, 1)["logits"].masked_fill(~batch.candidate_mask[:, None, :], -torch.inf)
        accuracy = float((logits.argmax(-1) == batch.answer).float().mean())
        chance = float(batch.candidate_mask.sum(-1).float().reciprocal().mean())
    print(f"T1 300 steps: accuracy={accuracy:.4f}, chance={chance:.4f}, seconds={summary['train_seconds']:.3f}")
    assert accuracy > chance + .10
