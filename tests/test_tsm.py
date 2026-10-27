"""TSM (F10): stay loss, sticky gate, parity with the existing training loop, endpoints.

Run only inside a Miyabi-G compute allocation, including the CPU cases.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.sym import tsm
from looplora.sym.config import resolve_config
from looplora.sym.evaluate import evaluate, stratify_queries
from looplora.sym.followup_stats import select_rows
from looplora.sym.model import SymbolicModel, make_task_spec
from looplora.sym.train import _fit, load_checkpoint, train

WRITERS = ("oracle_random", "joint_fact")


def _config(writer, **train):
    base = {
        "name": "tsm-tiny", "device": "cpu", "seed": 17, "writer": writer, "variant": "M1",
        "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 8},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 96, "n_rel": 4, "seen": [0, 72],
                 "unseen": [72, 96], "chain_lengths": [4, 4], "hops": [1, 2, 3]},
        "train": {"steps": 3, "batch_size": 4, "lr": 0.01, "warmup_steps": 0, "depths": [2],
                  "log_every": 1, **train},
        "eval": {"episodes": 6, "batch_size": 4, "splits": ["val"], "depths": [1, 2, 4],
                 "hops": [1, 2, 3, 4], "counterfactuals": False},
    }
    return resolve_config(base)


def _arrays(config, split="val", episodes=6):
    spec = make_task_spec(config, evaluation=True)
    seed = config["eval"]["seed"]
    return stratify_queries(generate(spec, split, 0, episodes, seed), spec, config["eval"]["hops"], seed)


@pytest.mark.parametrize("writer", WRITERS)
def test_read_states_and_decode_equal_the_existing_reader_bitwise(writer):
    model = SymbolicModel(_config(writer)).eval()
    prepared = model.prepare(_arrays(model.config))
    states = tsm.read_states(model, prepared, 4)
    memory = prepared.memory[:, None].expand(-1, prepared.u.shape[1], -1, -1)
    for depth in range(5):
        assert torch.equal(states[depth], model.adapter.read(prepared.u, memory, depth))
        assert torch.equal(tsm.decode(model, prepared, states[depth]), model.read(prepared, depth)["logits"])


@pytest.mark.parametrize("writer", WRITERS)
def test_sticky_gate_limits_and_formula(writer):
    model = SymbolicModel(_config(writer)).eval()
    prepared = model.prepare(_arrays(model.config))
    plain = tsm.read_states(model, prepared, 3)
    frozen = tsm.read_states(model, prepared, 3, tsm.StickyGate(0.0, -40.0))
    open_gate = tsm.read_states(model, prepared, 3, tsm.StickyGate(0.0, 40.0))
    for step in range(4):
        torch.testing.assert_close(frozen[step], prepared.u, rtol=0, atol=1e-12)
        torch.testing.assert_close(open_gate[step], plain[step], rtol=1e-5, atol=1e-6)
    gate = tsm.StickyGate(2.0, -0.5)
    recorded = []
    gated = tsm.read_states(model, prepared, 2, gate, gates=recorded)
    adapter, u = model.adapter, prepared.u
    memory = prepared.memory[:, None].expand(-1, u.shape[1], -1, -1)
    z = u
    for step in range(2):
        readout = adapter._query(z, memory, adapter.Q, adapter.U)
        update = u + adapter.cfg.rho * adapter._phi(F.linear(z, adapter.R) + readout)
        g = torch.sigmoid(2.0 * readout.norm(dim=-1) / u.norm(dim=-1) - 0.5)
        torch.testing.assert_close(recorded[step], g, rtol=1e-5, atol=1e-6)
        z = z + g[..., None] * (update - z)
        torch.testing.assert_close(gated[step + 1], z, rtol=1e-5, atol=1e-6)
    assert len(recorded) == 2 and recorded[0].shape == u.shape[:2]


def test_stay_loss_masks_steps_before_arrival_and_weights_queries_equally():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 2, 3, 5, generator=generator)
    candidate_mask = torch.tensor([[True, True, True, False, False]])
    answer = torch.tensor([[0, 2]])
    restricted = logits.masked_fill(~candidate_mask[:, None, None, :], -torch.inf)

    def ce(query, step):
        return F.cross_entropy(restricted[0, query, step][None], answer[:, query])

    loss = tsm.stay_loss(logits, candidate_mask, answer, torch.tensor([[1, 3]]))
    torch.testing.assert_close(loss, ((ce(0, 0) + ce(0, 1) + ce(0, 2)) / 3 + ce(1, 2)) / 2)
    # A query that has not arrived within the horizon contributes nothing.
    late = tsm.stay_loss(logits, candidate_mask, answer, torch.tensor([[2, 9]]))
    torch.testing.assert_close(late, ((ce(0, 1) + ce(0, 2)) / 2) / 2)


@pytest.mark.parametrize("writer", WRITERS)
def test_reference_loop_is_bitwise_the_existing_fit(tmp_path, writer):
    config = _config(writer, steps=4)
    first, second = SymbolicModel(config), SymbolicModel(config)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    summary_a = _fit(first, config, tmp_path / "a", "train")
    summary_b = tsm.fit_tsm(second, config, tmp_path / "b", stay=False)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name
    rows_a = [json.loads(line) for line in (tmp_path / "a" / "train_log.jsonl").read_text().splitlines()]
    rows_b = [json.loads(line) for line in (tmp_path / "b" / "train_log.jsonl").read_text().splitlines()]
    assert len(rows_a) == len(rows_b) == 4
    for a, b in zip(rows_a, rows_b):
        for key in ("step", "L", "mean_L", "loss", "accuracy", "lr", "grad_norm", "data_seed"):
            assert a[key] == b[key], key
    for key in ("data_seed", "final_loss", "mean_L", "steps", "train_tokens", "train_queries"):
        assert summary_a[key] == summary_b[key], key


@pytest.mark.parametrize("writer", WRITERS)
def test_stay_arms_use_the_same_data_stream_and_train_the_gate(tmp_path, writer):
    config = _config(writer, depths=[3], lr=0.05)
    initial = SymbolicModel(config)
    for arm in ("T1", "T2"):
        model, gate, summary = tsm.train_arm(config, tmp_path / arm, arm)
        assert summary["data_seed"] == derive_seed(config["seed"], "train_data")
        assert summary["mean_L"] == 3.0 and summary["arm"] == arm and summary["stay_loss"] is True
        assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), initial.parameters()))
        assert (tmp_path / arm / "checkpoint.pt").is_file()
        if arm == "T1":
            assert gate is None and summary["sticky_gate"] is False
        else:
            assert gate is not None and summary["gate_init"] == [0.0, 4.0]
            assert (float(gate.a), float(gate.b)) != (0.0, 4.0)
            rows = [json.loads(line) for line in (tmp_path / arm / "train_log.jsonl").read_text().splitlines()]
            assert "gate_a" in rows[-1] and "gate_b" in rows[-1]
            assert summary["trainable_parameters"] == sum(p.numel() for p in initial.parameters()
                                                         if p.requires_grad) + 2
    with pytest.raises(FileExistsError):
        tsm.train_arm(config, tmp_path / "T1", "T1")
    with pytest.raises(ValueError):
        tsm.train_arm(config, tmp_path / "T0", "T0")


def test_checkpoints_round_trip_through_both_loaders(tmp_path):
    config = _config("joint_fact", depths=[3])
    model, gate, _ = tsm.train_arm(config, tmp_path / "T2", "T2")
    restored, restored_gate, arm = tsm.load_tsm(tmp_path / "T2" / "checkpoint.pt", device="cpu")
    assert arm == "T2" and restored_gate is not None
    assert torch.equal(restored_gate.a, gate.a) and torch.equal(restored_gate.b, gate.b)
    assert not restored_gate.a.requires_grad
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name
    plain = load_checkpoint(tmp_path / "T2" / "checkpoint.pt", device="cpu")
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), restored.parameters()))
    tsm.train_arm(config, tmp_path / "T1", "T1")
    _, no_gate, arm = tsm.load_tsm(tmp_path / "T1" / "checkpoint.pt")
    assert no_gate is None and arm == "T1"
    train(_config("joint_fact", depths=[2]), tmp_path / "T0")
    _, none_gate, arm = tsm.load_tsm(tmp_path / "T0" / "checkpoint.pt")
    assert none_gate is None and arm == "T0"


@pytest.mark.parametrize("writer", WRITERS)
def test_evaluate_arm_without_gate_equals_the_standard_evaluator(tmp_path, writer):
    config = _config(writer)
    model = SymbolicModel(config).eval()
    evaluate(model, config, tmp_path, samples_only=True)
    with np.load(tmp_path / "samples.npz") as archive:
        standard = {name: archive[name] for name in archive.files}
    mine = tsm.evaluate_arm(model, None, _arrays(config), [1, 2, 4], batch_size=4, split="val")
    for depth in (1, 2, 4):
        a = select_rows(standard, "val", "original", depth)
        b = select_rows(mine, "val", "original", depth)
        for name in ("episode_id", "query_id", "hop", "answer", "prediction", "correct", "full_correct",
                     "candidate_size"):
            assert np.array_equal(standard[name][a], mine[name][b]), (depth, name)


def test_gate_profile_reports_the_initial_gate_for_every_hop_and_step():
    config = _config("oracle_random")
    model = SymbolicModel(config).eval()
    profile = tsm.gate_profile(model, tsm.StickyGate(), _arrays(config), steps=4, batch_size=4)
    values = np.asarray(profile["mean_gate"])
    assert profile["hops"] == [1, 2, 3, 4] and profile["steps"] == [1, 2, 3, 4]
    np.testing.assert_allclose(values, float(torch.sigmoid(torch.tensor(4.0))), rtol=1e-5)


def _synthetic(accuracy, seeds=(17, 29, 43), episodes=200):
    """Samples whose accuracy in each (depth, hop) cell is exactly the given percentage."""
    result = {}
    for index, seed in enumerate(seeds):
        columns = {name: [] for name in ("split", "condition", "episode_id", "query_id", "hop", "L", "answer",
                                         "correct")}
        for (depth, hop), percent in sorted(accuracy.items()):
            for episode in range(episodes):
                columns["split"].append("test")
                columns["condition"].append("original")
                columns["episode_id"].append(10_000 + episode)
                columns["query_id"].append(hop - 1)
                columns["hop"].append(hop)
                columns["L"].append(depth)
                columns["answer"].append(episode % 7)
                columns["correct"].append(int((episode + 7 * index) % 100 < percent))
        result[seed] = {name: np.asarray(values) for name, values in columns.items()}
    return result


def _grid(**cells):
    """Accuracy percent for depths {1, 4, 8, 16} by hop; ``cells`` maps depth -> {hop range: percent}."""
    table = {}
    for depth, spec in cells.items():
        for hops, percent in spec.items():
            for hop in hops:
                table[(int(depth.removeprefix("d")), hop)] = percent
    return table


def _scenario(stay16=90, far8=30, gain_b4=20):
    near, far, reach = (1, 2, 3), (4, 5, 6, 7, 8), (2, 3)
    t0 = _grid(d1={near: 40, far: 5}, d4={near: 60, far: 8}, d8={near: 30, far: 30}, d16={near: 20, far: 30})
    arm = _grid(d1={near: 40, far: 5}, d4={near: 90, far: 10}, d8={near: 90, far: far8},
                d16={near: stay16, far: 25})
    b4 = {(1, hop): gain_b4 for hop in (1, 2, 3, 4, 5, 6, 7, 8)}
    assert set(reach) <= set(near)
    return {"T0": _synthetic(t0), "T1": _synthetic(arm), "T2": _synthetic(arm), "B4": _synthetic(b4)}


def test_arm_endpoints_pass_when_all_three_registered_conditions_hold():
    result = tsm.arm_endpoints(_scenario(), "T1", writer="oracle_random", confidence=1 - 0.05 / 4)
    assert result["checks"] == dict(stay=True, far=True, gain=True) and result["passes"]
    assert result["stay"]["difference"] == pytest.approx(0.0)
    assert result["far"]["difference"] == pytest.approx(0.0)
    assert result["gain"]["difference"] == pytest.approx(0.70)
    assert result["gain"]["confidence"] == pytest.approx(1 - 0.05 / 4)
    assert result["descriptive"]["hop1_l16_vs_l1"]["difference"] == pytest.approx(0.50)


@pytest.mark.parametrize("changes,failed", [
    (dict(stay16=80), "stay"), (dict(far8=20), "far"), (dict(gain_b4=89), "gain")])
def test_arm_endpoints_report_which_condition_fails(changes, failed):
    result = tsm.arm_endpoints(_scenario(**changes), "T2", writer="joint_fact", confidence=0.9875)
    assert not result["checks"][failed] and not result["passes"]
    assert all(value for name, value in result["checks"].items() if name != failed)
    with pytest.raises(ValueError):
        tsm.arm_endpoints(_scenario(), "T0", writer="joint_fact", confidence=0.9875)
