"""Locked validation-only pilot grids, cross-stage writers, and selection rules."""
import json
from pathlib import Path

import pytest

from looplora.sym.pilot import (pilot1_configs, pilot2_configs, replication_configs,
                               select_pilot1, select_pilot2, write_plan, loss_diagnostic)
from looplora.sym.run import load_suite


def capacity_rows():
    rows = []
    for config in pilot1_configs():
        E, lr = config["task"]["E"], config["train"]["lr"]
        rows.append({"name": config["name"], "writer": config["writer"], "r": config["model"]["r"],
                     "E": E, "lr": lr, "accuracy": {8: .99, 12: .98, 16: .95, 24: .94}[E]
                     - (0 if lr == 1e-3 else .02), "writer_checkpoint": "/selected/writer.pt"})
    return rows


def test_pilot1_complete_grid_only_val_and_feasible_orthogonal_loads():
    configs = pilot1_configs()
    assert len(configs) == 60
    assert len({c["name"] for c in configs}) == 60
    for config in configs:
        assert config["seed"] == 17 and config["eval"]["seed"] == 4242
        assert config["eval"]["splits"] == ["val"]
        assert config["train"]["steps"] == 5000 and config["train"]["batch_size"] == 64
        assert config["eval"]["episodes"] == 1000
        assert config["task"]["head_sharing"] == .3 and config["task"]["n_rel"] == 4
        if config["writer"] == "oracle_orth":
            assert config["task"]["E"] * 2 <= config["model"]["r"]


def test_capacity_rule_uses_best_lr_per_E_inclusive_threshold_and_lr_at_E():
    selection = select_pilot1(capacity_rows())
    assert selection["E_star"] == 16
    assert selection["selected"]["joint/r32"]["lr"] == 1e-3
    assert not selection["fallback_to_12"]
    assert not selection["writer_diagnosis_required"]
    rows = capacity_rows()
    for row in rows:
        row["accuracy"] = .90
    selection = select_pilot1(rows)
    assert selection["E_star"] == 12
    assert selection["fallback_to_12"] and selection["writer_diagnosis_required"]
    assert selection["selected"]["joint/r32"]["lr"] == 3e-4


def test_capacity_lr_is_selected_at_chosen_E_and_separately_for_each_rank():
    rows = capacity_rows()
    for row in rows:
        if row["writer"] != "joint":
            continue
        preferred = 3e-4 if row["E"] == 8 or row["r"] == 16 else 3e-3
        row["accuracy"] = {8: .99, 12: .98, 16: .96, 24: .95}[row["E"]] \
            if row["lr"] == preferred else .80
    selection = select_pilot1(rows)
    assert selection["E_star"] == 24
    assert selection["joint_r32_best_by_E"][8]["lr"] == 3e-4
    assert selection["selected"]["joint/r32"]["lr"] == 3e-3
    assert selection["selected"]["joint/r16"]["lr"] == 3e-4


@pytest.mark.parametrize("E", [12, 16, 24])
def test_pilot2_and_replication_have_locked_shapes_budgets_and_shared_writer(E):
    selection1 = select_pilot1(capacity_rows())
    selection1["E_star"] = E
    configs = pilot2_configs(selection1)
    assert len(configs) == 60
    rows = []
    for config in configs:
        assert config["task"]["E"] == E
        assert sum(config["task"]["chain_lengths"]) == E
        assert config["eval"]["splits"] == ["val"]
        assert config["train"]["steps"] == 5000
        assert config["pretrain"]["task"]["head_sharing"] == .3
        if config["writer"] == "t1_frozen":
            assert config["pretrain"]["writer_checkpoint"] == "/selected/writer.pt"
        rows.append({"writer": config["writer"], "variant": config["variant"],
                     "lr": config["train"]["lr"], "accuracy": .5})
    selection2 = select_pilot2(rows)
    assert len(selection2["selected"]) == 20
    replicas = replication_configs(selection1, selection2)
    assert len(replicas) == 15
    for config in replicas:
        assert config["train"]["lr"] == 3e-4
        assert config["pretrain"]["task"]["E"] == E
        if config["writer"] == "oracle_random":
            assert config["task"]["E"] == E * 3 // 2
            assert config["task"]["chain_lengths"].count(8) >= 2
        else:
            assert config["task"]["E"] == E
            assert config["task"]["name"] == "t2_typed"
            assert config["task"]["n_distract"] == E // 4
            assert sum(config["task"]["chain_lengths"]) == E - E // 4


def test_external_writer_from_is_explicit_and_shards_are_disjoint(tmp_path):
    selection = select_pilot1(capacity_rows())
    checkpoint = tmp_path / "writer.pt"
    checkpoint.touch()
    selection["selected"]["joint/r32"]["writer_checkpoint"] = str(checkpoint)
    manifest = write_plan(pilot2_configs(selection), tmp_path / "plan", shards=3)
    observed = []
    for shard in range(3):
        jobs = load_suite(tmp_path / "plan" / f"shard-{shard:02d}.yaml", tmp_path / f"shard-{shard:02d}")
        observed.extend(job["name"] for job in jobs)
        for job in jobs:
            assert job["dependencies"] == []
            if job["config"]["writer"] == "t1_frozen":
                assert job["config"]["pretrain"]["writer_checkpoint"] == str(checkpoint)
    assert sorted(observed) == sorted(manifest["names"])
    with pytest.raises(FileExistsError):
        write_plan(pilot2_configs(selection), tmp_path / "plan", shards=3)


def test_loss_plateau_is_diagnostic_only_and_requires_late_training_data(tmp_path):
    path = tmp_path / "train_log.jsonl"
    path.write_text('\n'.join(json.dumps({"stage": "train", "step": step, "loss": value})
                              for step, value in ((4500, 1.), (5000, .99))))
    result = loss_diagnostic(path)
    assert result["plateau_2pct"]
    assert result["relative_loss_improvement"] == pytest.approx(.01)
    path.write_text(json.dumps({"stage": "train", "step": 100, "loss": 2.}))
    with pytest.raises(ValueError, match="5000"):
        loss_diagnostic(path)
