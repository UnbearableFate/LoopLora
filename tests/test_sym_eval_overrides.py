"""Approved evaluation overrides do not alter checkpoint training settings."""
import copy
import json
from types import SimpleNamespace

import pytest

from looplora.infra.seeds import stable_hash
from looplora.sym.config import resolve_config
from looplora.sym.evaluate import parse_evaluation_overrides, resolve_evaluation_config


def _checkpoint_config():
    return resolve_config({"device": "cuda", "writer": "joint", "variant": "M1",
                           "eval": {"splits": ["val"], "episodes": 64, "counterfactuals": False}})


def test_yaml_json_values_and_repeated_keys_resolve_deterministically():
    assignments = ["eval.seed=123", "eval.episodes=2000", 'eval.splits=["test"]',
                   "eval.batch_size=16", "eval.counterfactuals=false", "eval.seed=777001"]
    overrides = parse_evaluation_overrides(assignments)
    assert overrides == {"eval.seed": 777001, "eval.episodes": 2000, "eval.splits": ["test"],
                         "eval.batch_size": 16, "eval.counterfactuals": False}
    assert parse_evaluation_overrides(assignments) == overrides
    assert parse_evaluation_overrides(["eval.splits=[val, test_unseen]"])["eval.splits"] == ["val", "test_unseen"]
    assert parse_evaluation_overrides(["eval.counterfactuals=true"])["eval.counterfactuals"] is True
    assert parse_evaluation_overrides(["eval.seed=0"])["eval.seed"] == 0


def test_resolved_overrides_preserve_checkpoint_config_and_all_training_fields():
    checkpoint = _checkpoint_config()
    original = copy.deepcopy(checkpoint)
    overrides = parse_evaluation_overrides(["eval.seed=777001", "eval.episodes=2000", "eval.splits=[test]"])
    config = resolve_evaluation_config(checkpoint, overrides, device="cpu")
    assert checkpoint == original
    assert config["eval"]["seed"] == 777001
    assert config["eval"]["episodes"] == 2000
    assert config["eval"]["splits"] == ["test"]
    assert config["device"] == "cpu"
    for key in set(checkpoint) - {"eval", "device"}:
        assert config[key] == checkpoint[key]
    for key in set(checkpoint["eval"]) - {"seed", "episodes", "splits"}:
        assert config["eval"][key] == checkpoint["eval"][key]
    assert resolve_evaluation_config(checkpoint, {}) == checkpoint


@pytest.mark.parametrize("key", ["seed", "train.steps", "model.r", "writer", "task.name",
                                  "eval.depths", "eval.hops", "eval", "eval.seed.extra"])
def test_only_explicitly_approved_evaluation_keys_can_change(key):
    with pytest.raises(ValueError, match="not allowed"):
        parse_evaluation_overrides([f"{key}=1"])


@pytest.mark.parametrize("assignment,message", [
    ("eval.seed", "expected key=value"),
    ("eval.seed=-1", "integer >= 0"),
    ("eval.seed=true", "integer >= 0"),
    ("eval.seed=1.5", "integer >= 0"),
    ("eval.seed=null", "integer >= 0"),
    ("eval.episodes=0", "integer >= 1"),
    ("eval.batch_size=false", "integer >= 1"),
    ("eval.counterfactuals=1", "must be a boolean"),
    ("eval.counterfactuals='false'", "must be a boolean"),
    ("eval.splits=test", "nonempty YAML/JSON list"),
    ("eval.splits=[]", "nonempty YAML/JSON list"),
    ("eval.splits=[val, 3]", "nonempty YAML/JSON list"),
    ("eval.splits=[", "invalid YAML/JSON"),
])
def test_invalid_override_values_raise_clear_errors(assignment, message):
    with pytest.raises(ValueError, match=message):
        parse_evaluation_overrides([assignment])


@pytest.mark.parametrize("splits", ["[train]", "[other]", "[val,val]"])
def test_overridden_config_uses_existing_split_validation(splits):
    with pytest.raises(ValueError, match="eval.splits"):
        resolve_evaluation_config(_checkpoint_config(), parse_evaluation_overrides([f"eval.splits={splits}"]))


def test_programmatic_resolution_rejects_unapproved_keys():
    with pytest.raises(ValueError, match="non-allowlisted"):
        resolve_evaluation_config(_checkpoint_config(), {"train.steps": 10})


def test_cli_records_original_and_effective_configs_and_only_approved_overrides(monkeypatch, tmp_path):
    import looplora.sym.evaluate as evaluation
    import looplora.sym.train as training

    checkpoint = tmp_path / "training" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"mock checkpoint")
    saved_config = _checkpoint_config()
    loaded_config = {**copy.deepcopy(saved_config), "device": "cpu"}
    model = SimpleNamespace(config=loaded_config, checkpoint_config=copy.deepcopy(saved_config))
    seen = []

    def load(path, device=None):
        assert path == checkpoint.resolve()
        assert device == "cpu"
        return model

    def evaluate(loaded_model, config, out):
        assert loaded_model is model
        seen.append(copy.deepcopy(config))
        out.mkdir()
        return {"evaluation_seed": config["eval"]["seed"], "table": []}, {"evaluation_seconds": 1.0}

    monkeypatch.setattr(training, "load_checkpoint", load)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    out = tmp_path / "evaluation"
    evaluation.main(["--checkpoint", str(checkpoint), "--out", str(out), "--device", "cpu",
                     "--set", "eval.seed=17", "--set", "eval.seed=777001",
                     "--set", "eval.episodes=2000", "--set", "eval.splits=[test]",
                     "--set", "eval.batch_size=16", "--set", "eval.counterfactuals=false"])
    metrics = json.loads((out / "metrics.json").read_text())
    record = json.loads((out / "config.json").read_text())
    assert metrics["checkpoint_config"] == saved_config
    assert metrics["config"] == seen[0]
    assert metrics["config_hash"] == stable_hash(seen[0])
    assert metrics["checkpoint"] == str(checkpoint.resolve())
    assert metrics["evaluation_seed"] == 777001
    assert metrics["eval_overrides"] == {
        "eval.seed": 777001, "eval.episodes": 2000, "eval.splits": ["test"],
        "eval.batch_size": 16, "eval.counterfactuals": False,
    }
    assert record == {key: metrics[key] for key in
                      ("checkpoint", "checkpoint_config", "config", "config_hash", "eval_overrides")}
    assert model.config == loaded_config
    assert model.checkpoint_config == saved_config
    timing = json.loads((out / "timing.json").read_text())
    assert timing["evaluation_seconds"] == 1.0
    assert timing["runtime"]["threads"] == seen[0]["runtime"]["threads"]
    assert timing["runtime"]["device"] == "cpu" and timing["runtime"]["gpu"] is None


def test_cli_rejects_non_allowlisted_override_before_loading_checkpoint(monkeypatch, tmp_path):
    import looplora.sym.evaluate as evaluation
    import looplora.sym.train as training

    def do_not_load(*args, **kwargs):
        pytest.fail("invalid override must be rejected before loading a checkpoint")

    monkeypatch.setattr(training, "load_checkpoint", do_not_load)
    with pytest.raises(SystemExit) as error:
        evaluation.main(["--checkpoint", str(tmp_path / "checkpoint.pt"), "--out", str(tmp_path / "out"),
                         "--set", "eval.depths=[1,8]"])
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()
