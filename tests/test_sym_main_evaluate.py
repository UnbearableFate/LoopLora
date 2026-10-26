"""Confirmation embargo, main depth sweeps, and screening counterfactuals."""
import copy
import json

import numpy as np
import pytest
import torch

import looplora.sym.evaluate as evaluation
from looplora.infra.seeds import stable_hash
from looplora.sym.config import resolve_config
from looplora.sym.model import SymbolicModel

DEPTHS = [0, 1, 2, 3, 4, 6, 8, 12, 16]


def _config(**updates):
    config = {"device": "cpu", "writer": "oracle_random", "variant": "M1",
              "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 8},
              "task": {"name": "t2_chain_end", "E": 12, "Q": 3, "V": 128,
                       "seen": [0, 64], "unseen": [64, 128], "chain_lengths": [8, 4], "hops": [1, 2, 3]},
              "eval": {"episodes": 2, "batch_size": 2, "splits": ["test"], "depths": DEPTHS,
                       "hops": list(range(1, 9)), "seed": 900001, "counterfactuals": False,
                       "fixed_depth_sweep": True}}
    for key, value in updates.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    return resolve_config(config)


def _samples(path):
    with np.load(path / "samples.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def test_confirmation_samples_only_does_not_call_any_aggregation(monkeypatch, tmp_path):
    config = _config()
    checkpoint_config = copy.deepcopy(config)
    checkpoint_config["eval"]["seed"] = 4242
    provenance = {"checkpoint": "/retained/checkpoint.pt", "checkpoint_config": checkpoint_config,
                  "eval_overrides": {"eval.seed": 900001}}

    def forbidden(*args, **kwargs):
        pytest.fail("confirmation aggregation occurred before the global barrier")

    with monkeypatch.context() as guard:
        guard.setattr(evaluation, "aggregate_samples", forbidden)
        guard.setattr(evaluation, "_aggregate_diagnostics", forbidden)
        metadata, timing = evaluation.evaluate(SymbolicModel(config), config, tmp_path,
                                                samples_only=True, provenance=provenance)
    assert not (tmp_path / "metrics.json").exists()
    assert "table" not in metadata and "diagnostics" not in metadata and "accuracy" not in metadata
    assert metadata["aggregation_deferred"] is True and metadata["gate0_passed"] is True
    assert metadata["config"] == config
    assert metadata["checkpoint_config"] == checkpoint_config
    assert metadata["checkpoint_config_hash"] == stable_hash(checkpoint_config)
    assert metadata["config_hash"] == stable_hash(config)
    assert len(metadata["samples_hash"]) == 32
    assert {row["L"] for row in metadata["memory_hashes"]} == set(DEPTHS)
    assert json.loads((tmp_path / "evaluation.json").read_text()) == metadata
    assert json.loads((tmp_path / "config.json").read_text())["checkpoint_config"] == checkpoint_config
    assert json.loads((tmp_path / "timing.json").read_text()) == timing
    samples = _samples(tmp_path)
    assert len(samples["answer"]) == 2 * 8 * 9
    with pytest.raises(ValueError, match="global.*barrier"):
        evaluation.finalize_evaluation(tmp_path)
    assert not (tmp_path / "metrics.json").exists()
    metrics = evaluation.finalize_evaluation(tmp_path, barrier_passed=True)
    assert metrics["table"] == evaluation.aggregate_samples(samples)
    assert len(metrics["table"]) == 72 and all(row["n"] == 2 for row in metrics["table"])
    assert metrics["finalized_after_global_barrier"] is True
    assert metrics["aggregation_deferred"] is False
    with pytest.raises(FileExistsError, match="already exist"):
        evaluation.finalize_evaluation(tmp_path, barrier_passed=True)


def test_finalization_rejects_modified_sample_artifact(tmp_path):
    config = _config(eval={"depths": [1]})
    evaluation.evaluate(SymbolicModel(config), config, tmp_path, samples_only=True)
    with (tmp_path / "samples.npz").open("ab") as stream:
        stream.write(b"modified after generation")
    with pytest.raises(ValueError, match="samples changed"):
        evaluation.finalize_evaluation(tmp_path, barrier_passed=True)
    assert not (tmp_path / "metrics.json").exists()


def test_samples_only_cli_preserves_embargo(monkeypatch, tmp_path):
    import looplora.sym.train as training

    config = _config(eval={"depths": [1]})
    model = SymbolicModel(config)
    model.checkpoint_config = copy.deepcopy(config)
    checkpoint = tmp_path / "training" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"mock checkpoint")
    monkeypatch.setattr(training, "load_checkpoint", lambda path, device=None: model)
    monkeypatch.setattr(evaluation, "aggregate_samples", lambda *args: pytest.fail("embargoed aggregation"))
    monkeypatch.setattr(evaluation, "_aggregate_diagnostics", lambda *args: pytest.fail("embargoed diagnostics"))
    out = tmp_path / "confirmation"
    evaluation.main(["--checkpoint", str(checkpoint), "--out", str(out), "--device", "cpu", "--samples-only"])
    assert not (out / "metrics.json").exists()
    metadata = json.loads((out / "evaluation.json").read_text())
    assert metadata["checkpoint"] == str(checkpoint.resolve())
    assert metadata["checkpoint_config"] == config
    assert json.loads((out / "timing.json").read_text())["runtime"]["device"] == "cpu"


@pytest.mark.parametrize("sweep", [False, True])
def test_b4_training_stays_single_depth_but_explicit_main_sweep_evaluates_all(sweep, tmp_path):
    config = _config(variant="B4", eval={"fixed_depth_sweep": sweep})
    assert config["train"]["depths"] == [1]
    assert config["eval"]["depths"] == (DEPTHS if sweep else [1])
    metrics, _ = evaluation.evaluate(SymbolicModel(config), config, tmp_path)
    assert {row["L"] for row in metrics["table"]} == (set(DEPTHS) if sweep else {1})


@pytest.mark.parametrize("writer", ["oracle_orth", "oracle_random", "joint"])
def test_r16_screening_includes_removal_and_preserves_source_query_codes(writer, monkeypatch, tmp_path):
    config = _config(writer=writer, eval={"seed": 4242, "counterfactuals": True, "depths": [1, 4]})
    model = SymbolicModel(config)
    captured = []
    real_prepare = model.prepare

    def capture(arrays, **options):
        prepared = real_prepare(arrays, **options)
        captured.append((arrays, prepared))
        return prepared

    monkeypatch.setattr(model, "prepare", capture)
    metrics, _ = evaluation.evaluate(model, config, tmp_path)
    samples = _samples(tmp_path)
    assert set(samples["condition"]) == {"original", "replace_key_fact", "remove_required_fact", "swap_memory"}
    original_arrays, original = captured[0]
    for condition in ("replace_key_fact", "remove_required_fact"):
        offset = 0
        for arrays, prepared in captured:
            if arrays.meta.get("counterfactual", {}).get("kind") != condition:
                continue
            for row in range(len(arrays.episode_seed)):
                source = arrays.meta["source_episode"][offset + row]
                query = arrays.meta["source_query"][offset + row]
                assert torch.equal(prepared.u[row, 0], original.u[source, query])
                if writer == "oracle_orth":
                    entities = np.unique(original_arrays.facts[source, :, (0, 2)])
                    assert torch.equal(prepared.codes[row, entities], original.codes[source, entities])
            offset += len(arrays.episode_seed)
        assert offset == 16
    removal = samples["condition"] == "remove_required_fact"
    assert np.all(samples["answer"][removal] != samples["original_answer"][removal])
    assert metrics["counterfactual_codebook_allocation"] == "per_query_fork"
    for depth in (1, 4):
        mask = removal & (samples["L"] == depth)
        assert np.array_equal(samples["hop"][mask], np.tile(np.arange(1, 9), 2))


@pytest.mark.parametrize("kind", ["t1", "t2_walk"])
@pytest.mark.parametrize("writer", ["oracle_orth", "joint"])
def test_fixed_hop_removal_adds_insufficient_candidate_and_orthogonal_slot(kind, writer, monkeypatch, tmp_path):
    task = {"name": kind, "E": 4, "Q": 2, "hops": [1] if kind == "t1" else [1, 2],
            "chain_lengths": None if kind == "t1" else [4]}
    config = _config(writer=writer, task=task,
                     eval={"seed": 4242, "counterfactuals": True, "depths": [1], "hops": [1, 2]})
    model = SymbolicModel(config)
    removal_prepared = []
    real_read = model.read

    def capture(prepared, depth):
        if bool(torch.all(prepared.answer == config["task"]["V"])):
            removal_prepared.append(prepared)
        return real_read(prepared, depth)

    monkeypatch.setattr(model, "read", capture)
    evaluation.evaluate(model, config, tmp_path)
    assert removal_prepared
    for prepared in removal_prepared:
        assert prepared.candidate_mask[:, config["task"]["V"]].all()
        if writer == "oracle_orth":
            for row in range(len(prepared.answer)):
                candidates = prepared.codes[row, prepared.candidate_mask[row]]
                gram = candidates @ candidates.T
                assert torch.allclose(gram, torch.eye(len(candidates)), atol=2e-6)
    samples = _samples(tmp_path)
    assert np.all(samples["answer"][samples["condition"] == "remove_required_fact"] == config["task"]["V"])


@pytest.mark.parametrize("kind", ["t2_chain_end", "t2_typed"])
def test_successor_swap_preserves_pairing_and_codes_and_records_changed_hops(kind, monkeypatch, tmp_path):
    task = {"name": kind}
    if kind == "t2_typed":
        task.update(E=12, chain_lengths=[8, 1], n_distract=3)
    config = _config(writer="oracle_orth", task=task,
                     eval={"seed": 900002, "counterfactuals": True, "swap_successors": True})
    model = SymbolicModel(config)
    real_prepare, real_counterfactual = model.prepare, evaluation.counterfactual
    captured, swap_seeds = [], []

    def prepare(arrays, **options):
        prepared = real_prepare(arrays, **options)
        captured.append((arrays, prepared))
        return prepared

    def counterfactual(arrays, condition, seed):
        if condition == "swap_successors":
            swap_seeds.append(seed)
        return real_counterfactual(arrays, condition, seed)

    monkeypatch.setattr(model, "prepare", prepare)
    monkeypatch.setattr(evaluation, "counterfactual", counterfactual)
    metrics, _ = evaluation.evaluate(model, config, tmp_path)
    assert swap_seeds == [900002]
    samples = _samples(tmp_path)
    assert len(samples["counterfactual_hop_true"]) == len(samples["answer"])
    swapped = samples["condition"] == "swap_successors"
    assert int(swapped.sum()) == 2 * 8 * 3
    assert set(samples["L"][swapped]) == {1, 4, 8}
    assert np.all(samples["answer"][swapped] != samples["original_answer"][swapped])
    assert np.any(samples["counterfactual_hop_true"][swapped] != samples["hop"][swapped])
    original = samples["condition"] == "original"
    predictions = {(int(e), int(q), int(depth)): int(prediction)
                   for e, q, depth, prediction in zip(samples["episode_id"][original], samples["query_id"][original],
                                                      samples["L"][original], samples["prediction"][original])}
    original_arrays, original_prepared = captured[0]
    offset = 0
    for arrays, prepared in captured:
        if arrays.meta.get("counterfactual", {}).get("kind") != "swap_successors":
            continue
        for row in range(len(arrays.episode_seed)):
            source = arrays.meta["source_episode"][offset + row]
            query = arrays.meta["source_query"][offset + row]
            assert torch.equal(prepared.codes[row], original_prepared.codes[source])
            assert torch.equal(prepared.u[row, 0], original_prepared.u[source, query])
            assert set(arrays.facts[row, :, (0, 2)].ravel()) == set(original_arrays.facts[source, :, (0, 2)].ravel())
            graph = {int(h): int(t) for h, rel, t in arrays.facts[row] if rel == 0}
            node, actual_hop = int(arrays.q_start[row, 0]), 0
            while node in graph:
                node, actual_hop = graph[node], actual_hop + 1
                assert actual_hop <= config["task"]["E"]
            assert node == arrays.answer[row, 0]
            assert actual_hop == arrays.q_hop_true[row, 0]
            for depth in (1, 4, 8):
                mask = (swapped & (samples["episode_id"] == original_arrays.episode_seed[source])
                        & (samples["query_id"] == query) & (samples["L"] == depth))
                assert mask.sum() == 1
                assert samples["counterfactual_hop_true"][mask][0] == actual_hop
                assert samples["hop"][mask][0] == query + 1
                assert samples["original_prediction"][mask][0] == predictions[
                    (int(original_arrays.episode_seed[source]), query, depth)]
        offset += len(arrays.episode_seed)
    assert offset == 16
    assert metrics["swap_successors_diagnostic_only"] is True
    assert {row["L"] for row in metrics["table"] if row["condition"] == "swap_successors"} == {1, 4, 8}


@pytest.mark.parametrize("counterfactuals,splits", [(False, ["test"]), (True, ["val", "test_unseen"])])
def test_successor_swap_only_runs_with_counterfactuals_on_test(counterfactuals, splits, tmp_path):
    config = _config(eval={"swap_successors": True, "counterfactuals": counterfactuals, "splits": splits})
    evaluation.evaluate(SymbolicModel(config), config, tmp_path)
    assert "swap_successors" not in set(_samples(tmp_path)["condition"])


def test_successor_swap_defaults_off_and_requires_diagnostic_depths(tmp_path):
    config = _config()
    assert config["eval"]["swap_successors"] is False
    config["eval"].update(swap_successors=True, counterfactuals=True, depths=[1, 4])
    with pytest.raises(ValueError, match="requires evaluation depths 1, 4, 8"):
        evaluation.evaluate(SymbolicModel(config), config, tmp_path)
