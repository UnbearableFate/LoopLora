"""Evaluation protocol regressions; execute only inside a PBS allocation."""
from dataclasses import dataclass
import json

import numpy as np
import pytest
import torch

from looplora.data.relations import TaskSpec, generate
from looplora.sym.evaluate import aggregate_samples, evaluate, main, stratify_queries, tensor_hash


@dataclass
class _Prepared:
    x: torch.Tensor
    u: torch.Tensor
    memory: torch.Tensor
    answer: torch.Tensor
    candidate_mask: torch.Tensor
    codes: torch.Tensor
    true_value: torch.Tensor
    retrieval_valid: torch.Tensor
    write_count: torch.Tensor


class _Model(torch.nn.Module):
    """Known scores isolate evaluator behavior from reader training quality."""

    device = torch.device("cpu")

    def __init__(self, vocab: int, *, mutate: str | None = None):
        super().__init__()
        self.vocab = vocab
        self.mutate = mutate
        self.prepared = []

    def prepare(self, arrays):
        n, q = arrays.answer.shape
        candidates = torch.zeros(n, self.vocab + 1, dtype=torch.bool)
        for i in range(n):
            ids = arrays.facts[i, arrays.fact_valid[i]][:, (0, 2)].ravel()
            candidates[i, ids] = True
        x = torch.tensor(np.stack((arrays.q_start, arrays.q_rel), axis=-1), dtype=torch.float32)
        prepared = _Prepared(
            x=x, u=x.clone(), memory=torch.tensor(arrays.episode_seed % 1_000_000, dtype=torch.float64)[:, None, None],
            answer=torch.tensor(arrays.answer, dtype=torch.long), candidate_mask=candidates,
            codes=torch.zeros(n, self.vocab + 1, 2), true_value=torch.ones(n, q, 1),
            retrieval_valid=torch.ones(n, q, dtype=torch.bool), write_count=torch.tensor(arrays.fact_valid.sum(-1)),
        )
        self.prepared.append(prepared)
        return prepared

    def read(self, prepared, depth):
        if self.mutate:
            getattr(prepared, self.mutate).add_(1)
        n, q = prepared.answer.shape
        logits = torch.zeros(n, q, self.vocab + 1)
        logits.scatter_(-1, prepared.answer[..., None], 5)
        logits[..., self.vocab] = 10  # Full-vocabulary winner excluded by the candidate mask.
        cosine = torch.full((n, q), float("nan") if depth == 0 else 1.0)
        return {"logits": logits, "z": prepared.u + depth, "first_read_cosine": cosine}


def _spec(name="t2_chain_end"):
    return TaskSpec(name, E=10 if name == "t2_typed" else 8, Q=8, V=64,
                    seen=(0, 32), unseen=(32, 64), n_distract=2 if name == "t2_typed" else 0,
                    chain_lengths=(8,), hops=tuple(range(1, 9)))


def _config():
    return {"variant": "M1", "eval": {"episodes": 4, "batch_size": 3,
            "depths": [0, 1, 4], "splits": ["val", "test", "test_unseen"],
            "hops": list(range(1, 9)), "seed": 42, "counterfactuals": True,
            "fixed_depth_sweep": False, "swap_successors": False}}


def _patch_spec(monkeypatch, spec):
    from looplora.sym import model as symbolic_model
    monkeypatch.setattr(symbolic_model, "make_task_spec", lambda config, evaluation=False: spec)


@pytest.mark.parametrize("name", ["t2_chain_end", "t2_walk", "t2_typed"])
def test_exact_query_per_hop_has_correct_answers_and_unchanged_support(name):
    spec = _spec(name)
    generated = generate(spec, "test", 0, 5, 77)
    old_facts, old_answers = generated.facts.copy(), generated.answer.copy()
    arrays = stratify_queries(generated, spec, list(range(1, 9)), 123)
    again = stratify_queries(generated, spec, list(range(1, 9)), 123)
    assert np.array_equal(arrays.facts, old_facts)
    assert np.array_equal(arrays.episode_seed, generated.episode_seed)
    assert np.array_equal(generated.answer, old_answers)
    assert np.array_equal(arrays.q_hop_true, np.tile(np.arange(1, 9), (5, 1)))
    assert np.array_equal(arrays.q_start, again.q_start)
    for i in range(5):
        graph = {(int(h), int(r)): int(t) for h, r, t in arrays.facts[i]}
        for j, hop in enumerate(range(1, 9)):
            node = int(arrays.q_start[i, j])
            for _ in range(hop):
                node = graph[node, 0]
            assert node == arrays.answer[i, j]
            if name != "t2_walk":
                assert (node, 0) not in graph
            assert arrays.q_hop_given[i, j] == (hop if name == "t2_walk" else -1)


def test_samples_reproduce_metrics_and_gate0_hashes(monkeypatch, tmp_path):
    spec, config = _spec(), _config()
    _patch_spec(monkeypatch, spec)
    model = _Model(spec.V)
    metrics, timing = evaluate(model, config, tmp_path)
    with np.load(tmp_path / "samples.npz", allow_pickle=False) as archive:
        samples = {name: archive[name] for name in archive.files}
    assert metrics == json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["table"] == aggregate_samples(samples)
    assert metrics["gate0_passed"] is True
    assert model.training is True
    assert timing["writer_seconds"] > 0
    assert set(timing["reader_seconds"]) == {"0", "1", "4"}
    for row in metrics["table"]:
        assert row["n"] == 4
        assert row["accuracy"] == 1
        assert row["full_accuracy"] == 0
        assert 0 < row["chance_accuracy"] < 1
    assert all(row["first_read_cosine"] is None for row in metrics["diagnostics"] if row["L"] == 0)
    assert all(row["first_read_cosine"] == 1 for row in metrics["diagnostics"] if row["L"] > 0)
    grouped = {}
    for row in metrics["memory_hashes"]:
        key = row["split"], row["condition"], row["batch_start"]
        grouped.setdefault(key, []).append({name: row[name] for name in ("x", "u", "memory", "write_count")})
    assert all(len(rows) == 3 and all(row == rows[0] for row in rows) for rows in grouped.values())
    replace_mask = samples["condition"] == "replace_key_fact"
    assert np.all(samples["answer"][replace_mask] != samples["original_answer"][replace_mask])
    assert np.all(samples["prediction"][replace_mask] != samples["original_prediction"][replace_mask])
    swap_mask = samples["condition"] == "swap_memory"
    assert np.all(samples["memory_episode_id"][swap_mask] != samples["episode_id"][swap_mask])
    for depth in config["eval"]["depths"]:
        original = (samples["condition"] == "original") & (samples["split"] == "test") & (samples["L"] == depth)
        swapped = swap_mask & (samples["L"] == depth)
        assert np.array_equal(samples["candidate_size"][original], samples["candidate_size"][swapped])
        assert np.array_equal(samples["answer"][original], samples["answer"][swapped])


@pytest.mark.parametrize("field", ["x", "u", "memory", "write_count"])
def test_query_or_memory_mutation_fails_gate0(monkeypatch, tmp_path, field):
    spec, config = _spec(), _config()
    config["eval"].update(splits=["val"], counterfactuals=False)
    _patch_spec(monkeypatch, spec)
    model = _Model(spec.V, mutate=field)
    with pytest.raises(AssertionError, match="Gate 0"):
        evaluate(model, config, tmp_path)
    assert model.training is True
    assert not (tmp_path / "metrics.json").exists()


def test_t1_evaluation_has_one_question_per_episode(monkeypatch, tmp_path):
    spec = TaskSpec("t1", E=8, Q=4, V=64, seen=(0, 32), unseen=(32, 64))
    config = _config()
    config["eval"].update(splits=["val"], counterfactuals=False)
    _patch_spec(monkeypatch, spec)
    metrics, _ = evaluate(_Model(64), config, tmp_path)
    assert {row["hop"] for row in metrics["table"]} == {1}
    assert all(row["n"] == config["eval"]["episodes"] for row in metrics["table"])


def test_missing_query_coverage_fails_explicitly():
    spec = TaskSpec("t3_missing", E=8, Q=1, V=64, seen=(0, 32), unseen=(32, 64),
                    chain_lengths=(8,), hops=(1,))
    arrays = generate(spec, "test", 0, 1, 99)
    with pytest.raises(ValueError, match="query coverage"):
        stratify_queries(arrays, spec, [1, 2], 100)


@pytest.mark.parametrize("variant,depth", [("B4", 1), ("B6", 1), ("B7", 1), ("B8", 4)])
def test_fixed_depth_variants_respect_their_supported_depth(monkeypatch, tmp_path, variant, depth):
    config = _config()
    config["variant"] = variant
    config["eval"].update(splits=["val"], counterfactuals=False)
    _patch_spec(monkeypatch, _spec())
    metrics, _ = evaluate(_Model(64), config, tmp_path)
    assert {row["L"] for row in metrics["table"]} == {depth}


def test_tensor_hash_includes_shape_dtype_and_exact_values():
    values = torch.arange(8, dtype=torch.float32)
    assert tensor_hash(values) == tensor_hash(values.clone())
    assert tensor_hash(values) != tensor_hash(values.reshape(2, 4))
    assert tensor_hash(values) != tensor_hash(values.to(torch.float64))
    assert tensor_hash(values) != tensor_hash(values + 1)


@pytest.mark.parametrize("field", ["logits", "z"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reader_outputs_fail_before_scoring(monkeypatch, tmp_path, field, invalid):
    config = _config()
    config["eval"].update(splits=["val"], counterfactuals=False)
    _patch_spec(monkeypatch, _spec())
    model = _Model(64)
    finite_read = model.read

    def invalid_read(prepared, depth):
        result = finite_read(prepared, depth)
        result[field][0, 0, 0] = invalid
        return result

    monkeypatch.setattr(model, "read", invalid_read)
    with pytest.raises(FloatingPointError, match="non-finite.*L=0"):
        evaluate(model, config, tmp_path)
    assert model.training is True
    assert not (tmp_path / "samples.npz").exists()


def test_checkpoint_cli_runs_saved_protocol_and_records_provenance(monkeypatch, tmp_path):
    from looplora.sym import train as training

    config = _config()
    config["eval"].update(splits=["val"], counterfactuals=False)
    _patch_spec(monkeypatch, _spec())
    checkpoint = tmp_path / "training" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"mock checkpoint")
    original_metrics = checkpoint.parent / "metrics.json"
    original_metrics.write_text('{"training": true}\n')
    calls = []

    def load(path, device=None):
        calls.append((path, device))
        model = _Model(64)
        model.config = {**config, "device": device}
        return model.eval()

    def no_training(*args, **kwargs):
        pytest.fail("checkpoint evaluation must not train")

    monkeypatch.setattr(training, "load_checkpoint", load)
    monkeypatch.setattr(training, "train", no_training)
    out = tmp_path / "evaluation"
    main(["--checkpoint", str(checkpoint), "--out", str(out), "--device", "cpu"])
    metrics = json.loads((out / "metrics.json").read_text())
    assert calls == [(checkpoint.resolve(), "cpu")]
    assert metrics["checkpoint"] == str(checkpoint.resolve())
    assert metrics["config"]["eval"] == config["eval"]
    assert metrics["config"]["device"] == "cpu"
    assert len(metrics["config_hash"]) == 32
    assert {row["L"] for row in metrics["table"]} == set(config["eval"]["depths"])
    assert (out / "samples.npz").exists()
    assert (out / "timing.json").exists()
    assert original_metrics.read_text() == '{"training": true}\n'
    with pytest.raises(FileExistsError, match="new or empty"):
        main(["--checkpoint", str(checkpoint), "--out", str(out)])
    with pytest.raises(FileExistsError, match="new or empty"):
        main(["--checkpoint", str(checkpoint), "--out", str(checkpoint.parent)])
    assert len(calls) == 1


def test_orthogonal_counterfactual_keeps_query_and_original_entity_codes(monkeypatch, tmp_path):
    from looplora.sym.config import resolve_config
    from looplora.sym.model import SymbolicModel

    config = resolve_config({
        "device": "cpu", "writer": "oracle_orth",
        "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 4},
        "task": {"name": "t2_chain_end", "E": 4, "Q": 4, "V": 64,
                 "seen": [0, 32], "unseen": [32, 64], "chain_lengths": [4], "hops": [1, 2, 3, 4]},
        "eval": {"episodes": 2, "batch_size": 2, "splits": ["test"],
                 "depths": [0, 1, 4], "hops": [1, 2, 3, 4], "counterfactuals": True},
    })
    model = SymbolicModel(config)
    real_prepare, real_codebooks = model.prepare, model.orthogonal_codebooks
    captured, codebook_calls = [], []

    def prepare(arrays, **options):
        assert options.get("oracle_codes") is not None
        result = real_prepare(arrays, **options)
        captured.append((arrays, result))
        return result

    def make_codebooks(original, replacements=None, **options):
        codebook_calls.append((original, replacements))
        return real_codebooks(original, replacements, **options)

    monkeypatch.setattr(model, "prepare", prepare)
    monkeypatch.setattr(model, "orthogonal_codebooks", make_codebooks)
    metrics, _ = evaluate(model, config, tmp_path)
    assert codebook_calls[0][1] is None
    assert any(replacement is not None for _, replacement in codebook_calls)
    assert metrics["codebook_policy"] == "fixed_source_episode_basis_with_replacement_entity_slots"
    original_arrays, original = captured[0]
    offset = 0
    replacements = next(arrays for arrays, _ in captured
                        if arrays.meta.get("counterfactual", {}).get("kind") == "replace_key_fact")
    source_episodes = np.asarray(replacements.meta["source_episode"])
    source_queries = np.asarray(replacements.meta["source_query"])
    for arrays, prepared in captured:
        if arrays.meta.get("counterfactual", {}).get("kind") != "replace_key_fact":
            continue
        for row in range(len(arrays.episode_seed)):
            source, query = int(source_episodes[offset + row]), int(source_queries[offset + row])
            assert torch.equal(prepared.u[row, 0], original.u[source, query])
            assert torch.equal(prepared.x[row, 0], original.x[source, query])
            entities = np.unique(original_arrays.facts[source, :, (0, 2)])
            assert torch.equal(prepared.codes[row, entities], original.codes[source, entities])
        offset += len(arrays.episode_seed)
    assert offset == len(source_episodes)
