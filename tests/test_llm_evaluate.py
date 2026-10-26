"""Evaluation records, counterfactual controls, and strict depth pairing."""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from looplora.data.relations import EpisodeArrays, TaskSpec, generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed
from looplora.llm.evaluate import aggregate_samples, evaluate
from looplora.llm.protocols import SupportSnapshot
from looplora.sym.evaluate import stratify_queries
from looplora.sym.stats import compare_runs, load_samples


def config():
    return dict(seed=17, stage="joint", variant="M1", protocol="memory_only", smoke=True,
                task=dict(name="t2_chain_end", E=8, Q=2, V=1024, n_rel=4,
                          chain_lengths=[8], hops=[1, 2, 3]),
                model=dict(r=4, layer=14, rho=.25, alpha=1.),
                eval=dict(episodes=3, batch_size=2, depths=[0, 1, 4, 8], splits=["val"],
                          protocols=["memory_only", "context_retained"], hops=[1, 2, 8],
                          seed=4242, counterfactuals=True, allow_confirmation=False),
                stats=dict(resamples=2000, seed=2718))


@dataclass(frozen=True)
class Prepared:
    snapshot: SupportSnapshot
    answer_ids: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_mask: torch.Tensor
    arrays: EpisodeArrays
    write_count: torch.Tensor


class Experiment:
    """A scorer whose primary and candidate argmax deliberately disagree."""

    def __init__(self, *, change_inputs=False, mutate_memory=False):
        self.model = torch.nn.Identity()
        self.device = "cpu"
        self.adapter = SimpleNamespace(u=lambda x: x[..., :2])
        self.change_inputs, self.mutate_memory = change_inputs, mutate_memory
        self.pool = load_name_pool()
        self.tokens = torch.tensor((*self.pool.token_ids, self.pool.insufficient_token_id))
        self.vocab = int(self.tokens.max()) + 1
        self.preparations = []

    def prepare(self, arrays, protocol=None):
        memory = torch.tensor(arrays.facts[:, :, 2].sum(-1), dtype=torch.float32)[:, None, None]
        # Distinguishable KV must remain the source's when only M is swapped.
        kv = None if protocol == "memory_only" else ((memory.clone(), memory.clone()),)
        snapshot = SupportSnapshot({14: memory}, kv, torch.ones(len(memory), 1, dtype=torch.bool), protocol)
        entities = [np.unique(facts[valid][:, (0, 2)]) for facts, valid in zip(arrays.facts, arrays.fact_valid)]
        width = max(map(len, entities))
        candidates = torch.zeros(len(memory), width, dtype=torch.long)
        valid = torch.zeros_like(candidates, dtype=torch.bool)
        for i, ids in enumerate(entities):
            candidates[i, :len(ids)] = self.tokens[torch.tensor(ids)]
            valid[i, :len(ids)] = True
        result = Prepared(snapshot, self.tokens[torch.tensor(arrays.answer)], candidates, valid,
                          arrays, torch.tensor(arrays.fact_valid.sum(-1)))
        self.preparations.append(result)
        return result

    def score(self, prepared, L, *, capture_inputs=False):
        x = torch.tensor(prepared.arrays.q_start, dtype=torch.float32)[:, :, None, None].expand(-1, -1, 1, 4).clone()
        if self.change_inputs:
            x += L
        if self.mutate_memory:
            prepared.snapshot.states[14].add_(1)
        logits = torch.zeros((*prepared.answer_ids.shape, self.vocab))
        logits.scatter_(-1, prepared.answer_ids[:, :, None], 1.)
        # Token 0 is absent from the pinned name asset.
        assert not self.tokens.eq(0).any()
        logits[..., 0] = 2.
        return logits, x if capture_inputs else None


def test_fixed_hop_sets_records_reproduce_metrics_and_primary_is_full_vocab(tmp_path):
    cfg = config()
    experiment = Experiment()
    metrics, timing = evaluate(experiment, cfg, tmp_path)
    samples = load_samples(tmp_path / "samples.npz")
    assert metrics == json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["table"] == aggregate_samples(samples)
    assert all(row["accuracy"] == row["full_accuracy"] == 0 for row in metrics["table"])
    assert all(row["candidate_accuracy"] == 1 for row in metrics["table"])
    assert np.array_equal(samples["prediction"], samples["full_prediction"])
    assert np.array_equal(samples["correct"], samples["full_correct"])
    assert set(samples["hop"]) == {1, 2, 8}
    assert metrics["gate0_passed"]
    assert timing["evaluation_seconds"] > 0 and timing["writer_seconds"] > 0
    assert set(timing["reader_seconds"]) == {"0", "1", "4", "8"}
    assert len(timing["eval_sets"]) == 2
    assert experiment.model.training
    comparisons = compare_runs(samples, samples, 4, 1)
    assert comparisons and all(row["difference"] == 0 for row in comparisons)
    assert metrics["depth_comparisons"]
    for record in metrics["invariant_hashes"]:
        paired = [row for row in metrics["memory_hashes"]
                  if row["split"] == record["split"] and row["condition"] == record["condition"]
                  and row["batch_start"] == record["episode_start"]]
        assert len(paired) == 4
        for field in ("x", "u", "write_count"):
            assert {row[field] for row in paired} == {record[field]}
        assert {row["memory"] for row in paired} == {record["M"]}


def test_counterfactual_records_keep_original_labels_and_query_ids(tmp_path):
    from looplora.llm.train import make_task_spec

    cfg = config()
    evaluate(Experiment(), cfg, tmp_path)
    samples = load_samples(tmp_path)
    spec = make_task_spec(cfg, evaluation=True)
    arrays = stratify_queries(generate(spec, "val", 0, 3, 4242), spec, cfg["eval"]["hops"], 4242)
    token_ids = np.asarray(load_name_pool().token_ids)
    expected = {(int(ei), qi): (int(token_ids[answer]), int(hop))
                for ei, answers, hops in zip(arrays.episode_seed, arrays.answer, arrays.q_hop_true)
                for qi, (answer, hop) in enumerate(zip(answers, hops))}
    for i in range(len(samples["correct"])):
        key = int(samples["episode_id"][i]), int(samples["query_id"][i])
        answer, hop = expected[key]
        assert samples["original_answer"][i] == answer
        assert samples["hop"][i] == hop
        assert samples["original_prediction"][i] == 0
        if samples["condition"][i].endswith("/swap_memory"):
            assert samples["answer"][i] == answer
            assert samples["memory_episode_id"][i] != samples["episode_id"][i]
        elif samples["condition"][i].endswith("/replace_key_fact"):
            assert samples["answer"][i] != answer
    # Swapping M leaves each source episode's context-retained KV unchanged.
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    rows = metrics["memory_hashes"]
    for row in rows:
        if row["condition"] == "context_retained/swap_memory":
            original = next(other for other in rows if other["condition"] == "context_retained/original"
                            and other["L"] == row["L"] and other["batch_start"] == row["batch_start"])
            assert row["backbone_kv"] == original["backbone_kv"]


@pytest.mark.parametrize("kwargs", [{"change_inputs": True}, {"mutate_memory": True}])
def test_gate_zero_rejects_changed_activations_or_mutated_memory(tmp_path, kwargs):
    experiment = Experiment(**kwargs)
    experiment.model.eval()
    with pytest.raises(AssertionError, match="Gate 0"):
        evaluate(experiment, config(), tmp_path)
    assert not experiment.model.training
    assert not (tmp_path / "samples.npz").exists()


@pytest.mark.parametrize("smoke,allowed", [(True, False), (True, True), (False, False)])
def test_confirmation_guard_precedes_preparation(tmp_path, smoke, allowed):
    cfg = config()
    cfg["smoke"] = smoke
    cfg["eval"].update(splits=["confirm"], allow_confirmation=allowed)
    experiment = Experiment()
    with pytest.raises(ValueError, match="confirmation"):
        evaluate(experiment, cfg, tmp_path)
    assert not experiment.preparations


def test_b4_uses_only_single_read(tmp_path):
    cfg = config()
    cfg["variant"] = "B4"
    cfg["eval"]["counterfactuals"] = False
    metrics, _ = evaluate(Experiment(), cfg, tmp_path)
    assert {row["L"] for row in metrics["table"]} == {1}
    assert metrics["depth_comparisons"] == []


@pytest.fixture(scope="module")
def backbone():
    from looplora.llm.inject import load_model, load_tokenizer

    model = load_model(dtype=torch.bfloat16)
    yield model, load_tokenizer()
    del model
    torch.cuda.empty_cache()


def test_real_teacher_forced_em_equals_greedy_first_token_and_gate_zero(backbone, tmp_path):
    from looplora.llm.batching import repeat_snapshot
    from looplora.llm.protocols import greedy_answer
    from looplora.llm.train import LLMExperiment

    model, tokenizer = backbone
    cfg = config()
    cfg.update(device="cuda", write_policy="fact_final")
    cfg["eval"].update(episodes=2, batch_size=2, depths=[0, 1, 4], counterfactuals=False)
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        rng = torch.Generator(device="cuda").manual_seed(derive_seed(17, "evaluation-test", "nonzero-B"))
        with torch.no_grad():
            experiment.adapter.B.normal_(0, .08, generator=rng)
            arrays = generate(TaskSpec(**cfg["task"]), "val", 0, 2, 4242)
            for protocol in cfg["eval"]["protocols"]:
                batch = experiment.prepare(arrays, protocol=protocol)
                B, Q, T = batch.question_ids.shape
                for depth in cfg["eval"]["depths"]:
                    scores, _ = experiment.score(batch, depth)
                    greedy = greedy_answer(model, repeat_snapshot(batch.snapshot, Q),
                                           batch.question_ids.reshape(B*Q, T),
                                           batch.question_mask.reshape(B*Q, T), loop_steps=depth)
                    torch.testing.assert_close(scores.argmax(-1), greedy[:, 0].reshape(B, Q), atol=0, rtol=0)
                    assert torch.equal(scores.argmax(-1) == batch.answer_ids,
                                       greedy[:, 0].reshape(B, Q) == batch.answer_ids)
        metrics, _ = evaluate(experiment, cfg, tmp_path)
        assert metrics["gate0_passed"] and metrics["invariant_hashes"]
    finally:
        experiment.close()


def test_real_b0_memory_only_is_at_chance_and_support_independent(backbone, tmp_path):
    from dataclasses import replace
    from looplora.llm.train import LLMExperiment

    model, tokenizer = backbone
    cfg = config()
    cfg.update(device="cuda", write_policy="fact_final", stage="baseline", variant="B0")
    cfg["task"] = dict(name="t1", E=4, Q=1, V=1024, n_rel=4, head_sharing=.3, hops=[1])
    cfg["eval"].update(episodes=64, batch_size=16, depths=[1], hops=[1],
                       protocols=["memory_only"], counterfactuals=False)
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        with torch.no_grad():
            arrays = generate(TaskSpec(**cfg["task"]), "val", 0, 2, 4242)
            changed = replace(arrays, facts=arrays.facts[::-1].copy(), fact_valid=arrays.fact_valid[::-1].copy())
            batch, other = experiment.prepare(arrays), experiment.prepare(changed)
            a, _ = experiment.score(batch, 1)
            b, _ = experiment.score(other, 1)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert not batch.snapshot.states and not batch.write_count.any()
        metrics, _ = evaluate(experiment, cfg, tmp_path)
        row = metrics["table"][0]
        chance, n = row["chance_accuracy"], row["n"]
        bound = chance + 3 * np.sqrt(chance * (1-chance) / n) + 1/n
        assert row["candidate_accuracy"] <= bound
        assert row["accuracy"] <= bound
    finally:
        experiment.close()
