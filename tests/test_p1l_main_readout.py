"""P1-L state replay, residual gradients and design-boundary regressions."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from readout_p1l import (name_gradients, normalize_directions, path_entities,
                         read_batch, reader_states, score_answer_branches)
import evaluate_p1l_main as evaluator
from run_p1l_main import LABELS, config, labels_for_rank
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.data.relations import generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed
from looplora.llm.config import resolve_config
from looplora.llm.inject import configure_numerics, load_tokenizer
from looplora.llm.train import LLMExperiment, make_task_spec


@pytest.mark.parametrize("rank", [32, 128])
def test_registered_runs_resolve_and_confirmation_is_held(rank):
    for label in (*labels_for_rank(rank), *(("B0",) if rank == 32 else ())):
        for seed in ((17,) if label == "B0" else (17, 29, 43)):
            cfg = resolve_config(config(label, seed, rank))
            assert cfg["model"]["r"] == rank
            assert cfg["eval"]["splits"] == ["val"]
            assert not cfg["eval"]["allow_confirmation"]
            assert cfg["train"]["batch_size"] == 64 and cfg["task"]["E"] == 12
            assert cfg["train"]["depths"] == ([1] if label in {"B0", "B1", "B2", "B4"} else [4])


@pytest.mark.parametrize("variant", ["iterq", "unshared"])
def test_true_state_prefixes_end_at_deployed_reader(variant):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "readout-state"))
        adapter = make_adapter(AdapterConfig(d_in=7, d_out=5, r=8, variant=variant, unshared_steps=4))
        u, memory = torch.randn(3, 8), torch.randn(3, 8, 8)
    before = memory.clone()
    states = reader_states(adapter, u, memory)
    assert torch.equal(states[-1], adapter.read(u, memory, 4))
    assert torch.equal(memory, before)
    if variant == "iterq":
        for step, state in enumerate(states, 1):
            assert torch.equal(state, adapter.read(u, memory, step))
    else:
        with pytest.raises(ValueError, match="unshared"):
            adapter.read(u, memory, 1)


def test_latest_fact_paths_do_not_invent_post_terminal_entities():
    arrays = SimpleNamespace(facts=np.array([[[1, 0, 2], [2, 0, 3], [1, 0, 4], [4, 0, 5],
                                             [1, 1, 7], [5, 0, 6]]]),
        fact_valid=np.array([[1, 1, 1, 1, 1, 0]], dtype=bool),
        q_start=np.array([[1, 1]]), q_rel=np.array([[0, 1]]), answer=np.array([[5, 7]]))
    assert path_entities(arrays).tolist() == [[[4, 5, -1, -1], [7, -1, -1, -1]]]


def tiny(vocab=64):
    configure_numerics()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "readout-backbone"))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).eval()
    model.requires_grad_(False)
    return model


def test_name_gradient_batches_match_scalar_vjps_and_name_contrast():
    model = tiny()
    ids, valid = torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=bool)
    names = torch.tensor([6, 7, 8, 9])
    gradients = name_gradients(model, ids, valid, names, layer=0, chunk=2)
    separate = sum(name_gradients(model, ids[i:i+1], valid[i:i+1], names, layer=0, chunk=1)
                   for i in range(2))
    torch.testing.assert_close(gradients, separate, rtol=1e-5, atol=1e-6)
    primary, raw = normalize_directions(gradients)
    common = torch.arange(16).float()[None] * .03
    shifted, _ = normalize_directions(gradients + common)
    torch.testing.assert_close(primary, shifted, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(primary, raw)
    assert not model.model.layers[0].self_attn.o_proj._forward_hooks


@pytest.mark.parametrize("variant", ["M1", "B8"])
@pytest.mark.parametrize("rank", [32, 128])
def test_final_logit_lens_replays_actual_model_with_nonzero_branch(variant, rank):
    model, tokenizer = tiny(151936).cuda().to(torch.bfloat16), load_tokenizer()
    cfg = resolve_config(dict(writer="oracle_random", stage="reader", variant=variant,
        model=dict(r=rank, layer=0), task=dict(E=4, Q=2, chain_lengths=[4]),
        train=dict(batch_size=2, depths=[4]),
        eval=dict(episodes=2, batch_size=2, hops=[1, 2, 3, 4], counterfactuals=False)))
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        generator = torch.Generator(device="cuda").manual_seed(derive_seed(17, "readout-B"))
        with torch.no_grad():
            experiment.adapter.B.normal_(0, .1, generator=generator)
        arrays = generate(make_task_spec(cfg), "val", 0, 2, 4242)
        prepared = experiment.prepare(arrays)
        names = torch.tensor(load_name_pool().token_ids[:1024])
        raw = torch.randn((1024, 16), generator=torch.Generator().manual_seed(derive_seed(17, "test-J")))
        primary, secondary = normalize_directions(raw)
        directions = dict(names=names, contrastive=primary, raw_logit=secondary, random=secondary)
        outputs, actual = read_batch(experiment, prepared, directions)
        assert np.array_equal(outputs[-1]["logit_lens"].argmax(-1), actual)
        _, deployed_branch = score_answer_branches(experiment, prepared)
        expected = deployed_branch @ primary.cuda().T
        assert np.array_equal(outputs[-1]["contrastive"], expected.cpu().numpy())
        assert any(not np.array_equal(outputs[0][kind], outputs[-1][kind]) for kind in outputs[0])
    finally:
        experiment.close()


def test_swap_wrapper_preserves_source_queries_and_writes_reproducible_counts(monkeypatch, tmp_path):
    model, tokenizer = tiny(151936).cuda().to(torch.bfloat16), load_tokenizer()
    cfg = resolve_config(config("M1-L4"))
    cfg["model"]["layer"] = 0
    cfg["eval"]["batch_size"] = 4
    # The scientific helper's requested test range is intercepted: this test
    # only consumes a two-episode validation fixture, never confirmation data.
    spec = make_task_spec(cfg, evaluation=True)
    fixture = evaluator.stratify_queries(generate(spec, "val", 0, 2, 4242),
                                        spec, cfg["eval"]["hops"], 4242)
    monkeypatch.setattr(evaluator, "generate", lambda *args, **kwargs: fixture)
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        result = evaluator.swap_successors(experiment, cfg, tmp_path)
    finally:
        experiment.close()
    assert result["episodes"] == 2
    with np.load(tmp_path / "swap-successors-samples.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    assert set(arrays["episode_id"]) == set(fixture.episode_seed)
    assert set(arrays["L"]) == {1, 4, 8}
    assert set(arrays["query_id"]) == set(range(8))
    assert np.array_equal(arrays["hop"], arrays["query_id"] + 1)
    assert len(arrays["L"]) == 2 * 8 * 3
    for row in result["table"]:
        keep = (arrays["L"] == row["L"]) & (arrays["hop"] == row["hop"])
        assert row["n"] == 2
        assert row["follows_new"] == (arrays["prediction"][keep] == arrays["answer"][keep]).mean()


def test_inference_report_counts_adapter_and_real_batched_score_calls(tmp_path):
    import json
    model, tokenizer = tiny(151936).cuda().to(torch.bfloat16), load_tokenizer()
    cfg = resolve_config(config("B8"))
    cfg["model"]["layer"] = 0
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        evaluator.inference_timing(experiment, cfg, tmp_path)
        expected = sum(p.numel() for p in experiment.adapter.parameters())
    finally:
        experiment.close()
    result = json.loads((tmp_path / "inference.json").read_text())
    assert result["adapter_parameters"] == expected
    assert result["adapter_parameters"] > result["trainable_adapter_parameters"] > 0
    row, = result["timings"]
    assert row["batch_questions"] == 128 and row["L"] == 4
    assert len(row["batch_seconds"]) == 20 and row["warmups"] == 5
    assert row["mean_seconds_per_question"] > 0
    assert row["mean_seconds_per_question"] == pytest.approx(np.mean(row["batch_seconds"]) / 128)


def test_readout_wrapper_keeps_every_episode_across_partial_batches(tmp_path):
    import json
    model, tokenizer = tiny(151936).cuda().to(torch.bfloat16), load_tokenizer()
    cfg = resolve_config(config("B8"))
    cfg["model"]["layer"] = 0
    cfg["eval"].update(episodes=5, batch_size=2)
    raw = torch.randn((1024, 16), generator=torch.Generator().manual_seed(
        derive_seed(17, "test-readout-batches")))
    primary, secondary = normalize_directions(raw)
    directions = dict(names=torch.tensor(load_name_pool().token_ids[:1024]),
                      contrastive=primary, raw_logit=secondary, random=secondary)
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        evaluator.readout(experiment, cfg, directions, tmp_path)
    finally:
        experiment.close()
    result = json.loads((tmp_path / "readout.json").read_text())
    assert {row["hop"] for row in result["faithfulness"]} == set(range(1, 9))
    assert {row["reader_step"] for row in result["table"]} == {1, 2, 3, 4}
    assert all(row["n"] == 5 for row in result["faithfulness"] + result["table"])
