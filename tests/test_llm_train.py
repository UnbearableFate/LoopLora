"""Training contracts on a small real Qwen3 architecture and pinned tokenizer."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.data.relations import TaskSpec, generate
from looplora.infra.seeds import derive_seed
from looplora.llm.batching import repeat_snapshot
from looplora.llm.inject import load_tokenizer
from looplora.llm.protocols import answer_scores, greedy_answer
from looplora.llm.train import LLMExperiment, load_checkpoint, train


@pytest.fixture(scope="module")
def backbone():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(23, "tiny-qwen"))
        cfg = Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4, attention_dropout=0.)
        model = Qwen3ForCausalLM(cfg).cuda().eval()
    yield model
    assert not hasattr(model, "looplora_adapters")


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def config(stage="joint", variant="M1", protocol="memory_only"):
    return {"name": "test", "stage": stage, "variant": variant, "seed": 23,
            "device": "cuda", "protocol": protocol, "write_policy": "fact_final",
            "model": {"r": 4, "layer": 1, "rho": .25, "alpha": 1.},
            "task": asdict(TaskSpec("t2_chain_end", E=4, Q=2, chain_lengths=(4,), hops=(1, 2, 3))),
            "train": {"steps": 2, "batch_size": 2, "lr": .003, "weight_decay": .01,
                      "grad_clip": 1., "warmup_steps": 1, "depths": [1, 2, 4], "log_every": 1},
            "eval": {"hops": [1, 2, 3, 4]}, "writer_checkpoint": None}


def episodes(cfg):
    return generate(TaskSpec(**cfg["task"]), "train", 0, 2, derive_seed(23, "episodes"))


@pytest.mark.parametrize("protocol", ["memory_only", "context_retained"])
def test_flattened_questions_match_separate_forks_and_greedy(backbone, tokenizer, protocol):
    cfg = config(protocol=protocol)
    experiment = LLMExperiment(cfg, model=backbone, tokenizer=tokenizer)
    try:
        with torch.no_grad():
            experiment.adapter.B.normal_(0, .03, generator=torch.Generator(device="cuda").manual_seed(derive_seed(23, "test-B")))
            batch = experiment.prepare(episodes(cfg))
            snapshot = {key: value.clone() for key, value in batch.snapshot.states.items()}
            logits, x = experiment.score(batch, 4, capture_inputs=True)
            assert x.shape[:3] == batch.question_ids.shape
            assert batch.write_count.tolist() == [4, 4]
            for q in range(2):
                single = answer_scores(backbone, batch.snapshot, batch.question_ids[:, q],
                                       batch.question_mask[:, q], batch.answer_ids[:, q, None], loop_steps=4)
                torch.testing.assert_close(logits[:, q], single.logits[:, 0], atol=2e-6, rtol=2e-5)
            repeated = repeat_snapshot(batch.snapshot, 2)
            generated = greedy_answer(backbone, repeated, batch.question_ids.flatten(0, 1),
                                      batch.question_mask.flatten(0, 1), loop_steps=4)
            torch.testing.assert_close(logits.argmax(-1).flatten(), generated[:, 0])
            for key, value in snapshot.items():
                assert torch.equal(value, batch.snapshot.states[key])
    finally:
        experiment.close()


def test_single_token_training_uses_full_vocabulary_and_saves_no_backbone(backbone, tokenizer, tmp_path):
    cfg = config()
    cfg["train"]["steps"] = 1
    probe = LLMExperiment(cfg, model=backbone, tokenizer=tokenizer)
    try:
        data_seed = derive_seed(cfg["seed"], "llm", "train-data")
        arrays = generate(TaskSpec(**cfg["task"]), "train", 0, 2, data_seed)
        with torch.no_grad():
            prepared = probe.prepare(arrays)
            logits, _ = probe.score(prepared, 1)
            expected = float(F.cross_entropy(logits.float().flatten(0, 1), prepared.answer_ids.flatten()))
    finally:
        probe.close()
    experiment, summary = train(cfg, tmp_path, model=backbone, tokenizer=tokenizer)
    try:
        row = json.loads((tmp_path / "train_log.jsonl").read_text().strip())
        assert row["loss"] == pytest.approx(expected, abs=2e-6)
        assert summary["steps"] == 1 and summary["train_queries"] == 4
        assert experiment.adapter.B.abs().sum() > 0
        record = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
        assert set(record["adapter"]) == {"A", "B", "K", "V", "a_lambda", "a_beta", "R", "U", "Q"}
        assert set(record["writer"]) == {"A", "K", "V", "a_lambda", "a_beta"}
        assert all(not key.startswith(("model.", "lm_head.", "base.")) for key in record["adapter"])
    finally:
        experiment.close()


@pytest.fixture
def writer_checkpoint(backbone, tokenizer, tmp_path):
    cfg = config("writer")
    experiment, _ = train(cfg, tmp_path / "writer", model=backbone, tokenizer=tokenizer)
    try:
        assert experiment.config["task"]["name"] == "t1"
        assert experiment.config["task"]["n_rel"] == 4
        assert experiment.config["task"]["head_sharing"] == .3
        return tmp_path / "writer" / "writer.pt", experiment.writer_state()
    finally:
        experiment.close()


@pytest.mark.parametrize("variant", ["M1", "M1-lin", "B5", "B3", "B4"])
def test_frozen_writer_unchanged_and_reader_reload_self_contained(backbone, tokenizer, writer_checkpoint, tmp_path, variant):
    path, original = writer_checkpoint
    cfg = config("reader", variant)
    cfg["writer_checkpoint"] = str(path)
    experiment, summary = train(cfg, tmp_path / variant, model=backbone, tokenizer=tokenizer)
    try:
        assert summary["writer_frozen"]
        for name, tensor in original.items():
            assert torch.equal(tensor, experiment.writer_state()[name])
            assert not getattr(experiment.writer, name).requires_grad
        if variant == "B3":
            assert experiment.writer.A is experiment.adapter.A
        arrays = episodes(cfg)
        with torch.no_grad():
            batch = experiment.prepare(arrays)
            expected, _ = experiment.score(batch, 2)
        saved = {key: value.detach().cpu().clone() for key, value in experiment.adapter.state_dict().items()}
    finally:
        experiment.close()
    path.rename(path.with_suffix(".hidden"))
    restored = load_checkpoint(tmp_path / variant / "checkpoint.pt", model=backbone, tokenizer=tokenizer)
    try:
        for key, value in saved.items():
            assert torch.equal(value, restored.adapter.state_dict()[key].cpu())
        with torch.no_grad():
            observed, _ = restored.score(restored.prepare(arrays), 2)
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    finally:
        restored.close()


def test_b3_constructs_same_frozen_memory_and_ignores_memory(backbone, tokenizer, writer_checkpoint):
    path, _ = writer_checkpoint
    cfg = config("reader")
    cfg["writer_checkpoint"] = str(path)
    experiment = LLMExperiment(cfg, model=backbone, tokenizer=tokenizer)
    arrays = episodes(cfg)
    try:
        with torch.no_grad():
            expected = experiment.prepare(arrays).snapshot.states[1].clone()
    finally:
        experiment.close()
    cfg["variant"] = "B3"
    b3 = LLMExperiment(cfg, model=backbone, tokenizer=tokenizer)
    try:
        with torch.no_grad():
            b3.adapter.B.fill_(.01)
            batch = b3.prepare(arrays)
            assert torch.equal(expected, batch.snapshot.states[1])
            logits, x = b3.score(batch, 4, capture_inputs=True)
            changed = replace(batch.snapshot, states={1: batch.snapshot.states[1] + 100})
            modified, x2 = b3.score(replace(batch, snapshot=changed), 4, capture_inputs=True)
            assert torch.equal(logits, modified) and torch.equal(x, x2)
    finally:
        b3.close()


@pytest.mark.parametrize("stage,variant,protocol", [("baseline", "B0", "memory_only"),
                                                   ("baseline", "B0", "context_retained"),
                                                   ("lora_ctx", "B1", "context_retained"),
                                                   ("lora_ctx", "B2", "context_retained")])
def test_baselines_and_backbone_reuse(backbone, tokenizer, tmp_path, stage, variant, protocol):
    cfg = config(stage, variant, protocol)
    original = backbone.model.layers[1].self_attn.o_proj
    experiment, summary = train(cfg, tmp_path, model=backbone, tokenizer=tokenizer)
    try:
        assert summary["steps"] == (0 if variant == "B0" else 2)
        assert summary["writer_checkpoint"] is None
        assert experiment.config["variant"] == variant
        assert experiment.writer is None
        with torch.no_grad():
            batch = experiment.prepare(episodes(cfg))
            logits, x = experiment.score(batch, 1, capture_inputs=True)
        assert logits.shape == (2, 2, 151936) and x.shape[-1] == 16
        assert not batch.write_count.any()
        if variant == "B0":
            assert (tmp_path / "train_log.jsonl").read_text() == ""
    finally:
        experiment.close()
    assert backbone.model.layers[1].self_attn.o_proj is original


@pytest.mark.parametrize("protocol", ["memory_only", "context_retained"])
def test_b2_nonlinear_bottleneck_and_both_protocols(backbone, tokenizer, protocol):
    cfg = config("lora_ctx", "B2", "context_retained")
    experiment = LLMExperiment(cfg, model=backbone, tokenizer=tokenizer)
    try:
        assert experiment.adapter.cfg.variant == "parallel"
        assert experiment.adapter.cfg.r == cfg["model"]["r"]
        assert set(experiment.adapter.state_dict()) == {"A", "B"}
        with torch.no_grad():
            # A negative bottleneck feature is suppressed, which distinguishes
            # the nonlinear control from the B1 linear LoRA branch.
            experiment.adapter.A.zero_()
            experiment.adapter.A[:, 0] = 1
            experiment.adapter.B.fill_(.01)
            x = torch.zeros(2, 16, device="cuda")
            x[:, 0] = torch.tensor([-1., 1.], device="cuda")
            branch = experiment.adapter.branch(experiment.adapter.read(experiment.adapter.u(x), None, 1))
            assert branch[0].eq(0).all() and branch[1].gt(0).all()
            batch = experiment.prepare(episodes(cfg), protocol=protocol)
            assert batch.snapshot.protocol == protocol
            assert not batch.write_count.any()
            logits, inputs = experiment.score(batch, 1, capture_inputs=True)
            assert logits.shape == (2, 2, 151936) and inputs.shape[-1] == 16
            assert (batch.snapshot.past_key_values is None) == (protocol == "memory_only")
    finally:
        experiment.close()


def test_writer_mismatch_fails_and_removes_adapter(backbone, tokenizer, writer_checkpoint):
    path, _ = writer_checkpoint
    cfg = config("reader")
    cfg["writer_checkpoint"] = str(path)
    cfg["model"]["r"] = 3
    with pytest.raises(ValueError, match="signature"):
        LLMExperiment(cfg, model=backbone, tokenizer=tokenizer)
    assert not hasattr(backbone, "looplora_adapters")
