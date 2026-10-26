"""Exact CPU prefetch and unchanged training contracts (run on PBS compute)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, fields

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.data.relations import EpisodeArrays, TaskSpec, generate
from looplora.infra.seeds import derive_seed
from looplora.llm.batching import TokenBatch, tokenize_batch
from looplora.llm.config import resolve_config
from looplora.llm.inject import load_tokenizer
from looplora.llm.prefetch import BatchStream
from looplora.llm.train import LLMExperiment, train


class FailingTokenizer:
    """Spawn-picklable failure at the first rendering operation."""

    def __call__(self, *args, **kwargs):
        raise ValueError("deliberate prefetch tokenizer failure")


class ExitingTokenizer:
    def __call__(self, *args, **kwargs):
        os._exit(17)


class UnpicklableTokenizer:
    def __reduce__(self):
        raise ValueError("deliberate tokenizer serialization failure")


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def backbone():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(23, "prefetch-tiny-qwen"))
        cfg = Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4, attention_dropout=0.)
        model = Qwen3ForCausalLM(cfg).cuda().eval()
    yield model
    assert not hasattr(model, "looplora_adapters")


def task(name="t2_chain_end"):
    return (TaskSpec("t1", E=4, Q=2, head_sharing=.3, hops=(1,)) if name == "t1"
            else TaskSpec(name, E=4, Q=2, chain_lengths=(4,), hops=(1, 2, 3)))


def assert_arrays_equal(actual, expected):
    for field in fields(EpisodeArrays):
        observed, reference = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "meta":
            assert observed == reference
        else:
            assert observed.dtype == reference.dtype, field.name
            np.testing.assert_array_equal(observed, reference, err_msg=field.name)


def assert_tokens_equal(actual, expected):
    assert isinstance(actual, TokenBatch)
    for field in fields(TokenBatch):
        observed, reference = getattr(actual, field.name), getattr(expected, field.name)
        assert observed.device.type == "cpu", field.name
        assert observed.dtype == reference.dtype, field.name
        assert not observed.requires_grad, field.name
        assert torch.equal(observed, reference), field.name


@pytest.mark.parametrize("name", ["t1", "t2_chain_end"])
@pytest.mark.parametrize("write_policy", ["fact_final", "all_tokens"])
def test_fifty_steps_are_exact_for_both_tasks_and_write_policies(tokenizer, name, write_policy):
    spec, batch_size, seed = task(name), 2, derive_seed(23, "prefetch-exact", name)
    before = {child.pid for child in mp.active_children()}
    with BatchStream(spec, "train", batch_size, seed, 50,
                     tokenizer=tokenizer, workers=2) as stream:
        count = 0
        for step, (arrays, tokens) in enumerate(stream):
            expected = generate(spec, "train", step * batch_size, batch_size, seed)
            expected_tokens = tokenize_batch(expected, tokenizer, device="cpu")
            assert_arrays_equal(arrays, expected)
            assert_tokens_equal(tokens, expected_tokens)
            # Both policies consume these exact masks in prepare/encode_support.
            actual_write = (tokens.support_mask if write_policy == "all_tokens"
                            else tokens.fact_final & tokens.support_mask)
            expected_write = (expected_tokens.support_mask if write_policy == "all_tokens"
                              else expected_tokens.fact_final & expected_tokens.support_mask)
            assert torch.equal(actual_write, expected_write)
            if write_policy == "fact_final":
                assert actual_write.sum(-1).tolist() == [spec.E] * batch_size
            count += 1
        assert count == 50
    assert {child.pid for child in mp.active_children()} <= before


@pytest.mark.parametrize("workers", [0, 2])
def test_resumed_stream_uses_absolute_episode_offsets(tokenizer, workers):
    spec, seed, batch_size = task(), 43, 3
    with BatchStream(spec, "train", batch_size, seed, 11, tokenizer=tokenizer,
                     workers=workers, start_step=7) as stream:
        count = 0
        for step, (arrays, tokens) in enumerate(stream, start=7):
            expected = generate(spec, "train", step * batch_size, batch_size, seed)
            assert_arrays_equal(arrays, expected)
            if workers:
                assert_tokens_equal(tokens, tokenize_batch(expected, tokenizer, device="cpu"))
            else:
                assert tokens is None
            count += 1
        assert count == 4


def test_default_zero_workers_never_tokenizes_or_starts_children():
    spec, seed = task("t1"), 17
    before = {child.pid for child in mp.active_children()}
    with BatchStream(spec, "train", 2, seed, 3, tokenizer=FailingTokenizer()) as stream:
        assert iter(stream) is stream
        for step in range(3):
            arrays, tokens = next(stream)
            assert tokens is None
            assert_arrays_equal(arrays, generate(spec, "train", step * 2, 2, seed))
        with pytest.raises(StopIteration):
            next(stream)
    assert {child.pid for child in mp.active_children()} == before
    assert resolve_config({})["train"]["prefetch_workers"] == 0


@pytest.mark.parametrize("start_step,steps", [(0, 0), (5, 5)])
def test_empty_stream_does_not_start_workers(start_step, steps):
    before = {child.pid for child in mp.active_children()}
    with BatchStream(task(), "train", 2, 17, steps, tokenizer=FailingTokenizer(),
                     workers=2, start_step=start_step) as stream:
        assert list(stream) == []
    stream.close()
    assert {child.pid for child in mp.active_children()} == before


@pytest.mark.parametrize("raise_in_body", [False, True])
def test_early_context_exit_joins_live_workers(tokenizer, raise_in_body):
    before = {child.pid for child in mp.active_children()}
    workers = []
    stream = BatchStream(task(), "train", 2, 17, 50, tokenizer=tokenizer, workers=2)
    try:
        with stream:
            _, tokens = next(stream)
            assert tokens.support_ids.device.type == "cpu"
            workers = [child for child in mp.active_children() if child.pid not in before]
            assert len(workers) == 2 and all(child.is_alive() for child in workers)
            if raise_in_body:
                raise RuntimeError("deliberate consumer failure")
    except RuntimeError as exc:
        assert raise_in_body and str(exc) == "deliberate consumer failure"
    assert workers and all(not child.is_alive() for child in workers)
    assert {child.pid for child in mp.active_children()} <= before
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)


@pytest.mark.parametrize("tokenizer_type,error,match", [
    (FailingTokenizer, ValueError, "deliberate prefetch tokenizer failure"),
    (ExitingTokenizer, BrokenProcessPool, "terminated abruptly"),
])
def test_worker_exception_propagates_and_joins_workers(tokenizer_type, error, match):
    before = {child.pid for child in mp.active_children()}
    stream = BatchStream(task(), "train", 2, 17, 8,
                         tokenizer=tokenizer_type(), workers=2)
    with pytest.raises(error, match=match):
        with stream:
            next(stream)
    stream.close()
    assert {child.pid for child in mp.active_children()} <= before


def test_startup_serialization_failure_closes_stream_and_workers():
    before = {child.pid for child in mp.active_children()}
    stream = BatchStream(task(), "train", 2, 17, 8,
                         tokenizer=UnpicklableTokenizer(), workers=2)
    with pytest.raises(ValueError, match="deliberate tokenizer serialization failure"):
        with stream:
            pytest.fail("startup must reject the unserializable tokenizer")
    stream.close()
    assert {child.pid for child in mp.active_children()} <= before
    with pytest.raises(StopIteration):
        next(stream)


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "2", None])
def test_prefetch_workers_rejects_non_integer_or_negative_config(value):
    with pytest.raises(ValueError, match="train.prefetch_workers"):
        resolve_config({"train": {"prefetch_workers": value}})
    with pytest.raises(ValueError, match="workers"):
        BatchStream(task(), "train", 2, 17, 3, tokenizer=None, workers=value)


@pytest.mark.parametrize("workers", [0, 2, 4, 8])
def test_prefetch_worker_config_accepts_nonnegative_integers(workers):
    assert resolve_config({"train": {"prefetch_workers": workers}})["train"]["prefetch_workers"] == workers


@pytest.mark.parametrize("kwargs,match", [
    ({"steps": -1}, "steps"), ({"steps": True}, "steps"),
    ({"batch_size": 0}, "batch_size"), ({"batch_size": 1.5}, "batch_size"),
    ({"start_step": -1}, "start_step"), ({"start_step": True}, "start_step"),
    ({"start_step": 4, "steps": 3}, "start_step"),
])
def test_stream_validates_bounds_before_starting_workers(kwargs, match):
    options = {"batch_size": 2, "steps": 3, "workers": 2, **kwargs}
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(ValueError, match=match):
        BatchStream(task(), "train", seed=17, tokenizer=None, **options)
    assert {child.pid for child in mp.active_children()} == before


def config(write_policy="fact_final"):
    return {"name": "prefetch-test", "stage": "joint", "variant": "M1", "seed": 23,
            "device": "cuda", "protocol": "memory_only", "write_policy": write_policy,
            "model": {"r": 4, "layer": 1, "rho": .25, "alpha": 1.},
            "task": asdict(task()), "writer_checkpoint": None,
            "train": {"steps": 2, "batch_size": 2, "lr": .003, "weight_decay": .01,
                      "grad_clip": 1., "warmup_steps": 1, "depths": [1, 2, 4], "log_every": 1},
            "eval": {"hops": [1, 2, 3, 4]}}


@pytest.mark.parametrize("write_policy", ["fact_final", "all_tokens"])
@pytest.mark.parametrize("protocol", ["memory_only", "context_retained"])
def test_prepared_cpu_tokens_preserve_gpu_support_and_logits(backbone, tokenizer, monkeypatch,
                                                            write_policy, protocol):
    experiment = LLMExperiment(config(write_policy), model=backbone, tokenizer=tokenizer)
    try:
        with BatchStream(task(), "train", 2, 17, 1, tokenizer=tokenizer, workers=2) as stream:
            arrays, tokens = next(stream)
        with torch.no_grad():
            experiment.adapter.B.fill_(.01)
            sequential = experiment.prepare(arrays, protocol=protocol)
            expected, _ = experiment.score(sequential, 2)

            def unexpected_tokenization(*args, **kwargs):
                raise AssertionError("prepare must consume the supplied CPU tokens")

            monkeypatch.setattr("looplora.llm.train.tokenize_batch", unexpected_tokenization)
            prepared = experiment.prepare(arrays, protocol=protocol, tokens=tokens)
            observed, _ = experiment.score(prepared, 2)
        for field in fields(TokenBatch):
            actual = getattr(prepared, field.name)
            assert actual.device.type == "cuda"
            assert torch.equal(actual, getattr(sequential, field.name)), field.name
        assert prepared.support_tokens == sequential.support_tokens
        assert torch.equal(prepared.write_count, sequential.write_count)
        for layer, state in sequential.snapshot.states.items():
            assert torch.equal(prepared.snapshot.states[layer], state)
        assert torch.equal(prepared.snapshot.attention_mask, sequential.snapshot.attention_mask)
        expected_cache, actual_cache = sequential.snapshot.past_key_values, prepared.snapshot.past_key_values
        if expected_cache is None:
            assert actual_cache is None
        else:
            assert len(actual_cache) == len(expected_cache)
            for actual_layer, expected_layer in zip(actual_cache, expected_cache, strict=True):
                assert all(torch.equal(a, b) for a, b in zip(actual_layer, expected_layer, strict=True))
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    finally:
        experiment.close()


@pytest.mark.parametrize("write_policy", ["fact_final", "all_tokens"])
def test_fit_with_prefetch_matches_default_training_exactly(backbone, tokenizer, tmp_path, write_policy):
    records, summaries, states = [], [], []
    for workers in (0, 2):
        cfg = config(write_policy)
        # Omission exercises the existing default training path.
        if workers:
            cfg["train"]["prefetch_workers"] = workers
        out = tmp_path / f"workers{workers}"
        experiment, summary = train(cfg, out, model=backbone, tokenizer=tokenizer)
        try:
            states.append({key: value.detach().cpu().clone()
                           for key, value in experiment.adapter.state_dict().items()})
            records.append([{key: value for key, value in json.loads(line).items()
                             if key != "step_seconds"}
                            for line in (out / "train_log.jsonl").read_text().splitlines()])
            summaries.append(summary)
        finally:
            experiment.close()
    assert len(records[0]) == 2 and records[0] == records[1]
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key
    for key in ("steps", "final_loss", "data_seed", "train_tokens", "train_queries", "mean_L"):
        assert summaries[0][key] == summaries[1][key], key
