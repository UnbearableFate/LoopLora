"""Pilot accounting and timing boundaries, without loading a language model."""
from dataclasses import dataclass, replace
import json

import pytest

from looplora.llm import bench


def test_pilot_requires_the_full_requested_grid_and_five_repeats():
    config = bench.PilotConfig()
    assert set(config.cells()) == {(r, loops, batch) for r in (16, 32)
                                   for loops in (0, 1, 2, 4, 8) for batch in (16, 64)}
    assert len(config.cells()) == 20
    with pytest.raises(ValueError, match="full grid"):
        replace(config, ranks=(16,))
    with pytest.raises(ValueError, match=">=5"):
        replace(config, repeats=4)


def test_measure_excludes_parameter_restoration_and_warmup(monkeypatch):
    class Cuda:
        def __init__(self):
            self.resets = 0
            self.syncs = 0

        def synchronize(self):
            self.syncs += 1

        def memory_allocated(self):
            return 100

        def reset_peak_memory_stats(self):
            self.resets += 1

        def max_memory_allocated(self):
            return 160

        def max_memory_reserved(self):
            return 200

    cuda = Cuda()
    clock = [0.0]
    calls = []
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(bench.gc, "collect", lambda: None)

    def prepare():
        calls.append("restore")
        clock[0] += 10

    def operation():
        calls.append("step")
        clock[0] += 2

    config = bench.PilotConfig()
    result = bench._measure(operation, cuda=cuda, cfg=config, tokens=64, prepare=prepare)
    assert calls == ["restore", "step"] * (config.warmup + config.repeats)
    assert cuda.resets == config.repeats
    assert cuda.syncs == config.warmup + 2 * config.repeats
    assert len(result["samples"]) == config.repeats
    assert result["median_seconds"] == 2
    assert result["tokens_per_second"] == 32
    assert result["peak_allocated_bytes"] == 160
    assert result["peak_reserved_bytes"] == 200
    assert result["peak_extra_allocated_bytes"] == 60


def test_incremental_record_replaces_a_valid_json_document(tmp_path):
    path = tmp_path / "throughput.json"
    record = {"status": "running", "cells": []}
    bench._atomic_write(path, record)
    assert json.loads(path.read_text()) == record
    record["cells"].append({"rank": 16, "status": "complete"})
    bench._atomic_write(path, record)
    assert json.loads(path.read_text()) == record
    assert not path.with_suffix(".json.tmp").exists()


def test_training_snapshot_has_no_writer_graph_or_tensor_alias():
    import torch

    @dataclass(frozen=True)
    class Snapshot:
        states: dict
        attention_mask: object
        past_key_values: object
        protocol: str

    parameter = torch.ones(2, 3, 3, requires_grad=True)
    source = Snapshot({14: parameter.square()}, torch.ones(2, 4), None, "memory_only")
    snapshot = bench._detached_snapshot(source)
    assert snapshot.states[14].grad_fn is None
    assert not snapshot.states[14].requires_grad
    snapshot.states[14].zero_()
    snapshot.attention_mask.zero_()
    assert torch.equal(source.states[14], torch.ones(2, 3, 3))
    assert torch.equal(source.attention_mask, torch.ones(2, 4))
