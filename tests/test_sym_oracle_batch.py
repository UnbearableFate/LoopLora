"""Oracle batching must preserve the original episode codebook bit for bit.

Run on a Miyabi-G compute node; no performance threshold is asserted here.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.sym.config import resolve_config
from looplora.sym.model import SymbolicModel, make_task_spec
from looplora.sym.writers import OracleWriter


DEVICES = ("cpu", "cuda")


def _device(name):
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA equivalence requires the Miyabi-G compute GPU")
    return torch.device(name)


def _original_codes(writer, facts, valid, episode_seed, include_insufficient=False,
                    extra_entities=None):
    """Frozen pre-optimization algorithm, including its per-episode transfers."""
    if not writer.orthogonal:
        return writer.global_codes.unsqueeze(0).expand(facts.shape[0], -1, -1)
    result = []
    for i, episode in enumerate(episode_seed):
        entities = facts[i, valid[i]][:, (0, 2)].unique().cpu()
        if include_insufficient:
            entities = torch.cat((entities, torch.tensor([writer.V], dtype=entities.dtype))).unique()
        if extra_entities is not None:
            original = set(entities.tolist())
            added = sorted(set(extra_entities[i]) - original)
            entities = torch.cat((entities, torch.tensor(added, dtype=entities.dtype)))
        if entities.numel() > writer.r_mem:
            raise ValueError("oracle_orth requires episode entities <= r_mem")
        generator = torch.Generator().manual_seed(derive_seed(writer.seed, "oracle_orth", int(episode)))
        basis, _ = torch.linalg.qr(torch.randn(writer.r_mem, writer.r_mem, generator=generator))
        table = writer.global_codes.clone()
        table[entities.to(table.device)] = basis[:, :len(entities)].T.to(table.device)
        result.append(table)
    return torch.stack(result)


def _bitwise_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _facts(device):
    facts = torch.tensor([
        [[9, 0, 4], [4, 0, 7], [7, 0, 3], [3, 0, 1], [1234, 0, 1234]],
        [[10, 0, 11], [11, 0, 13], [13, 0, 12], [12, 0, 14], [14, 0, 2]],
        [[6, 0, 8], [8, 0, 5], [5, 0, 15], [15, 0, 0], [0, 0, 2]],
        [[20, 1, 21], [20, 2, 22], [22, 0, 23], [23, 0, 24], [1234, 0, 1234]],
    ], dtype=torch.long, device=device)
    valid = torch.ones((4, 5), dtype=torch.bool, device=device)
    valid[0, -1] = valid[3, -1] = False
    seeds = np.asarray([derive_seed(17, "oracle-batch", i) for i in (0, 1, 0, 2)], dtype=np.int64)
    extras = [[42, 0, 42, 9], [33, 1, 10], [31, 17, 17], [30, 2, 20]]
    return facts, valid, seeds, extras


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("rank", (16, 32))
@pytest.mark.parametrize("include_insufficient,with_extras", [(False, False), (False, True),
                                                             (True, False), (True, True)])
def test_batched_orthogonal_codes_match_original_bits_and_preserve_unused_global_rows(
        device_name, rank, include_insufficient, with_extras):
    device = _device(device_name)
    writer = OracleWriter(64, rank, 31415, orthogonal=True).to(device)
    facts, valid, seeds, extras = _facts(device)
    extra = extras if with_extras else None
    before = writer.global_codes.clone()
    rng_before = torch.random.get_rng_state().clone()
    expected = _original_codes(writer, facts, valid, seeds, include_insufficient, extra)
    actual = writer.codes(facts, valid, seeds, include_insufficient, extra)
    _bitwise_equal(actual, expected)
    _bitwise_equal(writer.global_codes, before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for i in range(len(seeds)):
        used = set(facts[i, valid[i]][:, (0, 2)].cpu().flatten().tolist())
        if include_insufficient:
            used.add(writer.V)
        if extra is not None:
            used.update(extra[i])
        untouched = sorted(set(range(writer.V + 1)) - used)
        _bitwise_equal(actual[i, untouched], before[untouched])


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("rank", (16, 32))
def test_source_episode_codes_stay_identical_when_extra_and_removal_slots_are_added(device_name, rank):
    device = _device(device_name)
    writer = OracleWriter(64, rank, 31415, orthogonal=True).to(device)
    facts, valid, seeds, extras = _facts(device)
    base = writer.codes(facts, valid, seeds)
    expanded = writer.codes(facts, valid, seeds, include_insufficient=True, extra_entities=extras)
    _bitwise_equal(expanded, _original_codes(writer, facts, valid, seeds, True, extras))
    for i in range(len(seeds)):
        entities = facts[i, valid[i]][:, (0, 2)].unique()
        _bitwise_equal(base[i, entities], expanded[i, entities])
    # Removal/replacement contexts reuse the source-episode table by index.
    source_indices = torch.tensor([2, 0, 2, 1], dtype=torch.long, device=device)
    _bitwise_equal(expanded[source_indices],
                   _original_codes(writer, facts, valid, seeds, True, extras)[source_indices])


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("rank", (16, 32))
def test_global_random_path_keeps_expanded_buffer_and_ignores_orthogonal_options(device_name, rank):
    device = _device(device_name)
    writer = OracleWriter(64, rank, 31415, orthogonal=False).to(device)
    facts, valid, seeds, extras = _facts(device)
    actual = writer.codes(facts, valid, seeds, True, extras)
    _bitwise_equal(actual, _original_codes(writer, facts, valid, seeds, True, extras))
    assert actual.untyped_storage().data_ptr() == writer.global_codes.untyped_storage().data_ptr()
    assert actual.stride(0) == 0


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("rank", (16, 32))
def test_new_codes_preserve_delta_memory_and_nonzero_reader_query_logits(device_name, rank):
    device = _device(device_name)
    config = resolve_config({
        "device": str(device), "writer": "oracle_orth", "variant": "M1",
        "model": {"r": rank, "r_mem": rank, "d_e": 8},
        "task": {"name": "t2_chain_end", "V": 64, "seen": [0, 48], "unseen": [48, 64],
                 "E": 12, "Q": 3, "chain_lengths": [8, 4], "hops": [1, 2, 3]},
    })
    model = SymbolicModel(config)
    arrays = generate(make_task_spec(config), "train", 0, 4, derive_seed(17, "oracle-batch-reader"))
    facts = torch.as_tensor(arrays.facts, dtype=torch.long, device=device)
    valid = torch.as_tensor(arrays.fact_valid, dtype=torch.bool, device=device)
    reference = _original_codes(model.writer, facts, valid, arrays.episode_seed)
    expected = model.prepare(arrays, oracle_codes=reference)
    actual = model.prepare(arrays)
    for field in ("x", "u", "memory", "codes", "write_count"):
        _bitwise_equal(getattr(actual, field), getattr(expected, field))
    generator = torch.Generator().manual_seed(derive_seed(17, "oracle-batch-nonzero-readout"))
    with torch.no_grad():
        model.adapter.B.copy_(torch.randn(model.adapter.B.shape, generator=generator).to(device))
        for depth in (0, 1, 4, 8):
            left, right = model.read(actual, depth), model.read(expected, depth)
            _bitwise_equal(left["z"], right["z"])
            _bitwise_equal(left["logits"], right["logits"])
            assert torch.count_nonzero(left["logits"]) > 0


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("rank", (16, 32))
def test_batched_orthogonal_capacity_error_matches_reference_with_extra_slots(device_name, rank):
    device = _device(device_name)
    writer = OracleWriter(64, rank, 31415, orthogonal=True).to(device)
    facts, valid, seeds, _ = _facts(device)
    extras = [list(range(rank + 1)) for _ in seeds]
    for codebook in (_original_codes, lambda writer, *args, **kwargs: writer.codes(*args, **kwargs)):
        with pytest.raises(ValueError, match="entities <= r_mem"):
            codebook(writer, facts, valid, seeds, True, extras)
