"""WP-L6 rotated-key oracle writers: codes, delta-rule invariants, gradients, accounting, train/export/reload.

Run only through the compute test workflow (docs/reports/WP-L6.md). Writer-level tests run on CPU and CUDA; the
tiny-model tests need the GPU and the cached tokenizer, exactly like tests/test_llm_oracle.py.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm.config import resolve_config
from looplora.llm.inject import load_tokenizer
from looplora.llm.oracle import OracleRandomWriter, OracleRotWriter, OracleSharedWriter, relation_rotations
from looplora.llm.run import _parameter_counts, compare_suite, run_one
from looplora.llm.train import LLMExperiment, load_checkpoint, make_task_spec, train

ROOT = Path(__file__).resolve().parents[1]
WRITERS = {"rot": OracleRotWriter, "shared": OracleSharedWriter}
TABLE_ROWS = 1025  # V = 1024 names plus the insufficient-information row


def make_writer(kind: str, rank: int, V: int = 32, n_rel: int = 4):
    return WRITERS[kind](V=V, n_rel=n_rel, seed=17, r_mem=rank)


def facts_batch(device):
    """Two episodes; unused slots hold the pipeline's -1 sentinels."""
    facts = torch.tensor([[[1, 0, 2], [1, 1, 3], [1, 0, 4], [-1, -1, -1]],
                          [[2, 3, 5], [5, 3, 8], [-1, -1, -1], [-1, -1, -1]]], device=device)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool, device=device)
    return facts, valid


def random_facts(device, V=32, n_rel=4, batch=3, count=6):
    generator = torch.Generator().manual_seed(derive_seed(17, "wp-l6-test", "random-facts"))
    facts = torch.stack((torch.randint(0, V, (batch, count), generator=generator),
                         torch.randint(0, n_rel, (batch, count), generator=generator),
                         torch.randint(0, V, (batch, count), generator=generator)), dim=-1)
    valid = torch.rand(batch, count, generator=generator) > .25
    valid[:, 0] = True
    return facts.masked_fill(~valid[..., None], -1).to(device), valid.to(device)


def reference_state(keys, values, valid):
    """The decay-first delta rule with lambda = beta = 1 written out in float64: M <- M + (v - M k) k^T."""
    keys, values, valid = keys.double().cpu(), values.double().cpu(), valid.cpu()
    batch, count, rank = keys.shape
    state = torch.zeros(batch, rank, rank, dtype=torch.float64)
    for b in range(batch):
        for t in range(count):
            if valid[b, t]:
                state[b] += torch.outer(values[b, t] - state[b] @ keys[b, t], keys[b, t])
    return state


def gram_schmidt(matrix):
    """Modified Gram-Schmidt: the QR factor with a positive R diagonal, which is unique for a full-rank matrix."""
    q = torch.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        v = matrix[:, j].clone()
        for i in range(j):
            v = v - (q[:, i] @ v) * q[:, i]
        q[:, j] = v / v.norm()
    return q


# ---- codes -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [32, 128])
def test_rotations_are_orthogonal_seeded_and_sign_fixed(rank):
    rotations = relation_rotations(17, 4, rank)
    assert rotations.shape == (4, rank, rank) and rotations.dtype == torch.float32
    eye = torch.eye(rank, dtype=torch.float64)
    for rel in range(4):
        pi = rotations[rel].double()
        torch.testing.assert_close(pi.T @ pi, eye, atol=2e-6, rtol=0)
        torch.testing.assert_close(pi @ pi.T, eye, atol=2e-6, rtol=0)
        # The LAPACK sign convention must not leak in: the matrix is the positive-diagonal QR factor of the Gaussian.
        generator = torch.Generator().manual_seed(derive_seed(17, "oracle", "rotation", rel))
        gaussian = torch.randn(rank, rank, generator=generator).double()
        torch.testing.assert_close(pi, gram_schmidt(gaussian), atol=1e-5, rtol=0)
    assert torch.equal(rotations, relation_rotations(17, 4, rank))
    assert not torch.equal(rotations[0], rotations[1])
    assert not torch.equal(rotations, relation_rotations(18, 4, rank))
    assert torch.equal(relation_rotations(17, 2, rank), rotations[:2])  # relation rel never depends on n_rel


@pytest.mark.parametrize("rank", [32, 128])
def test_state_layout_and_initialisation(rank):
    random_ = OracleRandomWriter(V=32, n_rel=4, seed=17, r_mem=rank)
    rot, shared = make_writer("rot", rank), make_writer("shared", rank)
    assert set(random_.state_dict()) == {"global_codes", "pair_codes"}
    assert set(rot.state_dict()) == {"global_codes", "rotations"}
    assert set(shared.state_dict()) == {"C", "global_codes", "rotations"}
    assert torch.equal(rot.global_codes, random_.global_codes)
    assert torch.equal(shared.global_codes, random_.global_codes)
    assert torch.equal(rot.rotations, shared.rotations)
    assert isinstance(shared.C, torch.nn.Parameter) and shared.C.shape == (33, rank)
    assert torch.equal(shared.C, shared.global_codes)
    assert [name for name, _ in shared.named_parameters()] == ["C"] and not list(rot.parameters())
    facts, valid = facts_batch("cpu")
    # At initialization O-shared writes the same memory as O-rot up to normalizing already-unit values.
    torch.testing.assert_close(shared(facts, valid), rot(facts, valid), atol=1e-5, rtol=0)


@pytest.mark.parametrize("cls", [OracleRotWriter, OracleSharedWriter])
def test_rank_must_be_32_or_128(cls):
    with pytest.raises(ValueError, match="32 or 128"):
        cls(V=32, n_rel=4, seed=17, r_mem=16)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("kind", ["rot", "shared"])
def test_keys_are_unit_rotations_of_entity_codes_and_values_are_entity_codes(kind, rank, device):
    writer = make_writer(kind, rank).to(device)
    if kind == "shared":
        with torch.no_grad():
            writer.C.mul_(3.)  # a trained table is not unit norm; keys and values must still be
    facts, valid = facts_batch(device)
    with torch.no_grad():
        keys, values = writer.fact_codes(facts, valid)
    clean = facts.masked_fill(~valid[..., None], 0)
    codes = writer.entity_codes().detach()
    assert keys.shape == values.shape == (2, 4, rank)
    torch.testing.assert_close(keys.norm(dim=-1), torch.ones(2, 4, device=device), atol=1e-6, rtol=0)
    for b in range(2):
        for t in range(4):
            head, rel, tail = clean[b, t].tolist()
            torch.testing.assert_close(keys[b, t], F.normalize(writer.rotations[rel] @ codes[head], dim=0),
                                       atol=1e-6, rtol=0)
            value = F.normalize(codes[tail], dim=0) if kind == "shared" else codes[tail]
            torch.testing.assert_close(values[b, t], value, atol=1e-6, rtol=0)
    if kind == "rot":
        assert torch.equal(values, writer.global_codes[clean[..., 2]])


@pytest.mark.parametrize("kind", ["rot", "shared"])
@pytest.mark.parametrize("rank", [32, 128])
def test_value_to_key_map_is_the_fixed_rotation_of_the_relation(kind, rank):
    V = 400  # more names than r, so a linear fit of unrelated keys cannot succeed (see the random control below)
    writer = make_writer(kind, rank, V=V)
    if kind == "shared":
        generator = torch.Generator().manual_seed(derive_seed(17, "wp-l6-test", "trained-table"))
        with torch.no_grad():
            writer.C.add_(.3 * torch.randn(writer.C.shape, generator=generator))
            writer.C.mul_(.5 + 2 * torch.rand(len(writer.C), 1, generator=generator))
    heads = torch.arange(V)
    valid = torch.ones(1, V, dtype=torch.bool)
    with torch.no_grad():
        for rel in range(4):
            facts = torch.stack((heads, torch.full_like(heads, rel), heads), dim=-1)[None]
            keys, values = writer.fact_codes(facts, valid)
            keys, values = keys[0].double(), values[0].double()
            fit = torch.linalg.lstsq(values, keys).solution  # values @ fit ~ keys
            torch.testing.assert_close(fit.T, writer.rotations[rel].double(), atol=1e-4, rtol=0)
            assert (values @ fit - keys).norm() / keys.norm() < 1e-5


@pytest.mark.parametrize("rank", [32, 128])
def test_random_oracle_value_to_key_map_is_not_linear(rank):
    """Control: the P1-L oracle's keys are unrelated to its values, so no r x r map takes values to keys."""
    V = 400
    writer = OracleRandomWriter(V=V, n_rel=4, seed=17, r_mem=rank)
    keys, values = writer.pair_codes[:, 0].double(), writer.global_codes[:V].double()
    fit = torch.linalg.lstsq(values, keys).solution
    assert (values @ fit - keys).norm() / keys.norm() > .5


# ---- memory writes -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("kind", ["rot", "shared"])
def test_state_is_the_decay_first_delta_rule_on_rotated_keys(kind, rank, device):
    writer = make_writer(kind, rank).to(device)
    facts, valid = random_facts(device)
    with torch.no_grad():
        keys, values = writer.fact_codes(facts, valid)
        state = writer(facts, valid)
    torch.testing.assert_close(state.cpu().double(), reference_state(keys, values, valid), atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("kind", ["rot", "shared"])
def test_writes_overwrite_exactly_at_beta_one(kind, rank, device):
    writer = make_writer(kind, rank).to(device)
    # Episode 0 writes key (3, rel 1) twice with different tails; episode 1 ends with a repeat of its first key.
    facts = torch.tensor([[[3, 1, 5], [3, 1, 9], [7, 2, 11]],
                          [[3, 1, 5], [7, 2, 11], [3, 1, 9]]], device=device)
    valid = torch.ones(2, 3, dtype=torch.bool, device=device)
    with torch.no_grad():
        keys, values = writer.fact_codes(facts, valid)
        state = writer(facts, valid)
        # The key just written reads back its value exactly (k^T k = 1): the earlier value at that key is gone.
        two_writes = writer(facts[:1, :2], valid[:1, :2])
        torch.testing.assert_close(two_writes[0] @ keys[0, 1], values[0, 1], atol=1e-5, rtol=0)
        stale = (two_writes[0] @ keys[0, 1] - values[0, 0]).norm()
        torch.testing.assert_close(stale, (values[0, 1] - values[0, 0]).norm(), atol=1e-5, rtol=0)
        assert stale > .5
        torch.testing.assert_close(state[0] @ keys[0, 2], values[0, 2], atol=1e-5, rtol=0)
        torch.testing.assert_close(state[1] @ keys[1, 2], values[1, 2], atol=1e-5, rtol=0)
        # A single fact is stored exactly.
        single = writer(facts[:, :1], valid[:, :1])
        for b in range(2):
            torch.testing.assert_close(single[b] @ keys[b, 0], values[b, 0], atol=1e-5, rtol=0)


@pytest.mark.parametrize("kind", ["rot", "shared"])
@pytest.mark.parametrize("rank", [32, 128])
def test_invalid_slots_never_address_or_write(kind, rank):
    writer = make_writer(kind, rank)
    facts, valid = facts_batch("cpu")
    relabelled = facts.clone()
    relabelled[~valid] = torch.tensor([30, 3, 31])  # arbitrary in-range ids in the unused slots
    assert torch.equal(writer(facts, valid), writer(relabelled, valid))
    assert writer(facts, valid).abs().sum() > 0
    assert not writer(facts, torch.zeros_like(valid)).any()
    empty = writer(facts[:, :0], valid[:, :0])
    assert empty.shape == (2, rank, rank) and not empty.any()


# ---- gradients ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradient_reaches_C_only_for_the_shared_writer(device):
    facts = torch.tensor([[[1, 0, 2], [3, 1, 4], [5, 2, 6], [-1, -1, -1]]], device=device)
    valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool, device=device)
    generator = torch.Generator().manual_seed(derive_seed(17, "wp-l6-test", "loss-weights"))
    weights = torch.randn(1, 32, 32, generator=generator).to(device)
    shared = make_writer("shared", 32).to(device)
    (shared(facts, valid) * weights).sum().backward()
    grad = shared.C.grad
    assert grad.shape == shared.C.shape and torch.isfinite(grad).all()
    used = {1, 2, 3, 4, 5, 6}
    # Names that appear in a written fact receive gradient; every other row, including the entity 0 that the
    # invalid sentinel slot is mapped to before indexing, receives exactly none.
    for entity in range(len(grad)):
        assert bool(grad[entity].abs().sum() > 0) == (entity in used), entity
    for fixed in (make_writer("rot", 32), OracleRandomWriter(V=32, n_rel=4, seed=17, r_mem=32)):
        fixed = fixed.to(device)
        state = fixed(facts, valid)
        assert not list(fixed.parameters()) and not state.requires_grad
        assert all(not b.requires_grad for b in fixed.buffers())


# ---- configuration -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("writer", ["oracle_rot", "oracle_shared"])
@pytest.mark.parametrize("stage", ["reader", "joint"])
@pytest.mark.parametrize("rank", [32, 128])
def test_new_writers_resolve(writer, stage, rank):
    cfg = resolve_config(dict(writer=writer, stage=stage, model=dict(r=rank)))
    assert cfg["writer"] == writer and cfg["oracle_seed"] == 17 and cfg["eval"]["protocols"] == ["memory_only"]


@pytest.mark.parametrize("writer", ["oracle_rot", "oracle_shared"])
@pytest.mark.parametrize("updates", [
    {"oracle_seed": True}, {"model": {"r": 8}}, {"stage": "writer"}, {"writer_checkpoint": "/writer.pt"},
    {"eval": {"protocols": ["context_retained"]}}, {"protocol": "context_retained"},
    {"stage": "baseline", "variant": "B0"},
    {"stage": "lora_ctx", "variant": "B1", "protocol": "context_retained"},
])
def test_new_writers_reject_incompatible_contracts(writer, updates):
    raw = dict(writer=writer, stage="reader")
    raw.update(updates)
    with pytest.raises(ValueError):
        resolve_config(raw)


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        return {key: leaf for name, item in value.items() for key, leaf in _flatten(item, f"{prefix}{name}.").items()}
    return {prefix[:-1]: value}


@pytest.mark.parametrize("arm", ["oracle_rot", "oracle_shared"])
def test_pilot_configs_differ_from_p1l_m1_l4_only_in_the_registered_fields(arm):
    base = json.loads((ROOT / "configs/p1l/main/M1-L4-s17.json").read_text())
    pilot = json.loads((ROOT / f"configs/wp_l6/{arm}.json").read_text())
    before, after = _flatten(base), _flatten(pilot)
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    assert changed == {"name", "writer", "train.steps", "eval.depths"}
    resolved = resolve_config(pilot)
    assert (resolved["writer"], resolved["variant"], resolved["stage"]) == (arm, "M1", "reader")
    assert resolved["model"]["r"] == 32 and resolved["seed"] == 17 and resolved["oracle_seed"] == 17
    assert resolved["train"]["steps"] == 5666 and resolved["train"]["lr"] == .003
    assert resolved["train"]["depths"] == [4] and resolved["train"]["batch_size"] == 64
    assert resolved["task"]["name"] == "t2_chain_end" and resolved["task"]["hops"] == [1, 2, 3]
    assert resolved["task"]["chain_lengths"] == [8, 4] and resolved["task"]["E"] == 12
    assert resolved["protocol"] == "memory_only" and resolved["eval"]["protocols"] == ["memory_only"]
    assert resolved["eval"]["splits"] == ["val"] and resolved["eval"]["seed"] == 4242
    assert resolved["eval"]["episodes"] == 1000 and resolved["eval"]["depths"] == [1, 2, 3, 4]
    assert resolved["eval"]["hops"] == list(range(1, 9)) and not resolved["eval"]["allow_confirmation"]
    assert not resolved["smoke"]


# ---- tiny-model integration (GPU) ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "oracle-tiny-qwen"))
        value = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield value
    assert not hasattr(value, "looplora_adapters")


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def config(variant="M1", writer="oracle_rot", **updates):
    return resolve_config(dict(name="oracle-codes-unit", writer=writer, stage="reader",
        variant=variant, model=dict(r=32, layer=1),
        task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
        train=dict(steps=2, batch_size=2, depths=[4], log_every=1, warmup_steps=1),
        eval=dict(episodes=2, batch_size=2, hops=[1, 2, 3, 4], counterfactuals=False), **updates))


def data(cfg):
    return generate(make_task_spec(cfg), "train", 0, 2, derive_seed(17, "oracle-unit-data"))


@pytest.mark.parametrize("variant", ["M1", "B8", "B4", "B5", "B3"])
@pytest.mark.parametrize("writer", ["oracle_rot", "oracle_shared"])
def test_train_export_reload_without_support_forward(model, tokenizer, tmp_path, writer, variant):
    shared = writer == "oracle_shared"
    depth = 1 if variant == "B4" else 4
    cfg = config(variant, writer)
    experiment, summary = train(cfg, tmp_path, model=model, tokenizer=tokenizer)
    arrays = data(cfg)
    try:
        assert summary["writer_frozen"] is (not shared)
        assert ("writer_trainable_parameters" in summary) is shared
        assert summary["trainable_parameters"] == sum(p.numel() for p in experiment.trainable_parameters())
        assert experiment.adapter.A.requires_grad and experiment.adapter.A.grad.abs().sum() > 0
        assert all(p.grad is None for p in experiment.model.parameters() if not p.requires_grad)
        assert all(not b.requires_grad and b.grad is None for b in experiment.writer.buffers())
        initial = experiment.writer.global_codes.clone()
        if shared:
            table = experiment.writer.C
            assert summary["writer_trainable_parameters"] == table.numel() == TABLE_ROWS * 32
            trainable = experiment.writer_trainable_parameters()
            assert len(trainable) == 1 and trainable[0] is table
            if variant == "B3":  # B3 never reads the memory: C gets no gradient and stays at its initialization
                assert table.grad is None and torch.equal(table, initial)
            else:
                assert table.grad is not None and torch.isfinite(table.grad).all() and table.grad.abs().sum() > 0
                assert not torch.equal(table, initial)
                # Training episodes only use the seen pool (names 0..767): the other rows never move.
                assert torch.equal(table[768:], initial[768:])
        else:
            assert not list(experiment.writer.parameters()) and not experiment.writer_trainable_parameters()
        calls = []
        hook = model.model.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            with patch("looplora.core.memory.scan_final", side_effect=AssertionError("LM writer used")), \
                 patch("looplora.llm.train.encode_support", side_effect=AssertionError("support LM used")):
                prepared = experiment.prepare(arrays)
        finally:
            hook.remove()
        assert not calls and prepared.support_tokens == 0 and prepared.support_ids.shape == (2, 0)
        assert prepared.write_count.tolist() == [4, 4] and prepared.snapshot.past_key_values is None
        memory = prepared.snapshot.states[1].clone()
        assert memory.requires_grad is shared
        with torch.no_grad():
            expected, _ = experiment.score(prepared, depth)
        assert torch.equal(memory, prepared.snapshot.states[1])
        with pytest.raises(ValueError, match="memory_only"):
            experiment.prepare(arrays, protocol="context_retained")
        record = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
        trained = experiment.writer_state()
        assert set(record["writer"]) == set(trained) == (
            {"C", "global_codes", "rotations"} if shared else {"global_codes", "rotations"})
        assert all(torch.equal(record["writer"][name], value) for name, value in trained.items())
        signature = experiment.writer_signature()
        assert signature["writer"] == writer and signature["r_mem"] == 32 and signature["seed"] == 17
        assert signature["rotation"] == "qr_gaussian_positive_diagonal"
    finally:
        experiment.close()
    (tmp_path / "writer.pt").unlink()  # the full checkpoint must be self-contained
    restored = load_checkpoint(tmp_path / "checkpoint.pt", model=model, tokenizer=tokenizer)
    try:
        assert restored.writer_signature() == signature
        assert all(torch.equal(value, restored.writer_state()[name]) for name, value in trained.items())
        with torch.no_grad():
            batch = restored.prepare(arrays)
            actual, _ = restored.score(batch, depth)
        assert torch.equal(memory, batch.snapshot.states[1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        restored.close()


def test_shared_table_trains_in_its_own_group_without_decay_and_is_clipped_with_the_reader(model, tokenizer, tmp_path):
    cfg = config("M1", "oracle_shared")
    created, clipped = [], []
    real_optimizer, real_clip = torch.optim.AdamW, torch.nn.utils.clip_grad_norm_

    def optimizer(groups, **kwargs):
        # AdamW fills defaults into the dicts it is given, so snapshot them first.
        created.append(([{key: [id(p) for p in value] if key == "params" else value for key, value in group.items()}
                         for group in groups], kwargs))
        return real_optimizer(groups, **kwargs)

    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        clipped.append([id(p) for p in parameters])
        return real_clip(parameters, *args, **kwargs)

    with patch("torch.optim.AdamW", side_effect=optimizer), patch("torch.nn.utils.clip_grad_norm_", side_effect=clip):
        experiment, _ = train(cfg, tmp_path, model=model, tokenizer=tokenizer)
    try:
        table = experiment.writer.C
        assert len(created) == 1
        (reader, writer), kwargs = created[0]
        assert kwargs == dict(lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        assert writer == dict(params=[id(table)], weight_decay=0.)
        assert reader == dict(params=[id(p) for p in experiment.trainable_parameters()])
        assert len(clipped) == cfg["train"]["steps"]
        assert all(step == reader["params"] + writer["params"] for step in clipped)
    finally:
        experiment.close()


@pytest.mark.parametrize("writer", ["oracle_rot", "oracle_shared"])
def test_parameter_counts_keep_reader_counts_and_count_C_as_writer_parameters(model, tokenizer, writer):
    counts = {}
    for name in ("oracle_random", writer):
        experiment = LLMExperiment(config("M1", name), model=model, tokenizer=tokenizer)
        try:
            counts[name] = _parameter_counts(experiment)
        finally:
            experiment.close()
    base, other = counts["oracle_random"], counts[writer]
    extra = TABLE_ROWS * 32 if writer == "oracle_shared" else 0
    assert base["external_writer"] == 0 and other["external_writer"] == extra
    for key in ("model_total", "backbone", "adapter"):
        assert other[key] == base[key]
    assert other["total"] == base["total"] + extra and other["trainable"] == base["trainable"] + extra
    assert other["active_adapter"] == base["active_adapter"] + extra


@pytest.mark.parametrize("writer", ["oracle_rot", "oracle_shared"])
def test_run_record_names_the_writer_and_accounts_C_separately(model, tokenizer, tmp_path, writer):
    cfg = config("M1", writer)
    cfg["name"] = f"{writer}-record"
    record = run_one(cfg, tmp_path / writer, model=model, tokenizer=tokenizer)
    assert record["status"] == "complete" and record["writer"] == writer
    assert record["writer_signature"]["writer"] == writer and record["writer_hash"]
    assert record["parameters"]["external_writer"] == (TABLE_ROWS * 32 if writer == "oracle_shared" else 0)
    assert record["consumed_writer_checkpoint"] is None


def test_compare_suite_checks_fixed_rotated_codes_and_skips_trained_shared_codes(model, tokenizer, tmp_path):
    suites = {}
    for writer in ("oracle_rot", "oracle_shared"):
        suites[writer] = []
        for variant in ("M1", "B4"):
            cfg = config(variant, writer)
            cfg["name"] = f"{writer}-{variant}"
            run_one(cfg, tmp_path / cfg["name"], model=model, tokenizer=tokenizer)
            suites[writer].append(dict(name=cfg["name"], config=cfg))
    assert compare_suite(suites["oracle_rot"], tmp_path)["frozen_writer_hashes_verified"] == [
        ["oracle_rot-M1", "oracle_rot-B4"]]
    # Each oracle_shared reader trains its own table, so its memories differ by construction: nothing to verify.
    assert compare_suite(suites["oracle_shared"], tmp_path)["frozen_writer_hashes_verified"] == []
    path = tmp_path / "oracle_rot-B4/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["invariant_hashes"][0]["M"] = "different"
    path.write_text(json.dumps(metrics))
    with pytest.raises(AssertionError, match="hashes differ"):
        compare_suite(suites["oracle_rot"], tmp_path)


def test_shared_checkpoint_restores_the_table_and_rejects_incompatible_writers(model, tokenizer, tmp_path):
    cfg = config("M1", "oracle_shared")
    experiment, _ = train(cfg, tmp_path / "original", model=model, tokenizer=tokenizer)
    experiment.close()
    record = torch.load(tmp_path / "original/checkpoint.pt", weights_only=True)
    record["writer"]["C"] = record["writer"]["C"] + 1.  # the saved table is authoritative, never regenerated
    torch.save(record, tmp_path / "changed.pt")
    restored = load_checkpoint(tmp_path / "changed.pt", model=model, tokenizer=tokenizer)
    try:
        assert torch.equal(restored.writer.C.detach().cpu(), record["writer"]["C"])
    finally:
        restored.close()
    for change in ("signature", "dtype", "missing"):
        bad = copy.deepcopy(record)
        if change == "signature":
            bad["signature"]["writer"] = "oracle_rot"
        elif change == "dtype":
            bad["writer"]["C"] = bad["writer"]["C"].bfloat16()
        else:
            del bad["writer"]["C"]
        torch.save(bad, tmp_path / f"{change}.pt")
        with pytest.raises(ValueError, match="signature|fp32|missing"):
            load_checkpoint(tmp_path / f"{change}.pt", model=model, tokenizer=tokenizer)
        assert not hasattr(model, "looplora_adapters")
