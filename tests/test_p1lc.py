"""P1L-C: read control and patch mechanics, bridge-world construction and the sample pipeline."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_p1l_causal as causal
from readout_p1l import reader_states
from run_p1l_main import config as p1l_config
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.data.relations import TaskSpec, generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed
from looplora.llm import p1lc
from looplora.llm.batching import repeat_snapshot
from looplora.llm.config import recursive_merge, resolve_config
from looplora.llm.evaluate import _slice
from looplora.llm.inject import configure_numerics, load_tokenizer
from looplora.llm.train import LLMExperiment, make_task_spec
from looplora.sym.evaluate import stratify_queries
from looplora.sym.stats import load_samples

VARIANTS = ["iterq", "unshared"]
LEFTOVERS = {"process_sequence", "_masked_branch", "_read"}


# ---------------------------------------------------------------------------- adapter level (CPU)

def adapter_case(variant, rows=4, length=7, r=8):
    """A non-trivial adapter (its output matrix is random), ragged right-padded inputs and a memory."""
    generator = torch.Generator().manual_seed(derive_seed(17, "p1lc-adapter", variant))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "p1lc-adapter-init", variant))
        adapter = make_adapter(AdapterConfig(d_in=6, d_out=5, r=r, variant=variant, unshared_steps=4))
    with torch.no_grad():
        adapter.B.normal_(0, .5, generator=generator)
    x = torch.randn(rows, length, 6, generator=generator)
    memory = torch.randn(rows, r, r, generator=generator)
    valid = torch.arange(length)[None, :] < torch.tensor([7, 5, 2, 6])[:rows, None]
    return adapter, x, memory, valid


def run(adapter, x, memory, valid, steps=4):
    """Read-only sequence pass (no writes) as the deployed question forward makes it."""
    return adapter.process_sequence(x, memory, steps, valid, torch.zeros_like(valid))[0]


def at_last(valid, last):
    return torch.arange(valid.shape[1], device=valid.device)[None, :] == last[:, None]


@pytest.mark.parametrize("variant", VARIANTS)
def test_read_control_is_neutral_when_idle_capturing_or_patching_with_the_deployed_state(variant):
    adapter, x, memory, valid = adapter_case(variant)
    last = p1lc.last_positions(valid)
    deployed = run(adapter, x, memory, valid)
    assert deployed[valid].abs().sum() > 0 and not deployed[~valid].any()
    with p1lc.read_control(adapter, last):
        assert torch.equal(run(adapter, x, memory, valid), deployed)
    captured = {}
    with p1lc.read_control(adapter, last, capture=captured):
        assert torch.equal(run(adapter, x, memory, valid), deployed)
    own = p1lc.stack_states(captured, len(last), x.shape[1], 8, x.device)
    for where in ("final", "all"):
        with p1lc.read_control(adapter, last, donor=own, where=where):
            assert torch.equal(run(adapter, x, memory, valid), deployed)
    assert torch.equal(run(adapter, x, memory, valid), deployed)
    assert not LEFTOVERS & set(vars(adapter))


@pytest.mark.parametrize("variant", VARIANTS)
def test_final_position_memory_zeroes_every_other_position_and_keeps_the_final_read(variant):
    adapter, x, memory, valid = adapter_case(variant)
    last = p1lc.last_positions(valid)
    deployed = run(adapter, x, memory, valid)
    nothing = run(adapter, x, torch.zeros_like(memory), valid)
    with p1lc.read_control(adapter, last, last_only=True):
        only_final = run(adapter, x, memory, valid)
    final = at_last(valid, last)
    other = valid & ~final
    assert not torch.equal(deployed, nothing)
    torch.testing.assert_close(only_final[final], deployed[final], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(only_final[other], nothing[other], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(only_final[other], deployed[other])
    assert not only_final[~valid].any()


def manual_branch(adapter, u, memory, donor, steps=4):
    """Branch output of one token whose first read state is replaced by donor [r]; reads 2..L use memory [r, r]."""
    z = donor
    for step in range(1, steps):
        R, U, Q = ((adapter.R[step], adapter.U[step], adapter.Q[step]) if adapter.cfg.variant == "unshared"
                   else (adapter.R, adapter.U, adapter.Q))
        z = u + adapter.cfg.rho * adapter._phi(F.linear(z, R) + adapter._query(z, memory, Q, U))
    return adapter.branch(z)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("where", ["final", "all"])
def test_state_patch_replaces_z1_and_continues_on_the_recipient_memory(variant, where):
    adapter, x, memory, valid = adapter_case(variant)
    last = p1lc.last_positions(valid)
    donor = torch.randn(x.shape[0], x.shape[1], 8,
                        generator=torch.Generator().manual_seed(derive_seed(17, "p1lc-donor", variant, where)))
    deployed = run(adapter, x, memory, valid)
    with p1lc.read_control(adapter, last, donor=donor, where=where):
        patched = run(adapter, x, memory, valid)
    u = adapter.u(x)
    hit = valid & (at_last(valid, last) if where == "final" else torch.ones_like(valid))
    for row, position in hit.nonzero().tolist():
        expected = manual_branch(adapter, u[row, position], memory[row], donor[row, position])
        torch.testing.assert_close(patched[row, position], expected, rtol=1e-5, atol=1e-6)
    untouched = valid & ~hit
    assert untouched.any() == (where == "final")
    assert torch.equal(patched[untouched], deployed[untouched])
    assert not torch.allclose(patched[hit], deployed[hit])
    assert not patched[~valid].any()


@pytest.mark.parametrize("variant", VARIANTS)
def test_read_states_replicates_the_deployed_loop_and_the_p1l_readout_replica(variant):
    adapter, x, memory, valid = adapter_case(variant)
    u = adapter.u(x)[:, 0]
    states = p1lc.read_states(adapter, u, memory, 4)
    assert torch.equal(states[-1], adapter.read(u, memory, 4))
    assert all(torch.equal(left, right) for left, right in zip(states, reader_states(adapter, u, memory)))
    patched = p1lc.read_states(adapter, u, memory, 4, replace_first=lambda z: torch.zeros_like(z))
    assert torch.equal(patched[0], torch.zeros_like(u)) and not torch.equal(patched[-1], states[-1])
    with pytest.raises(ValueError):
        p1lc.read_states(adapter, u, memory, 0)


def test_read_control_rejects_inconsistent_options_and_shapes_and_always_restores_the_adapter():
    adapter, x, memory, valid = adapter_case("iterq")
    last = p1lc.last_positions(valid)
    donor = torch.zeros(x.shape[0], x.shape[1], 8)
    for options in (dict(where="middle"), dict(last_only=True, capture={}), dict(capture={}, donor=donor)):
        with pytest.raises(ValueError):
            with p1lc.read_control(adapter, last, **options):
                pass
    with pytest.raises(ValueError, match="different number"):
        with p1lc.read_control(adapter, last[:3], last_only=True):
            run(adapter, x, memory, valid)
    with pytest.raises(ValueError, match="donor states"):
        with p1lc.read_control(adapter, last, donor=donor[:, :3]):
            run(adapter, x, memory, valid)
    with pytest.raises(RuntimeError, match="boom"):
        with p1lc.read_control(adapter, last, last_only=True):
            raise RuntimeError("boom")
    assert not LEFTOVERS & set(vars(adapter))


def test_last_positions_stack_states_and_random_directions():
    mask = torch.tensor([[[1, 1, 0, 0], [1, 1, 1, 1]], [[1, 0, 0, 0], [1, 1, 1, 0]]], dtype=torch.bool)
    assert p1lc.last_positions(mask).tolist() == [1, 3, 0, 2]
    stacked = p1lc.stack_states({0: torch.ones(2, 3), 2: 2 * torch.ones(2, 3)}, 2, 4, 3, "cpu")
    assert stacked.shape == (2, 4, 3) and stacked[:, 0].eq(1).all() and stacked[:, 2].eq(2).all()
    assert not stacked[:, 1].any() and not stacked[:, 3].any()
    units = p1lc.random_units([5, 7], 10, 8, 4242)
    assert units.shape == (2, 10, 8) and torch.allclose(units.norm(dim=-1), torch.ones(2, 10), atol=1e-6)
    assert torch.equal(p1lc.random_units([7], 10, 8, 4242)[0], units[1])
    assert torch.equal(p1lc.random_units([5], 6, 8, 4242)[0], units[0, :6])
    assert not torch.equal(units[0], units[1])
    assert not torch.equal(p1lc.random_units([5], 10, 8, 4243)[0], units[0])
    with pytest.raises(ValueError):
        p1lc.random_units([5], p1lc.RANDOM_POSITIONS + 1, 8, 4242)


# --------------------------------------------------------------------- episodes and worlds (CPU)

def walk_to_end(facts, start):
    edges = {int(h): int(t) for h, r, t in facts if r == 0}
    node = start
    while node in edges:
        node = edges[node]
    return node


def chain_length_of(facts, start):
    """Length of the chain a node lies on, counted independently of p1lc.chain_paths."""
    forward = {int(h): int(t) for h, r, t in facts if r == 0}
    backward = {tail: head for head, tail in forward.items()}
    steps, node = 0, start
    while node in forward:
        node, steps = forward[node], steps + 1
    node = start
    while node in backward:
        node, steps = backward[node], steps + 1
    return steps


@pytest.fixture(scope="module")
def world_data():
    cfg, spec = causal.evaluation_config(resolve_config(p1l_config("M1-L4", 17, 128)))
    arrays = stratify_queries(generate(spec, "val", 0, 60, 4242), spec, cfg["eval"]["hops"], 4242)
    qualifying = [row for row in range(60) if chain_length_of(arrays.facts[row], int(arrays.q_start[row, 1])) == 8]
    assert 12 <= len(qualifying) < 60
    everything = p1lc.select_bridge_worlds(arrays, spec, first_index=0, base_seed=4242, limit=len(qualifying))
    return spec, arrays, qualifying, everything


def test_bridge_worlds_keep_both_continuations_and_differ_only_in_the_first_edge(world_data):
    spec, arrays, qualifying, worlds = world_data
    assert worlds.rows.tolist() == qualifying and worlds.index.tolist() == qualifying
    for i, row in enumerate(worlds.rows):
        a, b, c = arrays.facts[row], worlds.facts_b[i], worlds.facts_c[i]
        s, bridge_a, target_a, bridge_b, target_b, other = (int(getattr(worlds, name)[i]) for name in (
            "start", "bridge_a", "target_a", "bridge_b", "target_b", "unrelated"))
        slot = int(worlds.edge[i])
        triples = [{tuple(map(int, fact)) for fact in facts} for facts in (a, b, c)]
        for world in triples:
            assert (bridge_a, 0, target_a) in world and (bridge_b, 0, target_b) in world
        assert (s, 0, bridge_a) in triples[0] and (s, 0, bridge_b) in triples[1] and (s, 0, other) in triples[2]
        assert target_a != target_b and bridge_a != bridge_b and s not in (bridge_a, bridge_b, other)
        assert np.flatnonzero((a != b).any(-1)).tolist() == [slot] == np.flatnonzero((a != c).any(-1)).tolist()
        assert a[slot].tolist() == [s, 0, bridge_a] and b[slot].tolist() == [s, 0, bridge_b]
        assert c[slot].tolist() == [s, 0, other]
        # the answer of a hop-2 question is t_A in world A, t_B in world B, and c (a dead end) in world C
        assert int(arrays.q_hop_true[row, 1]) == 2 and int(arrays.q_start[row, 1]) == s
        assert int(arrays.answer[row, 1]) == target_a
        assert walk_to_end(a, s) == target_a and walk_to_end(b, s) == target_b and walk_to_end(c, s) == other


def test_the_unrelated_entity_has_no_outgoing_fact_and_appears_nowhere_else(world_data):
    spec, arrays, _, worlds = world_data
    for i, row in enumerate(worlds.rows):
        a, c, other = arrays.facts[row], worlds.facts_c[i], int(worlds.unrelated[i])
        assert spec.seen[0] <= other < spec.seen[1]
        assert other not in set(a[:, 0].tolist()) | set(a[:, 2].tolist())
        assert other not in set(c[:, 0].tolist()) and c[:, 2].tolist().count(other) == 1


def test_world_selection_is_deterministic_a_prefix_property_and_seeded_by_the_episode_index(world_data):
    spec, arrays, qualifying, worlds = world_data
    again = p1lc.select_bridge_worlds(arrays, spec, first_index=0, base_seed=4242, limit=len(qualifying))
    assert again.fingerprint() == worlds.fingerprint()
    few = p1lc.select_bridge_worlds(arrays, spec, first_index=0, base_seed=4242, limit=5)
    assert few.rows.tolist() == qualifying[:5] and few.unrelated.tolist() == worlds.unrelated[:5].tolist()
    shifted = p1lc.select_bridge_worlds(_slice(arrays, slice(10, 60)), spec, first_index=10, base_seed=4242, limit=5)
    by_index = dict(zip(worlds.index.tolist(), worlds.unrelated.tolist()))
    assert shifted.index.tolist() == [row for row in qualifying if row >= 10][:5]
    assert [by_index[index] for index in shifted.index.tolist()] == shifted.unrelated.tolist()
    other_seed = p1lc.select_bridge_worlds(arrays, spec, first_index=0, base_seed=4243, limit=len(qualifying))
    assert other_seed.rows.tolist() == qualifying and other_seed.unrelated.tolist() != worlds.unrelated.tolist()
    assert other_seed.bridge_b.tolist() == worlds.bridge_b.tolist()


def test_world_selection_rejects_unsuited_specs_and_too_few_episodes(world_data):
    spec, arrays, qualifying, _ = world_data
    with pytest.raises(ValueError, match="only"):
        p1lc.select_bridge_worlds(arrays, spec, first_index=0, base_seed=4242, limit=len(qualifying) + 1)
    equal = TaskSpec(name="t2_chain_end", E=8, Q=4, chain_lengths=(4, 4), hops=(1, 2, 3, 4))
    sample = stratify_queries(generate(equal, "val", 0, 4, 4242), equal, [1, 2, 3, 4], 4242)
    with pytest.raises(ValueError, match="different length"):
        p1lc.select_bridge_worlds(sample, equal, first_index=0, base_seed=4242, limit=1)


def test_keep_queries_selects_hop_columns_without_changing_the_source(world_data):
    _, arrays, _, _ = world_data
    kept = p1lc.keep_queries(arrays, (2,))
    assert kept.q_start.shape == (60, 1) and set(kept.q_hop_true.ravel().tolist()) == {2}
    assert np.array_equal(kept.answer[:, 0], arrays.answer[:, 1]) and np.array_equal(kept.facts, arrays.facts)
    assert kept.meta["spec"]["Q"] == 1 and arrays.meta["spec"]["Q"] == 8 and arrays.q_start.shape == (60, 8)
    with pytest.raises(ValueError):
        p1lc.keep_queries(arrays, (9,))


def test_chain_paths_lists_relation_zero_chains_root_first():
    facts = np.array([[2, 0, 3], [1, 0, 2], [7, 0, 8], [9, 1, 10], [3, 0, 4]])
    valid = np.array([True, True, True, True, False])
    assert p1lc.chain_paths(facts, valid) == [[1, 2, 3], [7, 8]]
    with pytest.raises(ValueError, match="cycle"):      # a chain whose end loops back into itself
        p1lc.chain_paths(np.array([[0, 0, 1], [1, 0, 2], [2, 0, 1]]), np.ones(3, dtype=bool))


def test_registered_range_and_task_spec_follow_the_p1_l_confirmation():
    assert p1lc.REGISTERED_RANGE == dict(split="test", start=102000, count=2000, seed=4242)
    assert p1lc.REGISTERED_RANGE["start"] >= 100000 + 2000 and p1lc.SMOKE_RANGE["split"] == "val"
    base = resolve_config(p1l_config("B8", 29, 128))
    cfg, spec = causal.evaluation_config(base)
    template = json.loads((causal.ROOT / "configs/p1l/main/eval-confirmation.json").read_text())
    confirmation = make_task_spec(resolve_config(recursive_merge(base, template)), evaluation=True)
    assert asdict(spec) == asdict(confirmation) and spec.chain_lengths == (8, 4) and spec.E == 12
    assert cfg["eval"]["splits"] == ["val"] and cfg["eval"]["allow_confirmation"] is False
    assert cfg["eval"]["hops"] == list(range(1, 9)) and cfg["eval"]["seed"] == 4242


# ------------------------------------------------------------------ tiny Qwen with an adapter (GPU)

def tiny_config(variant):
    return resolve_config(dict(writer="oracle_random", stage="reader", variant=variant,
        model=dict(r=32, layer=0), task=dict(E=6, Q=2, chain_lengths=[4, 2], n_rel=4, head_sharing=.3),
        train=dict(batch_size=2, depths=[4]),
        eval=dict(episodes=6, batch_size=3, hops=[1, 2, 3, 4], counterfactuals=False)))


@pytest.fixture(scope="module")
def model():
    configure_numerics()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "p1lc-tiny-qwen"))
        value = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield value
    assert not hasattr(value, "looplora_adapters")


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def data():
    cfg = tiny_config("M1")
    spec = make_task_spec(cfg, evaluation=True)
    return stratify_queries(generate(spec, "val", 0, 6, 4242), spec, cfg["eval"]["hops"], 4242)


@pytest.fixture(params=["M1", "B8"])
def experiment(request, model, tokenizer):
    value = LLMExperiment(tiny_config(request.param), model=model, tokenizer=tokenizer)
    with torch.no_grad():
        value.adapter.B.normal_(0, .3, generator=torch.Generator(device="cuda").manual_seed(
            derive_seed(17, "p1lc-B", request.param)))
    yield value
    value.close()


def two_worlds(experiment, data):
    """Prepared batches of the same questions on two different memories (episode facts rolled by one)."""
    first = experiment.prepare(data)
    second = experiment.prepare(replace(data, facts=np.roll(data.facts, 1, axis=0)))
    assert torch.equal(first.question_ids, second.question_ids)
    assert not torch.equal(first.snapshot.states[experiment.layer], second.snapshot.states[experiment.layer])
    return first, second


def wrapper_outputs(experiment, prepared, depth, control=None):
    """Layer-0 projection outputs [rows, T, d_out] and the logits of one scored forward."""
    module, seen = experiment.bank.modules[experiment.layer], []
    handle = module.register_forward_hook(lambda _, __, output: seen.append(output.detach().clone()))
    try:
        logits = p1lc.score_logits(experiment, prepared, depth, control)
    finally:
        handle.remove()
    assert len(seen) == 1
    return seen[0], logits


def question_geometry(prepared):
    valid = prepared.question_mask.reshape(-1, prepared.question_mask.shape[-1])
    last = p1lc.last_positions(valid)
    final = at_last(valid, last)
    return valid, last, final, valid & ~final


def test_idle_capture_and_identity_patches_reproduce_the_deployed_logits(experiment, data):
    prepared = experiment.prepare(data)
    plain = p1lc.score_logits(experiment, prepared, 4)
    assert torch.equal(p1lc.score_logits(experiment, prepared, 4, dict()), plain)
    captured = {}
    assert torch.equal(p1lc.score_logits(experiment, prepared, 4, dict(capture=captured)), plain)
    z1 = p1lc.stack_states(captured, data.answer.size, prepared.question_mask.shape[-1], 32, experiment.device)
    for where in ("final", "all"):
        assert torch.equal(p1lc.score_logits(experiment, prepared, 4, dict(donor=z1, where=where)), plain)
    assert not LEFTOVERS & set(vars(experiment.adapter))


def test_final_position_memory_matches_deployed_at_the_final_token_and_zero_memory_elsewhere(experiment, data):
    prepared = experiment.prepare(data)
    zero = replace(prepared, snapshot=replace(prepared.snapshot, states={
        experiment.layer: torch.zeros_like(prepared.snapshot.states[experiment.layer])}))
    valid, last, final, other = question_geometry(prepared)
    deployed, plain_logits = wrapper_outputs(experiment, prepared, 4)
    nothing, _ = wrapper_outputs(experiment, zero, 4)
    only_final, logits = wrapper_outputs(experiment, prepared, 4, dict(last_only=True))
    assert not torch.equal(deployed[valid], nothing[valid])
    assert torch.equal(only_final[final], deployed[final])
    assert torch.equal(only_final[other], nothing[other])
    assert not torch.equal(only_final[other], deployed[other])
    assert not torch.equal(logits, plain_logits)


def test_captured_first_reads_match_the_p1_l_replica_at_every_real_token(experiment, data):
    prepared = experiment.prepare(data)
    captured = {}
    p1lc.score_logits(experiment, prepared, 4, dict(capture=captured))
    _, inputs = experiment.score(prepared, 4, capture_inputs=True)
    B, Q, T, _ = inputs.shape
    u = experiment.adapter.u(inputs.float()).reshape(B * Q, T, -1)
    memory = repeat_snapshot(prepared.snapshot, Q).states[experiment.layer]
    valid = prepared.question_mask.reshape(B * Q, T)
    for position in range(T):
        rows = valid[:, position].nonzero(as_tuple=True)[0]
        assert len(rows)
        expected = reader_states(experiment.adapter, u[rows, position], memory[rows])[0]
        assert torch.equal(captured[position][rows], expected)


def test_final_and_all_position_patches_act_where_they_say_at_four_reads(experiment, data):
    first, second = two_worlds(experiment, data)
    valid, last, final, other = question_geometry(first)
    captured = {}
    p1lc.score_logits(experiment, second, 4, dict(capture=captured))
    donor = p1lc.stack_states(captured, data.answer.size, valid.shape[1], 32, experiment.device)
    deployed, plain_logits = wrapper_outputs(experiment, first, 4)
    patched, logits = wrapper_outputs(experiment, first, 4, dict(donor=donor, where="final"))
    everywhere, _ = wrapper_outputs(experiment, first, 4, dict(donor=donor, where="all"))
    assert torch.equal(patched[other], deployed[other]) and not torch.equal(patched[final], deployed[final])
    assert not torch.equal(everywhere[other], deployed[other])
    assert torch.equal(everywhere[final], patched[final])
    assert not torch.equal(logits, plain_logits)


def test_one_read_patches_reproduce_the_donor_world_exactly(experiment, data):
    if experiment.config["variant"] == "B8":
        pytest.skip("B8 reads exactly four times")
    first, second = two_worlds(experiment, data)
    valid, last, final, other = question_geometry(first)
    captured = {}
    p1lc.score_logits(experiment, second, 1, dict(capture=captured))
    donor = p1lc.stack_states(captured, data.answer.size, valid.shape[1], 32, experiment.device)
    out_first, logits_first = wrapper_outputs(experiment, first, 1)
    out_second, logits_second = wrapper_outputs(experiment, second, 1)
    out_final, _ = wrapper_outputs(experiment, first, 1, dict(donor=donor, where="final"))
    # With a single read the patched state is the output state: patching every position is the donor run,
    # patching the final position is the donor's read at the final token and the recipient's elsewhere.
    assert torch.equal(p1lc.score_logits(experiment, first, 1, dict(donor=donor, where="all")), logits_second)
    assert torch.equal(out_final[final], out_second[final]) and torch.equal(out_final[other], out_first[other])
    assert not torch.equal(logits_first, logits_second)


def test_bridge_patch_pipeline_runs_end_to_end_and_writes_the_registered_artifacts(experiment, tmp_path):
    label = "B8" if experiment.config["variant"] == "B8" else "M1-L4"
    window = dict(split="val", start=0, count=24, seed=4242)
    record = causal.evaluate_run(experiment, label, tmp_path, window=window, patch_episodes=2)
    cfg, spec = causal.evaluation_config(experiment.config)
    arrays = stratify_queries(generate(spec, "val", 0, 24, 4242), spec, cfg["eval"]["hops"], 4242)
    samples = load_samples(tmp_path / "read-position.npz")
    assert set(samples["condition"]) == {"memory_only/deployed", "memory_only/final_only"}
    assert len(samples["correct"]) == 24 * 8 * 2 and set(samples["hop"].tolist()) == set(range(1, 9))
    assert set(samples["L"].tolist()) == {4} and set(samples["episode_id"].tolist()) == set(arrays.episode_seed.tolist())
    assert np.array_equal(samples["correct"], samples["prediction"] == samples["answer"])
    with np.load(tmp_path / "bridge-patch.npz", allow_pickle=False) as archive:
        patch = {key: archive[key] for key in archive.files}
    pool = np.asarray(load_name_pool().token_ids)
    worlds = p1lc.select_bridge_worlds(arrays, spec, first_index=0, base_seed=4242, limit=2)
    assert patch["prediction"].shape == (9, 2) and list(patch["conditions"]) == list(p1lc.PATCH_CONDITIONS)
    assert np.array_equal(patch["episode_index"], worlds.index) and np.array_equal(patch["token_t_B"], pool[worlds.target_b])
    assert np.array_equal(patch["token_t_A"], pool[worlds.target_a]) and np.array_equal(patch["token_c"], pool[worlds.unrelated])
    assert (patch["norm_b"] > 0).all() and (patch["norm_c"] > 0).all() and (patch["norm_a"] > 0).all()
    assert record["worlds_hash"] == worlds.fingerprint() and record["patch_episodes"] == 2
    assert record["spec_hash"] and record["episodes_hash"] and record["window"] == window
    assert len(record["read_position_memory_hashes"]) == 8 and record["bridge_memory_hashes"]
    assert not LEFTOVERS & set(vars(experiment.adapter))
