"""Study CTC (docs/prereg/CTC.md): positions, table lookup, losses, selection and decision rules, frozen readers.

Run only through the compute test workflow (docs/reports/CTC.md). The CPU cases need the cached tokenizer; the
tiny-model cases need the GPU exactly like tests/test_llm_oracle.py.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.data.relations import generate
from looplora.data.render import load_name_pool, render_episodes
from looplora.infra.seeds import derive_seed
from looplora.llm import ctc
from looplora.llm.config import resolve_config
from looplora.llm.inject import load_tokenizer
from looplora.llm.oracle import OracleRandomWriter
from looplora.llm.train import LLMExperiment, make_task_spec

ROOT = Path(__file__).resolve().parents[1]
TINY_SITES = {"o14": ("o_proj", 1), "h14": ("hidden", 1), "h21": ("hidden", 2), "h28": ("hidden", 2)}


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def names(tokenizer):
    return ctc.NameTokens(tokenizer)


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "ctc-tiny-qwen"))
        value = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield value
    assert not hasattr(value, "looplora_adapters")


def tiny_config():
    return resolve_config(dict(name="ctc-unit", writer="oracle_random", stage="reader", variant="M1",
        model=dict(r=32, layer=1, input_site="o_proj"),
        task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
        train=dict(steps=2, batch_size=2, depths=[4], log_every=1, warmup_steps=1),
        eval=dict(episodes=2, batch_size=2, hops=[1, 2, 3, 4], counterfactuals=False)))


def real_spec():
    return make_task_spec(ctc.reader_config())


# -- positions -----------------------------------------------------------------------------------------------
def test_every_name_token_count_matches_its_tokenization_at_a_line_start(tokenizer, names):
    pool = load_name_pool()
    for start in range(0, ctc.V, 128):
        texts = [f"Ann knows Bob.\n{name} trusts Bob.\n" for name in pool.names[start:start + 128]]
        encoded = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        for i, name in enumerate(pool.names[start:start + 128]):
            begin = len("Ann knows Bob.\n")
            hits = [k for k, (a, b) in enumerate(encoded["offset_mapping"][i]) if a < begin + len(name) and b > begin]
            assert len(hits) == int(names.length[start + i]), name
            assert encoded["input_ids"][i][hits[0]] == int(names.first[start + i])
            assert encoded["input_ids"][i][hits[-1]] == int(names.last[start + i])


def test_fact_positions_match_character_offsets(tokenizer, names):
    arrays = generate(real_spec(), "train", 0, 24, derive_seed(17, "ctc", "unit-positions"))
    rendered = render_episodes(arrays, tokenizer=tokenizer)
    tokens = ctc.cpu_tokens(arrays, tokenizer)
    positions = ctc.fact_positions4(arrays, tokens, names)
    pool = load_name_pool()
    assert positions.shape == (24, 12, 4)
    for b, row in enumerate(rendered):
        encoded = tokenizer(row.context_text, add_special_tokens=False, return_offsets_mapping=True)
        for span in row.fact_spans:
            head, rel, tail = map(int, arrays.facts[b, span.fact_index])
            start = span.char_start
            head_hits = [k for k, (a, c) in enumerate(encoded["offset_mapping"]) if a < start + len(pool.names[head]) and c > start]
            tail_end = span.char_end - 2
            tail_hit = [k for k, (a, c) in enumerate(encoded["offset_mapping"]) if a < tail_end and c > tail_end - len(pool.names[tail])]
            expected = (head_hits[-1], tail_hit[0] - 1, tail_hit[0], span.token_end - 1)
            assert tuple(positions[b, span.fact_index].tolist()) == expected
            assert row.context_ids[tail_hit[0]] == pool.token_ids[tail]


# -- lookup into the oracle tables ---------------------------------------------------------------------------
def random_facts(count=5, E=12, seed=3):
    generator = torch.Generator().manual_seed(seed)
    facts = torch.stack((torch.randint(0, ctc.SEEN, (count, E), generator=generator),
                         torch.randint(0, ctc.N_REL, (count, E), generator=generator),
                         torch.randint(0, ctc.SEEN, (count, E), generator=generator)), -1)
    valid = torch.rand(count, E, generator=generator) > .2
    return facts, valid


def test_the_oracle_tables_are_p1ls_r128_oracle():
    oracle = OracleRandomWriter(ctc.V, ctc.N_REL, 17, r_mem=ctc.R_MEM)
    assert ctc.oracle_hash(oracle) == ctc.ORACLE_HASH


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_hard_lookup_of_correct_classes_is_the_oracle_memory_bitwise(device):
    oracle = OracleRandomWriter(ctc.V, ctc.N_REL, 17, r_mem=ctc.R_MEM).to(device)
    facts, valid = random_facts()
    valid[0, 0] = True
    facts, valid = facts.to(device), valid.to(device)
    sentinel = facts.masked_fill(~valid[..., None], -1)
    expected = oracle(sentinel, valid)
    assert torch.equal(ctc.hard_memory(oracle, sentinel, valid), expected)
    assert torch.equal(ctc.hard_memory(oracle, facts, valid), expected)
    wrong = facts.clone()
    wrong[0, 0, 0] = (wrong[0, 0, 0] + 1) % ctc.SEEN
    assert valid[0, 0] and not torch.equal(ctc.hard_memory(oracle, wrong, valid), expected)


def sharp_logits(facts, scale=60.):
    return {"head": F.one_hot(facts[..., 0], ctc.V).float() * scale,
            "relation": F.one_hot(facts[..., 1], ctc.N_REL).float() * scale,
            "tail": F.one_hot(facts[..., 2], ctc.V).float() * scale}


def test_soft_lookup_of_near_one_hot_classes_is_the_oracle_memory_and_has_gradients():
    oracle = OracleRandomWriter(ctc.V, ctc.N_REL, 17, r_mem=ctc.R_MEM)
    facts, valid = random_facts()
    logits = {k: v.clone().requires_grad_() for k, v in sharp_logits(facts).items()}
    key, value = ctc.soft_codes(oracle, logits)
    assert torch.allclose(key.norm(dim=-1), torch.ones(key.shape[:2]), atol=1e-6)
    memory = ctc.soft_memory(oracle, logits, valid)
    assert torch.allclose(memory, oracle(facts, valid), atol=1e-5)
    (memory ** 2).sum().backward()
    for tensor in logits.values():
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_unseen_names_are_masked_in_every_head_output_and_loss():
    mean, std = torch.zeros(8), torch.ones(8)
    head = ctc.Head(8, ctc.V, "full", mean, std, seed=1)
    logits = head(torch.randn(5, 8))
    assert torch.isneginf(logits[:, ctc.SEEN:]).all() and torch.isfinite(logits[:, :ctc.SEEN]).all()
    assert (logits.argmax(-1) < ctc.SEEN).all()
    assert F.softmax(logits, -1)[:, ctc.SEEN:].sum() == 0
    assert torch.isfinite(F.cross_entropy(logits, torch.tensor([0, 1, 2, 3, 767])))
    relation = ctc.Head(8, ctc.N_REL, "full", mean, std, seed=1)
    assert torch.isfinite(relation(torch.randn(3, 8))).all()


def test_registered_parameter_counts():
    assert ctc.head_parameters(1024, ctc.V, "full") == 1_049_600
    assert ctc.head_parameters(1024, ctc.N_REL, "full") == 4_100
    assert ctc.head_parameters(2048, ctc.V, "full") == 2_098_176
    assert ctc.head_parameters(1024, ctc.V, "rank64") == 1024 * 64 + 64 * 1024 + 1024
    assert 2 * 1_049_600 + 4_100 == 2_103_300


def test_shuffled_memory_rolls_inside_each_evaluation_batch():
    states = torch.arange(5.).reshape(5, 1, 1).expand(5, 2, 2)
    rolled = ctc.roll_within_batches(states, size=3)
    assert rolled[:, 0, 0].tolist() == [1., 2., 0., 4., 3.]


# -- selection and decision ----------------------------------------------------------------------------------
def cell(target, site, position, form, accuracy, dim=1024):
    return dict(target=target, site=site, position=position, form=form, accuracy=accuracy, train_accuracy=1.,
                n=1000, parameters=ctc.head_parameters(dim, ctc.CLASSES[target], form))


def test_selection_takes_the_best_cell_and_breaks_ties_by_parameters_site_then_position():
    rows = []
    for target in ctc.TARGETS:
        rows.append(cell(target, "h28", "final", "full", .50))
    rows += [cell("head", "h21", "head_last", "full", .90), cell("head", "h28", "head_last", "rank64", .896),
             cell("head", "h14", "head_last", "rank64", .896), cell("head", "h14", "tail", "rank64", .896),
             cell("head", "o14", "tail", "rank64", .8939, 2048)]
    rows += [cell("relation", "h21", "rel_last", "full", 1.), cell("relation", "h14", "tail", "full", .999),
             cell("relation", "h14", "rel_last", "full", .999), cell("relation", "o14", "rel_last", "full", 1., 2048)]
    rows += [cell("tail", "h14", "tail", "full", .97)]
    chosen = ctc.select_cells(rows)
    # head: .90 best; .896 is within 0.5 pp; among rank64 cells (fewest parameters) the lowest site, earliest position.
    assert (chosen["head"]["site"], chosen["head"]["position"], chosen["head"]["form"]) == ("h14", "head_last", "rank64")
    # relation: every cell is within 0.5 pp of 1.0 and h14 has 4,100 parameters; the earlier position wins.
    assert (chosen["relation"]["site"], chosen["relation"]["position"]) == ("h14", "rel_last")
    assert chosen["tail"]["site"] == "h14"
    # just outside the margin is not tied
    rows = [cell("head", "h21", "head_last", "full", .90), cell("head", "h14", "head_last", "rank64", .8949)]
    rows += [cell(t, "h14", "tail", "full", .9) for t in ("relation", "tail")]
    assert ctc.select_cells(rows)["head"]["form"] == "full"


def metrics(h1, gain, joint=.97, zero=0., shuffled=0.):
    return dict(hop1_em=h1, hop1_se=.015, gain=gain, gain_se=.012, joint=joint, zero_hop1_em=zero, shuffled_hop1_em=shuffled)


def test_decision_candidate_stop_arm_rule_integrity_and_falsifiers():
    decision = ctc.decide(dict(arms=dict(ce=metrics(.42, .15), ceans=metrics(.425, .15))))
    assert decision["selected_arm"] == "ce" and decision["outcome"] == "candidate"  # below 1 pp: ce
    decision = ctc.decide(dict(arms=dict(ce=metrics(.38, .15), ceans=metrics(.41, .15))))
    assert decision["selected_arm"] == "ceans" and decision["outcome"] == "candidate"
    decision = ctc.decide(dict(arms=dict(ce=metrics(.41, .09), ceans=metrics(.30, .02, joint=.97))))
    assert decision["outcome"] == "stop" and decision["arms"]["ceans"]["falsifier_F1"]
    assert decision["arms"]["ceans"]["falsifier_F2"] and not decision["arms"]["ce"]["falsifier_F2"]
    decision = ctc.decide(dict(arms=dict(ce=metrics(.45, .2, zero=.06), ceans=metrics(.40, .10, shuffled=.0))))
    assert decision["arms"]["ce"]["candidate"] is False and not decision["arms"]["ce"]["integrity_ok"]
    assert decision["arms"]["ceans"]["candidate"] is True  # both thresholds are inclusive
    assert decision["selected_arm"] == "ce" and decision["outcome"] == "stop"


# -- the frozen backbone, readers and training ---------------------------------------------------------------
def tiny_batch(tokenizer, names, count=2, seed="ctc-tiny-data"):
    cfg = tiny_config()
    arrays = generate(make_task_spec(cfg), "train", 0, count, derive_seed(17, seed))
    tokens = ctc.cpu_tokens(arrays, tokenizer)
    return cfg, arrays, tokens, ctc.fact_positions4(arrays, tokens, names)


def test_features_are_the_bare_backbones_even_with_adapters_installed(model, tokenizer, names):
    cfg, arrays, tokens, positions = tiny_batch(tokenizer, names)
    ids, mask = tokens.support_ids.cuda(), tokens.support_mask.cuda()
    bare = ctc.extract_features(model, ids, mask, positions, TINY_SITES)
    assert {site: tuple(x.shape) for site, x in bare.items()} == {"o14": (2, 4, 4, 32), "h14": (2, 4, 4, 16),
                                                                  "h21": (2, 4, 4, 16), "h28": (2, 4, 4, 16)}
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        for parameter in experiment.adapter.parameters():  # a trained, nonzero adapter would change layer 1's output
            parameter.data.normal_(0, .5)
        projection = model.model.layers[1].self_attn.o_proj
        with pytest.raises(ValueError, match="bare"):
            ctc.extract_features(model, ids, mask, positions, TINY_SITES)
        with ctc.bare_backbone(model):
            inside = ctc.extract_features(model, ids, mask, positions, TINY_SITES)
        assert model.model.layers[1].self_attn.o_proj is projection  # restored
        for site in bare:
            assert torch.equal(inside[site], bare[site]), site
    finally:
        experiment.close()


def test_perfect_hard_lookup_reproduces_the_oracle_readers_logits_bitwise(model, tokenizer, names):
    cfg, arrays, tokens, _ = tiny_batch(tokenizer, names)
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        generator = torch.Generator(device="cuda").manual_seed(5)
        for parameter in experiment.adapter.parameters():  # an untrained adapter ignores its memory; make it read
            parameter.data.copy_(torch.randn(parameter.shape, generator=generator, device="cuda"))
        ctc.freeze_reader(experiment)
        with torch.no_grad():
            prepared = experiment.prepare(arrays)
            expected, _ = experiment.score(prepared, 2)
            facts = torch.as_tensor(arrays.facts, dtype=torch.long, device="cuda")
            valid = torch.as_tensor(arrays.fact_valid, dtype=torch.bool, device="cuda")
            state = ctc.hard_memory(experiment.writer, facts, valid)
            observed, _ = experiment.score(ctc.with_memory(prepared, state, experiment.layer), 2)
            zero, _ = experiment.score(ctc.with_memory(prepared, torch.zeros_like(state), experiment.layer), 2)
        assert torch.equal(state, prepared.snapshot.states[1])
        assert torch.equal(observed, expected) and not torch.equal(zero, expected)
    finally:
        experiment.close()


def tiny_writer(oracle, device="cuda"):
    stats = {f"{site}/{position}": dict(mean=torch.zeros(dim), std=torch.ones(dim))
             for site, dim in (("o14", 32), ("h14", 16), ("h21", 16), ("h28", 16)) for position in ctc.POSITIONS}
    plan = dict(head=dict(site="h21", position="head_last", form="full"),
                relation=dict(site="h14", position="rel_last", form="rank64"),
                tail=dict(site="o14", position="tail", form="full"))
    return ctc.CTCWriter(plan, stats, oracle).to(device)


@pytest.mark.parametrize("arm", ["ce", "ceans"])
def test_stage_b_trains_only_the_heads_and_leaves_the_reader_frozen(model, tokenizer, names, tmp_path, arm):
    cfg = tiny_config()
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        ctc.freeze_reader(experiment)
        writer = tiny_writer(experiment.writer)
        before = {n: p.detach().clone() for n, p in writer.heads.named_parameters()}
        reader_before = ctc.reader_state_hash(experiment)
        summary = ctc.train_writer(writer, arm, model=model, tokenizer=tokenizer, names=names, spec=make_task_spec(cfg),
                                   out=tmp_path / arm, experiment=experiment if arm == "ceans" else None, steps=3,
                                   batch_size=2, workers=0, log_every=1, site_spec=TINY_SITES)
        assert ctc.reader_state_hash(experiment) == reader_before
        assert all(not p.requires_grad for p in experiment.adapter.parameters())
        assert any(not torch.equal(before[n], p) for n, p in writer.heads.named_parameters())
        assert np.isfinite(summary["final"]["loss"]) and summary["final"]["grad_norm"] > 0
        assert ("answer_loss" in summary["final"]) == (arm == "ceans")
        assert (tmp_path / arm / "writer.pt").exists()
        with pytest.raises(FileExistsError):
            ctc.train_writer(writer, arm, model=model, tokenizer=tokenizer, names=names, spec=make_task_spec(cfg),
                             out=tmp_path / arm, experiment=experiment if arm == "ceans" else None, steps=1,
                             batch_size=2, workers=0, site_spec=TINY_SITES)
    finally:
        experiment.close()


def test_arm_and_reader_must_agree(model, tokenizer, names, tmp_path):
    cfg = tiny_config()
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        writer = tiny_writer(experiment.writer)
        for arm, reader in (("ceans", None), ("ce", experiment)):
            with pytest.raises(ValueError, match="arm"):
                ctc.train_writer(writer, arm, model=model, tokenizer=tokenizer, names=names, spec=make_task_spec(cfg),
                                 out=tmp_path / arm, experiment=reader, steps=1, batch_size=2, workers=0)
    finally:
        experiment.close()


def test_the_ctc_module_is_imported_by_nothing_else_and_changes_no_oracle_result(model, tokenizer):
    """Off path: no existing module imports ctc, and importing it does not change a reader's oracle logits."""
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if path.name not in {"ctc.py", "run_ctc.py"}:
            assert "llm.ctc" not in path.read_text() and "import ctc" not in path.read_text(), path
    program = textwrap.dedent("""
        import os, sys, torch
        if os.environ.get("WITH_CTC") == "1":
            import looplora.llm.ctc
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from looplora.data.relations import generate
        from looplora.infra.seeds import derive_seed
        from looplora.llm.config import resolve_config
        from looplora.llm.inject import load_tokenizer
        from looplora.llm.train import LLMExperiment, make_task_spec
        from looplora.sym.evaluate import tensor_hash
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed(17, "ctc-tiny-qwen"))
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
        cfg = resolve_config(dict(name="ctc-unit", writer="oracle_random", stage="reader", variant="M1",
            model=dict(r=32, layer=1, input_site="o_proj"),
            task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
            train=dict(steps=2, batch_size=2, depths=[4], log_every=1, warmup_steps=1),
            eval=dict(episodes=2, batch_size=2, hops=[1, 2, 3, 4], counterfactuals=False)))
        arrays = generate(make_task_spec(cfg), "train", 0, 2, derive_seed(17, "ctc-tiny-data"))
        experiment = LLMExperiment(cfg, model=model, tokenizer=load_tokenizer())
        with torch.no_grad():
            prepared = experiment.prepare(arrays)
            logits, _ = experiment.score(prepared, 3)
        print("HASH", tensor_hash(logits), tensor_hash(prepared.snapshot.states[1]))
    """)
    outputs = []
    for flag in ("0", "1"):
        env = {**__import__("os").environ, "WITH_CTC": flag, "PYTHONDONTWRITEBYTECODE": "1"}
        run = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env, check=True)
        outputs.append([line for line in run.stdout.splitlines() if line.startswith("HASH")])
    assert outputs[0] and outputs[0] == outputs[1]
