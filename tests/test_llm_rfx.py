"""WP-L7 RFX on the learned LLM writer: the final-token read hook, cached observations, checkpoint loading and the run.

Tiny-model tests (compute only); the three registered WP-L4 checkpoints are evaluated by rfx_run on compute nodes. The
algebra, the fits and the registered verdict rule are tested on synthetic tensors in test_rfx.py.
"""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.data.relations import generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed
from looplora.llm import read_fix, read_fix_run, rfx, rfx_run, writer_fix_run
from looplora.llm.inject import load_tokenizer
from looplora.llm.lsq import LSQExperiment, configuration as lsq_configuration
from looplora.llm.read_fix import ReadFixExperiment, question_terms
from looplora.llm.train import make_task_spec
from looplora.llm.writer_fix import WriterFixExperiment

RANK = 4  # tiny-model adapter rank = r_mem
INTERCEPTED = ('process_sequence', '_masked_branch', '_query')


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'rfx-test-model'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield model
    assert not hasattr(model, 'looplora_adapters')


def tiny(arm='R'):
    cfg = read_fix.configuration(arm, 10)
    cfg['model'].update(layer=1, r=RANK)
    return cfg


def data(cfg, episodes=2):
    return generate(make_task_spec(cfg), 'train', 0, episodes, derive_seed(17, 'rfx-test', 'fixture'))


@pytest.fixture
def experiment(model, tokenizer):
    """A WP-L4 R experiment on the tiny model; B is nonzero so that the read reaches the logits."""
    experiment = ReadFixExperiment(tiny(), 'R', model=model, tokenizer=tokenizer)
    with torch.no_grad():
        experiment.adapter.B.fill_(.1)
    try:
        yield experiment
    finally:
        experiment.close()


def intercepted(adapter):
    return [name for name in INTERCEPTED if name in adapter.__dict__]


def bare(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'rfx-test', 'bare-adapter'))
        adapter = make_adapter(AdapterConfig(d_in=6, d_out=5, r=4, **kwargs))
        with torch.no_grad():
            adapter.B.copy_(torch.randn(5, 4))
    return adapter


def sequences(lengths=(7, 1, 3, 5, 2, 7)):
    """Right-padded rows of different lengths with random stores: (x, M0, valid, no_write, last position of each row)."""
    g = torch.Generator().manual_seed(derive_seed(17, 'rfx-test', 'sequences'))
    lengths = torch.tensor(lengths)
    T = int(lengths.max())
    valid = torch.arange(T)[None, :] < lengths[:, None]
    return (torch.randn(len(lengths), T, 6, generator=g), torch.randn(len(lengths), 4, 4, generator=g), valid,
            torch.zeros_like(valid), lengths - 1)


def branch_with_read(adapter, x, last, read):
    """The one-step reader's branch output at every row's last token when its read is replaced by read [rows, r_mem]."""
    u = adapter.u(x)[torch.arange(len(last)), last]
    z = u + adapter.cfg.rho * torch.relu(F.linear(u, adapter.R) + F.linear(read, adapter.U))
    return adapter.branch(z)


def test_hook_records_and_replaces_exactly_the_last_real_token_of_every_row():
    adapter = bare()
    with torch.no_grad():
        adapter.U.fill_(.5)  # every pre-activation grows with the read, so a +100 replacement cannot be clipped away
    x, M0, valid, no_write, last = sequences()
    rows = torch.arange(len(last))
    with torch.no_grad():
        plain, _, _ = adapter.process_sequence(x, M0, 1, valid, no_write)
        with rfx.FinalRead(adapter, last) as probe:
            recorded, _, _ = adapter.process_sequence(x, M0, 1, valid, no_write)
        probe.check()
        assert torch.equal(recorded, plain) and not intercepted(adapter)
        q = F.linear(adapter.u(x)[rows, last], adapter.Q)
        torch.testing.assert_close(probe.q, q, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(probe.m, torch.einsum('bij,bj->bi', M0, q), rtol=1e-4, atol=1e-5)
        # Replace the reads of rows 1 (a one-token question) and 4 only; everything else keeps its deployed read.
        read = probe.m.clone()
        read[[1, 4]] += 100.
        with rfx.FinalRead(adapter, last, read) as swap:
            replaced, _, _ = adapter.process_sequence(x, M0, 1, valid, no_write)
        swap.check()
        final = torch.zeros_like(valid)
        final[rows, last] = True
        changed = (replaced != plain).any(-1)
        expected = torch.zeros_like(valid)
        expected[[1, 4], last[[1, 4]]] = True
        assert torch.equal(changed, expected)  # no other position of any row, no padding, no other row's last token
        assert torch.equal(swap.m, probe.m) and torch.equal(swap.q, probe.q)  # the record is the deployed read
        torch.testing.assert_close(replaced[final], branch_with_read(adapter, x, last, read), rtol=1e-4, atol=1e-5)
        # Re-inserting the recorded reads is bitwise the identity.
        with rfx.FinalRead(adapter, last, probe.m.clone()) as same:
            again, _, _ = adapter.process_sequence(x, M0, 1, valid, no_write)
        same.check()
        assert torch.equal(again, plain)


def test_hook_restarts_its_token_counter_for_every_forward_and_flags_unread_rows():
    adapter = bare()
    x, M0, valid, no_write, last = sequences()
    with torch.no_grad(), rfx.FinalRead(adapter, last) as probe:
        adapter.process_sequence(x, M0, 1, valid, no_write)
        reads = probe.m.clone()
        probe.seen.zero_()
        probe.m.zero_()
        adapter.process_sequence(x, M0, 1, valid, no_write)  # a second forward through the same context
    probe.check()
    assert torch.equal(probe.m, reads)
    # A position that is padding (row 1 has one token) or past the sequence is never read: check() must say so.
    for bad in (3, 7):
        wrong = last.clone()
        wrong[1] = bad
        with torch.no_grad(), rfx.FinalRead(adapter, wrong) as unread:
            adapter.process_sequence(x, M0, 1, valid, no_write)
        with pytest.raises(RuntimeError):
            unread.check()
    with pytest.raises(ValueError):  # a deeper loop reads several times per token; RFX is defined for one read
        with torch.no_grad(), rfx.FinalRead(adapter, last):
            adapter.process_sequence(x, M0, 2, valid, no_write)
    assert not intercepted(adapter)


def test_hook_guards_cleanup_and_double_interception():
    ok = bare()
    rfx.FinalRead(ok, torch.tensor([0, 1]))  # the registered reader: shared iterq, delta store, raw queries, eps 1e-6
    for other in (dict(variant='fixedq'), dict(variant='unshared'), dict(variant='lora'), dict(dpfp_nu=2),
                  dict(store='ridge'), dict(store='sweep'), dict(unit_query=True), dict(eps=1e-5)):
        with pytest.raises(ValueError):
            rfx.FinalRead(bare(**other), torch.tensor([0, 1]))
    for shape in ((3, 4), (2, 5), (2,)):
        with pytest.raises(ValueError):
            rfx.FinalRead(ok, torch.tensor([0, 1]), torch.zeros(*shape))
    rfx.FinalRead(ok, torch.tensor([0, 1]), torch.zeros(2, 4))
    last = torch.tensor([0, 1])
    with rfx.FinalRead(ok, last):
        assert sorted(intercepted(ok)) == sorted(INTERCEPTED)
        with pytest.raises(RuntimeError):  # one interception at a time, and the first stays in place
            rfx.FinalRead(ok, last).__enter__()
        assert sorted(intercepted(ok)) == sorted(INTERCEPTED)
    assert not intercepted(ok)
    with pytest.raises(KeyError):  # an exception inside the context still restores the adapter
        with rfx.FinalRead(ok, last):
            raise KeyError
    assert not intercepted(ok)


def test_observation_matches_the_deployed_read_and_the_cached_store(experiment):
    cfg = experiment.config
    batch = experiment.prepare(data(cfg))
    with torch.no_grad():
        logits, x = experiment.score(batch, 1, capture_inputs=True)
        terms = question_terms(experiment, batch, x)
    obs = rfx.observe(experiment, batch)
    assert not intercepted(experiment.adapter)
    assert torch.equal(obs['logits'], logits)  # recording is the deployed forward
    B, Q = 2, 4
    assert obs['q'].shape == (B, Q, RANK) and obs['keys'].shape == (B, 16, RANK) == obs['values'].shape
    torch.testing.assert_close(obs['q'], terms['q'], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(obs['delta']['raw'], terms['stored'], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(obs['recomputed_raw'], obs['delta']['raw'], rtol=1e-4, atol=1e-5)
    assert torch.equal(obs['clean'], terms['clean']) and torch.equal(obs['target'], terms['target'])
    state = batch.snapshot.states[experiment.layer]
    unit_q = F.normalize(obs['q'], dim=-1, eps=rfx.EPS)
    assert torch.equal(obs['delta']['unit'], torch.einsum('bij,bqj->bqi', state, unit_q))
    assert torch.equal(obs['delta']['true'], torch.einsum('bij,bqj->bqi', state, obs['true_key']))
    # The queried fact, found independently of true_fact_indices: the one fact with the question's head and relation.
    arrays = batch.arrays
    for b in range(B):
        for k in range(Q):
            hits = [j for j in range(16) if arrays.facts[b, j, 0] == arrays.q_start[b, k]
                    and arrays.facts[b, j, 1] == arrays.q_rel[b, k]]
            assert len(hits) == 1 and int(obs['target'][b, k]) == hits[0]
            assert torch.equal(obs['true_key'][b, k], batch.keys[b, hits[0]])
            assert torch.equal(obs['clean'][b, k], batch.values[b, hits[0]])
            assert int(obs['answer_entity'][b, k]) == int(arrays.answer[b, k]) == int(arrays.facts[b, hits[0], 2])
    assert torch.equal(obs['answer'], batch.answer_ids)
    # The cached unit keys and values rebuild the deployed store (the decay-first delta rule in the natural order) ...
    rebuilt = rfx.delta_state(experiment.adapter, obs['keys'], obs['values'])
    torch.testing.assert_close(rebuilt, state, rtol=1e-4, atol=1e-5)
    # ... and a different write order gives a different store (the delta rule is order dependent).
    order = torch.arange(15, -1, -1, device=obs['keys'].device)
    reordered = rfx.delta_state(experiment.adapter, obs['keys'][:, order], obs['values'][:, order])
    assert (reordered - state).norm() > 1e-3 * state.norm()  # relative: the tiny model's value scale is not assumed


def test_collect_keeps_episode_order_and_content_across_batches(experiment):
    # The bf16 backbone makes batch shape matter at the 1e-2 level, so this is a structural check (a scrambled or
    # repeated episode would differ by order one); the integer targets are data-derived and must match exactly.
    arrays = data(experiment.config, 4)
    whole, split = rfx.collect(experiment, arrays, 4), rfx.collect(experiment, arrays, 2)
    relative = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))
    assert whole['q'].shape == (4, 4, RANK) and whole['keys'].shape == (4, 16, RANK)
    for name in ('keys', 'values', 'q', 'true_key', 'clean'):
        assert relative(split[name], whole[name]) < .25, name
    for name in ('raw', 'unit', 'true'):
        assert relative(split['delta'][name], whole['delta'][name]) < .25, name
    assert torch.equal(whole['target'], split['target']) and torch.equal(whole['answer_entity'], split['answer_entity'])
    assert torch.equal(rfx.collect(experiment, arrays, 1)['target'], whole['target'])  # one episode per batch


def test_reinserting_the_recorded_reads_is_the_identity_and_replacements_reach_the_logits(experiment):
    batch = experiment.prepare(data(experiment.config))
    obs = rfx.observe(experiment, batch)
    deployed = obs['delta']['raw'].reshape(-1, RANK)
    with torch.no_grad():
        plain, _ = experiment.score(batch, 1)
        again = rfx.replaced_logits(experiment, batch, deployed)
        zero = rfx.replaced_logits(experiment, batch, torch.zeros_like(deployed))
        clean = rfx.replaced_logits(experiment, batch, obs['clean'].reshape(-1, RANK))
    assert torch.equal(again, plain) and torch.equal(obs['logits'], plain)
    assert not torch.equal(zero, plain) and not torch.equal(clean, plain)
    with pytest.raises(ValueError):
        rfx.replaced_logits(experiment, batch, deployed[:-1])


def test_replacement_changes_only_the_last_question_position_of_the_replaced_rows(experiment):
    with torch.no_grad():
        experiment.adapter.U.fill_(.5)  # every pre-activation grows with the read, so a +100 replacement cannot be clipped away
    batch = experiment.prepare(data(experiment.config))
    obs = rfx.observe(experiment, batch)
    deployed = obs['delta']['raw'].reshape(-1, RANK)
    last = rfx.last_positions(batch)
    swapped = deployed.clone()
    swapped[[1, 6]] += 100.
    outputs = []
    hook = experiment.bank.modules[1].register_forward_hook(lambda module, args, output: outputs.append(output.detach().clone()))
    try:
        with torch.no_grad():
            plain, _ = experiment.score(batch, 1)
            replaced = rfx.replaced_logits(experiment, batch, swapped)
    finally:
        hook.remove()
    assert len(outputs) == 2
    changed = (outputs[0] != outputs[1]).any(-1)  # [rows, positions] of the adapted projection
    expected = torch.zeros_like(changed)
    rows = torch.tensor([1, 6], device=last.device)
    expected[rows, last[rows]] = True
    assert torch.equal(changed, expected)
    assert not torch.equal(plain, replaced)


def test_final_position_follows_ragged_question_lengths(experiment):
    batch = experiment.prepare(data(experiment.config))
    lengths = batch.question_mask.long().sum(-1)  # [B, Q], right padded
    shorter = (lengths - torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], device=lengths.device)).clamp_min(1)
    width = batch.question_mask.shape[-1]
    ragged = replace(batch, question_mask=torch.arange(width, device=lengths.device)[None, None, :] < shorter[..., None])
    last = rfx.last_positions(ragged)
    assert last.tolist() == (shorter - 1).flatten().tolist() and len(set(last.tolist())) > 1
    with torch.no_grad():
        logits, x = experiment.score(ragged, 1, capture_inputs=True)
        terms = question_terms(experiment, ragged, x)
    obs = rfx.observe(experiment, ragged)
    torch.testing.assert_close(obs['q'], terms['q'], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(obs['delta']['raw'], terms['stored'], rtol=1e-4, atol=1e-5)
    assert torch.equal(rfx.replaced_logits(experiment, ragged, obs['delta']['raw'].reshape(-1, RANK)), logits)


def test_run_checkpoint_smoke_on_the_tiny_model_is_finite_complete_and_evaluation_only(experiment):
    adapter = experiment.adapter
    before = {k: v.clone() for k, v in adapter.state_dict().items()}
    record, samples = rfx.run_checkpoint('R-test', experiment, experiment.value_head, smoke=True,
                                         fit_episodes=8, eval_episodes=8)
    json.dumps(record, allow_nan=False)
    assert all(torch.equal(before[k], v) for k, v in adapter.state_dict().items())
    assert all(p.grad is None for p in adapter.parameters()) and not intercepted(adapter) and adapter.phase == 'idle'
    assert rfx_run.instrumentation_failures(record['checks']) == [] and record['checks']['identity_max_abs_logit_diff'] == 0.
    assert record['label'] == 'R-test' and record['smoke'] is True and record['rank'] == RANK
    fit, evaluation = record['data']['fit'], record['data']['eval']
    assert (fit['split'], fit['seed'], fit['start'], fit['episodes'], fit['queries_per_episode']) == (
        'train', rfx.FIT_SEED, 0, 8, 4)
    assert (evaluation['split'], evaluation['start'], evaluation['episodes'], evaluation['queries_per_episode']) == (
        'train', rfx.SMOKE_START, 8, 1)
    assert set(record['cells']) == set(rfx.VARIANTS) and set(record['refit']) == set(rfx.REFIT_CELLS)
    assert all(0 <= cell['lm_em'] <= 1 and 0 <= cell['lm_em_entity'] <= 1 and 0 <= cell['head_acc'] <= 1
               and np.isfinite(cell['err']) and cell['lm_nll'] > 0 for cell in record['cells'].values())
    assert record['cells']['clean']['err'] == 0. and 0 < record['cells']['zero']['err'] <= 1
    assert all(0 <= row['acc'] <= 1 and row['nll'] > 0 for row in record['refit'].values())
    assert record['verdict']['verdict'] in rfx.VERDICTS
    assert set(record['verdict']['rescue']) == set(rfx.VARIANTS) | {'decoder_refit'}
    assert record['verdict']['baseline']['lm_em_entity'] == max(
        record['cells']['delta/raw/asis']['lm_em_entity'], record['cells']['delta/raw/fit']['lm_em_entity'])
    assert set(record['parameters']['eta']) == set(rfx.QUERIES) and all(
        eta in rfx.ETA_GRID for eta in record['parameters']['eta'].values())
    # The deployed read through the deployed LM path is the model's own forward on the same set.
    arrays, _ = rfx.eval_arrays(make_task_spec(experiment.config), 8, True)
    batch = experiment.prepare(arrays)
    with torch.no_grad():
        logits, _ = experiment.score(batch, 1)
    correct = (logits.argmax(-1) == batch.answer_ids).flatten().cpu().numpy()
    base = record['cells']['delta/raw/asis']
    assert base['lm_em'] == pytest.approx(float(correct.mean()), abs=1e-12)
    assert base['lm_nll'] == pytest.approx(float(F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten())),
                                           rel=1e-5)
    assert np.array_equal(samples['delta/raw/asis/correct'], correct)
    # The entity-restricted EM: the argmax of the same logits over the 1,024 entity name tokens is the answer entity.
    names = torch.as_tensor(load_name_pool().token_ids[:experiment.config['task']['V']], device=logits.device)
    entity_correct = (logits[..., names].argmax(-1) == torch.as_tensor(arrays.answer, device=logits.device)).flatten().cpu().numpy()
    assert np.array_equal(samples['delta/raw/asis/entity_correct'], entity_correct)
    assert base['lm_em_entity'] == pytest.approx(float(entity_correct.mean()), abs=1e-12)
    assert all(len(samples[f'{v}/{k}']) == 8 for v in rfx.VARIANTS for k in ('correct', 'entity_correct', 'nll', 'err', 'head'))
    # Write order: four permutations of the same writes, tabulated by the permuted write age of the queried fact.
    order = record['write_order']
    assert order['permutations'] == rfx.PERMUTATIONS == len(order['err']) == len(order['lm_em'])
    assert all(0 <= a <= 1 for a in order['agreement_with_natural_order']) and all(
        d >= 0 for d in order['read_discrepancy'])
    assert [row['age'] for row in order['by_permuted_age']] == list(range(16))
    assert sum(row['n'] for row in order['by_permuted_age']) == rfx.PERMUTATIONS * 8
    assert all(sum(row['n'] for row in table) == 8 for table in record['age'].values())
    conditioning = record['conditioning']
    assert set(conditioning['ridge']) == set(rfx.QUERIES) and np.isfinite(conditioning['gram']['max'])
    assert all(c['max'] >= c['median'] for c in (conditioning['gram'], *conditioning['ridge'].values()))


def record_json(**override):
    return {**dict(status='complete', rank=128, arm='R', config_hash='h', git_commit='c', job_id='1.opbs',
                   training=dict(actual_steps=24645)), **override}


def test_registered_checkpoints_are_the_three_wp_l4_final_runs():
    assert set(rfx_run.CHECKPOINTS) == set(rfx.LABELS) == {'R-r128', 'R-r32', 'W5-r32'}
    assert rfx_run.CHECKPOINTS == {'R-r128': ('read-fix-20261027/arms-r128/R', 128, 24645),
                                   'R-r32': ('read-fix-20261027/arms/R', 32, 23590),
                                   'W5-r32': ('writer-fix-20261027/rerun/W5', 32, 32900)}


def test_registered_digests_cover_the_three_checkpoints_and_a_mismatch_voids_the_run():
    assert set(rfx_run.DIGESTS) == set(rfx.LABELS)
    assert all(len(x) == 32 and set(x) <= set('0123456789abcdef') for pair in rfx_run.DIGESTS.values() for x in pair)
    for label, (writer, checkpoint) in rfx_run.DIGESTS.items():
        rfx_run.verify_digests(label, dict(writer_blake2b=writer, checkpoint_blake2b=checkpoint))
        for bad in (dict(writer_blake2b='0' * 32, checkpoint_blake2b=checkpoint),
                    dict(writer_blake2b=writer, checkpoint_blake2b='0' * 32)):
            with pytest.raises(ValueError, match=label):
                rfx_run.verify_digests(label, bad)


def test_checkpoint_identity_requires_the_registered_complete_final_run(tmp_path):
    (tmp_path / 'writer.pt').write_bytes(b'writer')
    (tmp_path / 'checkpoint.pt').write_bytes(b'checkpoint')

    def write(**override):
        (tmp_path / 'record.json').write_text(json.dumps(record_json(**override)))

    write()
    identity = rfx_run.checkpoint_identity('R-r128', tmp_path)
    assert identity['writer_blake2b'] == hashlib.blake2b(b'writer', digest_size=16).hexdigest()
    assert identity['checkpoint_blake2b'] == hashlib.blake2b(b'checkpoint', digest_size=16).hexdigest()
    assert (identity['rank'], identity['steps'], identity['arm'], identity['job_id'], identity['config_hash'],
            identity['git_commit']) == (128, 24645, 'R', '1.opbs', 'h', 'c')
    for override in (dict(status='failed'), dict(smoke=True), dict(rank=32), dict(training=dict(actual_steps=24640))):
        write(**override)
        with pytest.raises(ValueError):
            rfx_run.checkpoint_identity('R-r128', tmp_path)
    write()
    with pytest.raises(ValueError):  # the rank 128 run is not the rank 32 checkpoint
        rfx_run.checkpoint_identity('R-r32', tmp_path)
    write(smoke=False)
    assert rfx_run.checkpoint_identity('R-r128', tmp_path)['steps'] == 24645


def fake_record(verdict, eta):
    return dict(verdict=dict(verdict=verdict, rescue={'ridge/unit/fit': verdict == 'storage'}),
                parameters=dict(eta=dict(raw=1., unit=eta, true=1e-6)))


def test_decisions_trigger_arms_and_carry_the_eta_only_for_ranks_with_a_storage_verdict():
    records = {'R-r128': fake_record('storage', 2e-6), 'R-r32': fake_record('normalization', 3e-3),
               'W5-r32': fake_record('storage', .5)}
    decided = rfx_run.decisions(records)
    assert decided['verdicts'] == {'R-r128': 'storage', 'R-r32': 'normalization', 'W5-r32': 'storage'}
    assert decided['lsq_arms'] == ['L32-U', 'L128-K4', 'L128-RG']  # candidate order; W5-r32 triggers nothing
    assert decided['rfx_eta_by_rank'] == {'128': 2e-6} and decided['eta_by_rank'] == {'128': 1e-4}  # clipped; none for U
    assert decided['rescue'] == {label: r['verdict']['rescue'] for label, r in records.items()}
    both = rfx_run.decisions({'R-r128': fake_record('storage', 5.), 'R-r32': fake_record('storage', 3e-3)})
    assert both['lsq_arms'] == ['L32-K4', 'L32-RG', 'L128-K4', 'L128-RG']
    assert both['rfx_eta_by_rank'] == {'32': 3e-3, '128': 5.} and both['eta_by_rank'] == {'32': 3e-3, '128': 1e-1}
    for verdict in ('query', 'store+query', 'decoder', 'none'):
        nothing = rfx_run.decisions({'R-r128': fake_record(verdict, 1e-3), 'R-r32': fake_record(verdict, 1e-3)})
        assert nothing['lsq_arms'] == [] and nothing['rfx_eta_by_rank'] == {} and nothing['eta_by_rank'] == {}
    json.dumps(both, allow_nan=False)


def test_instrumentation_guards_flag_only_gross_failures():
    ok = dict(identity_max_abs_logit_diff=0., raw_recompute_max_rel=1e-6, delta_reconstruction_max_rel=1e-6,
              ridge_permutation_max_abs=1e-9)
    assert rfx_run.instrumentation_failures(ok) == []
    assert len(rfx_run.instrumentation_failures({**ok, 'identity_max_abs_logit_diff': 1e-9})) == 1
    assert len(rfx_run.instrumentation_failures({**ok, 'identity_max_abs_logit_diff': None})) == 1
    for name in ('raw_recompute_max_rel', 'delta_reconstruction_max_rel', 'ridge_permutation_max_abs'):
        assert len(rfx_run.instrumentation_failures({**ok, name: 1e-3})) == 1
        assert len(rfx_run.instrumentation_failures({**ok, name: float('nan')})) == 1
        assert rfx_run.instrumentation_failures({**ok, name: 1e-4}) == []  # the registered tolerance is inclusive


def export_read_fix(model, tokenizer, folder, cfg=None, arm='R', cls=ReadFixExperiment):
    """Save an inference export and the full checkpoint of a tiny experiment; returns its logits and head state."""
    cfg = tiny() if cfg is None else cfg
    experiment = cls(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        with torch.no_grad():
            experiment.adapter.B.fill_(.1)
        folder.mkdir(parents=True, exist_ok=True)
        read_fix_run.save_checkpoint(experiment, torch.optim.AdamW(experiment.trainable_parameters()), folder, 10)
        with torch.no_grad():
            expected, _ = experiment.score(experiment.prepare(data(cfg)), 1)
        return expected, {k: v.detach().cpu().clone() for k, v in experiment.value_head.state_dict().items()}
    finally:
        experiment.close()


def export_writer_fix(model, tokenizer, folder):
    cfg = writer_fix_run.configuration('W5')
    cfg['model'].update(layer=1, r=RANK)
    experiment = WriterFixExperiment(cfg, 'W5', model=model, tokenizer=tokenizer)
    try:
        experiment.standardize()
        with torch.no_grad():
            experiment.adapter.B.fill_(.1)
        folder.mkdir(parents=True, exist_ok=True)
        writer_fix_run.save_checkpoint(experiment, torch.optim.AdamW(experiment.trainable_parameters()), folder, 10)
        with torch.no_grad():
            expected, _ = experiment.score(experiment.prepare(data(cfg)), 1)
        return cfg, expected, {k: v.detach().cpu().clone() for k, v in experiment.value_head.state_dict().items()}
    finally:
        experiment.close()


@pytest.mark.parametrize('kind', ['read-fix', 'writer-fix'])
def test_load_checkpoint_reads_both_wp_l4_export_formats_with_the_training_head(model, tokenizer, tmp_path, kind):
    if kind == 'read-fix':
        cfg = tiny()
        expected, head_state = export_read_fix(model, tokenizer, tmp_path)
    else:
        cfg, expected, head_state = export_writer_fix(model, tokenizer, tmp_path)
    loaded, head = rfx.load_checkpoint(tmp_path, model=model, tokenizer=tokenizer)
    try:
        assert type(loaded) is (ReadFixExperiment if kind == 'read-fix' else WriterFixExperiment)
        assert loaded.value_head is None and isinstance(head, nn.Linear)
        assert (head.in_features, head.out_features) == (RANK, loaded.config['task']['V'])
        assert head.weight.device == loaded.adapter.A.device
        assert all(torch.equal(head.state_dict()[k].cpu(), v) for k, v in head_state.items())
        with torch.no_grad():
            actual, _ = loaded.score(loaded.prepare(data(cfg)), 1)
        assert torch.equal(actual, expected)
        assert rfx.observe(loaded, loaded.prepare(data(cfg)))['logits'].equal(expected)  # the hook works on both adapters
    finally:
        loaded.close()


def test_load_checkpoint_refuses_exports_it_cannot_read_and_leaves_no_adapter_behind(model, tokenizer, tmp_path):
    headless = tmp_path / 'headless'
    export_read_fix(model, tokenizer, headless)
    payload = torch.load(headless / 'checkpoint.pt', map_location='cpu', weights_only=True)
    payload['value_head'] = None
    torch.save(payload, headless / 'checkpoint.pt')
    with pytest.raises(ValueError):  # the clean-value head is part of the registered decoders
        rfx.load_checkpoint(headless, model=model, tokenizer=tokenizer)
    assert not hasattr(model, 'looplora_adapters')
    cfg = lsq_configuration('L32-K4', 10)
    cfg['model'].update(layer=1, r=RANK)
    export_read_fix(model, tokenizer, tmp_path / 'lsq', cfg=cfg, arm='L32-K4', cls=LSQExperiment)
    with pytest.raises(ValueError):  # LSQ exports use another store and unit queries
        rfx.load_checkpoint(tmp_path / 'lsq', model=model, tokenizer=tokenizer)
    (tmp_path / 'other').mkdir()
    torch.save(dict(config={}), tmp_path / 'other' / 'writer.pt')
    with pytest.raises(ValueError):
        rfx.load_checkpoint(tmp_path / 'other', model=model, tokenizer=tokenizer)
    assert not hasattr(model, 'looplora_adapters')


def test_run_evaluates_both_export_formats_and_writes_the_registered_files(model, tokenizer, tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    export_read_fix(model, tokenizer, runs / 'r32')
    export_writer_fix(model, tokenizer, runs / 'w5')
    folders = {'R-r32': 'r32', 'W5-r32': 'w5'}
    for label, arm in (('R-r32', 'R'), ('W5-r32', 'W5')):
        (runs / folders[label] / 'record.json').write_text(json.dumps(record_json(
            arm=arm, rank=RANK, training=dict(actual_steps=10))))
    monkeypatch.setattr(rfx_run, 'CHECKPOINTS', {'R-r128': ('r128', 128, 10), 'R-r32': ('r32', RANK, 10),
                                                 'W5-r32': ('w5', RANK, 10)})
    digests = {label: tuple(hashlib.blake2b((runs / folders[label] / name).read_bytes(), digest_size=16).hexdigest()
                            for name in ('writer.pt', 'checkpoint.pt')) for label in folders}
    monkeypatch.setattr(rfx_run, 'DIGESTS', digests)
    monkeypatch.setattr(rfx_run, 'SMOKE', dict(fit_episodes=8, eval_episodes=8))
    monkeypatch.setattr(rfx_run, '_load_backbone', lambda device: (model, tokenizer, 0.))
    out = tmp_path / 'out'
    summary = rfx_run.run(runs, out, ('R-r32', 'W5-r32'), smoke=True)
    assert summary['smoke'] is True and summary['labels'] == ['R-r32', 'W5-r32']
    assert summary['instrumentation_failures'] == {'R-r32': [], 'W5-r32': []}
    assert set(summary['verdicts']) == {'R-r32', 'W5-r32'}
    assert set(summary['verdicts'].values()) <= set(rfx.VERDICTS)
    assert 'lsq_ranks' not in summary and summary['lsq_arms'] == rfx.lsq_arms(summary['verdicts'])
    assert set(summary['rfx_eta_by_rank']) == set(summary['eta_by_rank']) == {
        str(rank) for rank in rfx.storage_ranks(summary['verdicts'])}
    assert set(summary['rescue']) == set(summary['verdicts']) and all(
        set(flags) == set(rfx.VARIANTS) | {'decoder_refit'} for flags in summary['rescue'].values())
    on_disk = json.loads((out / 'summary.json').read_text())
    assert on_disk['verdicts'] == summary['verdicts'] and on_disk['lsq_arms'] == summary['lsq_arms']
    assert on_disk['rescue'] == summary['rescue'] and on_disk['eta_by_rank'] == summary['eta_by_rank']
    for label, arm in (('R-r32', 'R'), ('W5-r32', 'W5')):
        saved = json.loads((out / f'{label}.json').read_text())
        assert saved['label'] == label and saved['smoke'] is True and saved['checkpoint']['arm'] == arm
        digest = hashlib.blake2b((runs / folders[label] / 'writer.pt').read_bytes(), digest_size=16).hexdigest()
        assert saved['checkpoint']['writer_blake2b'] == digest and saved['verdict']['verdict'] == summary['verdicts'][label]
        with np.load(out / f'{label}-samples.npz') as samples:
            assert 'delta__raw__asis__correct' in samples.files and not any('/' in name for name in samples.files)
            assert samples['delta__raw__asis__correct'].shape == (8,)
    assert not hasattr(model, 'looplora_adapters')
    with pytest.raises(FileExistsError):  # outputs are never overwritten
        rfx_run.run(runs, out, ('R-r32',), smoke=True)
    with pytest.raises(ValueError):  # the registered study evaluates all three checkpoints
        rfx_run.run(runs, tmp_path / 'partial', ('R-r32',), smoke=False)
    assert not (tmp_path / 'partial').exists()
    monkeypatch.setattr(rfx_run, 'DIGESTS', {**digests, 'R-r32': ('0' * 32, digests['R-r32'][1])})
    with pytest.raises(ValueError, match='R-r32'):  # a checkpoint that is not the registered one voids the run
        rfx_run.run(runs, tmp_path / 'void', ('R-r32',), smoke=True)
    assert not (tmp_path / 'void').exists()
