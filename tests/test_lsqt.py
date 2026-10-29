"""WP-L12 LSQT: the three trained least-squares arms at r = 128 (docs/prereg/LSQT.md): configs, registry, decision rule, exports.

Compute only (imports torch; tiny models). The old LSQ registry, its configs and its loaders must be unchanged by LSQT.
"""
import copy
import json
import math

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.infra.seeds import derive_seed, stable_hash
from looplora.data.relations import generate
from looplora.llm import dpfp_run, lsq, lsq_run, read_fix, read_fix_run
from looplora.llm.inject import load_tokenizer
from looplora.llm.lsq import SWEEPS, STORE_RULES, LSQExperiment
from looplora.llm.lsq_release import SPECS as LSQ_SPECS
from looplora.llm.lsqt import LSQTExperiment, configuration
from looplora.llm.lsqt_release import ARMS, CONTROL, ETA_BOUNDS, SPECS, prereg_digest, scheduled_steps
from looplora.llm.lsqt_run import D128_3_CORRECT, R_R128_CORRECT, choose_t, load_inference
from looplora.llm.read_fix import extra_losses, question_terms
from looplora.llm.read_fix_run import fit, validate
from looplora.llm.train import make_task_spec

REGISTERED_BLAKE2B_128 = '94ed7364da460aeec18cbd5da6b086ff'  # docs/prereg/LSQT.md at registration (847b17f)

# stable_hash of lsq.configuration(arm, 1000, E=E, smoke=smoke, eta=(3e-3 for the ridge arms)) for every LSQ arm, computed on compute at
# 2396e54 (job hash-lsq, before any LSQT code): the LSQ registry, its configs and the STORE_RULES text must not move.
LSQ_HASHES = {
    'L128-K4|16|0': '39a8b3a23df942f1c2711ab0453fa085',
    'L128-K4|16|1': '47c8cd11f04af469aee4ad3df34bc611',
    'L128-K4|8|0': 'a30b51003d669d89ccbf1a6decd68016',
    'L128-K4|8|1': '2e6a0b7a7ee71bfbfce6004b187f0882',
    'L128-RG|16|0': '67953387025c4afd211518c40331b728',
    'L128-RG|16|1': '07d8515a93ffac7f4fa7958d88bee422',
    'L128-RG|8|0': '6ea8f62820d45e1bd432f78580abdeb3',
    'L128-RG|8|1': '9ff70663a2ebb2a4f90b8565e6478f9e',
    'L128-U|16|0': '65ad401e4802b66c18b116390ed5d283',
    'L128-U|16|1': 'c896994bc250f3e2acaf4a99539bf01a',
    'L128-U|8|0': '4387a8f1d2405a90b6ae46b512c3c6c6',
    'L128-U|8|1': '9f70ce64042b7bade7938a6b0c8a8539',
    'L32-K4|16|0': 'eaef72a6b68c80a72a2100d6da1f7810',
    'L32-K4|16|1': '15b953d2cf255c0bb7e77e0190f8c5fe',
    'L32-K4|8|0': '9676174a37f20e1d0ecd1add23e3a235',
    'L32-K4|8|1': 'b10d3770e2549ecd502884d15a4a5333',
    'L32-RG|16|0': '9c3fe0f7d4e6ab7eb7b2ea50fcbb42b0',
    'L32-RG|16|1': 'f0fa0415a677fbe2655ba22c0848ae5e',
    'L32-RG|8|0': 'ebb7980348bdb9d8f4057deee8ab9be8',
    'L32-RG|8|1': 'ae44ee124855c8500b3aaeb9e4e42ad0',
    'L32-U|16|0': 'bcf2d483f7f1d9fc2ccd37163ffe53ac',
    'L32-U|16|1': '03c9904cb51e448a7a5603e70443b3ee',
    'L32-U|8|0': '3fc3f425a2826b050486a04e09c63682',
    'L32-U|8|1': 'f79326d89b3b628d57490f2757822fcc',
}


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'lsqt-test-model'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield model
    assert not hasattr(model, 'looplora_adapters')


def tiny(arm):
    cfg = configuration(arm, 10)
    cfg['model'].update(layer=1)  # the registered rank 128 stays: a [128, 128] memory on the tiny model
    return cfg


def data(cfg):
    return generate(make_task_spec(cfg), 'train', 0, 2, derive_seed(17, 'lsqt-fixture'))


# ---------------------------------------------------------------------------------------------------- registry and configs

def test_the_registered_arms_etas_and_prereg_digest():
    assert ARMS == ('LT-K4', 'LT-RG001', 'LT-RG1') and CONTROL == 'LT-U'
    assert SPECS == {'LT-K4': ('sweep', None), 'LT-RG001': ('ridge', 1e-2), 'LT-RG1': ('ridge', 1.0), 'LT-U': ('delta', None)}
    assert ETA_BOUNDS == (1e-4, 1.0)
    assert prereg_digest() == REGISTERED_BLAKE2B_128


@pytest.mark.parametrize('arm', list(SPECS))
def test_configuration_is_wp_l4_R_at_r128_plus_only_the_registered_store_fields(arm):
    store, eta = SPECS[arm]
    cfg = configuration(arm, 24000)
    base = read_fix.configuration('R', 24000, E=16, rank=128)
    rest = copy.deepcopy(cfg)
    del rest['lsqt']
    for key in ('store', 'unit_query', 'store_eta', 'store_sweeps'):
        rest['model'].pop(key, None)
    rest['name'] = base['name']
    assert rest == base and cfg['name'] == f'lsqt-{arm}-E16'
    assert set(cfg['model']) - set(base['model']) == {
        'delta': {'unit_query'}, 'sweep': {'unit_query', 'store', 'store_sweeps'}, 'ridge': {'unit_query', 'store', 'store_eta'}}[store]
    assert (cfg['model']['r'], cfg['model'].get('store', 'delta'), cfg['model']['unit_query'], cfg['task']['E']) == (128, store, True, 16)
    assert cfg['model'].get('store_eta') == eta and cfg['model'].get('store_sweeps') == (SWEEPS if store == 'sweep' else None)
    assert (cfg['lsqt']['arm'], cfg['lsqt']['rank'], cfg['lsqt']['store'], cfg['lsqt']['eta']) == (arm, 128, store, eta)
    assert cfg['lsqt']['store_rule'] == STORE_RULES[store] and 'lsq' not in cfg and 'dpfp' not in cfg
    assert configuration(arm, 20, smoke=True)['eval']['episodes'] == 100 and cfg['eval']['episodes'] == 1000
    json.dumps(cfg, allow_nan=False)


def test_the_ridge_arms_use_the_lsq_store_fields_of_the_same_eta():
    # LT-K4 is L128-K4's training exactly (up to the name and the study key); the ridge arms differ from L128-RG only by eta.
    ours, theirs = configuration('LT-K4', 1000), lsq.configuration('L128-K4', 1000)
    for cfg, key in ((ours, 'lsqt'), (theirs, 'lsq')):
        cfg.pop(key)
        cfg['name'] = ''
    assert ours == theirs
    ours, theirs = configuration('LT-RG001', 1000), lsq.configuration('L128-RG', 1000, eta=1e-2)
    for cfg, key in ((ours, 'lsqt'), (theirs, 'lsq')):
        cfg.pop(key)
        cfg['name'] = ''
    assert ours == theirs


def test_rejected_configurations():
    for arm, kwargs in (('R', {}), ('L128-K4', {}), ('D128-3', {}), ('LT-K4', {'E': 8}), ('LT-RG1', {'E': 4})):
        with pytest.raises(ValueError):
            configuration(arm, 10, **kwargs)


def test_the_lsq_registry_configs_and_hashes_are_unchanged_by_lsqt():
    for arm in LSQ_SPECS:
        for E in (8, 16):
            for smoke in (False, True):
                eta = 3e-3 if LSQ_SPECS[arm][1] == 'ridge' else None
                key = f'{arm}|{E}|{int(smoke)}'
                assert stable_hash(lsq.configuration(arm, 1000, E=E, smoke=smoke, eta=eta)) == LSQ_HASHES[key], key
    assert len(LSQ_HASHES) == len(LSQ_SPECS) * 4
    assert lsq.SWEEPS == 4 and set(lsq.STORE_RULES) == {'ridge', 'sweep', 'delta'}


@pytest.mark.parametrize('arm, change', [
    ('LT-K4', dict(store='ridge')), ('LT-K4', dict(unit_query=False)), ('LT-K4', dict(store_sweeps=3)), ('LT-K4', dict(dpfp_nu=3)),
    ('LT-RG001', dict(store_eta=.5)), ('LT-RG001', dict(store='sweep')), ('LT-RG1', dict(store_eta=1e-2)),
    ('LT-RG001', dict(store_sweeps=4)), ('LT-RG1', dict(r=32)), ('LT-U', dict(store='ridge'))])
def test_experiment_rejects_a_config_that_is_not_its_arm(model, tokenizer, arm, change):
    cfg = tiny(arm)
    cfg['model'].update(change)
    with pytest.raises(ValueError):
        LSQTExperiment(cfg, arm, model=model, tokenizer=tokenizer)


def test_experiment_rejects_mixed_studies_and_arms(model, tokenizer):
    with pytest.raises(ValueError):
        LSQTExperiment(tiny('LT-K4'), 'LT-RG1', model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError):
        LSQTExperiment(tiny('LT-K4'), 'R', model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError):
        LSQExperiment(tiny('LT-K4'), 'L128-K4', model=model, tokenizer=tokenizer)  # the LSQ class does not take an LSQT config
    cfg = tiny('LT-K4')
    cfg['lsq'] = cfg['lsqt']
    with pytest.raises(ValueError):
        LSQTExperiment(cfg, 'LT-K4', model=model, tokenizer=tokenizer)
    cfg = tiny('LT-RG001')
    del cfg['lsqt']
    with pytest.raises(ValueError):
        LSQTExperiment(cfg, 'LT-RG001', model=model, tokenizer=tokenizer)


# ------------------------------------------------------------------------------------------------------------- the schedule

def release(steps, **kwargs):
    return dict(study='LSQT', E=16, steps_by_arm=steps, **kwargs)


def test_scheduled_steps_follow_the_release_and_never_run_the_control():
    steps = {'LT-K4': 8000, 'LT-RG001': 7500, 'LT-RG1': 7000}
    assert [scheduled_steps(arm, None, release(steps)) for arm in ARMS] == [8000, 7500, 7000]
    assert scheduled_steps('LT-K4', 2400, None, smoke=True) == 2400
    for arm, given, schedule, smoke in (('LT-U', None, release(steps), False), ('LT-K4', 8000, release(steps), False),
            ('LT-K4', None, None, False), ('LT-K4', 2400, release(steps), True), ('LT-K4', None, None, True), ('LT-K4', 3, None, True),
            ('R', None, release(steps), False), ('LT-K4', None, release({'LT-K4': 8000}), False),
            ('LT-K4', None, release({**steps, 'LT-K4': 7999}), False), ('LT-K4', None, release({**steps, 'LT-K4': True}), False),
            ('LT-K4', None, release(steps) | {'E': 8}, False), ('LT-K4', None, release(steps) | {'study': 'LSQ'}, False),
            ('LT-U', None, release({**steps, 'LT-U': 5}), False)):
        with pytest.raises(ValueError):
            scheduled_steps(arm, given, schedule, smoke=smoke)


def test_the_conditional_control_is_released_alone_in_its_own_study():
    control = release({'LT-U': 12000}) | {'study': 'LSQT-U'}
    assert scheduled_steps('LT-U', None, control) == 12000
    registered = {'LT-K4': 8000, 'LT-RG001': 7500, 'LT-RG1': 7000}
    for arm, schedule in (('LT-K4', control), ('LT-U', release(registered)), ('LT-U', release({**registered, 'LT-U': 5}) | {'study': 'LSQT-U'}),
                          ('LT-U', release({'LT-U': 12001}) | {'study': 'LSQT-U'}), ('LT-U', release({'LT-K4': 8000}) | {'study': 'LSQT-U'}),
                          ('LT-U', release({'LT-U': 12000}) | {'study': 'LSQT-U', 'E': 8}), ('LT-U', release({'LT-U': 12000}) | {'study': 'LSQ'})):
        with pytest.raises(ValueError):
            scheduled_steps(arm, None, schedule)
    assert [scheduled_steps(arm, None, release(registered)) for arm in ARMS] == [8000, 7500, 7000]  # the LSQT release is unchanged


# ---------------------------------------------------------------------------------------------------------- the decision rule

def records(correct, **changes):
    out = []
    for arm in ARMS:
        out.append({**dict(arm=arm, status='complete', smoke=False, config=dict(task=dict(E=16)),
                           training=dict(actual_steps=10, metrics={'10': dict(correct=correct[arm])})), **changes})
    return out


def test_choose_t_flags_the_candidate_the_w3_trigger_and_the_falsifier():
    def pick(k4, rg001, rg1):
        return choose_t(records({'LT-K4': k4, 'LT-RG001': rg001, 'LT-RG1': rg1}))

    result = pick(799, 799, 799)
    assert result['outcome'] == 'no_candidate' and result['candidate'] is None and result['e8_retrain'] is False
    assert result['best_rg'] == 'LT-RG001' and result['best_rg_band'] == 'w3_trigger'
    assert pick(800, 100, 100)['candidate'] == 'LT-K4' and pick(100, 800, 100)['candidate'] == 'LT-RG001'
    assert pick(800, 800, 800)['candidate'] == 'LT-K4'  # ties in the candidate order
    assert pick(850, 900, 800)['candidate'] == 'LT-RG001' and pick(900, 850, 950)['candidate'] == 'LT-RG1'
    assert pick(100, 800, 800)['candidate'] == 'LT-RG001'
    # the band of the best ridge arm: >= R-r128 + 15 pp = 48.5 % is the W3 trigger, < 40 % closes the line, between neither
    assert R_R128_CORRECT == 335 and D128_3_CORRECT == 608
    assert pick(0, 485, 0)['best_rg_band'] == 'w3_trigger' and pick(0, 484, 0)['best_rg_band'] == 'open'
    assert pick(0, 400, 0)['best_rg_band'] == 'open' and pick(0, 399, 399)['best_rg_band'] == 'closed'
    assert pick(0, 399, 399)['trained_lsq_line'] == 'closed' and pick(0, 400, 0)['trained_lsq_line'] == 'open'
    assert pick(900, 100, 100)['best_rg_band'] == 'closed'  # the band reads the ridge arms only; K4 does not rescue the line
    assert pick(300, 399, 399)['best_rg'] == 'LT-RG001' and pick(0, 450, 450)['best_rg'] == 'LT-RG001'  # ties: smaller eta first
    assert pick(0, 450, 451)['best_rg'] == 'LT-RG1'
    # the conditional U control runs only if the best arm (any of the three) reaches 40 %
    assert pick(400, 0, 0)['u_control_runs'] is True and pick(399, 399, 399)['u_control_runs'] is False
    result = pick(335, 400, 608)
    assert result['vs_r_r128_pp'] == {'LT-K4': 0., 'LT-RG001': 6.5, 'LT-RG1': 27.3} and result['vs_d128_3_pp']['LT-RG1'] == 0.
    assert result['correct'] == {'LT-K4': 335, 'LT-RG001': 400, 'LT-RG1': 608}


def test_choose_t_requires_the_three_complete_scientific_e16_arms():
    ok = {'LT-K4': 1, 'LT-RG001': 2, 'LT-RG1': 3}
    for bad in (records(ok)[:-1], records(ok) + records(ok)[:1], records(ok, status='failed'), records(ok, smoke=True),
                [{**r, 'arm': 'LT-U'} for r in records(ok)], []):
        with pytest.raises(ValueError):
            choose_t(bad)
    short = records(ok)
    short[1]['config']['task']['E'] = 8
    with pytest.raises(ValueError):
        choose_t(short)


# ----------------------------------------------------------------------- tiny training, export and reload of the new arms

@pytest.mark.parametrize('arm', ['LT-K4', 'LT-RG001', 'LT-RG1', 'LT-U'])
def test_training_validation_and_export_reload_for_every_lsqt_store(model, tokenizer, tmp_path, arm):
    cfg = tiny(arm)
    store = SPECS[arm][0]
    experiment = LSQTExperiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        assert experiment.read_arm == arm and experiment.adapter.cfg.store == store and experiment.adapter.cfg.unit_query
        signature = experiment.writer_signature()
        assert signature['read_fix'] == arm and signature['unit_query'] is True and signature.get('store', 'delta') == store
        assert experiment.standardize()['episodes'] == 256
        with torch.no_grad():
            experiment.adapter.B.fill_(.05)
        result = fit(experiment, tmp_path, 5, evaluate=False)
        assert result['actual_steps'] == 5
        rows = [json.loads(line) for line in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == [1, 5] and all(math.isfinite(row['loss']) for row in rows)
        assert all(math.isfinite(row['key_gram_log10_cond_median']) for row in rows)
        assert all(('ridge_log10_cond_median' in row) == (store == 'ridge') for row in rows)
        assert all('coefficient_loss' not in row for row in rows)  # the coefficient loss is SQ2's, off here
        metrics = validate(experiment, tmp_path / 'val', 16)
        assert metrics['n'] == 16 and math.isfinite(metrics['loss']) and metrics['gram_relative_error'] < 1e-4
        with np.load(tmp_path / 'val' / 'samples.npz') as samples:
            assert set(samples['arm'].tolist()) == {arm}
        with torch.no_grad():
            expected, _ = experiment.score(experiment.prepare(data(cfg)), 1)
    finally:
        experiment.close()
    exported = torch.load(tmp_path / 'writer.pt', weights_only=True)
    assert 'value_head' not in exported and exported['config']['lsqt']['arm'] == arm
    # No other study's loader accepts an LSQT export; the LSQT loader reproduces the logits of the trained writer exactly.
    for other in (read_fix_run.load_inference, dpfp_run.load_inference, lsq_run.load_inference):
        with pytest.raises(ValueError):
            other(tmp_path / 'writer.pt', model=model, tokenizer=tokenizer)
    loaded = load_inference(tmp_path / 'writer.pt', model=model, tokenizer=tokenizer)
    try:
        assert loaded.read_arm == arm and loaded.value_head is None
        with torch.no_grad():
            actual, _ = loaded.score(loaded.prepare(data(cfg)), 1)
        assert torch.equal(actual, expected)
    finally:
        loaded.close()
    with pytest.raises(ValueError):
        load_inference(tmp_path / 'checkpoint.pt', model=model, tokenizer=tokenizer)  # carries the head and optimizer
    tampered = {**exported, 'signature': {**exported['signature'], 'unit_query': False}}
    torch.save(tampered, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError):
        load_inference(tmp_path / 'tampered.pt', model=model, tokenizer=tokenizer)


def test_the_gradient_reaches_the_writer_through_both_ridge_arms_and_the_sweep(model, tokenizer):
    for arm in ('LT-K4', 'LT-RG001', 'LT-RG1'):
        cfg = tiny(arm)
        experiment = LSQTExperiment(cfg, arm, model=model, tokenizer=tokenizer)
        try:
            experiment.standardize()
            with torch.no_grad():
                experiment.adapter.B.fill_(.05)
            batch = experiment.prepare(data(cfg))
            logits, x = experiment.score(batch, 1, capture_inputs=True)
            read, _ = extra_losses(experiment, batch, x)
            adapter = experiment.adapter
            grads = torch.autograd.grad(read, [adapter.Q, adapter.V, adapter.K, adapter.a_lambda], retain_graph=True)
            assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
            q = question_terms(experiment, batch, x)['q']
            torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(q.norm(dim=-1)), rtol=0, atol=1e-5)
        finally:
            experiment.close()
