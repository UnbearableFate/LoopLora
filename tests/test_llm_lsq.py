"""WP-L7 LSQ arms on the learned LLM writer: configs, store algebra in the model, diagnostics, exports and gates.

Tiny-model tests (compute only). The real-model LSQ-off reproduction of WP-L4 R is test_llm_dpfp_repro.py (the LSQ
flags default off and share that path).
"""
import copy
import itertools
import json
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.memory import delta_update
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm import dpfp_run, read_fix, read_fix_run, rfx, rfx_run
from looplora.llm.inject import load_tokenizer
from looplora.llm.lsq import SWEEPS, LSQExperiment, configuration
from looplora.llm.lsq_release import (ARMS, ETA_BOUNDS, SPECS, check_arms, eligible_e8, lsq_eta, release_arms, release_eta,
                                      scheduled_steps)
from looplora.llm.lsq_run import choose, load_inference
from looplora.llm.read_fix import ReadFixExperiment, diagnostics, extra_losses, gram_read, question_terms, store_read
from looplora.llm.read_fix_run import fit, validate
from looplora.llm.train import make_task_spec

RANK = 4  # tiny-model adapter rank = r_mem
ETA = 1e-2


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'lsq-test-model'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield model
    assert not hasattr(model, 'looplora_adapters')


def tiny(arm, **kwargs):
    cfg = configuration(arm, 10, eta=ETA if SPECS[arm][1] == 'ridge' else None, **kwargs)
    cfg['model'].update(layer=1, r=RANK)
    return cfg


def data(cfg):
    return generate(make_task_spec(cfg), 'train', 0, 2, derive_seed(17, 'lsq-fixture'))


def ridge_state(keys, values, lam, eta):
    """M = sum_j w_j v_j k_j^T (eta I + sum_j w_j k_j k_j^T)^-1 with w_j = lam^(n-1-j), from M0 = 0."""
    keys, values = keys.double(), values.double()
    weights = lam ** torch.arange(keys.shape[1] - 1, -1, -1, dtype=torch.float64, device=keys.device)
    system = eta * torch.eye(keys.shape[2], dtype=torch.float64, device=keys.device) \
        + torch.einsum('n,bnj,bnk->bjk', weights, keys, keys)
    return torch.einsum('n,bni,bnj->bij', weights, values, keys) @ torch.linalg.inv(system)


def delta_rule_state(keys, values, lam, beta):
    """The deployed decay-first delta rule over the written pairs in order (arm U's store), from a zero state."""
    state = torch.zeros(keys.shape[0], values.shape[2], keys.shape[2], dtype=torch.float64, device=keys.device)
    for j in range(keys.shape[1]):
        state = delta_update(state, keys[:, j].double(), values[:, j].double(), None, lam, beta)
    return state


def sweep_state(keys, values, lam, sweeps):
    state = torch.zeros(keys.shape[0], values.shape[2], keys.shape[2], dtype=torch.float64, device=keys.device)
    for _ in range(sweeps):
        for j in range(keys.shape[1]):
            state = delta_update(state, keys[:, j].double(), values[:, j].double(), None, lam, 1.)
    return state


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('E', [16, 8])
def test_configuration_is_wp_l4_R_plus_only_the_registered_store_fields(arm, E):
    rank, store = SPECS[arm]
    eta = 3e-3 if store == 'ridge' else None
    cfg = configuration(arm, 24000, E=E, eta=eta)
    base = read_fix.configuration('R', 24000, E=E, rank=rank)
    assert not {'store', 'store_eta', 'store_sweeps', 'unit_query'} & set(base['model']) and 'lsq' not in base
    rest = copy.deepcopy(cfg)
    del rest['lsq']
    for key in ('store', 'unit_query', 'store_eta', 'store_sweeps'):
        rest['model'].pop(key, None)
    rest['name'] = base['name']
    assert rest == base
    assert cfg['name'] == f'lsq-{arm}-E{E}'
    # U is R plus the unit query only; K4 and RG add their store fields to it.
    assert set(cfg['model']) - set(base['model']) == {
        'delta': {'unit_query'}, 'sweep': {'unit_query', 'store', 'store_sweeps'},
        'ridge': {'unit_query', 'store', 'store_eta'}}[store]
    assert (cfg['model']['r'], cfg['model'].get('store', 'delta'), cfg['model']['unit_query'], cfg['task']['E']) == (
        rank, store, True, E)
    assert (cfg['seed'], cfg['model']['layer'], cfg['train']['prefetch_workers']) == (17, 14, 2)
    assert cfg['read_fix']['arm'] == 'R' and cfg['read_fix']['read_weight'] == 1
    assert cfg['model'].get('store_eta') == eta and cfg['model'].get('store_sweeps') == (SWEEPS if store == 'sweep' else None)
    assert (cfg['lsq']['arm'], cfg['lsq']['rank'], cfg['lsq']['store'], cfg['lsq']['eta']) == (arm, rank, store, eta)
    assert configuration(arm, 20, smoke=True, eta=eta)['eval']['episodes'] == 100 and cfg['eval']['episodes'] == 1000
    json.dumps(cfg, allow_nan=False)


def test_registered_arms_order_and_rejected_configurations():
    assert ARMS == ('L32-U', 'L32-K4', 'L32-RG', 'L128-U', 'L128-K4', 'L128-RG') and SWEEPS == 4
    assert [SPECS[a] for a in ARMS] == [(32, 'delta'), (32, 'sweep'), (32, 'ridge'),
                                        (128, 'delta'), (128, 'sweep'), (128, 'ridge')]
    for arm, kwargs in (('R', {}), ('D32-3', {}), ('L32-K4', {'E': 4}), ('L32-U', {'E': 4}), ('L32-RG', {}),
                        ('L32-RG', {'eta': 1e-5}), ('L32-RG', {'eta': .5}), ('L128-RG', {'eta': math.nan}),
                        ('L32-K4', {'eta': 1e-3}), ('L32-U', {'eta': 1e-3}), ('L128-U', {'eta': 1e-3})):
        with pytest.raises(ValueError):
            configuration(arm, 10, **kwargs)
    assert configuration('L128-RG', 10, eta=ETA_BOUNDS[0])['model']['store_eta'] == ETA_BOUNDS[0]
    assert configuration('L128-RG', 10, eta=ETA_BOUNDS[1])['model']['store_eta'] == ETA_BOUNDS[1]
    assert configuration('L128-U', 10)['model']['r'] == 128 and configuration('L32-U', 10)['model']['r'] == 32


def test_ridge_strength_is_the_clipped_rfx_selection():
    assert [lsq_eta(x) for x in (1e-6, 1e-4, 3e-3, 1e-1, 1., 10.)] == [1e-4, 1e-4, 3e-3, 1e-1, 1e-1, 1e-1]
    for bad in (0., -1e-3, math.inf, math.nan, True):
        with pytest.raises(ValueError):
            lsq_eta(bad)


def test_experiment_rejects_inconsistent_arm_and_config(model, tokenizer):
    def reject(cfg, arm):
        with pytest.raises(ValueError):
            LSQExperiment(cfg, arm, model=model, tokenizer=tokenizer)

    reject(tiny('L32-K4'), 'L32-RG')
    reject(tiny('L32-K4'), 'R')
    reject(tiny('L32-U'), 'L32-K4')
    reject(tiny('L32-K4'), 'L32-U')
    reject(tiny('L32-U'), 'R')
    for arm, change in (('L32-K4', dict(store='ridge')), ('L32-K4', dict(unit_query=False)),
                        ('L32-K4', dict(store_sweeps=3)), ('L32-RG', dict(store_eta=.5)), ('L32-RG', dict(dpfp_nu=1)),
                        ('L32-RG', dict(store='sweep')), ('L32-K4', dict(store_eta=1e-3)), ('L32-RG', dict(store_sweeps=4)),
                        ('L32-U', dict(store='ridge')), ('L32-U', dict(store='sweep')), ('L32-U', dict(unit_query=False)),
                        ('L32-U', dict(store_sweeps=4)), ('L32-U', dict(store_eta=1e-3)), ('L32-U', dict(dpfp_nu=1))):
        cfg = tiny(arm)
        cfg['model'].update(change)
        reject(cfg, arm)
    cfg = tiny('L32-RG')
    cfg['lsq']['rank'] = 128
    reject(cfg, 'L32-RG')
    for arm in ('L32-K4', 'L32-U'):
        cfg = tiny(arm)
        del cfg['lsq']
        reject(cfg, arm)
    cfg = tiny('L32-U')
    cfg['model']['store'] = 'delta'  # the explicit default is the same experiment
    LSQExperiment(cfg, 'L32-U', model=model, tokenizer=tokenizer).close()


def test_off_path_experiment_signature_and_adapter_defaults_are_the_wp_l4_ones(model, tokenizer):
    cfg = read_fix.configuration('R', 10)
    cfg['model'].update(layer=1, r=RANK)
    experiment = ReadFixExperiment(cfg, 'R', model=model, tokenizer=tokenizer)
    try:
        adapter = experiment.adapter
        assert (adapter.cfg.store, adapter.cfg.unit_query, adapter.cfg.dpfp_nu) == ('delta', False, 0)
        assert not {'store', 'store_eta', 'store_sweeps', 'unit_query', 'dpfp_nu'} & set(experiment.writer_signature())
        assert not hasattr(experiment, 'log_extras')
    finally:
        experiment.close()


@pytest.mark.parametrize('arm', ['L32-U', 'L32-K4', 'L32-RG'])
def test_lsq_arm_writes_the_registered_store_and_reads_with_unit_queries(model, tokenizer, arm):
    cfg = tiny(arm)
    rank, store = SPECS[arm]
    experiment = LSQExperiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        adapter = experiment.adapter
        assert (adapter.cfg.store, adapter.cfg.unit_query, adapter.cfg.dpfp_nu) == (store, True, 0)
        assert experiment.read_arm == arm and experiment.log_extras() == {}
        signature = experiment.writer_signature()
        assert signature['read_fix'] == arm and signature['unit_query'] is True and signature.get('store', 'delta') == store
        assert ('store' in signature) == (store != 'delta')  # U's signature differs from R's only by the unit query
        assert (signature.get('store_eta'), signature.get('store_sweeps')) == {
            'ridge': (ETA, None), 'sweep': (None, SWEEPS), 'delta': (None, None)}[store]
        assert experiment.standardize()['episodes'] == 256
        with torch.no_grad():
            adapter.B.fill_(.1)
        batch = experiment.prepare(data(cfg))
        state = batch.snapshot.states[1]
        assert state.shape == (2, RANK, RANK) and batch.keys.shape == (2, 16, RANK) and batch.base_keys is None
        lam = float(adapter.lam)
        expected = {'ridge': lambda: ridge_state(batch.keys, batch.values, lam, ETA),
                    'sweep': lambda: sweep_state(batch.keys, batch.values, lam, SWEEPS),
                    'delta': lambda: delta_rule_state(batch.keys, batch.values, lam, float(adapter.beta))}[store]()
        torch.testing.assert_close(state.detach().double(), expected, rtol=1e-4, atol=1e-5)

        logits, x = experiment.score(batch, 1, capture_inputs=True)
        terms = question_terms(experiment, batch, x)
        q = terms['q']
        torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(q.norm(dim=-1)), rtol=0, atol=1e-5)
        torch.testing.assert_close(q, F.normalize(F.linear(terms['u'], adapter.Q), dim=-1, eps=adapter.cfg.eps))
        assert torch.equal(terms['stored'], torch.einsum('bij,bqj->bqi', state, q))
        assert torch.equal(terms['m'], terms['stored'])

        # The raw-read loss reaches the writer parameters it should; the write-strength gate is unused by the least-squares
        # stores and trained by U's delta rule.
        read, decor = extra_losses(experiment, batch, x)
        assert read > 0 and decor == 0
        gradients = torch.autograd.grad(read, [adapter.Q, adapter.V, adapter.K, adapter.a_lambda], retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
        beta_gradient = torch.autograd.grad(read, [adapter.a_beta], retain_graph=True, allow_unused=True)[0]
        if store == 'delta':
            assert beta_gradient is not None and torch.isfinite(beta_gradient).all() and beta_gradient.abs().sum() > 0
        else:
            assert beta_gradient is None

        # Diagnostics: the exact coefficients reproduce the deployed read; conditioning is reported as log10.
        diag, exact = diagnostics(experiment, batch, x)
        assert diag['gram_relative_error'].max() < 1e-4
        assert exact['d'].shape == (2, 4, 16) and exact['gram'].shape == (2, 16, 16)
        assert torch.isfinite(diag['key_gram_log10_cond']).all() and (diag['key_gram_log10_cond'] >= 0).all()
        assert ('ridge_log10_cond' in diag) == (store == 'ridge')
        if store == 'ridge':
            assert (diag['ridge_log10_cond'] <= diag['key_gram_log10_cond'] + 1e-6).all()
        extras = experiment.log_extras()
        assert set(extras) == {'key_gram_log10_cond_median', 'key_gram_log10_cond_max'} | (
            {'ridge_log10_cond_median', 'ridge_log10_cond_max'} if store == 'ridge' else set())
        assert all(math.isfinite(v) for v in extras.values())
    finally:
        experiment.close()


@pytest.mark.parametrize('arm', ['L32-U', 'L32-K4', 'L32-RG'])
def test_training_loop_validation_and_export_reload_for_every_lsq_store(model, tokenizer, tmp_path, arm):
    cfg = tiny(arm)
    store = SPECS[arm][1]
    experiment = LSQExperiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        assert experiment.standardize()['episodes'] == 256
        result = fit(experiment, tmp_path, 5, evaluate=False)
        assert result['actual_steps'] == 5 and result['metrics'] == {}
        rows = [json.loads(line) for line in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == [1, 5]
        assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in rows)
        assert rows[0]['read_loss'] > 0 and rows[0]['address_loss'] > 0 and rows[0]['value_loss'] > 0
        assert all(math.isfinite(row['key_gram_log10_cond_median']) for row in rows)
        assert all(('ridge_log10_cond_median' in row) == (store == 'ridge') for row in rows)
        exported = torch.load(tmp_path / 'writer.pt', weights_only=True)
        assert 'value_head' not in exported and 'optimizer' not in exported
        assert exported['arm'] == arm and exported['signature'].get('store', 'delta') == store
        assert exported['signature']['unit_query'] is True
        assert exported['config']['lsq']['arm'] == arm
        metrics = validate(experiment, tmp_path / 'val', 16)
        assert metrics['n'] == 16 and 0 <= metrics['em'] <= 1 and math.isfinite(metrics['loss'])
        assert 'key_gram_log10_cond' in metrics and metrics['gram_relative_error'] < 1e-4
        with np.load(tmp_path / 'val' / 'samples.npz') as samples:
            assert set(samples['arm'].tolist()) == {arm}
        with torch.no_grad():
            expected, _ = experiment.score(experiment.prepare(data(cfg)), 1)
    finally:
        experiment.close()
    # Neither the WP-L4 nor the DPFP loader accepts an LSQ export; the LSQ loader reproduces its logits exactly.
    for other in (read_fix_run.load_inference, dpfp_run.load_inference):
        with pytest.raises(ValueError):
            other(tmp_path / 'writer.pt', model=model, tokenizer=tokenizer)
    loaded = load_inference(tmp_path / 'writer.pt', model=model, tokenizer=tokenizer)
    try:
        assert loaded.read_arm == arm and loaded.value_head is None and loaded.adapter.cfg.store == store
        with torch.no_grad():
            actual, _ = loaded.score(loaded.prepare(data(cfg)), 1)
        assert torch.equal(actual, expected)
    finally:
        loaded.close()
    payload = torch.load(tmp_path / 'writer.pt', weights_only=True)
    with pytest.raises(ValueError):
        load_inference(tmp_path / 'checkpoint.pt', model=model, tokenizer=tokenizer)  # carries the head and optimizer
    for change in (dict(store='delta'), dict(unit_query=False)):
        tampered = {**payload, 'signature': {**payload['signature'], **change}}
        torch.save(tampered, tmp_path / 'tampered.pt')
        with pytest.raises(ValueError):
            load_inference(tmp_path / 'tampered.pt', model=model, tokenizer=tokenizer)


@pytest.mark.parametrize('store', ['delta', 'ridge', 'sweep'])
def test_store_read_gives_the_exact_coefficients_of_each_store(store):
    torch.manual_seed(derive_seed(17, 'lsq-store-read', store))
    adapter = make_adapter(AdapterConfig(d_in=6, d_out=6, r=5, variant='iterq', store=store, store_eta=.03,
                                         store_sweeps=3, unit_query=store != 'delta')).double()
    g = torch.Generator().manual_seed(derive_seed(17, 'lsq-store-read-data', store))
    keys = F.normalize(torch.randn(2, 7, 5, generator=g, dtype=torch.float64), dim=-1)
    values = torch.randn(2, 7, 5, generator=g, dtype=torch.float64)
    q = F.normalize(torch.randn(2, 3, 5, generator=g, dtype=torch.float64), dim=-1)
    observed = store_read(adapter, keys, values, q)
    assert observed['d'].shape == (2, 3, 7) and observed['reconstructed'].shape == (2, 3, 5)
    if store == 'delta':
        expected = gram_read(keys, values, q, adapter.lam, adapter.beta)
        assert all(torch.equal(observed[k], expected[k]) for k in expected)
        return
    lam = float(adapter.lam)
    state = ridge_state(keys, values, lam, .03) if store == 'ridge' else sweep_state(keys, values, lam, 3)
    torch.testing.assert_close(observed['reconstructed'].detach(), torch.einsum('bij,bqj->bqi', state, q),
                               rtol=1e-9, atol=1e-9)


STORAGE = ('L32-K4', 'L32-RG', 'L128-K4', 'L128-RG')  # a storage verdict at both ranks
MIXED = ('L32-U', 'L128-K4', 'L128-RG')  # normalization at rank 32, storage at rank 128
RANK128 = ('L128-K4', 'L128-RG')


def records(scores, arms):
    return [dict(arm=arm, status='complete', smoke=False, config=dict(task=dict(E=16)),
                 training=dict(actual_steps=10, metrics={'10': dict(correct=scores.get(arm, 0))})) for arm in arms]


def test_selection_follows_candidate_order_thresholds_and_single_E8_gate():
    def pick(scores, arms=STORAGE):
        return choose(records(scores, arms), arms)

    assert pick({'L32-K4': 800, 'L32-RG': 999, 'L128-RG': 1000})['arm'] == 'L32-K4'
    assert pick({'L32-K4': 799, 'L32-RG': 800, 'L128-K4': 999})['arm'] == 'L32-RG'
    assert pick({'L128-RG': 800})['action'] == 'select'
    choice = pick({'L32-K4': 799, 'L32-RG': 799})  # ties keep the candidate order
    assert (choice['action'], choice['arm'], choice['E']) == ('retry_e8', 'L32-K4', 8)
    choice = pick({'L32-K4': 600, 'L128-RG': 700})
    assert (choice['action'], choice['arm'], choice['E']) == ('retry_e8', 'L128-RG', 8)
    assert pick({'L128-K4': 500})['action'] == 'retry_e8'
    stopped = pick({'L32-K4': 499, 'L32-RG': 300, 'L128-RG': 10})
    assert (stopped['action'], stopped['arm'], stopped['E']) == ('stop', None, None)
    assert stopped['correct'] == {'L32-K4': 499, 'L32-RG': 300, 'L128-K4': 0, 'L128-RG': 10}
    # Only the triggered arms compete, in candidate order (U comes first within its rank).
    assert pick({'L32-K4': 999, 'L128-RG': 800}, RANK128)['arm'] == 'L128-RG'
    assert pick({'L128-K4': 800, 'L128-RG': 900}, RANK128)['arm'] == 'L128-K4'
    assert pick({'L32-U': 800, 'L128-K4': 999}, MIXED)['arm'] == 'L32-U'
    assert pick({'L32-U': 799, 'L128-K4': 800, 'L128-RG': 900}, MIXED)['arm'] == 'L128-K4'
    choice = pick({'L32-U': 600, 'L128-K4': 600, 'L128-RG': 600}, MIXED)
    assert (choice['action'], choice['arm'], choice['E']) == ('retry_e8', 'L32-U', 8)
    assert [pick({arm: score}, (arm,))['action'] for arm, score in (('L128-U', 800), ('L128-U', 500), ('L32-U', 499))] == [
        'select', 'retry_e8', 'stop']
    with pytest.raises(ValueError):
        choose(records({}, STORAGE)[:-1], STORAGE)
    with pytest.raises(ValueError):
        choose(records({}, STORAGE) + [records({}, STORAGE)[0]], STORAGE)
    with pytest.raises(ValueError):
        choose(records({}, STORAGE), RANK128)  # the decision needs exactly the triggered arms
    with pytest.raises(ValueError):
        choose(records({}, RANK128), STORAGE)
    with pytest.raises(ValueError):
        choose(records({}, RANK128[::-1]), RANK128[::-1])  # candidate order is part of the rule
    for arms in ((), ('L32-K4',), ('L32-RG',), ('L32-U', 'L32-K4'), ('L32-U', 'L32-K4', 'L32-RG'), ARMS,
                 ('L32-U', 'L32-U')):  # RFX never triggers these sets
        with pytest.raises(ValueError):
            choose(records({}, arms), arms)
    with pytest.raises(ValueError):
        choose([], ())
    with pytest.raises(ValueError):
        choose([{**r, 'smoke': True} for r in records({}, STORAGE)], STORAGE)
    with pytest.raises(ValueError):
        choose([{**r, 'status': 'failed'} for r in records({}, STORAGE)], STORAGE)
    short = records({}, STORAGE)
    short[1]['config']['task']['E'] = 8
    with pytest.raises(ValueError):
        choose(short, STORAGE)
    lock = dict(choice=pick({'L32-K4': 600, 'L128-RG': 700}))
    assert eligible_e8(lock, 'L128-RG') == 8
    for arm in ('L32-K4', 'L32-RG', 'L128-K4', 'L32-U', 'L128-U'):
        with pytest.raises(ValueError):
            eligible_e8(lock, arm)
    assert eligible_e8(dict(choice=pick({'L32-U': 600, 'L128-K4': 599}, MIXED)), 'L32-U') == 8
    with pytest.raises(ValueError):
        eligible_e8(dict(choice=pick({'L32-K4': 800})), 'L32-K4')


def release(arms, steps, *, E=16, study='LSQ'):
    return dict(study=study, E=E, arms=list(arms), steps_by_arm=steps)


RANK_OPTIONS = {32: ([], ['L32-U'], ['L32-K4', 'L32-RG']), 128: ([], ['L128-U'], ['L128-K4', 'L128-RG'])}


def test_release_names_exactly_the_arms_rfx_can_trigger_in_candidate_order():
    assert release_arms(release(['L32-K4', 'L32-RG'], {})) == ['L32-K4', 'L32-RG']
    assert release_arms(release(['L32-U'], {})) == ['L32-U'] and release_arms(release(['L128-U'], {})) == ['L128-U']
    assert release_arms(release(RANK128, {})) == list(RANK128)
    assert release_arms(release(MIXED, {})) == list(MIXED)
    assert release_arms(release(STORAGE, {})) == list(STORAGE)
    assert release_arms(release(('L32-U', 'L128-U'), {})) == ['L32-U', 'L128-U']
    # Per rank: nothing, U alone, or K4 with RG. Every nonempty combination is a release; nothing else is.
    for low, high in itertools.product(RANK_OPTIONS[32], RANK_OPTIONS[128]):
        arms = low + high
        if arms:
            assert release_arms(release(arms, {})) == arms == check_arms(arms)
        else:
            with pytest.raises(ValueError):
                release_arms(release(arms, {}))
    for bad in (['L32-K4'], ['L32-RG'], ['L32-RG', 'L32-K4'], ['L32-U', 'L32-K4'], ['L32-U', 'L32-RG'],
                ['L32-U', 'L32-K4', 'L32-RG'], ['L128-K4'], ['L128-U', 'L128-RG'], list(ARMS), ['L32-U', 'L32-U'],
                ['L128-U', 'L32-U'], ['L128-K4', 'L128-RG', 'L32-K4', 'L32-RG'], ['L64-K4', 'L64-RG'], ['R'], ['L32-K4', 'L32-RG', 'L32-K4']):
        with pytest.raises(ValueError):
            release_arms(release(bad, {}))


def test_rfx_arms_and_etas_form_a_valid_release():
    def record(verdict, eta):
        return dict(verdict=dict(verdict=verdict, rescue={}), parameters=dict(eta=dict(raw=1., unit=eta, true=1e-6)))

    for r128, r32 in itertools.product(rfx.VERDICTS, repeat=2):
        decided = rfx_run.decisions({'R-r128': record(r128, 2e-6), 'R-r32': record(r32, 3e-3), 'W5-r32': record('storage', 1.)})
        arms = decided['lsq_arms']
        # U and K4/RG are exclusive per rank: a rank has a storage verdict, a normalization verdict or neither.
        assert arms == rfx.lsq_arms(decided['verdicts'])
        if not arms:
            continue
        schedule = dict(study='LSQ', E=16, arms=arms, steps_by_arm={arm: 15000 for arm in arms},
                        rfx_eta_by_rank=decided['rfx_eta_by_rank'], eta_by_rank=decided['eta_by_rank'])
        assert release_arms(schedule) == arms
        for arm in arms:
            assert scheduled_steps(arm, None, schedule) == 15000
            expected = lsq_eta(decided['rfx_eta_by_rank'][str(SPECS[arm][0])]) if SPECS[arm][1] == 'ridge' else None
            assert release_eta(schedule, arm) == expected
        assert {a for a in arms if SPECS[a][1] == 'ridge'} == {f'L{rank}-RG' for rank in (32, 128)
                                                                  if decided['verdicts'][f'R-r{rank}'] == 'storage'}


def test_ridge_strength_comes_from_the_release_and_must_be_the_clipped_rfx_value():
    schedule = dict(rfx_eta_by_rank={'32': 3e-3, '128': 2e-6}, eta_by_rank={'32': 3e-3, '128': 1e-4})
    assert release_eta(schedule, 'L32-RG') == 3e-3 and release_eta(schedule, 'L128-RG') == 1e-4
    for arm in ('L32-K4', 'L128-K4', 'L32-U', 'L128-U'):  # only the ridge arms have one
        assert release_eta(schedule, arm) is None
        assert release_eta(dict(arms=[arm]), arm) is None  # and a release without any eta field is fine for them
    for rank, eta in (('32', 4e-3), ('128', 2e-6), ('128', 1e-3)):
        with pytest.raises(ValueError):
            release_eta({**schedule, 'eta_by_rank': {**schedule['eta_by_rank'], rank: eta}}, f'L{rank}-RG')


def test_step_counts_come_from_the_release_or_the_explicit_smoke_count():
    counts = {'L32-K4': 20000, 'L32-RG': 19000}
    both = release(('L32-K4', 'L32-RG'), counts)
    assert [scheduled_steps(arm, None, both) for arm in ('L32-K4', 'L32-RG')] == [20000, 19000]
    assert scheduled_steps('L32-K4', 2400, None, smoke=True) == 2400 and scheduled_steps('L32-U', 2400, None, smoke=True) == 2400
    retrain = release(('L32-K4', 'L32-RG'), {'L32-RG': 15000}, E=8)  # the conditional E8 release names the retrained arm
    assert scheduled_steps('L32-RG', None, retrain, E=8) == 15000
    mixed = release(MIXED, {'L32-U': 21000, 'L128-K4': 18000, 'L128-RG': 17000})
    assert [scheduled_steps(arm, None, mixed) for arm in MIXED] == [21000, 18000, 17000]
    assert scheduled_steps('L32-U', None, release(MIXED, {'L32-U': 14000}, E=8), E=8) == 14000
    four = release(STORAGE, {**counts, 'L128-K4': 18000, 'L128-RG': 17000})
    assert scheduled_steps('L128-RG', None, four) == 17000
    for call in (lambda: scheduled_steps('L32-K4', 8225, both),
                 lambda: scheduled_steps('L32-K4', None, None),
                 lambda: scheduled_steps('L32-K4', None, both, E=8),
                 lambda: scheduled_steps('L32-K4', None, {**both, 'study': 'DPFP'}),
                 lambda: scheduled_steps('L32-K4', None, {**both, 'steps_by_arm': {**counts, 'L32-K4': 20001}}),
                 lambda: scheduled_steps('L32-K4', None, {**both, 'steps_by_arm': {**counts, 'L32-K4': True}}),
                 lambda: scheduled_steps('L32-K4', None, {**both, 'steps_by_arm': {'L32-K4': 20000}}),  # E16 names every arm
                 lambda: scheduled_steps('L32-U', None, {**mixed, 'steps_by_arm': {'L32-U': 21000}}),
                 lambda: scheduled_steps('L128-K4', None, both),
                 lambda: scheduled_steps('L32-U', None, both),
                 lambda: scheduled_steps('L32-K4', None, {**both, 'arms': ['L32-K4']}),
                 lambda: scheduled_steps('L32-K4', None, release(('L32-U', 'L32-K4', 'L32-RG'), counts)),
                 lambda: scheduled_steps('L32-K4', None, retrain, E=8),
                 lambda: scheduled_steps('L32-RG', None, release(('L32-K4', 'L32-RG'), counts, E=8), E=8),  # E8: one arm only
                 lambda: scheduled_steps('L128-K4', None, release(('L32-U',), {'L128-K4': 15000}, E=8), E=8),  # not an arm
                 lambda: scheduled_steps('L32-K4', 4, None, smoke=True),
                 lambda: scheduled_steps('L32-K4', None, None, smoke=True),
                 lambda: scheduled_steps('L32-K4', 2400, both, smoke=True)):
        with pytest.raises(ValueError):
            call()
