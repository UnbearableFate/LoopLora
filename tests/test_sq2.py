"""WP-L12 SQ2: D128-3 continued with delta or ridge storage on the DPFP features, with or without the coefficient loss
(docs/prereg/SQ2.md): the ridge store on DPFP features, the coefficient loss, the continuation, the numerics job and the runner.

Compute only (imports torch; tiny models, the registered rank 128 on a hidden-16 backbone). The real parent checkpoint is checked
when its folder exists (compute nodes see it).
"""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.core import memory
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm import dpfp, read_fix, sq2_numerics, sq2_release, sq2_run
from looplora.llm.dpfp import DPFPExperiment
from looplora.llm.inject import load_tokenizer
from looplora.llm.read_fix import (COEFFICIENT_EPS, ReadFixExperiment, coefficient_loss, coefficient_statistics, question_terms,
                                   store_read)
from looplora.llm.read_fix_run import fit
from looplora.llm.sq2 import SQ2Experiment, configuration, load_continuation, load_inference
from looplora.llm.sq2_release import (ARMS, ETA_BOUNDS, ETA_GRID, PARENT, SPECS, WEIGHT_BOUNDS, choose_eta, file_digest,
                                      prereg_digest, round_weight, scheduled_steps)
from looplora.llm.train import make_task_spec

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_BLAKE2B_128 = '9c21d638427ff1973536899b748a05d5'  # docs/prereg/SQ2.md at registration (847b17f)
ETA, WEIGHT = 1e-2, 2.0
DEVICES = ['cpu', 'cuda']


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'sq2-test-model'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield model
    assert not hasattr(model, 'looplora_adapters')


def tiny(arm, steps=10, smoke=False):
    """The registered SQ2 config of `arm` on the 2-layer test backbone (hook at layer 1)."""
    store, coefficient = SPECS[arm]
    cfg = configuration(arm, steps, eta=ETA if store == 'ridge' else None, w_c=WEIGHT if coefficient else None, smoke=smoke,
                        numerics_digest='n' * 32)
    cfg['model'].update(layer=1)
    return cfg


def data(cfg, count=2):
    return generate(make_task_spec(cfg), 'train', 0, count, derive_seed(17, 'sq2-fixture'))


@pytest.fixture(scope='module')
def trained_parent(model, tokenizer, tmp_path_factory):
    """A tiny 'D128-3' trained for 5 steps with a memory that matters."""
    folder = tmp_path_factory.mktemp('parent')
    cfg = dpfp.configuration('D128-3', 5)
    cfg['model'].update(layer=1)
    experiment = DPFPExperiment(cfg, 'D128-3', model=model, tokenizer=tokenizer)
    try:
        with torch.no_grad():
            experiment.adapter.B.fill_(.05)
        fit(experiment, folder, 5, evaluate=False)
        with torch.no_grad():
            logits, _ = experiment.score(experiment.prepare(data(cfg)), 1)
    finally:
        experiment.close()
    return dict(folder=folder, logits=logits, payload=torch.load(folder / 'checkpoint.pt', weights_only=True), cfg=cfg)


@pytest.fixture
def parent(trained_parent, monkeypatch):
    """The registration pointed at the tiny parent (config hash, step, digests) for one test."""
    folder = trained_parent['folder']
    monkeypatch.setitem(PARENT, 'steps', 5)
    monkeypatch.setitem(PARENT, 'config_hash', stable_hash(trained_parent['payload']['config']))
    monkeypatch.setitem(PARENT, 'digests', {name: file_digest(folder / name) for name in ('checkpoint.pt', 'writer.pt')})
    return trained_parent


# ------------------------------------------------------------------------------------------ registry, rules and configs

def test_the_registered_arms_parent_and_prereg_digest():
    assert ARMS == ('S-DE', 'S-DC', 'S-RE', 'S-RC')
    assert SPECS == {'S-DE': ('delta', False), 'S-DC': ('delta', True), 'S-RE': ('ridge', False), 'S-RC': ('ridge', True)}
    assert (PARENT['arm'], PARENT['steps'], PARENT['config_hash']) == ('D128-3', 22370, '2207a847e4967fb8e60d1663237fd979')
    assert PARENT['digests'] == {'checkpoint.pt': 'b6acff8b772a692311e630293add553f', 'writer.pt': '31d98c01c604c05a54bc20e4b4c4f436'}
    assert ETA_GRID == (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.) and ETA_BOUNDS == (1e-4, 1.) and WEIGHT_BOUNDS == (1 / 16, 16.)
    assert prereg_digest() == REGISTERED_BLAKE2B_128 and sq2_release.NUMERICS_BATCHES == 64 and sq2_release.N_MAX == 6000
    assert COEFFICIENT_EPS == 1e-6


def test_the_eta_rule_is_the_lowest_error_larger_on_ties_clipped_to_the_bounds():
    def curve(errors):
        return [dict(eta=eta, error=e) for eta, e in zip(ETA_GRID, errors)]

    assert choose_eta(curve([.9, .8, .7, .5, .3, .4, .6, .9])) == 1e-2
    assert choose_eta(curve([.9, .8, .3, .5, .3, .4, .6, .9])) == 1e-2  # ties: the larger eta
    assert choose_eta(curve([.1, .8, .7, .5, .3, .4, .6, .9])) == 1e-4  # clipped from 1e-6
    assert choose_eta(curve([.9, .8, .7, .5, .3, .4, .6, .05])) == 1.  # clipped from 10
    assert choose_eta(curve([.9, .8, .7, .5, .3, .4, .05, .9])) == 1.


def test_the_weight_rule_is_the_median_ratio_rounded_to_a_power_of_two_and_clipped():
    assert round_weight([3.]) == 4. and round_weight([1.4]) == 1. and round_weight([.7, 1., 100.]) == 1.
    assert round_weight([2., 4.]) == 4.  # the median of an even count is the mean (3), whose log2 rounds to 2
    assert round_weight([1e-4, 2e-4, 3e-4]) == WEIGHT_BOUNDS[0] and round_weight([1e5, 2e5, 3e5]) == WEIGHT_BOUNDS[1]
    # rounding is on the log2 scale (the ratio is multiplicative): the boundary between 4 and 8 is 4*sqrt(2) = 5.657
    assert round_weight([5.6]) == 4. and round_weight([5.7]) == 8.
    for bad in ([], [0.], [-1.], [math.nan], [math.inf], [1., math.nan]):
        with pytest.raises(ValueError):
            round_weight(bad)


def test_scheduled_steps_are_one_released_count_for_all_four_arms():
    release = dict(study='SQ2', E=16, steps_by_arm={arm: 6000 for arm in ARMS})
    assert [scheduled_steps(arm, None, release) for arm in ARMS] == [6000] * 4 and scheduled_steps('S-RC', 600, None, smoke=True) == 600
    for arm, steps, schedule, smoke in (
            ('S-DE', 6000, release, False), ('S-DE', None, None, False), ('S-DE', 600, release, True), ('S-DE', None, None, True),
            ('S-DE', 3, None, True), ('D128-3', None, release, False),
            ('S-DE', None, dict(release, steps_by_arm={**release['steps_by_arm'], 'S-RC': 5995}), False),  # one count, not four
            ('S-DE', None, dict(release, steps_by_arm={a: 6005 for a in ARMS}), False),  # beyond 6,000
            ('S-DE', None, dict(release, steps_by_arm={a: 5997 for a in ARMS}), False),  # not a multiple of 5
            ('S-DE', None, dict(release, steps_by_arm={a: True for a in ARMS}), False),
            ('S-DE', None, dict(release, steps_by_arm={'S-DE': 6000}), False),
            ('S-DE', None, dict(release, study='LSQT'), False), ('S-DE', None, dict(release, E=8), False)):
        with pytest.raises(ValueError):
            scheduled_steps(arm, steps, schedule, smoke=smoke)


@pytest.mark.parametrize('arm', ARMS)
def test_configuration_is_the_d128_3_config_plus_only_the_registered_changes(arm):
    store, coefficient = SPECS[arm]
    cfg = configuration(arm, 6000, eta=ETA if store == 'ridge' else None, w_c=WEIGHT if coefficient else None, numerics_digest='n' * 32)
    base = dpfp.configuration('D128-3', 6000)
    rest = copy.deepcopy(cfg)
    spec = rest.pop('sq2')
    assert spec['arm'] == arm and spec['steps'] == 6000 and spec['parent']['config_hash'] == PARENT['config_hash']
    assert spec['numerics'] == 'n' * 32 and rest.pop('name') == f'sq2-{arm}' and base.pop('name') == 'dpfp-D128-3-E16'
    for key in ('store', 'store_eta'):
        if store == 'ridge':
            assert rest['model'].pop(key) == {'store': 'ridge', 'store_eta': ETA}[key]
    if coefficient:
        assert rest['read_fix'].pop('coefficient_weight') == WEIGHT and rest['read_fix'].pop('coefficient_eps') == COEFFICIENT_EPS
    assert rest == base  # nothing else moves: the same task, optimizer, data size, evaluation, DPFP map and read objective
    assert (cfg['train']['steps'], cfg['model']['r'], cfg['model']['dpfp_nu'], cfg['task']['E'], cfg['seed']) == (6000, 128, 3, 16, 17)
    assert cfg['model'].get('unit_query', False) is False and cfg['eval']['episodes'] == 1000
    assert configuration(arm, 20, eta=ETA if store == 'ridge' else None, w_c=WEIGHT if coefficient else None,
                         smoke=True)['eval']['episodes'] == 100
    json.dumps(cfg, allow_nan=False)


def test_eta_and_the_weight_belong_to_exactly_their_arms_and_stay_in_bounds():
    for arm, kwargs in (('S-DE', dict(eta=ETA)), ('S-DE', dict(w_c=1.)), ('S-DC', dict(eta=ETA, w_c=1.)), ('S-DC', {}),
                        ('S-RE', {}), ('S-RE', dict(eta=ETA, w_c=1.)), ('S-RC', dict(eta=ETA)), ('S-RC', dict(w_c=1.)),
                        ('S-RE', dict(eta=1e-5)), ('S-RE', dict(eta=1.5)), ('S-RC', dict(eta=ETA, w_c=1 / 32)),
                        ('S-RC', dict(eta=ETA, w_c=32.)), ('S-DC', dict(w_c=0.)), ('R', {}), ('D128-3', {})):
        with pytest.raises(ValueError):
            configuration(arm, 10, **kwargs)
    assert configuration('S-RE', 10, eta=ETA_BOUNDS[0])['model']['store_eta'] == 1e-4
    assert configuration('S-RC', 10, eta=ETA_BOUNDS[1], w_c=WEIGHT_BOUNDS[1])['read_fix']['coefficient_weight'] == 16.
    assert 'coefficient_weight' not in configuration('S-DE', 10)['read_fix'] and 'store' not in configuration('S-DC', 10, w_c=1.)['model']


@pytest.mark.parametrize('arm, change', [
    ('S-DE', dict(model=dict(store='ridge', store_eta=ETA))), ('S-DE', dict(model=dict(unit_query=True))),
    ('S-DE', dict(read_fix=dict(coefficient_weight=1.))), ('S-RE', dict(model=dict(store_eta=.5))),
    ('S-RE', dict(model=dict(store='sweep'))), ('S-RC', dict(read_fix=dict(coefficient_weight=1.))), ('S-DC', dict(model=dict(dpfp_nu=2))),
    ('S-DC', dict(read_fix=dict(coefficient_weight=None)))])
def test_experiment_rejects_a_config_that_is_not_its_arm(model, tokenizer, arm, change):
    cfg = tiny(arm)
    for section, fields in change.items():
        for key, value in fields.items():
            if value is None:
                cfg[section].pop(key)
            else:
                cfg[section][key] = value
    with pytest.raises(ValueError):
        SQ2Experiment(cfg, arm, model=model, tokenizer=tokenizer)


def test_experiment_rejects_mixed_arms_and_other_studies_configs(model, tokenizer):
    with pytest.raises(ValueError):
        SQ2Experiment(tiny('S-RE'), 'S-RC', model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError):
        SQ2Experiment(tiny('S-DE'), 'D128-3', model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError):
        SQ2Experiment(dpfp.configuration('D128-3', 10), 'S-DE', model=model, tokenizer=tokenizer)  # a DPFP config has no sq2 section
    cfg = tiny('S-RC')
    cfg['sq2']['eta'] = None
    with pytest.raises(ValueError):
        SQ2Experiment(cfg, 'S-RC', model=model, tokenizer=tokenizer)
    cfg = tiny('S-DE')
    cfg['dpfp']['arm'] = 'D32-3'
    with pytest.raises(ValueError):
        SQ2Experiment(cfg, 'S-DE', model=model, tokenizer=tokenizer)


# ------------------------------------------------------------------------------------ the ridge store on the DPFP features

def explicit_features(u, K, V, nu, eps=1e-6):
    raw = memory.dpfp_features(F.linear(u, K), nu)
    return F.normalize(raw, dim=-1, eps=eps), F.linear(u, V)


def dual_ridge(keys, values, lam, eta):
    """M = V^T W K (K^T W K + eta I)^-1 over the unit feature keys, written in the n x n dual form as the store solves it."""
    n = keys.shape[1]
    weights = lam ** torch.arange(n - 1, -1, -1, dtype=keys.dtype, device=keys.device)
    gram = torch.einsum('bnj,bmj->bnm', keys, keys)
    system = torch.diag(1 / weights) * eta + gram
    # M = V^T (G + eta W^-1)^-1 K   (push-through identity of the primal ridge solution)
    return torch.einsum('bni,bnm,bmj->bij', values, torch.linalg.inv(system), keys)


@pytest.mark.parametrize('device', DEVICES)
def test_ridge_final_on_dpfp_features_is_the_ridge_solution_over_the_feature_keys(device):
    b, n, r, rm, nu = 3, 7, 5, 4, 3
    g = torch.Generator(device=device).manual_seed(derive_seed(17, 'sq2', 'ridge-dpfp'))
    u = torch.randn(b, n, r, generator=g, device=device, dtype=torch.float64)
    K = torch.randn(rm, r, generator=g, device=device, dtype=torch.float64) * .7
    V = torch.randn(rm, r, generator=g, device=device, dtype=torch.float64) * .7
    keys, values = explicit_features(u, K, V, nu)
    width = 2 * rm * nu
    assert keys.shape == (b, n, width)
    zeros = torch.zeros(b, rm, width, dtype=torch.float64, device=device)
    lam, eta = .9, .03
    state = memory.ridge_final(zeros, u, None, None, None, K=K, V=V, lam=lam, eta=eta,
                               feature_map=lambda raw: memory.dpfp_features(raw, nu))
    assert state.shape == (b, rm, width)  # rectangular: the solve runs through the n x n system, not a width x width one
    torch.testing.assert_close(state, dual_ridge(keys, values, lam, eta), rtol=1e-9, atol=1e-9)
    # the primal definition agrees: minimize sum_j w_j |M k_j - v_j|^2 + eta |M|^2
    weights = lam ** torch.arange(n - 1, -1, -1, dtype=torch.float64, device=device)
    system = eta * torch.eye(width, dtype=torch.float64, device=device) + torch.einsum('n,bnj,bnk->bjk', weights, keys, keys)
    primal = torch.einsum('n,bni,bnj->bij', weights, values, keys) @ torch.linalg.inv(system)
    torch.testing.assert_close(state, primal, rtol=1e-8, atol=1e-8)
    # with eta -> 0 the independent feature keys are interpolated: M k_j = v_j for every j, whatever lam
    small = memory.ridge_final(zeros, u, None, None, None, K=K, V=V, lam=.8, eta=1e-10, feature_map=lambda raw: memory.dpfp_features(raw, nu))
    torch.testing.assert_close(torch.einsum('bij,bnj->bni', small, keys), values, rtol=1e-5, atol=1e-5)
    # a state of the feature width needs the map; without it the plain ridge store refuses
    with pytest.raises(ValueError):
        memory.ridge_final(zeros, u, None, None, None, K=K, V=V, lam=lam, eta=eta)


@pytest.mark.parametrize('device', DEVICES)
def test_ridge_on_dpfp_is_differentiable_and_ignores_the_write_strength(device):
    cfg = AdapterConfig(d_in=6, d_out=6, r=5, r_mem=4, variant='iterq', store='ridge', store_eta=.05, dpfp_nu=3)
    adapter = make_adapter(cfg).double().to(device)
    u = torch.randn(2, 6, 6, dtype=torch.float64, device=device)
    zeros = torch.zeros(2, 4, cfg.key_dim, dtype=torch.float64, device=device)
    state = adapter.scan_final(zeros, u, None, None, None)
    assert state.shape == (2, 4, 24) and bool(torch.isfinite(state).all())
    state.square().sum().backward()
    assert adapter.K.grad.abs().sum() > 0 and adapter.V.grad.abs().sum() > 0 and adapter.a_lambda.grad.abs().sum() > 0
    assert adapter.a_beta.grad is None or float(adapter.a_beta.grad.abs().sum()) == 0.


def test_the_off_paths_of_the_ridge_store_and_the_config_guard_are_unchanged():
    plain = AdapterConfig(d_in=4, d_out=4, r=3, store='ridge', store_eta=.1, gate='scalar')
    assert plain.key_dim == 3 and plain.dpfp_nu == 0
    for bad in (dict(store='sweep', dpfp_nu=3), dict(store='ridge', dpfp_nu=3, unit_query=True), dict(store='ridge', dpfp_nu=3, gate='input')):
        with pytest.raises(ValueError):
            AdapterConfig(d_in=4, d_out=4, r=3, **bad)


# ---------------------------------------------------------------------------------------------- the coefficient loss

class Stub:
    """read_fix.coefficient_loss on a chosen coefficient tensor d: the store read and the question terms are replaced."""
    def __init__(self, monkeypatch, d, target):
        monkeypatch.setattr(read_fix, 'question_terms', lambda experiment, batch, x: dict(q=None, target=target))
        monkeypatch.setattr(read_fix, 'store_read', lambda adapter, keys, values, q: dict(d=d))
        self.adapter = None


BATCH = SimpleNamespace(keys=None, values=None)


def test_coefficient_loss_is_the_direction_error_of_the_exact_read_coefficients(monkeypatch):
    target = torch.tensor([[2, 0], [1, 3]])
    basis = F.one_hot(target, 5).double()
    stub = Stub(monkeypatch, 7. * basis, target)
    assert float(coefficient_loss(stub, BATCH, None)) < 1e-12  # d proportional to e_t: any scale is right
    stub = Stub(monkeypatch, 1e-3 * basis, target)
    assert float(coefficient_loss(stub, BATCH, None)) == pytest.approx(1e-6, rel=.01)  # a small read: eps shrinks the direction by 1e-3
    stub = Stub(monkeypatch, -3. * basis, target)
    assert float(coefficient_loss(stub, BATCH, None)) == pytest.approx(4., rel=1e-6)  # the opposite direction
    d = torch.randn(2, 2, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    expected = ((d / (d.norm(dim=-1, keepdim=True) + 1e-6) - basis) ** 2).sum(-1).mean()
    stub = Stub(monkeypatch, d, target)
    assert float(coefficient_loss(stub, BATCH, None)) == pytest.approx(float(expected), rel=1e-12)
    assert float(coefficient_loss(Stub(monkeypatch, 100. * d, target), BATCH, None)) == pytest.approx(float(expected), rel=1e-5)
    stub = Stub(monkeypatch, torch.zeros(2, 2, 5, dtype=torch.float64), target)
    assert float(coefficient_loss(stub, BATCH, None)) == pytest.approx(1., rel=1e-12)  # no read at all: |0 - e_t|^2


def ridge_coefficients_by_hand(keys, lam, eta, q):
    """d_i = w_i k_i^T (K^T W K + eta I)^-1 q: the primal form, per episode."""
    out = []
    for k, qq in zip(keys.double(), q.double()):
        n = k.shape[0]
        w = lam ** torch.arange(n - 1, -1, -1, dtype=torch.float64, device=k.device)
        system = eta * torch.eye(k.shape[1], dtype=torch.float64, device=k.device) + (k * w[:, None]).T @ k
        out.append(w[None, :] * (qq @ torch.linalg.inv(system) @ k.T))
    return torch.stack(out)


@pytest.mark.parametrize('arm', ARMS)
def test_the_arms_coefficients_loss_statistics_and_gradients_on_the_tiny_model(model, tokenizer, arm):
    store, coefficient = SPECS[arm]
    cfg = tiny(arm)
    experiment = SQ2Experiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        adapter = experiment.adapter
        assert (adapter.cfg.store, adapter.cfg.dpfp_nu, adapter.cfg.unit_query) == (store, 3, False)
        signature = experiment.writer_signature()
        assert signature['read_fix'] == arm and signature.get('store', 'delta') == store and signature['dpfp_nu'] == 3
        with torch.no_grad():
            adapter.B.fill_(.05)
        batch = experiment.prepare(data(cfg))
        assert batch.snapshot.states[1].shape == (2, 128, 768) and batch.keys.shape == (2, 16, 768)
        torch.testing.assert_close(batch.keys.norm(dim=-1), torch.ones(2, 16, device=batch.keys.device), rtol=0, atol=1e-5)
        logits, x = experiment.score(batch, 1, capture_inputs=True)
        terms = question_terms(experiment, batch, x)
        torch.testing.assert_close(terms['q'].norm(dim=-1), torch.ones_like(terms['q'].norm(dim=-1)), rtol=0, atol=1e-5)  # unit DPFP query
        if store == 'ridge':  # the deployed state is the ridge solution over the DPFP keys of this batch
            torch.testing.assert_close(batch.snapshot.states[1].detach().double(),
                                       dual_ridge(batch.keys.double(), batch.values.double(), float(adapter.lam), ETA), rtol=1e-4, atol=1e-5)
        exact = store_read(adapter, batch.keys, batch.values, terms['q'])
        d = exact['d']
        assert d.shape == (2, 4, 16)
        torch.testing.assert_close(exact['reconstructed'].detach(), terms['stored'].double(), rtol=1e-4, atol=1e-4)  # d reproduces the read
        if store == 'ridge':
            hand = ridge_coefficients_by_hand(batch.keys, float(adapter.lam), ETA, terms['q'])
            torch.testing.assert_close(d.detach(), hand, rtol=1e-6, atol=1e-8)
        target = F.one_hot(terms['target'], 16).double()
        by_hand = ((d / (d.norm(dim=-1, keepdim=True) + 1e-6) - target) ** 2).sum(-1).mean()
        loss = coefficient_loss(experiment, batch, x)
        assert float(loss) == pytest.approx(float(by_hand), rel=1e-12) and 0 < float(loss) < 4.0001
        weighted = experiment.coefficient_loss(batch, x)
        if coefficient:
            assert float(weighted) == pytest.approx(WEIGHT * float(loss), rel=1e-12)
            grads = torch.autograd.grad(weighted, [adapter.Q, adapter.K, adapter.a_lambda], retain_graph=True)  # d does not depend on V
            assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0
        else:
            assert weighted is None  # no weight in the config: the objective is the parent's, term for term
        stats = coefficient_statistics(experiment, batch, x)
        assert set(stats) == {'query_norm', 'direction_cosine', 'coefficient_cancellation_median', 'key_gram_log10_cond_median'} | (
            {'ridge_log10_cond_median'} if store == 'ridge' else set())
        assert all(math.isfinite(v) for v in stats.values()) and stats['query_norm'] == pytest.approx(1., abs=1e-4)
        assert -1.0001 <= stats['direction_cosine'] <= 1.0001 and stats['coefficient_cancellation_median'] >= 1 - 1e-6
        experiment.observe(batch, x)
        assert experiment.log_extras() == stats
    finally:
        experiment.close()


def test_the_coefficient_term_is_off_for_every_other_study(model, tokenizer):
    cfg = read_fix.configuration('R', 10)
    cfg['model'].update(layer=1, r=4)
    experiment = ReadFixExperiment(cfg, 'R', model=model, tokenizer=tokenizer)
    try:
        batch = experiment.prepare(data(cfg))
        _, x = experiment.score(batch, 1, capture_inputs=True)
        assert experiment.coefficient_loss(batch, x) is None
        experiment.config['read_fix']['coefficient_weight'] = 0.
        assert experiment.coefficient_loss(batch, x) is None
        assert not hasattr(experiment, 'observe') and not hasattr(experiment, 'log_extras')
    finally:
        experiment.close()


# ---------------------------------------------------------------------------------------------------- the continuation

def test_load_continuation_reproduces_the_parent_at_step_zero_and_returns_its_optimizer_state(model, tokenizer, parent):
    cfg = tiny('S-DE')
    experiment = SQ2Experiment(cfg, 'S-DE', model=model, tokenizer=tokenizer)
    try:
        state = load_continuation(experiment, parent['folder'])
        with torch.no_grad():
            actual, _ = experiment.score(experiment.prepare(data(cfg)), 1)
        assert torch.equal(actual, parent['logits'])  # the continued control at step 0 is the parent
        assert state['param_groups'] and all(float(s['step']) == 5 for s in state['state'].values())
        assert state['state'].keys() == parent['payload']['optimizer']['state'].keys()
        assert experiment.value_head is not None
        for name, value in parent['payload']['adapter'].items():
            assert torch.equal(experiment.adapter.state_dict()[name].cpu(), value)
        for name, value in parent['payload']['value_head'].items():
            assert torch.equal(experiment.value_head.state_dict()[name].cpu(), value)
    finally:
        experiment.close()


def test_load_continuation_refuses_another_checkpoint(model, tokenizer, parent, tmp_path, monkeypatch):
    experiment = SQ2Experiment(tiny('S-DE'), 'S-DE', model=model, tokenizer=tokenizer)
    try:
        folder = parent['folder']
        for name in ('checkpoint.pt', 'writer.pt'):  # a byte changed in a file of the registered digests
            copy_folder = tmp_path / f'tampered-{name}'
            copy_folder.mkdir()
            for other in ('checkpoint.pt', 'writer.pt'):
                (copy_folder / other).write_bytes((folder / other).read_bytes())
            with open(copy_folder / name, 'ab') as handle:
                handle.write(b'\0')
            with pytest.raises(ValueError, match='blake2b-128'):
                load_continuation(experiment, copy_folder)
        for key, value in (('steps', 6), ('config_hash', '0' * 32), ('arm', 'D32-3')):
            with monkeypatch.context() as patch:
                patch.setitem(PARENT, key, value)
                with pytest.raises(ValueError, match='not the registered D128-3 final checkpoint'):
                    load_continuation(experiment, folder)
        headless = dict(parent['payload'], value_head=None)
        torch.save(headless, tmp_path / 'checkpoint.pt')
        with monkeypatch.context() as patch:
            patch.setitem(PARENT, 'digests', {'checkpoint.pt': file_digest(tmp_path / 'checkpoint.pt')})
            with pytest.raises(ValueError, match='value head'):
                load_continuation(experiment, tmp_path)
    finally:
        experiment.close()


@pytest.mark.parametrize('arm', ARMS)
def test_the_continuation_trains_on_a_fresh_stream_with_constant_lr_and_the_parents_optimizer(model, tokenizer, parent, tmp_path, arm):
    cfg = tiny(arm, steps=5, smoke=True)
    experiment = SQ2Experiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        state = load_continuation(experiment, parent['folder'])
        result = fit(experiment, tmp_path, 5, data_seed=sq2_run.DATA_SEED, optimizer_state=state, warmup=0, time_limit=3400)
        assert result['actual_steps'] == 5 and result['data_seed'] == sq2_run.DATA_SEED
        assert sq2_run.DATA_SEED == derive_seed(17, 'llm', 'sq2-continuation') != derive_seed(17, 'llm', 'writer-data')
        rows = [json.loads(line) for line in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == [1, 5] and all(row['lr'] == .003 and row['data_seed'] == sq2_run.DATA_SEED for row in rows)
        assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in rows)
        assert all(('coefficient_loss' in row) == SPECS[arm][1] for row in rows)
        assert all(set(row) >= {'query_norm', 'direction_cosine', 'coefficient_cancellation_median'} for row in rows)  # the observe hook
        saved = torch.load(tmp_path / 'checkpoint.pt', weights_only=True)
        assert all(float(s['step']) == 10 for s in saved['optimizer']['state'].values())  # Adam's counter continues from the parent's 5
    finally:
        experiment.close()


def test_the_default_training_loop_is_unchanged(model, tokenizer, tmp_path):
    cfg = read_fix.configuration('R', 5, smoke=True)
    cfg['model'].update(layer=1, r=4)
    experiment = ReadFixExperiment(cfg, 'R', model=model, tokenizer=tokenizer)
    try:
        fit(experiment, tmp_path, 5, evaluate=False)
        rows = [json.loads(line) for line in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
        assert rows[0]['lr'] == pytest.approx(.003 / 20) and rows[1]['lr'] == pytest.approx(.003 * 5 / 20)  # 20-step warmup
        assert rows[0]['data_seed'] == derive_seed(17, 'llm', 'writer-data') and 'coefficient_loss' not in rows[0]
    finally:
        experiment.close()


# -------------------------------------------------------------------------------------------------- the numerics job

def test_numerics_are_training_only_and_follow_the_registered_rules(model, tokenizer, parent, monkeypatch, tmp_path):
    real = sq2_numerics.configuration

    def on_tiny(*args, **kwargs):
        cfg = real(*args, **kwargs)
        cfg['model'].update(layer=1)
        return cfg

    monkeypatch.setattr(sq2_numerics, 'configuration', on_tiny)
    result = sq2_numerics.numerics(count=2, model=model, tokenizer=tokenizer, folder=parent['folder'])
    assert result['split'] == 'train' and result['seed_labels'] == ['sq2', 'numerics'] and result['batches'] == 2 and result['episodes'] == 64
    assert [row['eta'] for row in result['eta_curve']] == list(ETA_GRID) and all(math.isfinite(row['error']) for row in result['eta_curve'])
    assert result['eta'] == choose_eta(result['eta_curve']) and ETA_BOUNDS[0] <= result['eta'] <= ETA_BOUNDS[1]
    assert result['w_c'] == round_weight(result['gradient_ratio']['delta_store']) and result['w_c'] in {2.0 ** k for k in range(-4, 5)}
    ratios = result['gradient_ratio']['delta_store']
    assert len(ratios) == 2 and all(math.isfinite(r) and r > 0 for r in ratios)
    assert result['gradient_ratio']['min'] == min(ratios) and result['gradient_ratio']['max'] == max(ratios)
    assert math.isfinite(result['ridge_store_gradient_ratio_median']) and math.isfinite(result['deployed_read_error_delta_store'])
    for key in ('statistics_delta_store', 'statistics_ridge_store'):
        assert all(math.isfinite(v) for v in result[key].values())
    assert 'ridge_log10_cond_median' in result['statistics_ridge_store'] and 'ridge_log10_cond_median' not in result['statistics_delta_store']
    json.dumps(result, allow_nan=False)
    # deterministic: the same stream and checkpoint give the same decisions
    again = sq2_numerics.numerics(count=2, model=model, tokenizer=tokenizer, folder=parent['folder'])
    assert (again['eta'], again['w_c']) == (result['eta'], result['w_c'])
    assert [r['error'] for r in again['eta_curve']] == pytest.approx([r['error'] for r in result['eta_curve']], rel=1e-5)
    # the numerics stream is not the continuation's and not D128-3's
    assert derive_seed(17, *sq2_numerics.SEED_LABELS) not in (sq2_run.DATA_SEED, derive_seed(17, 'llm', 'writer-data'))
    # the numerics file's digest and values are what the runner reads
    path = tmp_path / 'numerics.json'
    path.write_text(json.dumps(result))
    values = sq2_run.numerics_values(path)
    assert values == dict(eta=result['eta'], w_c=result['w_c'], digest=file_digest(path))


# ------------------------------------------------------------------------------------------------------- the runner

def runner(monkeypatch, model, tokenizer, parent):
    real = sq2_run.configuration
    patched = lambda *args, **kwargs: {**real(*args, **kwargs), 'model': {**real(*args, **kwargs)['model'], 'layer': 1}}
    monkeypatch.setattr(sq2_run, 'configuration', patched)
    monkeypatch.setattr(sq2_run, 'SQ2Experiment', lambda cfg, arm: SQ2Experiment(cfg, arm, model=model, tokenizer=tokenizer))
    monkeypatch.setattr(sq2_run, 'source_metadata', lambda: dict(git_commit='f' * 40, source_dirty=False, source_hash='0' * 32))
    return dict(eta=ETA, w_c=WEIGHT, digest='n' * 32)


@pytest.mark.parametrize('arm', ['S-DE', 'S-RC'])
def test_run_one_evaluates_step_zero_trains_and_exports_a_loadable_writer(model, tokenizer, parent, monkeypatch, tmp_path, arm):
    numerics = runner(monkeypatch, model, tokenizer, parent)
    record = sq2_run.run_one(arm, tmp_path / arm, 5, numerics, smoke=True, parent_folder=parent['folder'])
    store, coefficient = SPECS[arm]
    assert record['status'] == 'complete' and record['arm'] == arm and record['store'] == store and record['dpfp_nu'] == 3
    assert record['store_eta'] == (ETA if store == 'ridge' else None) and record['coefficient_weight'] == (WEIGHT if coefficient else None)
    assert record['config']['sq2']['numerics'] == 'n' * 32 and record['parent']['config_hash'] == PARENT['config_hash']
    assert record['data_seed'] == sq2_run.DATA_SEED and record['training']['data_seed'] == sq2_run.DATA_SEED
    assert record['step0']['n'] == 100 and (tmp_path / arm / 'val-0' / 'samples.npz').exists()
    assert sorted(record['training']['metrics']) == ['1', '2', '3', '4', '5'] and record['memory']['shape'] == [128, 768]
    assert record['inference_parameters'] > 0 and record['training_only_parameters'] > 0
    reloaded = load_inference(tmp_path / arm / 'writer.pt', model=model, tokenizer=tokenizer)
    try:
        assert reloaded.read_arm == arm and reloaded.value_head is None and reloaded.adapter.cfg.store == store
        with torch.no_grad():
            actual, _ = reloaded.score(reloaded.prepare(data(reloaded.config)), 1)
        assert torch.isfinite(actual).all()
    finally:
        reloaded.close()
    with pytest.raises(ValueError):  # the continuation's loader accepts nothing but its own export
        load_inference(tmp_path / arm / 'checkpoint.pt', model=model, tokenizer=tokenizer)
    saved = json.loads((tmp_path / arm / 'record.json').read_text())
    assert saved['status'] == 'complete' and saved['config_hash'] == record['config_hash']


def test_run_one_refuses_a_config_that_differs_from_the_frozen_release_and_records_the_failure(model, tokenizer, parent, monkeypatch, tmp_path):
    numerics = runner(monkeypatch, model, tokenizer, parent)
    with pytest.raises(ValueError, match='frozen release'):
        sq2_run.run_one('S-DE', tmp_path / 'a', 5, numerics, smoke=True, expected_hash='0' * 32, parent_folder=parent['folder'])
    bad = dict(numerics, eta=.5)
    with pytest.raises(ValueError):  # a ridge arm whose eta is not the frozen one has another config hash
        cfg_hash = stable_hash(sq2_run.configuration('S-RE', 5, eta=ETA, w_c=None, smoke=True, numerics_digest='n' * 32))
        sq2_run.run_one('S-RE', tmp_path / 'b', 5, bad, smoke=True, expected_hash=cfg_hash, parent_folder=parent['folder'])
    with pytest.raises(FileExistsError):
        sq2_run.run_one('S-DE', tmp_path / 'a', 5, numerics, smoke=True, parent_folder=parent['folder'])
    with pytest.raises(FileNotFoundError):
        sq2_run.run_one('S-DE', tmp_path / 'c', 5, numerics, smoke=True, parent_folder=tmp_path / 'absent')
    failed = json.loads((tmp_path / 'c' / 'record.json').read_text())
    assert failed['status'] == 'failed' and 'FileNotFoundError' in failed['failure_reason']


# ------------------------------------------------------------------------------------------- the real parent checkpoint

REAL = Path('/work/xg24i002/x10041/LoopLora/runs/p1l/dpfp/arms/D128-3')


@pytest.mark.skipif(not (REAL / 'checkpoint.pt').exists(), reason='the parent checkpoint lives on the cluster filesystem')
def test_the_registered_parent_checkpoint_and_the_s_de_config(monkeypatch):
    assert PARENT['folder'] == str(REAL)
    for name, digest in PARENT['digests'].items():
        assert file_digest(REAL / name) == digest
    payload = torch.load(REAL / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert payload['arm'] == 'D128-3' and payload['step'] == PARENT['steps'] and stable_hash(payload['config']) == PARENT['config_hash']
    assert payload['value_head'] is not None and payload['optimizer']['state']
    # S-DE at the parent's length differs from the parent's config in nothing but its name and the sq2 section
    cfg = json.loads(json.dumps(configuration('S-DE', PARENT['steps'])))
    assert cfg.pop('sq2')['arm'] == 'S-DE' and cfg.pop('name') == 'sq2-S-DE'
    expected = json.loads(json.dumps(payload['config']))
    expected.pop('name')
    assert cfg == expected
    record = json.loads((ROOT / PARENT['record']).read_text())
    assert record['config_hash'] == PARENT['config_hash'] and record['training']['metrics'][str(PARENT['steps'])]['correct'] == 608
    assert record['training']['metrics'][str(PARENT['steps'])]['nll'] == 2.1291537284851074
