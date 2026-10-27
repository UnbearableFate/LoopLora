"""WP-L7 RFX algebra: the ridge store, gain and eta fitting, the refit decoder, write orders and the registered verdict.

Synthetic cached tensors only; the instrumented LLM forward is tested in test_llm_rfx.py. Compute-only like the rest.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from looplora.core import memory
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm import read_fix, rfx
from looplora.llm.train import make_task_spec

DTYPE = torch.float64


def gen(*label):
    return torch.Generator().manual_seed(derive_seed(17, 'rfx-test', *label))


def randn(g, *shape, scale=1.):
    return torch.randn(*shape, generator=g, dtype=DTYPE) * scale


def unit(x):
    return F.normalize(x, dim=-1, eps=1e-6)


def test_ridge_reads_are_V_Kt_K_Kt_eta_I_inverse_applied_to_the_query():
    g = gen('ridge')
    b, n, r, queries, eta = 3, 7, 6, 4, .05
    keys, values, q = unit(randn(g, b, n, r)), randn(g, b, n, r), randn(g, b, queries, r)
    single = rfx.ridge_reads(keys.float(), values.float(), q.float(), eta)  # float32 inputs are read in float64
    assert single.dtype == DTYPE and single.shape == (b, queries, r)
    # The task's formula with the keys as columns of K (r x n) and the values as columns of V (r x n).
    K, V = keys.transpose(1, 2), values.transpose(1, 2)
    M = V @ K.transpose(1, 2) @ torch.linalg.inv(K @ K.transpose(1, 2) + eta * torch.eye(r, dtype=DTYPE))
    exact = rfx.ridge_reads(keys, values, q, eta)
    torch.testing.assert_close(exact, torch.einsum('bij,bqj->bqi', M, q), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(single, exact, rtol=1e-3, atol=1e-3)


def test_ridge_reads_interpolate_independent_keys_and_equal_the_lsq_store_at_unit_decay():
    g = gen('ridge-lsq')
    b, n, r = 3, 6, 8
    u, K, V = randn(g, b, n, r), randn(g, r, r), randn(g, r, r)
    keys, values = unit(F.linear(u, K)), F.linear(u, V)
    stored = rfx.ridge_reads(keys, values, keys, 1e-12)
    torch.testing.assert_close(stored, values, rtol=0, atol=1e-6)
    q, eta = randn(g, b, 5, r), .03
    state = memory.ridge_final(torch.zeros(b, r, r, dtype=DTYPE), u, None, None, None, K=K, V=V, lam=1., eta=eta)
    torch.testing.assert_close(rfx.ridge_reads(keys, values, q, eta), torch.einsum('bij,bqj->bqi', state, q),
                               rtol=1e-9, atol=1e-9)
    # Ridge is order free: any permutation of the written pairs gives the same read.
    order = torch.randperm(n, generator=g)
    torch.testing.assert_close(rfx.ridge_reads(keys[:, order], values[:, order], q, eta),
                               rfx.ridge_reads(keys, values, q, eta), rtol=1e-10, atol=1e-10)


def test_normalized_error_is_the_wp_l4_relative_read_error():
    g = gen('error')
    target, read = randn(g, 4, 3, 5), randn(g, 4, 3, 5)
    expected = (read - target).square().sum(-1) / (target.square().sum(-1) + 1e-6)
    assert torch.equal(rfx.normalized_error(read, target), expected)
    assert (rfx.normalized_error(target, target) == 0).all()
    torch.testing.assert_close(rfx.normalized_error(torch.zeros_like(target), target),
                               target.square().sum(-1) / (target.square().sum(-1) + 1e-6))
    assert rfx.normalized_error(torch.zeros(1, 1, 3, dtype=DTYPE), torch.zeros(1, 1, 3, dtype=DTYPE)) == 0


def test_the_fitted_gain_minimizes_the_mean_normalized_error():
    g = gen('gain')
    target = randn(g, 40, 3, 6)
    read = .4 * target + randn(g, 40, 3, 6, scale=.5)
    gain = rfx.fit_gain(read, target)
    mean_error = lambda c: float(rfx.normalized_error(c * read, target).mean())
    for step in (-.05, -.01, .01, .05):
        assert mean_error(gain) < mean_error(gain + step)
    assert abs((mean_error(gain + 1e-4) - mean_error(gain - 1e-4)) / 2e-4) < 1e-6  # stationary
    moments = rfx.Moments()
    moments.add(read[:15], target[:15])
    moments.add(read[15:], target[15:])
    fitted, error = moments.gain_error()
    assert math.isclose(fitted, gain, rel_tol=1e-12) and math.isclose(error, mean_error(gain), rel_tol=1e-10)
    scaled = rfx.Moments()
    scaled.add(3.7 * read, target)
    assert math.isclose(scaled.gain_error()[0], gain / 3.7, rel_tol=1e-12)
    zero = rfx.Moments()
    zero.add(torch.zeros_like(target), target)
    assert zero.gain_error()[0] == 1. and zero.gain_error()[1] > .99  # a zero read has no direction to scale


def cached(g, N=48, E=5, Q=3, r=8):
    """Synthetic cached observations: E < r keys, raw q near the true key with its own scale."""
    keys, values = unit(randn(g, N, E, r)), randn(g, N, E, r)
    target = torch.randint(0, E, (N, Q), generator=g)
    pick = lambda x: x.gather(1, target[..., None].expand(-1, -1, r))
    true_key, clean = pick(keys), pick(values)
    q = 3. * true_key + randn(g, N, Q, r, scale=.4)
    return dict(keys=keys, values=values, q=q, true_key=true_key, clean=clean, target=target,
                delta=dict(raw=2. * clean, unit=.5 * clean + randn(g, N, Q, r, scale=.3), true=4. * clean))


def test_fit_parameters_choose_eta_and_gains_on_the_fitting_set_only():
    fit = cached(gen('fit-parameters'))
    params = rfx.fit_parameters(fit, chunk=10)
    assert params['gain']['delta']['raw'] == pytest.approx(.5, abs=1e-12)
    assert params['gain']['delta']['true'] == pytest.approx(.25, abs=1e-12)
    assert set(params['eta']) == set(rfx.QUERIES) and all(e in rfx.ETA_GRID for e in params['eta'].values())
    # The true key of independent-enough keys is reproduced exactly by the nearly unregularized ridge read.
    curve = params['curves']['true']
    assert [row['eta'] for row in curve] == list(rfx.ETA_GRID)
    assert params['eta']['true'] == rfx.ETA_GRID[0]
    assert curve[0]['error'] < 1e-6 < curve[-3]['error']
    assert params['gain']['ridge']['true'] == pytest.approx(1., abs=1e-2)
    # The selection is the argmin of the fitted-gain error (the larger eta wins a tie) with that eta's own gain.
    for query in rfx.QUERIES:
        best = min(params['curves'][query], key=lambda row: (row['error'], -row['eta']))
        assert (best['eta'], best['gain']) == (params['eta'][query], params['gain']['ridge'][query])
    # Chunking does not change the fit.
    assert rfx.fit_parameters(fit, chunk=1000)['eta'] == params['eta']
    json_ready = {k: params[k] for k in ('eta', 'gain', 'curves')}
    assert all(math.isfinite(row['error']) and math.isfinite(row['gain']) for c in json_ready['curves'].values() for row in c)


def test_variant_reads_cover_the_factorial_with_and_without_the_fitted_gain():
    fit = cached(gen('variants'))
    params = rfx.fit_parameters(fit)
    obs = rfx.slice_obs(fit, slice(0, 6))
    reads = rfx.variant_reads(obs, params)
    assert tuple(reads) == rfx.VARIANTS and len(rfx.VARIANTS) == 14 and len(rfx.REFIT_CELLS) == 8
    assert all(read.shape == obs['clean'].shape and read.dtype == DTYPE for read in reads.values())
    for store, query in rfx.CELLS:
        asis, fitted = reads[f'{store}/{query}/asis'], reads[f'{store}/{query}/fit']
        torch.testing.assert_close(fitted, params['gain'][store][query] * asis)
    assert torch.equal(reads['delta/raw/asis'], obs['delta']['raw'].double())
    assert torch.equal(reads['clean'], obs['clean'].double()) and not reads['zero'].any()
    expected = rfx.ridge_reads(obs['keys'], obs['values'], obs['true_key'], params['eta']['true'])
    torch.testing.assert_close(reads['ridge/true/asis'], expected)
    unit_q = rfx.ridge_reads(obs['keys'], obs['values'], unit(obs['q']), params['eta']['unit'])
    torch.testing.assert_close(reads['ridge/unit/asis'], unit_q)
    cells = rfx.cell_reads(fit, params, chunk=7)
    assert tuple(cells) == rfx.REFIT_CELLS and all(c.shape == (48 * 3, 8) for c in cells.values())
    for name, value in rfx.cell_reads(fit, params, chunk=1000).items():
        torch.testing.assert_close(cells[name], value)
    assert not cells['zero'].any()


def test_refit_decoder_is_scale_free_and_reads_a_linearly_decodable_signal():
    g = gen('decoder')
    classes, d = 5, 6
    centres = randn(g, classes, d, scale=2.)
    labels = torch.randint(0, classes, (800,), generator=g)
    x = centres[labels] + randn(g, 800, d, scale=.5)
    decoder = rfx.fit_decoder(x[:600], labels[:600], classes)
    assert decoder['decay'] in rfx.DECODER_DECAYS
    score = rfx.decoder_scores(decoder, x[600:], labels[600:])
    assert score['acc'] > .9 and math.isfinite(score['nll'])
    rescaled = rfx.fit_decoder(3.7 * x[:600] + 11., labels[:600], classes)
    other = rfx.decoder_scores(rescaled, 3.7 * x[600:] + 11., labels[600:])
    assert other['acc'] == score['acc'] and other['nll'] == pytest.approx(score['nll'], abs=1e-6)
    # No signal: constant features can only learn the label frequencies.
    blind = rfx.fit_decoder(torch.zeros(600, d, dtype=DTYPE), labels[:600], classes)
    chance = rfx.decoder_scores(blind, torch.zeros(200, d, dtype=DTYPE), labels[600:])
    majority = int(torch.bincount(labels[:600], minlength=classes).argmax())
    assert chance['acc'] == pytest.approx(float((labels[600:] == majority).double().mean()))


def test_write_orders_are_per_episode_pure_functions_of_the_episode_seed():
    seeds = np.array([derive_seed(17, 'rfx-test', 'episode', i) for i in range(6)], dtype=np.int64)
    first = rfx.episode_permutations(seeds, 16, 0)
    assert first.shape == (6, 16) and all(sorted(row) == list(range(16)) for row in first)
    assert (rfx.episode_permutations(seeds, 16, 0) == first).all()
    assert (rfx.episode_permutations(seeds[2:3], 16, 0)[0] == first[2]).all()  # independent of the batch
    assert (rfx.episode_permutations(seeds, 16, 1) != first).any(axis=1).all()
    assert len({tuple(row) for row in first}) == 6
    assert rfx.PERMUTATIONS == 4


def test_delta_state_applies_the_deployed_decay_first_rule_in_the_given_order():
    g = gen('delta-state')
    b, n, r = 3, 6, 5
    keys, values = unit(randn(g, b, n, r)), randn(g, b, n, r)
    adapter = SimpleNamespace(lam=torch.tensor(.9, dtype=DTYPE), beta=torch.tensor(.4, dtype=DTYPE))
    state = torch.zeros(b, r, r, dtype=DTYPE)
    for j in range(n):
        decayed = .9 * state
        error = values[:, j] - torch.einsum('bij,bj->bi', decayed, keys[:, j])
        state = decayed + .4 * torch.einsum('bi,bj->bij', error, keys[:, j])
    torch.testing.assert_close(rfx.delta_state(adapter, keys, values), state, rtol=1e-12, atol=1e-12)
    reversed_state = rfx.delta_state(adapter, keys.flip(1), values.flip(1))
    assert not torch.allclose(reversed_state, state)  # the delta store is order dependent, ridge is not


def test_small_table_helpers():
    age = np.array([0, 0, 1, 3, 3, 3])
    rows = rfx.age_table(age, 5, err=np.array([1., 3., 5., 2., 4., 6.]), lm_em=np.array([1., 0., 1., 0., 0., 1.]))
    assert [row['n'] for row in rows] == [2, 1, 0, 3, 0] and sum(row['n'] for row in rows) == len(age)
    assert rows[0]['err'] == 2. and rows[3]['err'] == 4. and rows[2]['err'] is None and rows[3]['lm_em'] == pytest.approx(1 / 3)
    assert rfx.mean_se([1., 2., 3.]) == pytest.approx((2., 1 / math.sqrt(3)))
    assert rfx.mean_se([5.]) == (5., 0.)
    summary = rfx.cond_summary(np.array([1e1, 1e2, 1e3, 1e4]))
    assert summary['median'] == pytest.approx(2.5) and summary['max'] == pytest.approx(4.) and summary['p99'] <= 4.
    assert summary['unit'] == 'log10 condition number'


def test_the_running_maximum_of_the_instrumentation_checks_keeps_a_nan():
    nan = float('nan')
    assert rfx._worst(0., 2.) == 2. and rfx._worst(3., 2.) == 3. and rfx._worst(0., 0.) == 0.
    assert max(0., nan) == 0.  # why max() is not used: a NaN would vanish from the guards
    assert math.isnan(rfx._worst(0., nan)) and math.isnan(rfx._worst(nan, 1.))
    assert math.isnan(rfx._worst(rfx._worst(0., nan), 5.))


def table(**override):
    """A tilted evaluation table: the deployed read is poor, no rescue unless an override says otherwise."""
    cells = {name: dict(err=.8, lm_em=.30) for name in rfx.VARIANTS}
    cells['delta/raw/fit'] = dict(err=.6, lm_em=.31)
    refit = {cell: dict(acc=.30) for cell in rfx.REFIT_CELLS}
    for name, value in override.items():
        if name == 'refit':
            refit['delta/raw'] = dict(acc=value)
        else:
            cells[name.replace('__', '/')] = value
    return cells, refit


def test_verdicts_follow_the_registered_rescue_rules_and_precedence():
    # storage: ridge + raw q halves the (gain-fitted) deployed error and adds at least 10 pp EM.
    cells, refit = table(ridge__raw__fit=dict(err=.3, lm_em=.45))
    verdict = rfx.classify(cells, refit)
    assert verdict['verdict'] == 'storage' and verdict['rescue']['storage_ridge_raw']
    assert verdict['baseline'] == dict(lm_em=.31, err_fit=.6)
    # ... and it wins over every other rescue.
    cells, refit = table(ridge__raw__fit=dict(err=.3, lm_em=.45), delta__true__fit=dict(err=.1, lm_em=.8), refit=.9)
    assert rfx.classify(cells, refit)['verdict'] == 'storage'
    # Each half of the rule is necessary.
    assert rfx.classify(*table(ridge__raw__fit=dict(err=.31, lm_em=.45)))['verdict'] == 'none'  # error not halved
    assert rfx.classify(*table(ridge__raw__fit=dict(err=.1, lm_em=.40)))['verdict'] == 'none'  # +9 pp only
    assert rfx.classify(*table(ridge__raw__fit=dict(err=.3, lm_em=.41)))['verdict'] == 'storage'  # exactly +10 pp
    assert rfx.classify(*table(ridge__raw__fit=dict(err=.30, lm_em=.4099)))['verdict'] == 'none'
    # The baseline is the better of the deployed read as is and with its fitted gain, for both thresholds.
    cells, refit = table(ridge__raw__fit=dict(err=.3, lm_em=.44))
    cells['delta/raw/asis'] = dict(err=.9, lm_em=.35)
    assert rfx.classify(cells, refit)['verdict'] == 'none' and rfx.classify(cells, refit)['baseline']['lm_em'] == .35
    # Rounding of k/n rates: .50 - .40 is 0.0999... in floating point and still counts as 10 pp.
    cells, refit = table(ridge__raw__fit=dict(err=.3, lm_em=.5))
    cells['delta/raw/asis'] = dict(err=.8, lm_em=.4)
    cells['delta/raw/fit'] = dict(err=.6, lm_em=.4)
    assert rfx.classify(cells, refit)['verdict'] == 'storage'
    # The error threshold is a ratio to the deployed (gain-fitted) read: exactly one half is enough.
    cells, refit = table(ridge__raw__fit=dict(err=.25, lm_em=.45))
    cells['delta/raw/fit'] = dict(err=.5, lm_em=.31)
    assert rfx.classify(cells, refit)['verdict'] == 'storage'
    # query: only a true key rescues (either store); decoder: only the refit decoder rescues; otherwise none.
    assert rfx.classify(*table(delta__true__fit=dict(err=.2, lm_em=.6)))['verdict'] == 'query'
    assert rfx.classify(*table(ridge__true__fit=dict(err=.0, lm_em=.8)))['verdict'] == 'query'
    assert rfx.classify(*table(delta__true__fit=dict(err=.2, lm_em=.6), refit=.9))['verdict'] == 'query'
    assert rfx.classify(*table(refit=.45))['verdict'] == 'decoder'
    assert rfx.classify(*table(refit=.41))['verdict'] == 'decoder'  # exactly +10 pp over the deployed baseline .31
    assert rfx.classify(*table(refit=.40))['verdict'] == 'none'
    assert rfx.classify(*table())['verdict'] == 'none'
    assert rfx.classify(*table(delta__true__fit=dict(err=.2, lm_em=.4)))['verdict'] == 'none'  # error rescue without EM


def test_only_storage_verdicts_of_the_R_checkpoints_trigger_lsq():
    assert rfx.lsq_ranks({'R-r128': 'storage', 'R-r32': 'none', 'W5-r32': 'storage'}) == [128]
    assert rfx.lsq_ranks({'R-r128': 'storage', 'R-r32': 'storage', 'W5-r32': 'none'}) == [32, 128]
    assert rfx.lsq_ranks({'R-r128': 'query', 'R-r32': 'decoder', 'W5-r32': 'storage'}) == []
    assert rfx.lsq_ranks({}) == [] and rfx.lsq_ranks({'R-r32': 'storage'}) == [32]


def test_registered_data_ranges_and_seeds_are_new_and_disjoint_from_training():
    cfg = read_fix.configuration('R', 10)
    spec = make_task_spec(cfg)
    fit_arrays = generate(spec, 'train', 0, 8, rfx.FIT_SEED)
    trained = generate(spec, 'train', 0, 8, derive_seed(17, 'llm', 'writer-data'))
    assert rfx.FIT_SEED == derive_seed(17, 'rfx', 'fit') and rfx.FIT_SEED != derive_seed(17, 'llm', 'writer-data')
    assert not set(fit_arrays.episode_seed.tolist()) & set(trained.episode_seed.tolist())
    assert fit_arrays.q_start.shape == (8, 4) and (rfx.FIT_EPISODES, rfx.EVAL_EPISODES) == (4096, 4000)
    arrays, info = rfx.eval_arrays(spec, 12, smoke=False)
    assert info == dict(split='val', seed=4242, start=200000, episodes=12) and arrays.meta['split'] == 'val'
    assert arrays.q_start.shape == (12, 1) and len(set(arrays.episode_seed.tolist())) == 12
    # Earlier evaluations: val [0, 2000) and the tests [100000, 104000) and [110000, 112000).
    assert rfx.EVAL_START >= 112000 and rfx.EVAL_START + rfx.EVAL_EPISODES <= 10 ** 6
    smoke, smoke_info = rfx.eval_arrays(spec, 12, smoke=True)
    assert smoke_info['split'] == 'train' and smoke.meta['split'] == 'train' and smoke_info['start'] >= 10 ** 6
    assert not set(smoke.episode_seed.tolist()) & set(arrays.episode_seed.tolist())
    assert set(arrays.episode_seed.tolist()).isdisjoint(generate(spec, 'val', 0, 12, 4242).episode_seed.tolist())
