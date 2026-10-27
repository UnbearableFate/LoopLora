"""WP-L7 RFX algebra: the ridge store, gain and eta fitting, the refit decoder, write orders and the registered verdict.

Synthetic cached tensors only; the instrumented LLM forward is tested in test_llm_rfx.py. Compute-only like the rest.
"""
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from looplora.core import memory
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm import lsq_release, read_fix, rfx
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


def table(refit=None, **override):
    """A tilted evaluation table: the deployed read is poor, no rescue unless an override says otherwise.

    The deployed baselines are error .6 (gain-fitted read), full-vocabulary EM .31 and entity-restricted EM .40 (the
    better of the as-is and the gain-fitted read in each case); a rescue needs error <= .3 and EM >= .41.
    """
    cells = {name: dict(err=.8, lm_em=.30, lm_em_entity=.38) for name in rfx.VARIANTS}
    cells['delta/raw/fit'] = dict(err=.6, lm_em=.31, lm_em_entity=.40)
    decoder = {cell: dict(acc=.30) for cell in rfx.REFIT_CELLS}
    if refit is not None:
        decoder['delta/raw'] = dict(acc=refit)
    for name, value in override.items():
        cells[name.replace('__', '/')] = value
    return cells, decoder


RESCUED = dict(err=.1, lm_em=.5, lm_em_entity=.6)
DECISIVE = dict(ridge_raw='ridge/raw/fit', ridge_unit='ridge/unit/fit', delta_unit='delta/unit/fit',
                delta_true='delta/true/fit', ridge_true='ridge/true/fit')


def with_rescues(*names, decoder=False):
    """The tilted table in which exactly the named decisive cells rescue (and, optionally, the refit decoder does)."""
    return table(refit=.51 if decoder else None, **{DECISIVE[n].replace('/', '__'): dict(RESCUED) for n in names})


def verdict_of(*names, decoder=False):
    return rfx.classify(*with_rescues(*names, decoder=decoder))['verdict']


def test_verdicts_follow_the_registered_decision_tree():
    # storage: the exact store rescues with the deployed query, or with the unit query when the unit query alone does not.
    assert verdict_of('ridge_raw') == 'storage'
    assert verdict_of('ridge_unit') == 'storage'
    assert verdict_of('ridge_raw', 'ridge_unit') == 'storage'
    # ... storage wins over normalization when ridge/raw rescues, but not when only ridge/unit does: delta/unit explains it.
    assert verdict_of('ridge_raw', 'delta_unit') == 'storage'
    assert verdict_of('ridge_raw', 'ridge_unit', 'delta_unit') == 'storage'
    assert verdict_of('ridge_unit', 'delta_unit') == 'normalization'
    # normalization: only the unit query on the deployed store.
    assert verdict_of('delta_unit') == 'normalization'
    assert verdict_of('delta_unit', 'delta_true') == 'normalization'
    assert verdict_of('delta_unit', 'ridge_true', decoder=True) == 'normalization'
    # query: only the true key on the deployed store; the ridge + true key read is store+query, not query.
    assert verdict_of('delta_true') == 'query'
    assert verdict_of('delta_true', 'ridge_true') == 'query'
    assert verdict_of('delta_true', decoder=True) == 'query'
    assert verdict_of('ridge_true') == 'store+query'
    assert verdict_of('ridge_true', decoder=True) == 'store+query'
    # decoder and none.
    assert verdict_of(decoder=True) == 'decoder'
    assert verdict_of() == 'none'
    # Every verdict is registered, each one is reachable, and the strings are the ones the LSQ release logic reads.
    assert set(rfx.VERDICTS) == {'storage', 'normalization', 'query', 'store+query', 'decoder', 'none'}
    assert {verdict_of(*n, decoder=d) for n in ((), ('ridge_raw',), ('delta_unit',), ('delta_true',), ('ridge_true',))
            for d in (False, True)} == set(rfx.VERDICTS)


def registered_verdict(ridge_raw, ridge_unit, delta_unit, delta_true, ridge_true, decoder):
    """docs/prereg/RFX.md, 'Decision rule', transcribed: the first verdict that applies."""
    if ridge_raw or (ridge_unit and not delta_unit):
        return 'storage'
    if delta_unit:
        return 'normalization'
    if delta_true:
        return 'query'
    if ridge_true:
        return 'store+query'
    if decoder:
        return 'decoder'
    return 'none'


@pytest.mark.parametrize('flags', list(itertools.product((False, True), repeat=6)))
def test_every_combination_of_rescue_flags_gets_the_registered_verdict(flags):
    names = [name for name, on in zip(DECISIVE, flags[:5]) if on]
    verdict = rfx.classify(*with_rescues(*names, decoder=flags[5]))
    assert verdict['verdict'] == registered_verdict(*flags)
    # Every rescue flag is recorded, including ridge/unit/fit and delta/unit/fit, and it is exactly what was set.
    assert set(verdict['rescue']) == set(rfx.VARIANTS) | {'decoder_refit'}
    assert {name: verdict['rescue'][cell] for name, cell in DECISIVE.items()} == dict(zip(DECISIVE, flags[:5]))
    assert verdict['rescue']['decoder_refit'] == flags[5]
    assert sum(verdict['rescue'].values()) == sum(flags)  # no other cell (baseline, clean, zero, ...) rescues


def test_a_cell_rescues_with_half_the_error_and_ten_points_of_em_over_the_better_deployed_read():
    result = rfx.classify(*with_rescues('ridge_raw'))
    assert result['baseline'] == dict(err_fit=.6, lm_em=.31, lm_em_entity=.40)
    assert result['thresholds'] == dict(error_ratio=.5, em_gain=.10)
    # Each half of the rule is necessary.
    assert rfx.classify(*table(ridge__raw__fit=dict(err=.31, lm_em=.45, lm_em_entity=.6)))['verdict'] == 'none'  # error
    assert rfx.classify(*table(ridge__raw__fit=dict(err=.1, lm_em=.40, lm_em_entity=.6)))['verdict'] == 'none'  # +9 pp
    assert rfx.classify(*table(ridge__raw__fit=dict(err=.3, lm_em=.41, lm_em_entity=.4)))['verdict'] == 'storage'
    assert rfx.classify(*table(ridge__raw__fit=dict(err=.30, lm_em=.4099, lm_em_entity=.4)))['verdict'] == 'none'
    # The error threshold is a ratio to the gain-fitted deployed read only: exactly one half is enough.
    cells, refit = table(ridge__raw__fit=dict(err=.25, lm_em=.45, lm_em_entity=.4))
    cells['delta/raw/fit'] = dict(err=.5, lm_em=.31, lm_em_entity=.40)
    assert rfx.classify(cells, refit)['verdict'] == 'storage'
    cells['delta/raw/asis'] = dict(err=.2, lm_em=.30, lm_em_entity=.38)  # a lower as-is error does not move the baseline
    assert rfx.classify(cells, refit)['verdict'] == 'storage'
    # The EM baseline is the better of the as-is and the gain-fitted deployed read.
    cells, refit = table(ridge__raw__fit=dict(err=.3, lm_em=.44, lm_em_entity=.6))
    cells['delta/raw/asis'] = dict(err=.9, lm_em=.35, lm_em_entity=.38)
    assert rfx.classify(cells, refit)['verdict'] == 'none' and rfx.classify(cells, refit)['baseline']['lm_em'] == .35
    # Rounding of k/n rates: .50 - .40 is 0.0999... in floating point and still counts as 10 pp.
    cells, refit = table(ridge__raw__fit=dict(err=.3, lm_em=.5, lm_em_entity=.5))
    cells['delta/raw/asis'] = dict(err=.8, lm_em=.4, lm_em_entity=.4)
    cells['delta/raw/fit'] = dict(err=.6, lm_em=.4, lm_em_entity=.4)
    assert rfx.classify(cells, refit)['verdict'] == 'storage'
    # An error rescue without the EM gain does not rescue any cell, for any query.
    for cell in DECISIVE.values():
        assert rfx.classify(*table(**{cell.replace('/', '__'): dict(err=.2, lm_em=.4, lm_em_entity=.9)}))['verdict'] == 'none'
    # The rescue test uses the full-vocabulary EM, never the entity-restricted one.
    assert rfx.classify(*table(delta__unit__fit=dict(err=.1, lm_em=.30, lm_em_entity=.99)))['verdict'] == 'none'


def test_the_refit_decoder_is_compared_with_the_entity_restricted_baseline_em():
    # Baseline: entity-restricted EM .40 (better of as-is .38 and fitted .40), full-vocabulary EM .31.
    assert rfx.classify(*table(refit=.51))['verdict'] == 'decoder'
    assert rfx.classify(*table(refit=.50))['verdict'] == 'decoder'  # exactly +10 pp (floating-point tolerance)
    assert rfx.classify(*table(refit=.49))['verdict'] == 'none'
    assert rfx.classify(*table(refit=.41))['verdict'] == 'none'  # +10 pp over the full-vocabulary .31 would have counted
    cells, refit = table(refit=.52)
    cells['delta/raw/asis'] = dict(err=.8, lm_em=.30, lm_em_entity=.45)  # the as-is entity EM is the better baseline
    assert rfx.classify(cells, refit)['verdict'] == 'none' and rfx.classify(cells, refit)['baseline']['lm_em_entity'] == .45
    refit['delta/raw'] = dict(acc=.55)
    assert rfx.classify(cells, refit)['verdict'] == 'decoder'
    # Only the decoder on the deployed raw read counts.
    other = {cell: dict(acc=.99) for cell in rfx.REFIT_CELLS if cell != 'delta/raw'}
    assert rfx.classify(table()[0], {**other, 'delta/raw': dict(acc=.30)})['verdict'] == 'none'


LABEL_VERDICTS = ('storage', 'normalization', 'query', 'store+query', 'decoder', 'none')


def test_lsq_arms_follow_the_R_verdicts_in_candidate_order():
    def verdicts(r128, r32, w5='none'):
        return {'R-r128': r128, 'R-r32': r32, 'W5-r32': w5}

    assert rfx.lsq_arms(verdicts('storage', 'none')) == ['L128-K4', 'L128-RG']
    assert rfx.lsq_arms(verdicts('none', 'storage')) == ['L32-K4', 'L32-RG']
    assert rfx.lsq_arms(verdicts('storage', 'storage')) == ['L32-K4', 'L32-RG', 'L128-K4', 'L128-RG']
    assert rfx.lsq_arms(verdicts('normalization', 'none')) == ['L128-U']
    assert rfx.lsq_arms(verdicts('none', 'normalization')) == ['L32-U']
    assert rfx.lsq_arms(verdicts('normalization', 'normalization')) == ['L32-U', 'L128-U']
    assert rfx.lsq_arms(verdicts('normalization', 'storage')) == ['L32-K4', 'L32-RG', 'L128-U']
    assert rfx.lsq_arms(verdicts('storage', 'normalization')) == ['L32-U', 'L128-K4', 'L128-RG']
    for other in ('query', 'store+query', 'decoder', 'none'):
        assert rfx.lsq_arms(verdicts(other, other)) == []
    # W5-r32 triggers nothing, and a missing checkpoint triggers nothing.
    assert rfx.lsq_arms(verdicts('none', 'none', 'storage')) == [] == rfx.lsq_arms(verdicts('none', 'none', 'normalization'))
    assert rfx.lsq_arms({}) == [] and rfx.lsq_arms({'R-r32': 'storage'}) == ['L32-K4', 'L32-RG']
    # Candidate order is LSQ's, and per rank U never comes with K4 or RG (the arm sets check_arms accepts).
    for r128, r32 in itertools.product(LABEL_VERDICTS, repeat=2):
        arms = rfx.lsq_arms(verdicts(r128, r32))
        assert arms == [a for a in lsq_release.ARMS if a in arms]
        if arms:
            assert lsq_release.check_arms(arms) == arms
    assert rfx.storage_ranks(verdicts('storage', 'none', 'storage')) == [128]
    assert rfx.storage_ranks(verdicts('storage', 'storage')) == [32, 128]
    assert rfx.storage_ranks(verdicts('normalization', 'query', 'storage')) == []
    assert rfx.storage_ranks({}) == []


def test_entity_hits_restrict_the_argmax_to_the_entity_name_tokens():
    entity_ids = torch.tensor([7, 3, 11, 5])  # entity i has the name token entity_ids[i]
    logits = torch.zeros(4, 1, 16)
    logits[0, 0, 9], logits[0, 0, 11], logits[0, 0, 3] = 5., 2., 1.  # a non-entity token is largest; entity 2 leads the rest
    logits[1, 0, 5], logits[1, 0, 7] = 3., 2.  # the largest logit is the answer's entity token (entity 3)
    logits[2, 0, 7], logits[2, 0, 3] = 4., 3.9  # the wrong entity (0) wins; the answer is entity 1
    logits[3, 0, 0], logits[3, 0, 3] = 9., 1.  # a non-entity token is largest; entity 1 is the only positive entity logit
    answers = torch.tensor([[2], [3], [1], [1]])
    hits = rfx.entity_hits(logits, entity_ids, answers)
    assert hits.shape == (4, 1) and hits.tolist() == [[True], [True], [False], [True]]
    assert (logits.argmax(-1) == entity_ids[answers]).tolist() == [[False], [True], [False], [False]]  # full vocabulary
    # Without ties, a full-vocabulary hit is always an entity-restricted hit.
    g = torch.Generator().manual_seed(derive_seed(17, 'rfx-test', 'entity-hits'))
    logits = torch.randn(64, 3, 40, generator=g)
    ids = torch.randperm(40, generator=g)[:25]
    for answer_entity in (torch.randint(0, 25, (64, 3), generator=g), logits.index_select(-1, ids).argmax(-1)):
        full = logits.argmax(-1) == ids[answer_entity]
        assert (rfx.entity_hits(logits, ids, answer_entity) | ~full).all()


def tied_logits(top, vocabulary=16):
    """[1, 1, vocabulary] logits with an exact tie of 2. at the tokens `top` and zeros elsewhere."""
    logits = torch.zeros(1, 1, vocabulary)
    logits[0, 0, torch.tensor(list(top), dtype=torch.long)] = 2.
    return logits


def test_entity_hits_break_exact_ties_like_the_full_vocabulary_argmax_not_by_entity_index():
    # Entity i has the name token entity_ids[i]; the entity order is not the token order (entity 0 = token 7 > entity 3 = 5).
    entity_ids = torch.tensor([7, 3, 11, 5])

    def hit_and_full_hit(top, answer):
        logits, answers = tied_logits(top), torch.tensor([[answer]])
        return bool(rfx.entity_hits(logits, entity_ids, answers)), bool(logits.argmax(-1) == entity_ids[answers])

    # A tie between the answer (entity 3, token 5) and entity 0 (the lower entity index, the higher token id): the full
    # argmax takes the lower token id, the answer, and so does the restricted one (by entity index it would take entity 0).
    assert hit_and_full_hit((7, 5), 3) == (True, True)
    # The same tie with the answer on the other side (entity 0, token 7): both argmaxes take token 5, entity 3.
    assert hit_and_full_hit((7, 5), 0) == (False, False)
    # A three-way tie goes to the lowest token id of the three (token 5, entity 3), whichever entity is the answer.
    assert [hit_and_full_hit((7, 11, 5), answer) for answer in range(4)] == [(False, False)] * 3 + [(True, True)]
    # A tie with a non-entity token: the full argmax may take it (token 2 is lower: a miss), the restricted one ignores it.
    assert hit_and_full_hit((5, 2), 3) == (True, False)
    assert hit_and_full_hit((5, 9), 3) == (True, True)  # the non-entity token is higher: the answer wins both
    # All logits equal: the full argmax is token 0, no entity; the restricted one takes the lowest entity token (3, entity 1).
    assert [hit_and_full_hit((), answer) for answer in range(4)] == [(False, False), (True, False), (False, False),
                                                                      (False, False)]


def test_a_full_vocabulary_hit_is_an_entity_restricted_hit_even_with_ties():
    g = gen('entity-ties')
    logits = torch.randint(0, 4, (64, 3, 40), generator=g).float()  # four levels: almost every row ties at its maximum
    ids = torch.randperm(40, generator=g)[:25]  # 25 entities whose order is not the token order
    entity_logits = logits.index_select(-1, ids)
    at_max = entity_logits == entity_logits.max(-1, keepdim=True).values
    lowest = torch.where(at_max, ids, torch.full_like(ids, 10 ** 6)).min(-1).values  # the lowest token id at the maximum
    assert (at_max.sum(-1) > 1).float().mean() > .9
    differs_from_entity_index_rule = False
    for entity in range(25):
        hits = rfx.entity_hits(logits, ids, torch.full((64, 3), entity, dtype=torch.long))
        assert (hits == (lowest == ids[entity])).all()
        assert (hits | ~(logits.argmax(-1) == ids[entity])).all()
        differs_from_entity_index_rule |= bool((hits != (entity_logits.argmax(-1) == entity)).any())
    assert differs_from_entity_index_rule  # the test can tell the two tie rules apart


def guard_record(rows, deficit=None, gain=0, queries=1):
    """A run_checkpoint-like record: every variant's full argmax hits a third of the rows; its entity-restricted argmax
    loses deficit[variant] of those rows and wins `gain` others."""
    hits, deficit = rows // 3, deficit or {}
    return dict(data=dict(eval=dict(episodes=rows // queries, queries_per_episode=queries)),
                cells={v: dict(lm_em=hits / rows, lm_em_entity=(hits - deficit.get(v, 0) + gain) / rows,
                               full_hit_entity_miss=deficit.get(v, 0)) for v in rfx.VARIANTS})


def test_the_entity_em_guard_allows_half_a_point_or_one_row_and_catches_a_wrong_restricted_set():
    one = lambda rows, deficit=None, gain=0: {'a': guard_record(rows, deficit, gain)}
    assert rfx.ENTITY_EM_SLACK == .005 and rfx.entity_em_failures(one(4000)) == []
    # The allowance is 0.005 of the rows: 20 of 4,000 (an exact 0.5 pp deficit passes), 21 are not.
    assert rfx.entity_em_failures(one(4000, {'delta/raw/asis': 20})) == []
    failed = rfx.entity_em_failures({**one(4000, {'delta/raw/asis': 21}), 'b': guard_record(4000)})
    assert failed == [dict(label='a', variant='delta/raw/asis', rows=4000, lm_em=1333 / 4000, lm_em_entity=1312 / 4000,
                           full_hit_entity_miss=21)]
    # The restricted set is easier than the full vocabulary when the full argmax lands on a non-entity token.
    assert rfx.entity_em_failures(one(4000, gain=400)) == []
    # The 64-row smoke: one row is 1.6 pp, so one row passes and two fail; from 200 rows on the allowance is 0.005 of the rows.
    assert rfx.entity_em_failures(one(64, {'clean': 1})) == []
    assert [f['variant'] for f in rfx.entity_em_failures(one(64, {'clean': 2, 'zero': 1}))] == ['clean']
    assert rfx.entity_em_failures(one(200, {'zero': 1})) == []
    assert [f['variant'] for f in rfx.entity_em_failures(one(200, {'zero': 2}))] == ['zero']
    # Rows are episodes times queries per episode.
    assert [f['variant'] for f in rfx.entity_em_failures({'a': guard_record(64, {'clean': 2}, queries=2)})] == ['clean']
    assert rfx.entity_em_failures({'a': guard_record(64, {'clean': 1}, queries=2)}) == []
    # A wrong slice, axis or pairing leaves chance-level restricted hits: every variant that has hits fails, with the count.
    wrong = guard_record(4000)
    for cell in wrong['cells'].values():
        cell.update(lm_em_entity=0., full_hit_entity_miss=round(cell['lm_em'] * 4000))
    failures = rfx.entity_em_failures({'R-r32': wrong})
    assert [f['variant'] for f in failures] == list(rfx.VARIANTS) and all(
        (f['label'], f['full_hit_entity_miss']) == ('R-r32', 1333) for f in failures)
    # A NaN deficit fails closed.
    broken = guard_record(4000)
    broken['cells']['ridge/true/fit']['lm_em_entity'] = float('nan')
    assert [f['variant'] for f in rfx.entity_em_failures({'a': broken})] == ['ridge/true/fit']


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
