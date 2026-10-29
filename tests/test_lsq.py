"""WP-L7 LSQ stores: ridge least squares and repeated-scan (K4) writers, their coefficients and the unchanged off path.

Compute-only tests (login nodes never import torch). The two stores are final-only; the delta rule stays the default.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from looplora.core import memory
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed

DTYPE = torch.float64
DEVICES = ['cpu', 'cuda']


def generator(device, *label):
    return torch.Generator(device=device).manual_seed(derive_seed(17, 'lsq', *label))


def rand(g, device, *shape, scale=1., dtype=DTYPE):
    return torch.randn(*shape, generator=g, device=device, dtype=dtype) * scale


def explicit_keys(u, K, V, eps=1e-6):
    return F.normalize(F.linear(u, K), dim=-1, eps=eps), F.linear(u, V)


def reference_ridge(state0, u, valid, write, reset, K, V, lam, eta, eps=1e-6):
    """Row by row from the definition: minimize sum_j w_j |M k_j - v_j|^2 + eta |M - lam^n M0|^2, w_j = lam^(n-1-j)."""
    out = []
    for b in range(u.shape[0]):
        base = torch.zeros_like(state0[b]) if bool(reset[b, 0]) else state0[b]
        index = (valid[b] & write[b]).nonzero(as_tuple=True)[0]
        n = len(index)
        if n == 0:
            out.append(base)
            continue
        keys, values = explicit_keys(u[b, index], K, V, eps)
        weights = lam ** torch.arange(n - 1, -1, -1, dtype=u.dtype, device=u.device)
        prior = lam ** n * base
        residual = values - keys @ prior.T
        system = eta * torch.eye(keys.shape[1], dtype=u.dtype, device=u.device) + (keys * weights[:, None]).T @ keys
        out.append(prior + (residual * weights[:, None]).T @ keys @ torch.linalg.inv(system))
    return torch.stack(out)


def layout_masks(g, device, layout, b, t):
    valid = torch.rand(b, t, generator=g, device=device) > .2
    write = torch.rand(b, t, generator=g, device=device) > .4
    reset = torch.zeros_like(valid)
    if layout == 'all':
        valid.fill_(True)
        write.fill_(True)
    else:
        write[0].fill_(False)  # a row without any write in a mixed batch
    if layout == 'initial_reset':
        reset[:, 0] = True
    return valid, write, reset


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('layout', ['random', 'all', 'initial_reset'])
def test_ridge_final_is_the_recency_weighted_ridge_solution_of_the_written_tokens(device, layout):
    b, t, r, rm = 5, 12, 6, 7
    g = generator(device, 'ridge-reference', layout)
    u, state0 = rand(g, device, b, t, r), rand(g, device, b, rm, rm, scale=.3)
    K, V = rand(g, device, rm, r, scale=.7), rand(g, device, rm, r, scale=.7)
    valid, write, reset = layout_masks(g, device, layout, b, t)
    lam, eta = torch.tensor(.8, dtype=DTYPE, device=device), .07
    actual = memory.ridge_final(state0, u, valid, write, reset, K=K, V=V, lam=lam, eta=eta)
    expected = reference_ridge(state0, u, valid, write, reset, K, V, lam, eta)
    assert actual.shape == state0.shape and actual.dtype == DTYPE
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    if layout != 'all':  # the row without writes keeps its (possibly reset) prior untouched
        torch.testing.assert_close(actual[0], torch.zeros_like(state0[0]) if layout == 'initial_reset' else state0[0],
                                   rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('lam', [1., .8])
def test_ridge_final_interpolates_independent_keys_as_eta_vanishes(device, lam):
    b, n, r, rm = 3, 6, 8, 8
    g = generator(device, 'ridge-interpolation', lam)
    u, K, V = rand(g, device, b, n, r), rand(g, device, rm, r), rand(g, device, rm, r)
    state = memory.ridge_final(torch.zeros(b, rm, rm, dtype=DTYPE, device=device), u, None, None, None,
                               K=K, V=V, lam=lam, eta=1e-12)
    keys, values = explicit_keys(u, K, V)
    torch.testing.assert_close(torch.einsum('bij,bnj->bni', state, keys), values, rtol=0, atol=1e-6)
    # Any recency weighting gives the same interpolant: it is the minimum-norm solution of M k_j = v_j.
    gram = keys @ keys.transpose(1, 2)
    minimum_norm = torch.einsum('bni,bnm,bmj->bij', values, torch.linalg.inv(gram), keys)
    torch.testing.assert_close(state, minimum_norm, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('device', DEVICES)
def test_ridge_final_satisfies_the_normal_equations_when_writes_exceed_the_key_dimension(device):
    b, n, r, rm = 3, 14, 5, 6
    g = generator(device, 'ridge-normal-equations')
    u, state0 = rand(g, device, b, n, r), rand(g, device, b, rm, rm, scale=.3)
    K, V = rand(g, device, rm, r), rand(g, device, rm, r)
    lam, eta = .9, .05
    state = memory.ridge_final(state0, u, None, None, None, K=K, V=V, lam=lam, eta=eta)
    keys, values = explicit_keys(u, K, V)
    weights = lam ** torch.arange(n - 1, -1, -1, dtype=DTYPE, device=device)
    error = torch.einsum('bij,bnj->bni', state, keys) - values
    gradient = torch.einsum('n,bni,bnj->bij', weights, error, keys) + eta * (state - lam ** n * state0)
    assert gradient.abs().max() < 1e-9
    assert error.abs().max() > 1e-3  # more writes than dimensions: genuinely a regression, not an interpolant


@pytest.mark.parametrize('device', DEVICES)
def test_ridge_final_rejects_unsupported_calls(device):
    b, t, r, rm = 2, 5, 3, 4
    g = generator(device, 'ridge-errors')
    u, K, V = rand(g, device, b, t, r), rand(g, device, rm, r), rand(g, device, rm, r)
    zeros = torch.zeros(b, rm, rm, dtype=DTYPE, device=device)
    call = lambda **kw: memory.ridge_final(zeros, u, kw.pop('valid', None), None, kw.pop('reset', None),
                                           **{**dict(K=K, V=V, lam=.9, eta=.1), **kw})
    call()
    interior = torch.zeros(b, t, dtype=torch.bool, device=device)
    interior[:, 2] = True
    with pytest.raises(ValueError, match='interior'):
        call(reset=interior)
    initial = torch.zeros_like(interior)
    initial[:, 0] = True
    call(reset=initial)
    for eta in (0., -1., math.inf, math.nan, True):
        with pytest.raises(ValueError):
            call(eta=eta)
    with pytest.raises(ValueError):  # input-dependent gates are not defined for a joint solve
        call(lam=torch.full((b, t), .9, dtype=DTYPE, device=device))
    with pytest.raises(ValueError):
        call(K=K[:, :2])
    with pytest.raises(ValueError):
        memory.ridge_final(zeros[:, :, :3], u, None, None, None, K=K, V=V, lam=.9, eta=.1)
    assert memory.ridge_final(zeros, u[:, :0], None, None, None, K=K, V=V, lam=.9, eta=.1) is zeros
    none = torch.zeros(b, t, dtype=torch.bool, device=device)
    state0 = rand(g, device, b, rm, rm)
    assert torch.equal(memory.ridge_final(state0, u, none, None, None, K=K, V=V, lam=.9, eta=.1), state0)


@pytest.mark.parametrize('device', DEVICES)
def test_ridge_final_is_differentiable_and_matches_finite_differences(device):
    g = generator(device, 'ridge-gradcheck')
    b, t, r, rm = 2, 5, 3, 4
    leaves = [rand(g, device, b, t, r), rand(g, device, rm, r, scale=.7), rand(g, device, rm, r, scale=.7),
              torch.tensor(.85, dtype=DTYPE, device=device), rand(g, device, b, rm, rm, scale=.1)]
    for leaf in leaves:
        leaf.requires_grad_()
    valid = torch.ones(b, t, dtype=torch.bool, device=device)
    valid[0, 3] = False  # one padded slot: its zero weight must not create NaN gradients
    write = torch.ones_like(valid)

    def store(u, K, V, lam, state0):
        return memory.ridge_final(state0, u, valid, write, None, K=K, V=V, lam=lam, eta=.05)

    assert torch.autograd.gradcheck(store, tuple(leaves), eps=1e-6, atol=1e-6, rtol=1e-4)
    weight = rand(g, device, b, rm, rm)
    gradients = torch.autograd.grad((store(*leaves) * weight).sum(), leaves)
    assert all(torch.isfinite(x).all() and x.abs().sum() > 0 for x in gradients)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('layout', ['random', 'all', 'initial_reset'])
def test_sweep_final_is_the_repeated_beta_one_scan_with_one_reset(device, layout):
    b, t, r, rm = 4, 11, 5, 6
    g = generator(device, 'sweep-structure', layout)
    dtype = torch.float32
    u, state0 = rand(g, device, b, t, r, dtype=dtype), rand(g, device, b, rm, rm, scale=.1, dtype=dtype)
    K, V = rand(g, device, rm, r, scale=.6, dtype=dtype), rand(g, device, rm, r, scale=.6, dtype=dtype)
    valid, write, reset = layout_masks(g, device, layout, b, t)
    lam = torch.tensor(.9, device=device)
    kwargs = dict(K=K, V=V, lam=lam)
    once = memory.scan_final(state0, u, valid, write, reset, beta=1.0, **kwargs)
    assert torch.equal(memory.sweep_final(state0, u, valid, write, reset, sweeps=1, **kwargs), once)
    twice = memory.scan_final(once, u, valid, write, None, beta=1.0, **kwargs)
    thrice = memory.scan_final(twice, u, valid, write, None, beta=1.0, **kwargs)
    assert torch.equal(memory.sweep_final(state0, u, valid, write, reset, sweeps=2, **kwargs), twice)
    assert torch.equal(memory.sweep_final(state0, u, valid, write, reset, sweeps=3, **kwargs), thrice)
    assert not torch.equal(once, twice)
    if layout == 'all':  # explicit decay-first loop over the same ordered writes
        expected, (keys, values) = state0, explicit_keys(u, K, V)
        for _ in range(3):
            for j in range(t):
                expected = memory.delta_update(expected, keys[:, j], values[:, j], None, lam, 1.)
        torch.testing.assert_close(thrice, expected, rtol=1e-5, atol=1e-6)
    interior = torch.zeros_like(valid)
    interior[:, 3] = True
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            memory.sweep_final(state0, u, valid, write, reset, sweeps=bad, **kwargs)
    with pytest.raises(ValueError, match='interior'):
        memory.sweep_final(state0, u, valid, write, interior, **kwargs)


@pytest.mark.parametrize('device', DEVICES)
def test_sweeps_approach_the_interpolant_monotonically_but_four_sweeps_are_far_from_it(device):
    b, n, r = 4, 5, 16
    g = generator(device, 'sweep-convergence')
    u = rand(g, device, b, n, r) + rand(g, device, b, 1, r)  # correlated keys: one sweep is not an interpolant
    K, V = torch.eye(r, dtype=DTYPE, device=device), rand(g, device, r, r)
    keys, values = explicit_keys(u, K, V)
    zeros = torch.zeros(b, r, r, dtype=DTYPE, device=device)
    interpolant = torch.einsum('bni,bnm,bmj->bij', values, torch.linalg.inv(keys @ keys.transpose(1, 2)), keys)
    assert torch.einsum('bij,bnj->bni', interpolant, keys).sub(values).abs().max() < 1e-9
    ridge = memory.ridge_final(zeros, u, None, None, None, K=K, V=V, lam=1., eta=1e-12)
    torch.testing.assert_close(ridge, interpolant, rtol=0, atol=1e-8)
    steps = (1, 2, 4, 8, 16, 64, 256)
    distance = {s: (memory.sweep_final(zeros, u, None, None, None, K=K, V=V, lam=1., sweeps=s) - interpolant)
                .flatten(1).norm(dim=1) for s in steps}
    for before, after in zip(steps, steps[1:]):
        assert (distance[after] <= distance[before] + 1e-12).all(), (before, after)
    assert (distance[16] < distance[1]).all() and (distance[4] < distance[1]).all()
    assert (distance[4] > 1e-3).all()  # K4 is a four-step approximation, not the exact solve
    assert (distance[256] < 1e-6 * distance[1]).all()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('store', ['ridge', 'sweep'])
def test_gradients_flow_through_both_stores_and_stay_finite_with_padding(device, store):
    b, t, r, rm = 3, 8, 4, 5
    g = generator(device, 'store-gradients', store)
    dtype = torch.float32
    leaves = [rand(g, device, b, t, r, dtype=dtype), rand(g, device, rm, r, scale=.6, dtype=dtype),
              rand(g, device, rm, r, scale=.6, dtype=dtype), torch.tensor(.9, device=device),
              rand(g, device, b, rm, rm, scale=.1, dtype=dtype)]
    for leaf in leaves:
        leaf.requires_grad_()
    valid = torch.rand(b, t, generator=g, device=device) > .25
    write = torch.ones_like(valid)
    write[2] = False  # one row without writes
    u, K, V, lam, state0 = leaves
    if store == 'ridge':
        state = memory.ridge_final(state0, u, valid, write, None, K=K, V=V, lam=lam, eta=1e-2)
    else:
        state = memory.sweep_final(state0, u, valid, write, None, K=K, V=V, lam=lam, sweeps=4)
    weight = rand(g, device, b, rm, rm, dtype=dtype)
    gradients = torch.autograd.grad((state * weight).sum(), leaves)
    assert all(torch.isfinite(x).all() for x in gradients)
    assert all(x.abs().sum() > 0 for x in gradients[1:3] + gradients[4:])
    assert gradients[3].abs() > 0
    assert gradients[0][2].abs().sum() == 0 and gradients[0][~valid].abs().sum() == 0  # padding gets no gradient


@pytest.mark.parametrize('device', DEVICES)
def test_ridge_and_sweep_coefficients_reconstruct_the_stored_read(device):
    b, n, r, rm, queries = 3, 9, 5, 6, 4
    g = generator(device, 'coefficients')
    u, K, V = rand(g, device, b, n, r), rand(g, device, rm, r, scale=.7), rand(g, device, rm, r, scale=.7)
    q = rand(g, device, b, queries, rm)
    keys, values = explicit_keys(u, K, V)
    zeros = torch.zeros(b, rm, rm, dtype=DTYPE, device=device)
    lam, eta, sweeps = .9, .03, 3
    cases = (('ridge', memory.ridge_final(zeros, u, None, None, None, K=K, V=V, lam=lam, eta=eta),
              memory.ridge_coefficients(keys, q, lam, eta)),
             ('sweep', memory.sweep_final(zeros, u, None, None, None, K=K, V=V, lam=lam, sweeps=sweeps),
              memory.sweep_coefficients(keys, q, lam, sweeps)))
    for name, state, coefficients in cases:
        assert coefficients['d'].shape == (b, queries, n) and coefficients['gram'].shape == (b, n, n)
        reconstructed = torch.einsum('bqe,ber->bqr', coefficients['d'], values)
        torch.testing.assert_close(reconstructed, torch.einsum('bij,bqj->bqi', state, q), rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(coefficients['b'], torch.einsum('bnr,bqr->bqn', keys, q))
    # With eta -> 0 and independent keys the ridge read of a stored key is exactly its value.
    exact = memory.ridge_coefficients(keys[:, :4], keys[:, :4], 1., 1e-12)
    torch.testing.assert_close(exact['d'], torch.eye(4, dtype=DTYPE, device=device).expand(b, 4, 4), rtol=0, atol=1e-6)


@pytest.mark.parametrize('device', DEVICES)
def test_gram_conditions_are_finite_condition_numbers_of_the_gram_and_the_ridge_system(device):
    g = generator(device, 'conditions')
    b, n, r = 3, 6, 9
    keys = F.normalize(rand(g, device, b, n, r), dim=-1)
    lam, eta = .7, .02
    result = memory.gram_conditions(keys, lam, eta)
    gram = (keys @ keys.transpose(1, 2)).cpu().numpy()
    root = np.sqrt(lam ** np.arange(n - 1, -1, -1))
    for i in range(b):
        values = np.linalg.eigvalsh(gram[i])
        assert math.isclose(float(result['gram'][i]), values[-1] / values[0], rel_tol=1e-8)
        system = root[:, None] * gram[i] * root[None, :] + eta * np.eye(n)
        values = np.linalg.eigvalsh(system)
        assert math.isclose(float(result['ridge'][i]), values[-1] / values[0], rel_tol=1e-8)
    assert set(memory.gram_conditions(keys)) == {'gram'}
    unweighted = memory.gram_conditions(keys, None, eta)['ridge']
    eigenvalues = torch.linalg.eigvalsh(keys @ keys.transpose(1, 2))
    torch.testing.assert_close(unweighted, (eigenvalues[:, -1] + eta) / (eigenvalues[:, 0] + eta))
    orthonormal = torch.eye(4, dtype=DTYPE, device=device).expand(2, 4, 4)
    torch.testing.assert_close(memory.gram_conditions(orthonormal, None, eta)['gram'], torch.ones(2, dtype=DTYPE, device=device))
    repeated = memory.gram_conditions(torch.ones(1, 3, 2, dtype=DTYPE, device=device))['gram']
    assert torch.isfinite(repeated).all() and repeated > 1e12  # singular Gram: huge but finite (floored)
    # Adding the regularizer bounds the conditioning of the system a joint solve actually inverts.
    assert memory.gram_conditions(torch.ones(1, 3, 2, dtype=DTYPE, device=device), None, 1e-2)['ridge'] < 1e3


def adapters(**extra):
    """The three stores with identical parameters (same seed, same shapes)."""
    out = {}
    for store in ('delta', 'ridge', 'sweep'):
        torch.manual_seed(derive_seed(17, 'lsq-adapter'))
        out[store] = make_adapter(AdapterConfig(d_in=7, d_out=5, r=4, variant='iterq', store=store,
                                                unit_query=store != 'delta', **extra))
    return out


def test_adapter_scan_final_dispatches_to_the_configured_store():
    a = adapters(store_eta=.02, store_sweeps=3)
    assert torch.equal(a['delta'].K, a['ridge'].K) and torch.equal(a['delta'].V, a['sweep'].V)
    g = torch.Generator().manual_seed(derive_seed(17, 'lsq-dispatch'))
    u, state0 = torch.randn(3, 9, 4, generator=g), torch.randn(3, 4, 4, generator=g) * .1
    valid, write = torch.rand(3, 9, generator=g) > .2, torch.rand(3, 9, generator=g) > .3
    for store, adapter in a.items():
        kwargs = dict(K=adapter.K, V=adapter.V, lam=adapter.lam)
        expected = {'delta': lambda: memory.scan_final(state0, u, valid, write, None, beta=adapter.beta, eps=adapter.cfg.eps,
                                                       **kwargs),
                    'ridge': lambda: memory.ridge_final(state0, u, valid, write, None, eta=.02, eps=adapter.cfg.eps, **kwargs),
                    'sweep': lambda: memory.sweep_final(state0, u, valid, write, None, sweeps=3, eps=adapter.cfg.eps,
                                                        **kwargs)}[store]()
        assert torch.equal(adapter.scan_final(state0, u, valid, write, None), expected)
    assert not torch.equal(a['ridge'].scan_final(state0, u, valid, write, None),
                           a['delta'].scan_final(state0, u, valid, write, None))


def test_least_squares_stores_are_final_only_and_reject_online_writes():
    for store in ('ridge', 'sweep'):
        adapter = adapters()[store]
        g = torch.Generator().manual_seed(derive_seed(17, 'lsq-online', store))
        x, state = torch.randn(2, 5, 7, generator=g), torch.zeros(2, 4, 4)
        with pytest.raises(ValueError, match='final-only|final support'):
            adapter.process_sequence(x, state, 1)  # default write mask: every valid token writes
        with pytest.raises(ValueError, match='final-only|final support'):
            adapter.process_sequence_fast(x, state, 1)
        with pytest.raises(ValueError):
            adapter.write(state, torch.randn(2, 4, generator=g), None)
        with pytest.raises(ValueError):
            adapter.write(state, torch.randn(2, 4, generator=g), torch.tensor([True, False]))
        # Reading a frozen state writes nothing and is allowed: the read-only protocol the LLM scorer uses.
        nothing = torch.zeros(2, 5, dtype=torch.bool)
        branch, final, _ = adapter.process_sequence(x, state + .1, 1, write=nothing)
        assert branch.shape == (2, 5, 5) and torch.equal(final, state + .1)
        assert torch.equal(adapter.write(state, torch.randn(2, 4, generator=g), torch.zeros(2, dtype=torch.bool)), state)


def test_unit_query_normalizes_the_read_address_and_default_keeps_the_raw_tensor():
    a = adapters()
    raw = torch.randn(3, 4, generator=torch.Generator().manual_seed(derive_seed(17, 'lsq-query'))) * 3.
    assert a['delta'].query_code(raw) is raw
    for store in ('ridge', 'sweep'):
        code = a[store].query_code(raw)
        torch.testing.assert_close(code, F.normalize(raw, dim=-1, eps=a[store].cfg.eps))
        torch.testing.assert_close(code.norm(dim=-1), torch.ones(3), rtol=0, atol=1e-6)
    state = torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(derive_seed(17, 'lsq-query-state')))
    z = torch.randn(3, 4, generator=torch.Generator().manual_seed(derive_seed(17, 'lsq-query-z')))
    read = lambda adapter: adapter._query(z, state, adapter.Q, adapter.U)
    torch.testing.assert_close(read(a['ridge']), F.linear(torch.matmul(
        state, F.normalize(F.linear(z, a['ridge'].Q), dim=-1, eps=1e-6).unsqueeze(-1)).squeeze(-1), a['ridge'].U))
    assert not torch.allclose(read(a['ridge']), read(a['delta']))


def test_unit_query_alone_is_the_delta_store_with_a_normalized_read():
    # Arm U of the LSQ study: the deployed delta store (learned lambda and beta, no new parameter) read with the unit query.
    def build(**extra):
        torch.manual_seed(derive_seed(17, 'lsq-arm-u'))
        return make_adapter(AdapterConfig(d_in=7, d_out=5, r=4, variant='iterq', **extra))

    plain, unit = build(), build(unit_query=True)
    assert (unit.cfg.store, unit.cfg.unit_query) == ('delta', True)
    assert [n for n, _ in plain.named_parameters()] == [n for n, _ in unit.named_parameters()]
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), unit.parameters()))
    g = torch.Generator().manual_seed(derive_seed(17, 'lsq-arm-u-data'))
    u, state0 = torch.randn(3, 9, 4, generator=g), torch.randn(3, 4, 4, generator=g) * .1
    valid, write = torch.rand(3, 9, generator=g) > .2, torch.rand(3, 9, generator=g) > .3
    assert torch.equal(unit.scan_final(state0, u, valid, write, None), plain.scan_final(state0, u, valid, write, None))
    for a, b in zip(unit.scan_prewrite(state0, u, valid, write, None), plain.scan_prewrite(state0, u, valid, write, None)):
        assert torch.equal(a, b)
    assert torch.equal(unit.write(state0, u[:, 0], write[:, 0]), plain.write(state0, u[:, 0], write[:, 0]))
    z = torch.randn(3, 4, generator=g)
    state = torch.randn(3, 4, 4, generator=g)
    assert torch.equal(unit._query(z, state, unit.Q, unit.U), F.linear(torch.matmul(
        state, F.normalize(F.linear(z, unit.Q), dim=-1, eps=unit.cfg.eps).unsqueeze(-1)).squeeze(-1), unit.U))
    assert not torch.allclose(unit._query(z, state, unit.Q, unit.U), plain._query(z, state, plain.Q, plain.U))
    raw = torch.randn(3, 4, generator=g) * 3.
    assert plain.query_code(raw) is raw
    torch.testing.assert_close(unit.query_code(raw).norm(dim=-1), torch.ones(3), rtol=0, atol=1e-6)


def frozen_memory():
    spec = importlib.util.spec_from_file_location(
        'memory_before_dpfp', Path(__file__).parent / 'assets' / 'memory_before_dpfp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('device', DEVICES)
def test_default_adapter_is_bitwise_the_pre_lsq_write_and_read(device):
    torch.manual_seed(derive_seed(17, 'lsq-off'))
    cfg = AdapterConfig(d_in=7, d_out=5, r=4, variant='iterq')
    assert (cfg.store, cfg.unit_query, cfg.store_eta, cfg.store_sweeps) == ('delta', False, 1e-3, 4)
    adapter = make_adapter(cfg).to(device)
    g = torch.Generator(device=device).manual_seed(derive_seed(17, 'lsq-off-data'))
    b, t = 4, 13
    u = torch.randn(b, t, 4, generator=g, device=device)
    state0 = torch.randn(b, 4, 4, generator=g, device=device) * .1
    valid = torch.rand(b, t, generator=g, device=device) > .2
    write = torch.rand(b, t, generator=g, device=device) > .4
    reset = torch.zeros_like(valid)
    reset[:, 0] = True
    frozen = frozen_memory()
    for r in (None, reset):
        expected = frozen.scan_final(state0, u, valid, write, r, K=adapter.K, V=adapter.V, lam=adapter.lam,
                                     beta=adapter.beta, eps=adapter.cfg.eps)
        assert torch.equal(adapter.scan_final(state0, u, valid, write, r), expected)
    snapshots, last = adapter.scan_prewrite(state0, u, valid, write, None)
    expected = frozen.scan_prewrite(state0, u, valid, write, None, K=adapter.K, V=adapter.V, lam=adapter.lam,
                                    beta=adapter.beta, eps=adapter.cfg.eps)
    assert torch.equal(snapshots, expected[0]) and torch.equal(last, expected[1])
    raw = torch.randn(b, 4, generator=g, device=device)
    assert adapter.query_code(raw) is raw
    z = torch.randn(b, 4, generator=g, device=device)
    original = F.linear(torch.matmul(state0, F.linear(z, adapter.Q).unsqueeze(-1)).squeeze(-1), adapter.U)
    assert torch.equal(adapter._query(z, state0, adapter.Q, adapter.U), original)


@pytest.mark.parametrize('kwargs', [
    dict(store='bogus'), dict(unit_query='yes'), dict(unit_query=1),
    dict(store_sweeps=0), dict(store_sweeps=True), dict(store_sweeps=2.5),
    dict(store_eta=0.), dict(store_eta=-1e-3), dict(store_eta=math.inf), dict(store_eta=math.nan), dict(store_eta=True),
    dict(variant='lora', store='ridge'), dict(variant='parallel', store='sweep'), dict(variant='loop', unit_query=True),
    dict(store='ridge', gate='input'), dict(store='sweep', gate='input'),
    dict(store='sweep', dpfp_nu=3), dict(unit_query=True, dpfp_nu=1), dict(store='ridge', dpfp_nu=1, unit_query=True)])
def test_config_rejects_invalid_store_settings(kwargs):
    with pytest.raises(ValueError):
        AdapterConfig(d_in=4, d_out=4, r=3, **kwargs)


def test_config_accepts_the_registered_store_settings():
    for variant in ('iterq', 'fixedq', 'unshared'):
        assert AdapterConfig(d_in=4, d_out=4, r=3, variant=variant, store='ridge', store_eta=1e-4, unit_query=True).store == 'ridge'
    assert AdapterConfig(d_in=4, d_out=4, r=3, store='sweep', store_sweeps=4, unit_query=True).store_sweeps == 4
    assert AdapterConfig(d_in=4, d_out=4, r=3, unit_query=True).store == 'delta'
    assert AdapterConfig(d_in=4, d_out=4, r=3, store='ridge', gate='scalar').key_dim == 3
    # SQ2 (docs/prereg/SQ2.md): the ridge store takes DPFP features (state width 2 * r_mem * nu); the sweep store still does not.
    assert AdapterConfig(d_in=4, d_out=4, r=3, store='ridge', dpfp_nu=3).key_dim == 18


def test_parameter_counts_are_the_delta_rule_counts():
    counts = {store: sum(p.numel() for p in adapter.parameters()) for store, adapter in adapters().items()}
    assert counts['delta'] == counts['ridge'] == counts['sweep']
