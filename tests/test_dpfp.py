"""DPFP features, the rectangular delta-rule memory and the unchanged DPFP-off path (compute only)."""
import importlib.util
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from looplora.core import memory
from looplora.core.accounting import macs_per_token, param_count, state_bytes
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed
from looplora.llm.read_fix import gram_read

DTYPE = torch.float64


def reference_dpfp(k, nu):
    """Loop form of Schlag, Irie & Schmidhuber (2021, sec. 5.4), independent of torch.roll."""
    n = 2 * len(k)
    r = np.maximum(np.concatenate([k, -k]), 0.)
    out = np.empty(n * nu)
    for j in range(1, nu + 1):
        for i in range(n):
            out[(j - 1) * n + i] = r[i] * r[(i - j) % n]
    return out


@pytest.mark.parametrize('nu', [1, 2, 3])
@pytest.mark.parametrize('d', [1, 5, 32])
def test_dpfp_matches_the_reference_definition_and_is_nonnegative_and_degree_two_homogeneous(d, nu):
    g = torch.Generator().manual_seed(derive_seed(17, 'dpfp-reference', d, nu))
    k = torch.randn(4, d, generator=g, dtype=DTYPE)
    actual = memory.dpfp_features(k, nu)
    assert actual.shape == (4, 2 * d * nu)
    expected = torch.from_numpy(np.stack([reference_dpfp(row.numpy(), nu) for row in k]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual >= 0).all()
    for c in (.5, 3.):
        torch.testing.assert_close(memory.dpfp_features(c * k, nu), c ** 2 * actual, rtol=1e-12, atol=1e-12)
    assert memory.dpfp_features(k.reshape(2, 2, d), nu).shape == (2, 2, 2 * d * nu)


def test_dpfp_literal_examples_fix_the_block_order_and_the_roll_direction():
    assert memory.dpfp_features(torch.tensor([1., -2.]), 1).tolist() == [2., 0., 0., 0.]
    # relu([k; -k]) = [1, 0, 2, 0, 1, 0]; roll 1 gives all-zero products, roll 2 gives [1, 0, 2, 0, 2, 0].
    assert memory.dpfp_features(torch.tensor([1., -1., 2.]), 2).tolist() == [0.] * 6 + [1., 0., 2., 0., 2., 0.]
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            memory.dpfp_features(torch.ones(3), bad)


@pytest.mark.parametrize('nu', [1, 3])
@pytest.mark.parametrize('lam', [1., .7])
def test_unit_dpfp_keys_overwrite_exactly_at_beta_one(nu, lam):
    g = torch.Generator().manual_seed(derive_seed(17, 'dpfp-overwrite', nu))
    d, r_mem = 6, 5
    key = F.normalize(memory.dpfp_features(torch.randn(3, d, generator=g, dtype=DTYPE), nu), dim=-1)
    assert key.shape == (3, 2 * d * nu)
    state = torch.randn(3, r_mem, 2 * d * nu, generator=g, dtype=DTYPE)
    first, second = (torch.randn(3, r_mem, generator=g, dtype=DTYPE) for _ in range(2))
    once = memory.delta_update(state, key, first, None, lam, 1.)
    twice = memory.delta_update(once, key, second, None, lam, 1.)
    assert once.shape == twice.shape == state.shape
    for stored, value in ((once, first), (twice, second)):
        torch.testing.assert_close(torch.einsum('bij,bj->bi', stored, key), value, rtol=0, atol=1e-12)


def test_write_with_the_feature_map_stores_the_unit_feature_key_and_checks_shapes():
    g = torch.Generator().manual_seed(derive_seed(17, 'dpfp-write'))
    B, r, r_mem, nu = 3, 4, 5, 2
    K, V = (torch.randn(r_mem, r, generator=g, dtype=DTYPE) for _ in range(2))
    u = torch.randn(B, r, generator=g, dtype=DTYPE)
    state = torch.randn(B, r_mem, 2 * r_mem * nu, generator=g, dtype=DTYPE)
    feature_map = partial(memory.dpfp_features, nu=nu)
    written = memory.write(state, u, None, K=K, V=V, lam=1., beta=1., feature_map=feature_map)
    key = F.normalize(memory.dpfp_features(F.linear(u, K), nu), dim=-1)
    torch.testing.assert_close(torch.einsum('bij,bj->bi', written, key), F.linear(u, V), rtol=0, atol=1e-12)
    with pytest.raises(ValueError):  # a square-state writer cannot address a feature-space state
        memory.write(state, u, None, K=K, V=V, lam=1., beta=1.)
    assert memory.zeros_state(2, 5, DTYPE, 'cpu', 30).shape == (2, 5, 30)
    assert memory.zeros_state(2, 5, DTYPE, 'cpu').shape == (2, 5, 5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('layout', ['random', 'all', 'initial_reset', 'interior_reset'])
@pytest.mark.parametrize('nu', [1, 3])
def test_dpfp_fast_scan_matches_the_reference_scan_and_explicit_delta_updates(device, layout, nu):
    b, t, r, rm = 4, 11, 5, 4
    g = torch.Generator(device=device).manual_seed(derive_seed(17, 'dpfp-scan', layout, nu))

    def rand(*shape, scale=1.):
        return torch.randn(*shape, generator=g, device=device, dtype=DTYPE) * scale

    K, V = rand(rm, r, scale=.7), rand(rm, r, scale=.7)
    u0, state0 = rand(b, t, r), rand(b, rm, 2 * rm * nu, scale=.1)
    valid = torch.rand(b, t, generator=g, device=device) > .2
    write = torch.rand(b, t, generator=g, device=device) > .5
    reset = torch.zeros_like(valid)
    if layout == 'all':
        valid.fill_(True)
        write.fill_(True)
    else:
        write[0].fill_(False)  # a whole row without writes in a mixed batch
    if layout == 'initial_reset':
        reset[:, 0] = True
    elif layout == 'interior_reset':
        reset[:, t // 2] = True
    weight = rand(b, rm, 2 * rm * nu)
    feature_map = partial(memory.dpfp_features, nu=nu)
    results = []
    for fast in (False, True):
        leaves = [x.clone().requires_grad_() for x in (u0, state0, K, V,
            torch.tensor(.9, dtype=DTYPE, device=device), torch.tensor(.6, dtype=DTYPE, device=device))]
        u, initial, key, value, lam, beta = leaves
        kwargs = dict(K=key, V=value, lam=lam, beta=beta, feature_map=feature_map)
        output = (memory.scan_final(initial, u, valid, write, reset, **kwargs) if fast
                  else memory.scan_prewrite(initial, u, valid, write, reset, **kwargs)[1])
        loss = (output * weight).sum() + output.square().sum()
        results.append((output, torch.autograd.grad(loss, leaves, allow_unused=True)))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-10, atol=1e-12)
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    if layout == 'all':
        expected = state0
        for j in range(t):
            code = F.normalize(memory.dpfp_features(F.linear(u0[:, j], K), nu), dim=-1, eps=1e-6)
            expected = memory.delta_update(expected, code, F.linear(u0[:, j], V), None, .9, .6)
        torch.testing.assert_close(results[1][0], expected, rtol=1e-10, atol=1e-12)


def frozen_memory():
    spec = importlib.util.spec_from_file_location(
        'memory_before_dpfp', Path(__file__).parent / 'assets' / 'memory_before_dpfp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def memory_run(module, data, grads, **extra):
    leaves = [data[name].clone().requires_grad_(grads) for name in ('u', 'state', 'K', 'V', 'lam', 'beta')]
    u, state, K, V, lam, beta = leaves
    masks = (data['valid'], data['write'], data['reset'])
    kwargs = dict(K=K, V=V, lam=lam, beta=beta, **extra)
    snapshots, last = module.scan_prewrite(state, u, *masks, **kwargs)
    outputs = [module.scan_final(state, u, *masks, **kwargs), snapshots, last,
               module.write(state, u[:, 0], data['write'][:, 0], K=K, V=V, lam=lam[:, 0], beta=beta[:, 0], **extra)]
    gradients = None
    if grads:
        loss = sum((o * w).sum() for o, w in zip(outputs, data['weights']))
        gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
    return outputs, gradients


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('layout', ['random', 'all', 'initial_reset', 'interior_reset'])
def test_dpfp_off_memory_is_bitwise_the_pre_dpfp_memory(device, dtype, layout):
    b, t, r, rm = 4, 13, 6, 5
    g = torch.Generator(device=device).manual_seed(derive_seed(17, 'dpfp-off', str(dtype), layout))

    def rand(*shape, scale=1.):
        return (torch.randn(*shape, generator=g, device=device) * scale).to(dtype)

    valid = torch.rand(b, t, generator=g, device=device) > .2
    write = torch.rand(b, t, generator=g, device=device) > .4
    reset = torch.zeros_like(valid)
    if layout == 'all':
        valid.fill_(True)
        write.fill_(True)
    if layout == 'initial_reset':
        reset[:, 0] = True
    elif layout == 'interior_reset':
        reset[:, t // 2] = True
    data = dict(u=rand(b, t, r), state=rand(b, rm, rm, scale=.1), K=rand(rm, r, scale=.5), V=rand(rm, r, scale=.5),
                lam=(torch.rand(b, t, generator=g, device=device) * .3 + .65).to(dtype),
                beta=(torch.rand(b, t, generator=g, device=device) * .8 + .1).to(dtype),
                valid=valid, write=write, reset=reset,
                weights=[rand(b, rm, rm), rand(b, t, rm, rm), rand(b, rm, rm), rand(b, rm, rm)])
    grads = device == 'cpu'  # CPU gradients are deterministic; the CUDA path compares forward values
    reference, reference_grads = memory_run(frozen_memory(), data, grads)
    for extra in ({}, {'feature_map': None}):
        outputs, gradients = memory_run(memory, data, grads, **extra)
        assert all(torch.equal(a, b) for a, b in zip(outputs, reference))
        if grads:
            for a, b in zip(gradients, reference_grads, strict=True):
                assert (a is None) == (b is None)
                assert a is None or torch.equal(a, b)


def test_adapter_config_key_dim_and_validation():
    cfg = AdapterConfig(d_in=4, d_out=4, r=3, r_mem=5)
    assert cfg.dpfp_nu == 0 and cfg.key_dim == 5
    assert AdapterConfig(d_in=4, d_out=4, r=3, r_mem=5, dpfp_nu=3).key_dim == 30
    assert AdapterConfig(d_in=4, d_out=4, r=3, dpfp_nu=1).key_dim == 6
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            AdapterConfig(d_in=4, d_out=4, r=3, dpfp_nu=bad)
    with pytest.raises(ValueError):
        AdapterConfig(d_in=4, d_out=4, r=3, variant='lora', dpfp_nu=1)


def explicit_dpfp_adapter(adapter, x, state0, valid, write, reset, steps):
    """Token-by-token DPFP reader and writer written from the registered equations."""
    cfg = adapter.cfg
    state = [row.clone() for row in state0]
    outputs = []
    for t in range(x.shape[1]):
        out = torch.zeros(x.shape[0], cfg.d_out, dtype=x.dtype)
        for b in range(x.shape[0]):
            if reset[b, t]:
                state[b] = torch.zeros_like(state[b])
            if valid[b, t]:
                u = adapter.A @ x[b, t]
                z = u
                for _ in range(steps):
                    q = F.normalize(memory.dpfp_features(adapter.Q @ z, cfg.dpfp_nu), dim=-1, eps=cfg.eps)
                    z = u + cfg.rho * torch.relu(adapter.R @ z + adapter.U @ (state[b] @ q))
                out[b] = cfg.alpha * (adapter.B @ z)
        outputs.append(out)
        for b in range(x.shape[0]):
            if valid[b, t] and write[b, t]:
                u = adapter.A @ x[b, t]
                feature = memory.dpfp_features(adapter.K @ u, cfg.dpfp_nu)
                key = feature / torch.linalg.vector_norm(feature).clamp_min(cfg.eps)
                decayed = adapter.lam * state[b]
                state[b] = decayed + adapter.beta * torch.outer(adapter.V @ u - decayed @ key, key)
    return torch.stack(outputs, 1), torch.stack(state)


@pytest.mark.parametrize('nu', [1, 3])
@pytest.mark.parametrize('steps', [1, 2])
def test_dpfp_adapter_paths_match_the_explicit_feature_space_equations(nu, steps):
    torch.manual_seed(derive_seed(17, 'dpfp-adapter', nu, steps))
    cfg = AdapterConfig(d_in=4, d_out=6, r=3, r_mem=5, variant='iterq', alpha=.7, rho=.31,
                        lambda_init=.81, beta_init=.23, dpfp_nu=nu)
    adapter = make_adapter(cfg).double()
    with torch.no_grad():
        adapter.B.normal_(std=.4)
    x = torch.randn(3, 7, 4, dtype=DTYPE)
    state0 = torch.randn(3, 5, cfg.key_dim, dtype=DTYPE) * .4
    valid, write, reset = torch.rand(3, 7) > .3, torch.rand(3, 7) > .35, torch.rand(3, 7) > .7
    valid[0, 1], reset[0, 1] = False, True
    valid[1, 2], write[1, 2] = True, False
    with torch.no_grad():
        expected, final = explicit_dpfp_adapter(adapter, x, state0, valid, write, reset, steps)
        assert final.shape == (3, 5, cfg.key_dim)
        for process in (adapter.process_sequence, adapter.process_sequence_fast):
            branch, state, stats = process(x, state0, steps, valid=valid, write=write, reset=reset)
            torch.testing.assert_close(branch, expected, rtol=1e-11, atol=1e-12)
            torch.testing.assert_close(state, final, rtol=1e-11, atol=1e-12)
        assert adapter.process_sequence(x, None, 1)[1].shape == (3, 5, cfg.key_dim)
        with pytest.raises(ValueError):  # a square state does not fit the feature-space adapter
            adapter.process_sequence(x, torch.zeros(3, 5, 5, dtype=DTYPE), 1)


def test_gram_read_reconstructs_the_feature_space_read():
    g = torch.Generator().manual_seed(derive_seed(17, 'dpfp-gram-read'))
    E, Q, r_mem, d, nu = 7, 4, 5, 5, 2
    keys = F.normalize(memory.dpfp_features(torch.randn(3, E, d, generator=g, dtype=DTYPE), nu), dim=-1)
    q = F.normalize(memory.dpfp_features(torch.randn(3, Q, d, generator=g, dtype=DTYPE), nu), dim=-1)
    values = torch.randn(3, E, r_mem, generator=g, dtype=DTYPE)
    lam, beta = torch.tensor(.91, dtype=DTYPE), torch.tensor(.7, dtype=DTYPE)
    state = torch.zeros(3, r_mem, 2 * d * nu, dtype=DTYPE)
    for j in range(E):
        state = memory.delta_update(state, keys[:, j], values[:, j], None, lam, beta)
    observed = gram_read(keys, values, q, lam, beta)
    torch.testing.assert_close(observed['reconstructed'], torch.einsum('bij,bqj->bqi', state, q),
                               rtol=1e-12, atol=1e-12)
    assert observed['gram'].shape == (3, E, E) and observed['d'].shape == (3, Q, E)


def test_accounting_counts_the_rectangular_state_and_no_extra_parameters():
    off = AdapterConfig(d_in=7, d_out=11, r=3, r_mem=5)
    on = AdapterConfig(d_in=7, d_out=11, r=3, r_mem=5, dpfp_nu=3)  # key_dim 30
    assert param_count(on) == param_count(off)
    assert param_count(on) == sum(p.numel() for p in make_adapter(on).parameters() if p.requires_grad)
    assert state_bytes(off, batch=2, n_modules=4, dtype=torch.float32) == 4 * 2 * 25 * 4
    assert state_bytes(on, batch=2, n_modules=4, dtype=torch.float32) == 4 * 2 * 5 * 30 * 4
    read, writer = 3 ** 2 + 2 * 3 * 5 + 5 * 30, 2 * 3 * 5 + 2 * 5 * 30
    assert macs_per_token(on, 2, 1) == 3 * (7 + 11) + 2 * read + writer
