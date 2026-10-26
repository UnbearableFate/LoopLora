"""Final-only writes preserve reference states, gradients, masks and resets."""
import copy

import pytest
import torch
from torch.nn import functional as F

from looplora.core import memory
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(1, 1, 3, 5), (4, 19, 7, 3), (3, 48, 32, 32)])
@pytest.mark.parametrize('gate', ['scalar', 'input'])
@pytest.mark.parametrize('layout', ['random', 'all', 'none', 'initial_reset', 'interior_reset'])
def test_final_state_and_writer_gradients_match(device, shape, gate, layout):
    b, t, r, rm = shape
    generator = torch.Generator(device=device).manual_seed(derive_seed(17, 'scan-final', shape, gate, layout))
    adapter = make_adapter(AdapterConfig(d_in=r, d_out=r, r=r, r_mem=rm, gate=gate)).to(device)
    with torch.no_grad():
        for name in ('K', 'V', 'w_lambda', 'w_beta'):
            if hasattr(adapter, name):
                getattr(adapter, name).copy_(torch.randn(getattr(adapter, name).shape,
                                                        generator=generator, device=device) * .2)
    valid = torch.rand(b, t, generator=generator, device=device) > .2
    write = torch.rand(b, t, generator=generator, device=device) > .5
    reset = torch.zeros_like(valid)
    if layout == 'all':
        valid.fill_(True)
        write.fill_(True)
    elif layout == 'none':
        write.fill_(False)
    else:
        write[0].fill_(False)  # A whole row has no writes in a mixed batch.
    if layout == 'initial_reset':
        reset[:, 0] = True
    elif layout == 'interior_reset':
        reset[:, t // 2] = True  # Includes resets on invalid/non-write tokens.
    u0 = torch.randn(b, t, r, generator=generator, device=device) * .3
    state0 = torch.randn(b, rm, rm, generator=generator, device=device) * .1
    weight = torch.randn(b, rm, rm, generator=generator, device=device)
    results = []
    for fast in (False, True):
        current = copy.deepcopy(adapter)
        u = u0.clone().requires_grad_()
        initial = state0.clone().requires_grad_()
        output = (current.scan_final(initial, u, valid, write, reset) if fast
                  else current.scan_prewrite(initial, u, valid, write, reset)[1])
        names = ['K', 'V', 'a_lambda', 'a_beta'] + (['w_lambda', 'w_beta'] if gate == 'input' else [])
        parameters = [u, initial, *[getattr(current, name) for name in names]]
        loss = (output * weight).sum() + output.square().sum()
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        results.append((output, gradients))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-6)
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if layout not in {'all', 'initial_reset', 'interior_reset'}:
        assert torch.equal(results[1][0][0], state0[0])


def test_nonwrite_nan_inputs_and_gates_are_excluded_from_values_and_gradients():
    device = 'cuda'
    generator = torch.Generator(device=device).manual_seed(derive_seed(17, 'scan-final-nan'))
    valid = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], device=device, dtype=torch.bool)
    write = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]], device=device, dtype=torch.bool)
    enabled = valid & write
    u = torch.randn(3, 4, 3, device=device, generator=generator).masked_fill(~enabled[..., None], float('nan'))
    lam = torch.full((3, 4), .9, device=device).masked_fill(~enabled, float('nan'))
    beta = torch.full((3, 4), .1, device=device).masked_fill(~enabled, float('nan'))
    state = torch.randn(3, 3, 3, device=device, generator=generator)
    K, V = [torch.randn(3, 3, device=device, generator=generator) * .1 for _ in range(2)]
    results = []
    for fast in (False, True):
        values = [v.clone().requires_grad_() for v in (state, u, K, V, lam, beta)]
        initial, inputs, key, value, decay, update = values
        kwargs = dict(K=key, V=value, lam=decay, beta=update)
        output = (memory.scan_final(initial, inputs, valid, write, **kwargs) if fast else
                  memory.scan_prewrite(initial, inputs, valid, write, **kwargs)[1])
        gradients = torch.autograd.grad(output.square().sum(), values)
        assert all(torch.isfinite(g).all() for g in gradients)
        assert torch.equal(output[0], state[0])
        results.append((output, gradients))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-6)
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('b,t', [(0, 0), (0, 3), (2, 0)])
def test_empty_scans_keep_state(b, t):
    initial = torch.randn(b, 3, 3)
    out = memory.scan_final(initial, torch.randn(b, t, 4), K=torch.randn(3, 4), V=torch.randn(3, 4),
                            lam=.9, beta=.1)
    assert torch.equal(out, initial)


def test_gate_shapes_and_projection_shapes_are_checked():
    initial, u = torch.zeros(2, 3, 3), torch.zeros(2, 4, 5)
    kwargs = dict(K=torch.zeros(3, 5), V=torch.zeros(3, 5), lam=.9, beta=.1)
    with pytest.raises(ValueError, match='scan gates'):
        memory.scan_final(initial, u, **{**kwargs, 'lam': torch.ones(2)})
    with pytest.raises(ValueError, match='K and V'):
        memory.scan_final(initial, u, **{**kwargs, 'K': torch.ones(4, 5)})
    with pytest.raises(ValueError, match='eps'):
        memory.scan_final(initial, u, **kwargs, eps=0)


@pytest.mark.parametrize('input_site', ['o_proj', 'residual'])
@pytest.mark.parametrize('policy', ['fact_final', 'all_tokens'])
def test_fixed_p1l_memory_only_loss_matches_reference(input_site, policy, monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from looplora.core.adapters import MemoryAdapter
    from looplora.data.relations import generate
    from looplora.llm.inject import load_tokenizer
    from looplora.llm.pilot1_select import writer_config
    from looplora.llm.train import LLMExperiment, make_task_spec

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'scan-final-loss', input_site, policy))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    config = writer_config('scan-loss', 16, policy, .0003, 32, 1)
    config['model'].update(layer=1, r=4, input_site=input_site)
    experiment = LLMExperiment(config, model=model, tokenizer=load_tokenizer())
    generator = torch.Generator(device='cuda').manual_seed(derive_seed(17, 'scan-final-loss-B'))
    with torch.no_grad():
        experiment.adapter.B.normal_(0, .03, generator=generator)
    arrays = generate(make_task_spec(config), 'train', 0, 3, derive_seed(17, 'scan-final-fixed-batch'))

    def reference(self, *args, **kwargs):
        return self.scan_prewrite(*args, **kwargs)[1]

    results = []
    try:
        for fast in (False, True):
            with monkeypatch.context() as patch:
                if not fast:
                    patch.setattr(MemoryAdapter, 'scan_final', reference)
                experiment.adapter.zero_grad(set_to_none=True)
                prepared = experiment.prepare(arrays)
                logits, _ = experiment.score(prepared, 1)
                loss = F.cross_entropy(logits.float().flatten(0, 1), prepared.answer_ids.flatten())
                results.append((loss.detach(), prepared.snapshot.states[1].detach().clone()))
                loss.backward()
                assert experiment.adapter.A.grad.abs().sum() > 0
        torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=0)
        torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-5, atol=1e-6)
    finally:
        experiment.close()
