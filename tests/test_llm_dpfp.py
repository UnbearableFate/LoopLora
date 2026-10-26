"""WP-L5 DPFP arms on the learned LLM writer: configs, feature-space terms, controls, exports and gates.

Tiny-model tests (compute only). The real-model DPFP-off reproduction lives in test_llm_dpfp_repro.py.
"""
import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.memory import delta_update, dpfp_features
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm import read_fix, read_fix_run
from looplora.llm.dpfp import ARMS, SPECS, DPFPExperiment, configuration
from looplora.llm.dpfp_run import choose, eligible_e8, load_inference, scheduled_steps
from looplora.llm.inject import load_tokenizer
from looplora.llm.read_fix import ReadFixExperiment, controlled_score, diagnostics, extra_losses, question_terms
from looplora.llm.read_fix_run import fit, validate
from looplora.llm.train import make_task_spec
from looplora.llm.writer_fix import FixAdapter, alignment, auxiliary_losses, query_codes, true_fact_indices

RANK = 4  # tiny-model adapter rank = r_mem
KEY_DIM = {arm: 2 * RANK * nu for arm, (_, nu) in SPECS.items()}


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'dpfp-test-model'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield model
    assert not hasattr(model, 'looplora_adapters')


def tiny(arm):
    cfg = configuration(arm, 10)
    cfg['model'].update(layer=1, r=RANK)
    return cfg


def data(cfg):
    return generate(make_task_spec(cfg), 'train', 0, 2, derive_seed(17, 'dpfp-fixture'))


def off_diagonal_rms(keys):
    unit = F.normalize(keys, dim=-1)
    gram = unit @ unit.transpose(-1, -2)
    off = ~torch.eye(keys.shape[1], dtype=torch.bool, device=keys.device)
    return gram[:, off].square().mean(-1).sqrt()


@pytest.mark.parametrize('arm, rank, nu', [('D32-3', 32, 3), ('D128-1', 128, 1), ('D128-3', 128, 3)])
@pytest.mark.parametrize('E', [16, 8])
def test_configuration_is_wp_l4_R_plus_only_the_registered_dpfp_fields(arm, rank, nu, E):
    cfg = configuration(arm, 24000, E=E)
    base = read_fix.configuration('R', 24000, E=E, rank=rank)
    assert 'dpfp_nu' not in base['model'] and 'dpfp' not in base
    rest = copy.deepcopy(cfg)
    del rest['dpfp'], rest['model']['dpfp_nu']
    rest['name'] = base['name']
    assert rest == base
    assert cfg['name'] == f'dpfp-{arm}-E{E}'
    assert (cfg['model']['r'], cfg['model']['dpfp_nu'], cfg['task']['E']) == (rank, nu, E)
    assert (cfg['seed'], cfg['model']['layer'], cfg['train']['prefetch_workers']) == (17, 14, 2)
    assert cfg['read_fix']['arm'] == 'R' and cfg['read_fix']['read_weight'] == 1
    assert cfg['dpfp'] == {**cfg['dpfp'], 'arm': arm, 'base_arm': 'R', 'nu': nu, 'rank': rank, 'key_dim': 2 * rank * nu}
    assert AdapterConfig(d_in=1, d_out=1, r=rank, dpfp_nu=nu).key_dim == 2 * rank * nu
    assert configuration(arm, 20, smoke=True)['eval']['episodes'] == 100
    assert cfg['eval']['episodes'] == 1000


def test_registered_arms_dimensions_and_rejected_arms():
    assert ARMS == ('D32-3', 'D128-1', 'D128-3')
    assert [AdapterConfig(d_in=1, d_out=1, r=SPECS[a][0], dpfp_nu=SPECS[a][1]).key_dim for a in ARMS] == [192, 256, 768]
    for arm, kwargs in (('R', {}), ('D32-1', {}), ('D32-3', {'E': 4})):
        with pytest.raises(ValueError):
            configuration(arm, 10, **kwargs)


def test_experiment_rejects_inconsistent_arm_and_config(model, tokenizer):
    cfg = tiny('D32-3')
    cfg['model']['dpfp_nu'] = 1
    with pytest.raises(ValueError):
        DPFPExperiment(cfg, 'D32-3', model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError):
        DPFPExperiment(tiny('D32-3'), 'D128-1', model=model, tokenizer=tokenizer)
    cfg = tiny('D32-3')
    del cfg['dpfp']
    with pytest.raises(ValueError):
        DPFPExperiment(cfg, 'D32-3', model=model, tokenizer=tokenizer)


def test_dpfp_rejects_the_paired_W4_writer_and_keeps_the_fact_final_writers():
    initial = make_adapter(AdapterConfig(d_in=8, d_out=7, r=RANK, variant='iterq', dpfp_nu=2))
    with pytest.raises(ValueError, match='W4'):
        FixAdapter(initial, 'W4')
    assert FixAdapter(initial, 'W5').cfg.key_dim == 2 * RANK * 2


@pytest.mark.parametrize('arm', ['D32-3', 'D128-1'])
def test_dpfp_arm_writes_unit_phi_keys_and_every_address_term_lives_in_phi_space(model, tokenizer, arm):
    cfg, nu = tiny(arm), SPECS[arm][1]
    kd = KEY_DIM[arm]
    experiment = DPFPExperiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        adapter = experiment.adapter
        assert adapter.cfg.dpfp_nu == nu and adapter.cfg.key_dim == kd and experiment.read_arm == arm
        assert experiment.writer_signature()['dpfp_nu'] == nu and experiment.writer_signature()['read_fix'] == arm
        with torch.no_grad():
            adapter.B.fill_(.1)
        batch = experiment.prepare(data(cfg))
        state = batch.snapshot.states[1]
        assert state.shape == (2, RANK, kd)
        assert batch.keys.shape == (2, 16, kd) and batch.base_keys.shape == (2, 16, RANK)
        assert (batch.keys >= 0).all()
        norms = batch.keys.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0, atol=1e-5)
        # The memory is the ordered decay-first delta rule over exactly these unit phi keys and the raw values.
        expected = torch.zeros_like(state)
        for j in range(16):
            expected = delta_update(expected, batch.keys[:, j], batch.values[:, j], None, adapter.lam, adapter.beta)
        torch.testing.assert_close(state.detach(), expected.detach(), rtol=1e-4, atol=1e-5)
        # The keys are phi of the projected keys, normalized.
        projected = F.linear(adapter.u(batch.raw_support.gather(
            1, batch.positions[:, :, 0, None].expand(-1, -1, batch.raw_support.shape[-1])).float()), adapter.K)
        torch.testing.assert_close(F.normalize(dpfp_features(projected, nu), dim=-1, eps=adapter.cfg.eps).detach(),
                                   batch.keys.detach(), rtol=1e-4, atol=1e-5)

        logits, x = experiment.score(batch, 1, capture_inputs=True)
        terms = question_terms(experiment, batch, x)
        q = terms['q']
        assert q.shape == (2, 4, kd) and (q >= 0).all()
        qn = q.norm(dim=-1)
        torch.testing.assert_close(qn, torch.ones_like(qn), rtol=0, atol=1e-5)
        assert torch.equal(q, F.normalize(dpfp_features(F.linear(terms['u'], adapter.Q), nu), dim=-1, eps=adapter.cfg.eps))
        assert torch.equal(terms['stored'], torch.einsum('bij,bqj->bqi', state, q))
        assert torch.equal(terms['m'], terms['stored'])

        # InfoNCE addressing and the alignment metrics use phi-space dot products q_hat . k_hat_j.
        assert torch.equal(query_codes(experiment, batch, x), q)
        address, value = auxiliary_losses(experiment, batch, x)
        targets = true_fact_indices(batch.arrays, x.device)
        phi_scores = torch.einsum('bqk,bek->bqe', q, batch.keys)
        torch.testing.assert_close(address, F.cross_entropy((phi_scores / .1).flatten(0, 1), targets.flatten()))
        base_q = F.normalize(F.linear(terms['u'], adapter.Q), dim=-1, eps=adapter.cfg.eps)
        base_scores = torch.einsum('bqr,ber->bqe', base_q, F.normalize(batch.base_keys, dim=-1))
        base_address = F.cross_entropy((base_scores / .1).flatten(0, 1), targets.flatten())
        assert not torch.isclose(address, base_address, rtol=0, atol=1e-6)
        metric = alignment(experiment, batch, x)
        assert torch.equal(metric['key_top1'], phi_scores.argmax(-1) == targets)

        # The raw-read loss reads M q_hat; its gradients reach the query, key, value and beta parameters.
        read, decor = extra_losses(experiment, batch, x)
        assert read > 0 and decor == 0 and address > 0 and value > 0
        gradients = torch.autograd.grad(read, [adapter.Q, adapter.V, adapter.K, adapter.a_beta], retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)

        # Diagnostics: the Gram reconstruction holds in phi space, key overlap is reported in both spaces.
        diag, exact = diagnostics(experiment, batch, x)
        assert diag['gram_relative_error'].max() < 1e-4
        assert exact['gram'].shape == (2, 16, 16) and exact['d'].shape == (2, 4, 16)
        torch.testing.assert_close(diag['key_gram_rms'][:, 0], off_diagonal_rms(batch.keys))
        torch.testing.assert_close(diag['key_gram_rms_base'][:, 0], off_diagonal_rms(batch.base_keys))
        assert diag['key_gram_max_abs_base'].shape == (2, 4)
        assert not torch.allclose(diag['key_gram_rms'], diag['key_gram_rms_base'])
    finally:
        experiment.close()


def test_dpfp_off_experiment_keeps_the_wp_l4_wiring(model, tokenizer):
    cfg = read_fix.configuration('R', 10)
    cfg['model'].update(layer=1, r=RANK)
    experiment = ReadFixExperiment(cfg, 'R', model=model, tokenizer=tokenizer)
    try:
        adapter = experiment.adapter
        assert adapter.cfg.dpfp_nu == 0 and adapter.cfg.key_dim == RANK and adapter.base_keys is None
        assert 'dpfp_nu' not in experiment.writer_signature()
        with torch.no_grad():
            adapter.B.fill_(.1)
        batch = experiment.prepare(data(cfg))
        assert batch.base_keys is None and batch.snapshot.states[1].shape == (2, RANK, RANK)
        logits, x = experiment.score(batch, 1, capture_inputs=True)
        terms = question_terms(experiment, batch, x)
        raw = F.linear(terms['u'], adapter.Q)
        assert torch.equal(terms['q'], raw)
        assert torch.equal(query_codes(experiment, batch, x), F.normalize(raw, dim=-1, eps=adapter.cfg.eps))
        diag, _ = diagnostics(experiment, batch, x)
        assert 'key_gram_rms' in diag and not any(key.endswith('_base') for key in diag)
    finally:
        experiment.close()


def test_zero_and_shuffled_memory_controls_work_on_the_rectangular_state(model, tokenizer):
    arm = 'D32-3'
    cfg = tiny(arm)
    experiment = DPFPExperiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        with torch.no_grad():
            experiment.adapter.B.fill_(.2)
            batch = experiment.prepare(data(cfg))
            donor = replace(batch, values=batch.values.flip(0),
                            snapshot=replace(batch.snapshot, states={1: batch.snapshot.states[1].flip(0)}))
            observed = []
            hook = experiment.bank.modules[1].register_forward_pre_hook(
                lambda module, inputs: observed.append(module.context.state.clone()))
            try:
                original, x = controlled_score(experiment, batch)
                zero, zx = controlled_score(experiment, batch, 'zero')
                shuffled, sx = controlled_score(experiment, batch, 'shuffle', donor)
            finally:
                hook.remove()
            assert [tuple(state.shape) for state in observed] == [(8, RANK, KEY_DIM[arm])] * 3
            assert torch.equal(x, zx) and torch.equal(x, sx)
            assert observed[1].count_nonzero() == 0
            assert torch.equal(observed[2], donor.snapshot.states[1].repeat_interleave(4, 0))
            assert not torch.equal(original, zero) and not torch.equal(original, shuffled)
            diag, exact = diagnostics(experiment, batch, x)
            assert diag['gram_relative_error'].max() < 1e-4
    finally:
        experiment.close()


def test_training_loop_validation_and_export_reload_with_dpfp(model, tokenizer, tmp_path):
    arm = 'D128-1'
    cfg = tiny(arm)
    experiment = DPFPExperiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        result = fit(experiment, tmp_path, 5, evaluate=False)
        assert result['actual_steps'] == 5 and result['metrics'] == {}
        rows = [json.loads(line) for line in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == [1, 5]
        assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in rows)
        assert rows[0]['read_loss'] > 0 and rows[0]['address_loss'] > 0 and rows[0]['value_loss'] > 0
        assert rows[0]['decorrelation_loss'] == 0
        exported = torch.load(tmp_path / 'writer.pt', weights_only=True)
        assert 'value_head' not in exported and 'optimizer' not in exported
        assert exported['arm'] == arm and exported['signature']['dpfp_nu'] == 1
        assert exported['adapter']['K'].shape == (RANK, exported['adapter']['A'].shape[0])
        metrics = validate(experiment, tmp_path / 'val', 16)
        assert metrics['n'] == 16 and 0 <= metrics['em'] <= 1 and math.isfinite(metrics['loss'])
        assert {'key_gram_rms', 'key_gram_rms_base', 'gram_relative_error'} <= set(metrics)
        with np.load(tmp_path / 'val' / 'samples.npz') as samples:
            assert set(samples['arm'].tolist()) == {arm} and samples['key_gram_rms_base'].shape == (16,)
        with torch.no_grad():
            expected, _ = experiment.score(experiment.prepare(data(cfg)), 1)
    finally:
        experiment.close()
    # A WP-L4 loader cannot take a DPFP export; the DPFP loader reproduces its logits exactly.
    with pytest.raises(ValueError):
        read_fix_run.load_inference(tmp_path / 'writer.pt', model=model, tokenizer=tokenizer)
    loaded = load_inference(tmp_path / 'writer.pt', model=model, tokenizer=tokenizer)
    try:
        assert loaded.read_arm == arm and loaded.value_head is None and loaded.adapter.cfg.dpfp_nu == 1
        with torch.no_grad():
            actual, _ = loaded.score(loaded.prepare(data(cfg)), 1)
        assert torch.equal(actual, expected)
    finally:
        loaded.close()


def records(scores):
    return [dict(arm=arm, status='complete', smoke=False, config=dict(task=dict(E=16)),
                 training=dict(actual_steps=10, metrics={'10': dict(correct=scores.get(arm, 0))})) for arm in ARMS]


def test_selection_follows_candidate_order_thresholds_and_single_E8_gate():
    assert choose(records({'D32-3': 800, 'D128-1': 999, 'D128-3': 1000}))['arm'] == 'D32-3'
    assert choose(records({'D32-3': 799, 'D128-1': 800, 'D128-3': 999}))['arm'] == 'D128-1'
    assert choose(records({'D128-3': 800}))['action'] == 'select'
    choice = choose(records({'D32-3': 799, 'D128-1': 799}))  # ties keep the candidate order
    assert (choice['action'], choice['arm'], choice['E']) == ('retry_e8', 'D32-3', 8)
    choice = choose(records({'D32-3': 600, 'D128-3': 700}))
    assert (choice['action'], choice['arm'], choice['E']) == ('retry_e8', 'D128-3', 8)
    assert choose(records({'D128-1': 500}))['action'] == 'retry_e8'
    stopped = choose(records({'D32-3': 499, 'D128-1': 300, 'D128-3': 10}))
    assert (stopped['action'], stopped['arm'], stopped['E']) == ('stop', None, None)
    assert stopped['correct'] == {'D32-3': 499, 'D128-1': 300, 'D128-3': 10}
    with pytest.raises(ValueError):
        choose(records({})[:-1])
    with pytest.raises(ValueError):
        choose(records({}) + [records({})[0]])
    with pytest.raises(ValueError):
        choose([{**r, 'smoke': True} for r in records({})])
    with pytest.raises(ValueError):
        choose([{**r, 'status': 'failed'} for r in records({})])
    short = records({})
    short[1]['config']['task']['E'] = 8
    with pytest.raises(ValueError):
        choose(short)
    lock = dict(choice=choose(records({'D32-3': 600, 'D128-3': 700})))
    assert eligible_e8(lock, 'D128-3') == 8
    for arm in ('D32-3', 'D128-1'):
        with pytest.raises(ValueError):
            eligible_e8(lock, arm)
    with pytest.raises(ValueError):
        eligible_e8(dict(choice=choose(records({'D32-3': 800}))), 'D32-3')


def test_step_counts_come_from_the_release_or_the_explicit_smoke_count():
    release = dict(study='DPFP', E=16, steps_by_arm={'D32-3': 20000, 'D128-1': 19000, 'D128-3': 18000})
    assert [scheduled_steps(arm, None, release) for arm in ARMS] == [20000, 19000, 18000]
    assert scheduled_steps('D32-3', 2400, None, smoke=True) == 2400
    for call in (lambda: scheduled_steps('D32-3', 8225, release),
                 lambda: scheduled_steps('D32-3', None, None),
                 lambda: scheduled_steps('D32-3', None, release, E=8),
                 lambda: scheduled_steps('D32-3', None, {**release, 'study': 'P1L'}),
                 lambda: scheduled_steps('D32-3', None, {**release, 'steps_by_arm': {'D32-3': 20001}}),
                 lambda: scheduled_steps('D32-3', None, {**release, 'steps_by_arm': {'D32-3': True}}),
                 lambda: scheduled_steps('D32-3', 4, None, smoke=True),
                 lambda: scheduled_steps('D32-3', None, None, smoke=True),
                 lambda: scheduled_steps('D32-3', 2400, release, smoke=True)):
        with pytest.raises(ValueError):
            call()
