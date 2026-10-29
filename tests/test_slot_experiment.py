"""SLOT-LOOP-v1: the slot read path on a tiny model, the old path's identity, the freeze verification, the export, the trainer and
the S0 pipeline (docs/prereg/SLOT-LOOP-v1.md, "Verification list").

Compute only (imports torch; a 2-layer backbone with hidden size 16 and the registered adapter: rank 128, DPFP nu 3, key
dimension 768, ridge store, T1 with E = 16). The adapter sits at the last layer, so only the final token's read reaches the
logits. Backend semantics are in `test_slot_backend.py`; the registry, guard and rules are in `test_slot_data.py`.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import slot_golden
from looplora.core.memory_backend import (HardSlotBackend, MatrixBackend, MatrixState, MemoryRead, SoftSlotBackend, build_slots)
from looplora.infra.seeds import derive_seed
from looplora.llm import slot_data, slot_preflight, slot_rules, slot_run
from looplora.llm.batching import repeat_snapshot
from looplora.llm.evaluate import _slice
from looplora.llm.inject import load_tokenizer
from looplora.llm.prefetch import BatchStream
from looplora.llm.protocols import SupportSnapshot
from looplora.llm.slot_experiment import (adopt, digest_report, episodes, evaluate, freeze_for_s1a, gradient_report, load_export,
                                          raw_episodes, save_export, support_digest, tensor_digests, trainable_report)
from looplora.llm.slot_read import ReadSpec, ScopedRead, SlotFixAdapter, bind_batch
from looplora.llm.slot_train import dev_row, fit
from looplora.llm.train import make_task_spec

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'tests' / 'assets' / 'slot_golden_digests.json'
TINY = slot_data.EvalSet('T', 'tiny', 'val', 0, 48, derive_seed(17, 'slot-test-eval'))


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    model = slot_golden.tiny_model()
    yield model
    assert not hasattr(model, 'looplora_adapters')


@pytest.fixture
def experiment(model, tokenizer):
    experiment = slot_golden.tiny_experiment(model, tokenizer)
    try:
        yield adopt(experiment)
    finally:
        experiment.close()


def eval_arrays(count=16, start=0):
    return _slice(episodes(TINY), slice(start, start + count))


def train_arrays(count=8):
    return _slice(raw_episodes(TINY), slice(0, count))


# ----------------------------------------------------------------------------------------------------- the registered layout

def test_the_fixture_is_the_registered_layout(experiment):
    cfg = experiment.adapter.cfg
    assert make_task_spec(experiment.config) == slot_data.task_spec()
    assert (cfg.key_dim, cfg.r_mem, cfg.r, cfg.dpfp_nu, cfg.store) == (768, 128, 128, 3, 'ridge')
    assert isinstance(experiment.adapter, SlotFixAdapter) and experiment.value_head is None
    assert experiment.adapter.A_query is None and {n for n, _ in experiment.adapter.named_buffers()} == {'input_mean', 'input_std'}


# --------------------------------------------------------------------------------------------- the old path, unchanged

@pytest.mark.parametrize('shape', ['train', 'eval'])
def test_matrix_backend_logits_and_gradients_equal_native_bitwise(experiment, shape):
    """Matrix (g = 1) is the original read: logits at L = 1, 2, 4 for four questions per episode and for one, and the
    gradients of one training loss."""
    arrays = train_arrays(8) if shape == 'train' else eval_arrays(16)
    with torch.no_grad():
        batch = experiment.prepare(arrays)
        bound = bind_batch(experiment, batch, ReadSpec('matrix'))
        for depth in (1, 2, 4):
            native, _ = experiment.score(batch, depth)
            matrix, _ = experiment.score(bound, depth)
            assert torch.equal(native, matrix)
        soft, _ = experiment.score(bind_batch(experiment, batch, ReadSpec('soft', tau=0.1)), 1)
        assert not torch.equal(experiment.score(batch, 1)[0], soft)  # the reads matter in this fixture
    if shape == 'train':
        report = slot_preflight.matrix_gradients(experiment, arrays)
        assert report['bitwise'], report['max_abs_difference']


def test_the_old_matrix_path_is_bitwise_unchanged(tmp_path):
    """The state, the logits (L = 1, 2, 4; four questions and one) and the training gradients of the S-RE-style tiny experiment
    hash to the digests of the source before any slot code (`tests/assets/slot_golden_digests.json`, captured with
    `tests/slot_golden.py --old-commit`). Fresh process, as the DPFP reproduction test does."""
    recorded = json.loads(GOLDEN.read_text())
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT / 'src'), os.environ.get('PYTHONPATH')]))}
    run = subprocess.run([sys.executable, str(ROOT / 'tests' / 'slot_golden.py')], env=env, check=True, capture_output=True,
                         text=True, timeout=900)
    now = json.loads(run.stdout.strip().splitlines()[-1])
    assert set(now) == set(recorded['digests'])
    assert {k: v for k, v in now.items() if v != recorded['digests'][k]} == {}


def test_tensor_states_still_take_the_original_paths(experiment):
    state = torch.randn(3, 4, 5)
    snapshot = SupportSnapshot({1: state}, None, torch.ones(3, 2, dtype=torch.bool), 'memory_only')
    assert torch.equal(repeat_snapshot(snapshot, 4).states[1], state.repeat_interleave(4, dim=0))
    assert repeat_snapshot(snapshot, 1).states[1].shape == state.shape
    with pytest.raises(ValueError):
        repeat_snapshot(snapshot, 0)
    wide = torch.zeros(2, 128, 768, dtype=torch.float64)
    for refused in (wide, MemoryRead.bind(MatrixBackend(), MatrixState(wide))):
        with pytest.raises(ValueError, match='fp32'):
            with experiment.bank.use({experiment.layer: refused}, valid=torch.ones(2, 3, dtype=torch.bool)):
                pass


# ------------------------------------------------------------------------------------------------- placement of the reads

@torch.no_grad()
def test_reads_are_placed_at_the_registered_tokens_for_each_owner(experiment):
    """Against a per-token reference built from the adapter's own pieces: every valid token reads (all_valid_question), the
    scoped two-state diagnostic replaces only the final valid token, the oracle puts the row's true value there."""
    adapter, device, cfg = experiment.adapter, experiment.device, experiment.adapter.cfg
    g = torch.Generator().manual_seed(3)
    B, Q, T, S = 2, 2, 5, 4
    unit = lambda t: t / t.norm(dim=-1, keepdim=True)
    keys = unit(torch.rand(B, S, cfg.key_dim, generator=g)).to(device)
    values = (torch.randn(B, S, cfg.r_mem, generator=g) * 3 + torch.tensor([10., -10.])[:, None, None]).to(device)
    matrix = torch.einsum('bsv,bsk->bvk', values, keys)
    x = torch.randn(B * Q, T, cfg.d_in, generator=g).to(device)
    valid = torch.ones(B * Q, T, dtype=torch.bool, device=device)
    valid[1, 3:], valid[2, :2], valid[3, 1] = False, False, False
    owner = torch.arange(B, device=device).repeat_interleave(Q)
    last = (valid * torch.arange(1, T + 1, device=device)).amax(1) - 1
    u = adapter.u_read(torch.where(valid[..., None], x, torch.zeros_like(x)))
    oracle = (torch.randn(B, cfg.r_mem, generator=g) * 5).to(device)
    q_of = lambda z: adapter.query_code(F.linear(z, adapter.Q))
    soft, hard, slots = SoftSlotBackend(0.3, 1.5), HardSlotBackend(), build_slots(keys, values, S)
    native = MemoryRead.bind(MatrixBackend(), MatrixState(matrix))
    matrix_m = lambda r, z: matrix[owner[r]] @ q_of(z)
    slot_m = lambda backend: (lambda r, z: backend.read(slots, q_of(z)[None], owner[r:r + 1])[0])

    def reference(rows, read):
        out = torch.zeros(len(rows), T, cfg.d_out, device=device)
        for i, r in enumerate(rows):
            for t in range(T):
                if valid[r, t]:
                    z0 = u[r, t]
                    z1 = z0 + cfg.rho * adapter._phi(F.linear(z0, adapter.R) + F.linear(read(i, r, t, z0), adapter.U))
                    out[i, t] = adapter.branch(z1)
        return out

    all_rows, first_rows = list(range(B * Q)), [0, Q]
    cases = {
        'matrix': (native.expand(Q), all_rows, lambda i, r, t, z: matrix_m(r, z)),
        'soft': (MemoryRead.bind(soft, slots).expand(Q), all_rows, lambda i, r, t, z: slot_m(soft)(r, z)),
        'hard': (MemoryRead.bind(hard, slots).expand(Q), all_rows, lambda i, r, t, z: slot_m(hard)(r, z)),
        'soft, final token only': (ScopedRead(soft, slots, native.owner, native=native).expand(Q), all_rows,
                                   lambda i, r, t, z: slot_m(soft)(r, z) if t == last[r] else matrix_m(r, z)),
        'oracle, final token only': (ScopedRead(MatrixBackend(), MatrixState(matrix), native.owner, native=native, oracle=oracle),
                                     first_rows, lambda i, r, t, z: oracle[owner[r]] if t == last[r] else matrix_m(r, z)),
    }
    for name, (read, rows, per_token) in cases.items():
        got = adapter.process_sequence(x[rows], read, 1, valid[rows])[0]
        want = reference(rows, per_token)
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-3 * float(want.abs().max())), name
        assert bool((got[~valid[rows]] == 0).all()), name


def test_the_adapter_checks_pass_on_the_tiny_adapter(experiment):
    assert slot_preflight.owner_mask_check(experiment.adapter, device=experiment.device)['passed']


# ---------------------------------------------------------------------------------- features, bytes, capacity, isolation

def test_captured_keys_values_and_matrix_are_consistent(experiment):
    report = slot_preflight.feature_check(experiment, train_arrays(8))
    assert report['passed'], report
    assert (report['key_dim'], report['value_dim']) == (768, 128) and report['base_keys_unused']
    assert report['reconstruction_max_relative_error'] < 1e-6


def test_bytes_follow_numel_times_element_size(experiment):
    report = slot_preflight.bytes_check(experiment, train_arrays(4))
    assert report['passed'] and (report['slot_bytes_per_episode'], report['matrix_bytes_per_episode']) == (57_368, 393_216)


def test_capacity_errors_when_e_exceeds_s_and_nothing_is_truncated(experiment):
    report = slot_preflight.capacity_check(experiment, train_arrays(4))
    assert report['passed'] and report['facts'] == 16 and report['exact_fit_keeps_all']


def test_reads_leave_the_support_state_untouched(experiment):
    report = slot_preflight.isolation_check(experiment, eval_arrays(8))
    assert report['passed'], report
    assert report['scored'] == 6 * 3  # six read conditions x three loop depths, the oracle included


# ----------------------------------------------------------------------------------------------- freeze verification

def test_the_freeze_verification_passes_on_the_registered_trainable_set(experiment):
    report = slot_preflight.frozen_set_check(experiment, eval_arrays(8), train_arrays(8))
    assert report['passed'] and report['frozen_unchanged'] and report['trainable_changed']
    names = report['trainable_report']
    assert names['trainable'] == ['Q', 'R', 'U', 'B'] and sorted(names['frozen']) == sorted(slot_rules.FROZEN)
    assert names['buffers'] == ['input_mean', 'input_std']
    cfg = experiment.adapter.cfg
    assert names['trainable_parameters'] == 3 * cfg.r * cfg.r_mem + cfg.d_out * cfg.r  # Q, R, U, B
    assert set(report['gradients']['trainable_grad_norms']) == {'Q', 'R', 'U', 'B'}


def test_the_freeze_verification_catches_each_defect(experiment):
    adapter = experiment.adapter
    params = freeze_for_s1a(experiment)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    assert trainable_report(experiment, optimizer)['trainable_parameters'] == sum(p.numel() for p in params)
    adapter.A.requires_grad_(True)  # (i) a tensor outside the set trains
    with pytest.raises(RuntimeError, match='trainable set'):
        trainable_report(experiment, optimizer)
    adapter.A.requires_grad_(False)
    with pytest.raises(RuntimeError, match='optimizer'):
        trainable_report(experiment, torch.optim.AdamW([*params, adapter.K], lr=1e-3))
    backbone = next(p for p in experiment.model.parameters() if all(p is not q for q in adapter.parameters()))
    backbone.requires_grad_(True)
    with pytest.raises(RuntimeError, match='backbone'):
        trainable_report(experiment, optimizer)
    backbone.requires_grad_(False)
    reference = tensor_digests(adapter)  # (ii) frozen bytes, and the trainable ones must move after training
    with pytest.raises(RuntimeError):
        digest_report(adapter, reference, trained=True)  # nothing moved yet
    assert digest_report(adapter, reference, trained=False)['frozen_unchanged']
    with torch.no_grad():
        adapter.input_mean += 1e-3  # a buffer is part of the frozen set
    with pytest.raises(RuntimeError):
        digest_report(adapter, reference, trained=False)
    with torch.no_grad():
        adapter.input_mean -= 1e-3
    support = eval_arrays(8)
    before = support_digest(experiment, support)  # (iii) the written state is sensitive to the writer's weights
    with torch.no_grad():
        adapter.K += 1e-2
    assert support_digest(experiment, support) != before
    with torch.no_grad():
        adapter.K -= 1e-2
    # (iv) gradients: none yet is an error, a frozen tensor with a gradient is an error
    with pytest.raises(RuntimeError, match='gradient'):
        gradient_report(experiment)
    adapter.A.requires_grad_(True)
    batch = experiment.prepare(train_arrays(4))
    logits, _ = experiment.score(bind_batch(experiment, batch, ReadSpec('matrix')), 1)
    logits.float().square().mean().backward()
    with pytest.raises(RuntimeError, match='frozen'):
        gradient_report(experiment)


def test_standardize_is_refused_because_it_would_overwrite_the_buffers(experiment):
    before = tensor_digests(experiment.adapter)
    with pytest.raises(RuntimeError, match='standardize'):
        experiment.standardize()
    assert tensor_digests(experiment.adapter) == before


# ----------------------------------------------------------------------------------------------------- export and reload

def test_the_export_reloads_with_buffers_settings_and_scope_and_reproduces_predictions(experiment, model, tokenizer, tmp_path):
    spec = ReadSpec('soft', 1.5, tau=0.3)
    arrays = eval_arrays(16)
    before = evaluate(experiment, arrays, {'read': spec}, diagnostics=False)['samples']['read']
    parent = dict(name='S-RE', writer_blake2b_128='0' * 32)
    digest = save_export(experiment, tmp_path / 'x.pt', arm='soft_slot_L1', spec=spec, step=10, parent=parent, run=dict(seed=17))
    payload = torch.load(tmp_path / 'x.pt', map_location='cpu', weights_only=True)
    assert 'optimizer' not in payload and 'value_head' not in payload and not ({'snapshot', 'states', 'keys', 'values'} & set(payload))
    assert set(payload['adapter']) == set(experiment.adapter.state_dict())  # the buffers are in the export
    assert payload['slot']['read'] == spec.as_dict() and payload['slot']['key_dim'] == 768 and payload['slot']['value_dim'] == 128
    assert payload['slot']['parent'] == parent and payload['slot']['loop_steps'] == 1 and len(digest) == 32
    experiment.close()
    reloaded = load_export(tmp_path / 'x.pt', model=model, tokenizer=tokenizer)
    try:
        assert reloaded.read_spec == spec and reloaded.value_head is None
        after = evaluate(reloaded, arrays, {'read': reloaded.read_spec}, diagnostics=False)['samples']['read']
        assert all(np.array_equal(before[k], after[k]) for k in ('episode_id', 'prediction', 'correct', 'nll'))
    finally:
        reloaded.close()


def test_an_export_without_buffers_or_backend_settings_does_not_load(experiment, model, tokenizer, tmp_path):
    save_export(experiment, tmp_path / 'ok.pt', arm='matrix_L1', spec=ReadSpec('matrix'), step=1, parent={}, run={})
    experiment.close()
    good = torch.load(tmp_path / 'ok.pt', map_location='cpu', weights_only=True)

    def broken(edit):
        payload = {k: (dict(v) if isinstance(v, dict) else v) for k, v in good.items()}
        edit(payload)
        torch.save(payload, tmp_path / 'bad.pt')
        with pytest.raises(ValueError):
            load_export(tmp_path / 'bad.pt', model=model, tokenizer=tokenizer)

    broken(lambda p: p['adapter'].pop('input_mean'))
    broken(lambda p: p['adapter'].pop('input_std'))
    broken(lambda p: p['slot'].pop('read'))
    broken(lambda p: p['slot'].pop('feature_map'))
    broken(lambda p: p.update(optimizer={}))
    broken(lambda p: p.update(slot_version=99))
    broken(lambda p: p['signature'].update(layer=99))
    reloaded = load_export(tmp_path / 'ok.pt', model=model, tokenizer=tokenizer)
    reloaded.close()


def test_the_query_projection_ablation_gives_the_reader_its_own_copy_of_a(experiment, model, tokenizer, tmp_path):
    adapter, arrays, spec = experiment.adapter, eval_arrays(16), ReadSpec('matrix')
    base = evaluate(experiment, arrays, {'read': spec}, diagnostics=False)['samples']['read']['nll']
    support = support_digest(experiment, arrays)
    a_before = adapter.A.detach().clone()
    adapter.add_query_projection()
    assert torch.equal(adapter.A_query, a_before) and adapter.A_query.requires_grad
    with pytest.raises(ValueError):
        adapter.add_query_projection()
    assert np.array_equal(evaluate(experiment, arrays, {'read': spec}, diagnostics=False)['samples']['read']['nll'], base)
    with torch.no_grad():
        adapter.A_query *= 1.1  # the reader changes, the writer does not
    changed = evaluate(experiment, arrays, {'read': spec}, diagnostics=False)['samples']['read']['nll']
    assert not np.array_equal(changed, base) and support_digest(experiment, arrays) == support
    save_export(experiment, tmp_path / 'q.pt', arm='matrix_L1', spec=spec, step=1, parent={}, run={}, trainable=slot_rules.ABLATION_TRAINABLE)
    copy = adapter.A_query.detach().clone()
    experiment.close()
    reloaded = load_export(tmp_path / 'q.pt', model=model, tokenizer=tokenizer)
    try:
        assert torch.equal(reloaded.adapter.A_query.cpu(), copy.cpu()) and reloaded.slot['query_projection'] == 'copy'
        assert np.array_equal(evaluate(reloaded, arrays, {'read': spec}, diagnostics=False)['samples']['read']['nll'], changed)
    finally:
        reloaded.close()


# ------------------------------------------------------------------------------------------------------------ the trainer

def test_both_arms_see_the_same_batches_of_one_seed(tokenizer):
    seed = derive_seed(17, 'slot-test-stream')
    streams = [BatchStream(slot_data.task_spec(), 'train', slot_data.BATCH, seed, 3, tokenizer=tokenizer, workers=0) for _ in range(2)]
    first, second = ([arrays.episode_seed.copy() for arrays, _ in stream] for stream in streams)
    assert len(first) == 3 and all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len({int(i) for batch in first for i in batch}) == 3 * slot_data.BATCH


def test_fit_keeps_the_gradient_record_selects_by_the_rule_and_exports_the_selected_checkpoint(experiment, model, tokenizer, tmp_path):
    spec, dev, steps = ReadSpec('soft', tau=0.3), eval_arrays(16), 12
    summary = fit(experiment, spec, steps=steps, lr=1e-3, stream_seed=derive_seed(17, 'slot-test-stream'), out=tmp_path, dev=dev,
                  dev_steps=(4, 8, 12), support=eval_arrays(8), workers=0,
                  export=dict(arm='soft_slot_L1', parent={}, run=dict(seed=17)))
    log = [json.loads(line) for line in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
    assert [row['step'] for row in log] == [1, 4, 8, 12]
    assert all(math.isfinite(row['grad_norm']) and isinstance(row['clipped'], bool) for row in log)
    assert log[0]['lr'] == pytest.approx(1e-3 / slot_rules.WARMUP) and log[-1]['lr'] == pytest.approx(1e-3 * 12 / slot_rules.WARMUP)
    steps_file = np.load(tmp_path / 'train_steps.npz')
    norms = steps_file['grad_norm']
    assert len(norms) == len(steps_file['loss']) == steps and np.isfinite(norms).all()
    assert summary['clipped_steps'] == int((norms > slot_rules.CLIP).sum()) and summary['all_finite']
    assert summary['clipped_fraction'] == pytest.approx(float((norms > slot_rules.CLIP).mean()))
    assert [row['grad_norm'] for row in log][0] == pytest.approx(float(norms[0]), rel=1e-6)  # the logged norm is the pre-clip one
    evaluations = summary['dev']
    assert [r['step'] for r in evaluations] == [0, 4, 8, 12] and [r['selectable'] for r in evaluations] == [False, True, True, True]
    chosen = slot_rules.select_checkpoint([(r['step'], r['nll']) for r in evaluations if r['selectable']])
    assert summary['selected_step'] == chosen
    freeze = summary['freeze']
    assert freeze['digests']['frozen_unchanged'] and freeze['digests']['trainable_changed'] and freeze['support']['equal']
    assert freeze['digests']['before']['A'] == freeze['digests']['after']['A']
    assert freeze['digests']['before']['Q'] != freeze['digests']['after']['Q']
    assert summary['trainable_parameters'] + summary['frozen_parameters'] == sum(p.numel() for p in experiment.adapter.parameters())
    experiment.close()
    reloaded = load_export(tmp_path / 'selected.pt', model=model, tokenizer=tokenizer)
    try:
        row = next(r for r in evaluations if r['step'] == chosen)
        again = dev_row(reloaded, reloaded.read_spec, dev, 0, True)
        assert (again['correct'], again['nll']) == (row['correct'], pytest.approx(row['nll'], abs=1e-6))
    finally:
        reloaded.close()


def test_fit_with_the_ablation_trains_the_query_copy_and_leaves_the_writer_alone(experiment, tmp_path):
    experiment.adapter.add_query_projection()
    support = eval_arrays(8)
    before_support = support_digest(experiment, support)
    a_before = experiment.adapter.A.detach().clone()
    summary = fit(experiment, ReadSpec('matrix'), steps=4, lr=1e-3, stream_seed=derive_seed(17, 'slot-test-stream'), out=tmp_path,
                  support=support, workers=0, trainable=slot_rules.ABLATION_TRAINABLE)
    assert summary['trainable'] == ['A_query', 'Q', 'R', 'U', 'B']
    assert summary['freeze']['digests']['before']['A_query'] != summary['freeze']['digests']['after']['A_query']
    assert torch.equal(experiment.adapter.A, a_before) and support_digest(experiment, support) == before_support
    cfg = experiment.adapter.cfg
    assert summary['trainable_parameters'] == 3 * cfg.r * cfg.r_mem + cfg.d_out * cfg.r + cfg.r * cfg.d_in


def test_a_defective_run_stops_instead_of_continuing(experiment, tmp_path):
    with torch.no_grad():
        experiment.adapter.B.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        fit(experiment, ReadSpec('matrix'), steps=2, lr=1e-3, stream_seed=1, out=tmp_path, workers=0)


# ------------------------------------------------------------------------------------------------------- the S0 pipeline

def test_the_s0_pipeline_on_a_tiny_model_has_the_registered_structure(experiment, tmp_path):
    core = slot_run.s0_core(experiment, eval_arrays(32), eval_arrays(16, 32))
    rows = core['calibration']['candidates']
    assert [(r['kind'], r['gain'], r['tau']) for r in rows] == [(c['kind'], c['gain'], c['tau']) for c in slot_rules.calibration_grid()]
    assert len(rows) == 18 and all(math.isfinite(r['nll']) for r in rows)
    selected = core['calibration']['selected']
    for kind in ('matrix', 'soft', 'hard'):
        want = slot_rules.choose_candidate(rows, kind)
        assert selected[kind]['gain'] == want['gain'] and selected[kind].get('tau') == (want['tau'] if kind == 'soft' else None)
    metrics = core['dev']['metrics']
    assert list(metrics) == [f'{role}/{scope}' for role, scope in [
        ('native', 'all_valid_question'), ('matrix_uncal', 'all_valid_question'), ('matrix_cal', 'all_valid_question'),
        ('soft_uncal', 'all_valid_question'), ('soft_cal', 'all_valid_question'), ('hard_uncal', 'all_valid_question'),
        ('hard_cal', 'all_valid_question'), ('soft_rec_uncal', 'all_valid_question'), ('soft_rec_cal', 'all_valid_question'),
        ('matrix_cal', 'final_only'), ('soft_cal', 'final_only'), ('hard_cal', 'final_only'), ('clean_final', 'final_only')]]
    assert all(m['n'] == 16 and math.isfinite(m['nll']) and math.isfinite(m['rel_err']) for m in metrics.values())
    native, matrix = metrics['native/all_valid_question'], metrics['matrix_uncal/all_valid_question']
    assert (native['correct'], native['nll'], native['rel_err']) == (matrix['correct'], matrix['nll'], matrix['rel_err'])
    clean = metrics['clean_final/final_only']
    assert clean['rel_err'] == 0 and clean['cos'] == pytest.approx(1.0, abs=1e-6) and clean['norm_ratio'] == pytest.approx(1.0, abs=1e-6)
    samples = core['dev']['samples']
    ids = {name: tuple(fields['episode_id']) for name, fields in samples.items()}
    assert len(set(ids.values())) == 1  # the same episodes and order in every condition
    assert {'w_true', 'w_gap', 'w_entropy'} <= set(metrics['soft_cal/all_valid_question'])
    assert 'slot_is_true' in metrics['hard_cal/all_valid_question']
    slot_run.write_json(tmp_path / 'metrics.json', dict(calibration=core['calibration'], conditions=metrics, common=core['dev']['common']))
    slot_run.save_samples(tmp_path / 'samples.npz', samples)
    assert 'clean_final/final_only/prediction' in np.load(tmp_path / 'samples.npz').files


def test_arms_read_with_their_calibrated_values():
    selected = dict(matrix=dict(gain=2.0), soft=dict(gain=0.5, tau=0.1), hard=dict(gain=1.0))
    assert slot_run.arm_spec('matrix_L1', selected) == ReadSpec('matrix', 2.0)
    assert slot_run.arm_spec('soft_slot_L1', selected) == ReadSpec('soft', 0.5, tau=0.1)
    with pytest.raises(ValueError):
        slot_run.arm_spec('hard_L1', selected)


# ---------------------------------------------------------------------------------------------------------- the runner gate

def test_the_runner_refuses_without_a_valid_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(slot_data, 'inventory', lambda: {'S-RE': {'writer.pt': 'a' * 32}})
    monkeypatch.setattr(slot_data, 'manifest', lambda: {'ids': {'x': 1}})
    good = dict(status='locked', prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128, preflight=dict(passed=True),
                manifest={'ids': {'x': 1}}, inventory={'S-RE': {'writer.pt': 'a' * 32}})
    path = tmp_path / 'lock.json'

    def check(**edit):
        path.write_text(json.dumps({**good, **edit}))
        return slot_run.require_lock(path)

    assert check()['status'] == 'locked'
    for edit in (dict(status='pending'), dict(prereg_blake2b_128='0' * 32), dict(preflight=dict(passed=False)), dict(preflight={}),
                 dict(manifest={'ids': {'x': 2}}), dict(inventory={'S-RE': {'writer.pt': 'b' * 32}})):
        with pytest.raises(slot_run.StopRun):
            check(**edit)


def test_a_stage_config_that_differs_from_the_registration_is_refused(tmp_path):
    for stage in ('S0', 'S1A'):
        path = tmp_path / f'{stage}.json'
        config = slot_rules.stage_config(stage)
        path.write_text(json.dumps(config))
        assert slot_run.check_config(path, stage)['config'] == config
        config['calibration' if stage == 'S0' else 'clip'] = None
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match='differs'):
            slot_run.check_config(path, stage)


def test_sealed_sets_are_not_evaluated_without_the_open_record(tmp_path):
    entry = dict(label='x', export=str(tmp_path / 'none.pt'), checkpoint_blake2b_128='0' * 32)
    for name in ('confirmation', 'unseen'):
        with pytest.raises(slot_data.SealedSetError):
            slot_run.evaluate_entries(('S1A', name), [entry], tmp_path / name)


# ------------------------------------------------------------------------------------------------------- the S1A runner

SELECTED = dict(matrix=dict(gain=1.0), soft=dict(gain=1.0, tau=0.3), hard=dict(gain=1.0))
PARENT_DIGESTS = {'writer.pt': 'a' * 32, 'checkpoint.pt': 'b' * 32}


@pytest.fixture
def tiny_study(monkeypatch, model, tokenizer, tmp_path):
    """The runner's own S1A code on the tiny model: the parent loader, the lock and the registered sets are replaced (the real
    parents and the registered sizes need the cluster); the stage config, the release checks, the trainer and the exports are
    the real ones. Returns the paths the modes take."""
    sets = {('S0', 'calibration'): slot_data.EvalSet('S0', 'calibration', 'val', 0, 32, derive_seed(17, 'slot-test-calibration')),
            ('S0', 'dev'): slot_data.EvalSet('S0', 'dev', 'val', 0, 32, derive_seed(17, 's0-test-dev')),
            ('S1A', 'pilot'): slot_data.EvalSet('S1A', 'pilot', 'val', 0, 48, derive_seed(17, 'slot-test-pilot')),
            ('S1A', 'dev'): slot_data.EvalSet('S1A', 'dev', 'val', 0, 48, derive_seed(17, 'slot-test-dev')),
            ('S1A', 'smoke'): slot_data.EvalSet('S1A', 'smoke', 'train', 0, 64, derive_seed(17, 'slot-test-smoke'))}
    shared, registered = (model, tokenizer), slot_data.eval_set

    def load_parent(name, *, model=None, tokenizer=None, expected=None):
        assert name in {'S-RE', 'D128-3'} and set(expected) <= set(PARENT_DIGESTS)
        return adopt(slot_golden.tiny_experiment(*shared))

    monkeypatch.setattr(slot_run, 'load_parent', load_parent)
    monkeypatch.setattr(slot_run, 'require_lock', lambda path: dict(status='locked', inventory={'S-RE': dict(PARENT_DIGESTS)}))
    monkeypatch.setattr(slot_data, 'eval_set', lambda stage, purpose, **kw: sets.get((stage, purpose)) or registered(stage, purpose, **kw))
    monkeypatch.setattr(slot_data, 'PILOT_STEPS', 20)
    monkeypatch.setattr(slot_run, 'load_value_head', lambda name, device: None)
    monkeypatch.setattr(slot_run, 'old_range_record', lambda experiment, name: dict(correct=608, n=1000, loss_float32=2.1291537284851074,
                                                                                    passed=True))
    metrics = tmp_path / 's0_metrics.json'
    metrics.write_text(json.dumps(dict(calibration=dict(selected=SELECTED))))
    return dict(config=ROOT / 'configs' / 'slot_loop_v1' / 's1a.json', manifest=tmp_path / 'lock.json', s0=metrics, sets=sets)


def test_the_smoke_trains_both_arms_under_the_freeze_checks_and_its_exports_reload_to_the_saved_predictions(tiny_study, model, tokenizer, tmp_path):
    result = slot_run.run_smoke(tmp_path / 'smoke', tiny_study['config'], tiny_study['manifest'], tiny_study['s0'], lr=1e-3, steps=6)
    assert list(result['arms']) == ['matrix_L1', 'soft_slot_L1'] and result['steps'] == 6
    arrays = episodes(tiny_study['sets']['S1A', 'smoke'])
    for arm, report in result['arms'].items():
        assert report['all_finite'] and report['freeze']['digests']['frozen_unchanged'] and report['freeze']['support']['equal']
        assert [row['step'] for row in report['dev']] == [0, 6] and report['selected_checkpoint_blake2b_128']
        check = slot_run.reload_check(tmp_path / 'smoke' / arm / 'selected.pt', tmp_path / 'smoke' / arm / 'smoke_predictions.npz', arrays,
                                      model=model, tokenizer=tokenizer)
        assert check['passed'] and all(check['equal'].values()) and check['read']['kind'] == slot_rules.ARM_KIND[arm]


def test_the_pilot_writes_its_full_record_and_picks_each_arms_rate_and_the_step_count_by_the_rules(tiny_study, tmp_path):
    out = tmp_path / 'pilot'
    try:
        result = slot_run.run_pilot(out, tiny_study['config'], tiny_study['manifest'], tiny_study['s0'])
    except slot_run.StopRun:  # a registered stop is written to the record before it is raised
        result = json.loads((out / 'pilot.json').read_text())
        assert result['stop']
    assert len(result['runs']) == 6 and {(r['arm'], r['lr']) for r in result['runs']} == {
        (arm, lr) for arm in slot_rules.ARMS for lr in slot_rules.LR_GRID}
    assert all(r['step0']['step'] == 0 and r['final']['step'] == 20 and r['final']['n'] == 48 for r in result['runs'])
    for arm in slot_rules.ARMS:
        rows = [r for r in result['runs'] if r['arm'] == arm]
        try:
            assert result['lr'][arm] == slot_rules.choose_lr(rows, rows[0]['step0']['nll'])['lr']
        except slot_rules.StopForOrchestrator:
            assert result['lr'][arm] is None and result['stop']
    assert result['step_seconds'] == max(r['step_seconds_mean'] for r in result['runs'])
    if result['stop'] is None:
        steps = slot_rules.schedule_steps(result['step_seconds'], result['eval_seconds'], result['load_seconds'])
        assert result['steps'] == steps and steps % 10 == 0 and result['walltime_seconds'] <= 3600
    for arm in slot_rules.ARMS:
        for lr in slot_rules.LR_GRID:
            assert (out / arm / f'lr{lr:g}' / 'training.json').exists() and not (out / arm / f'lr{lr:g}' / 'selected.pt').exists()


def release_for(tiny_study, tmp_path, steps=20, **edit):
    """A release of the tiny study: the science commit and hashes the runner recomputes before it trains."""
    arms = {arm: dict(lr=1e-3, read=slot_run.arm_spec(arm, SELECTED).as_dict()) for arm in slot_rules.ARMS}
    hashes = {f'{arm}/{seed}': slot_rules.science_hash(slot_rules.science_config(
        arm, seed, steps=steps, lr=1e-3, read=arms[arm]['read'], parent_digest=PARENT_DIGESTS['writer.pt']))
        for arm in slot_rules.ARMS for seed in slot_data.RUN_SEEDS}
    release = dict(science_commit=slot_run.provenance()['git_commit'], prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128, steps=steps,
                   parent=dict(writer_blake2b_128=PARENT_DIGESTS['writer.pt']), arms=arms, config_hashes=hashes)
    release.update(edit)
    path = tmp_path / 'release.json'
    path.write_text(json.dumps(release))
    return path


def test_an_arm_seed_run_records_dev_selection_gradient_and_clipping_statistics_and_its_export_evaluates_to_the_same_numbers(
        tiny_study, model, tokenizer, tmp_path):
    release = release_for(tiny_study, tmp_path)
    record = slot_run.run_train(tmp_path / 'run', release, 'soft_slot_L1', 29, config_path=tiny_study['config'],
                                manifest_path=tiny_study['manifest'])
    assert record['status'] == 'complete' and record['confirmation_accessed'] is False and record['steps'] == 20
    assert [r['step'] for r in record['dev']] == [0] + slot_rules.checkpoint_steps(20)
    assert record['selected_step'] == slot_rules.select_checkpoint([(r['step'], r['nll']) for r in record['dev'] if r['selectable']])
    assert record['config']['train_base_seed'] == slot_data.TRAIN_BASE['S1A', 29]
    training = record['training']
    assert training['all_finite'] and 0.0 <= training['clipped_fraction'] <= 1.0 and training['grad_norm_max'] >= training['grad_norm']['median']
    assert record['freeze']['digests']['frozen_unchanged'] and record['freeze']['digests']['trainable_changed']
    assert record['trainable_parameters'] == 3 * 128 * 128 + 16 * 128  # Q, R, U (128 x 128) and B (hidden 16 x 128); 180,224 at hidden 1024
    export = tmp_path / 'run' / 'selected.pt'
    assert slot_data.file_digest(export) == record['selected_checkpoint_blake2b_128']
    out = tmp_path / 'dev'
    report = slot_run.evaluate_entries(('S1A', 'dev'), [
        dict(label='soft_slot_L1_29', export=str(export), checkpoint_blake2b_128=record['selected_checkpoint_blake2b_128']),
        dict(label='native', parent_writer_blake2b_128=PARENT_DIGESTS['writer.pt'], read=None)], out, model=model, tokenizer=tokenizer)
    assert set(report) == {'soft_slot_L1_29', 'native'} and report['native']['n'] == 48
    arm = np.load(out / 'soft_slot_L1_29.npz')
    selected_row = next(r for r in record['dev'] if r['step'] == record['selected_step'])
    assert int(arm['correct'].sum()) == selected_row['correct'] and float(arm['nll'].mean()) == pytest.approx(selected_row['loss_float32'], abs=1e-6)
    assert set(np.load(out / 'native.npz').files) >= {'episode_id', 'answer', 'prediction', 'correct', 'nll', 'rel_err'}
    assert np.array_equal(arm['episode_id'], np.load(out / 'native.npz')['episode_id'])


@pytest.mark.parametrize('defect', ['commit', 'prereg', 'parent', 'science_config', 'unknown_run'])
def test_a_run_that_differs_from_its_release_does_not_start(tiny_study, tmp_path, defect):
    release = json.loads(release_for(tiny_study, tmp_path).read_text())
    seed = 17
    if defect == 'commit':
        release['science_commit'] = '0' * 40
    elif defect == 'prereg':
        release['prereg_blake2b_128'] = '0' * 32
    elif defect == 'parent':
        release['parent']['writer_blake2b_128'] = 'f' * 32
    elif defect == 'science_config':
        release['arms']['matrix_L1']['lr'] = 3e-3  # the hash in the release is of lr 1e-3
    else:
        release['config_hashes'].pop('matrix_L1/17')
    path = tmp_path / 'edited.json'
    path.write_text(json.dumps(release))
    with pytest.raises(slot_run.StopRun):
        slot_run.run_train(tmp_path / 'run', path, 'matrix_L1', seed, config_path=tiny_study['config'], manifest_path=tiny_study['manifest'])
    assert not (tmp_path / 'run' / 'train_log.jsonl').exists()


# ------------------------------------------------------------------------------------- the runner and the frozen analysis

def analysis_script():
    spec = importlib.util.spec_from_file_location('analyze_slot_study', ROOT / 'scripts' / 'analyze_slot_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_what_the_runner_writes_for_s0_is_what_the_frozen_analysis_reads(tiny_study, tmp_path):
    analysis = analysis_script()
    out = tmp_path / 's0'
    metrics = slot_run.run_s0(out, ROOT / 'configs' / 'slot_loop_v1' / 's0.json', tiny_study['manifest'])
    assert sorted(path.name for path in out.iterdir()) == ['calibration.json', 'd128_3_samples.npz', 'metrics.json', 'samples.npz']
    assert metrics['n'] == 32 and metrics['calibration']['n'] == 32 and len(metrics['calibration']['candidates']) == 18
    preflight = tmp_path / 'preflight.json'
    preflight.write_text(json.dumps(dict(passed=True, checks={name: dict(passed=True) for name in analysis.CHECKS})))
    assert analysis.main(['--stage', 'S0', '--records', str(out), '--out', str(tmp_path / 'decision'), '--preflight', str(preflight)]) == 0
    decision = json.loads((tmp_path / 'decision' / 'decision_s0.json').read_text())
    assert decision['row'] in {'R1', 'R2', 'R3', 'R4', 'R5'} and decision['n'] == 32 and decision['preflight']['failed'] == []
    assert decision['calibration']['selected'] == metrics['calibration']['selected']
    assert decision['s1a']['soft_slot'] == metrics['calibration']['selected']['soft']
    assert set(decision['d128_3']['conditions']) == {f'{role}/all_valid_question' for role in analysis.D128_ROLES}
    # the engineering gate: a failed check ends in R0 without reading any S0 data
    preflight.write_text(json.dumps(dict(passed=False, checks={**{name: dict(passed=True) for name in analysis.CHECKS},
                                                              'matrix_equals_native': dict(passed=False)})))
    assert analysis.main(['--stage', 'S0', '--records', str(tmp_path), '--out', str(tmp_path / 'gate'), '--preflight', str(preflight)]) == 0
    assert json.loads((tmp_path / 'gate' / 'decision_s0.json').read_text())['row'] == 'R0'


def test_the_run_record_is_what_the_frozen_analysis_reads_for_the_dev_selection(tiny_study, tmp_path):
    analysis = analysis_script()
    release = release_for(tiny_study, tmp_path)
    record = slot_run.run_train(tmp_path / 'matrix_L1' / 's43', release, 'matrix_L1', 43, config_path=tiny_study['config'],
                                manifest_path=tiny_study['manifest'])
    parsed = analysis.parse_run(json.loads((tmp_path / 'matrix_L1' / 's43' / 'record.json').read_text()), 'matrix_L1', 43)
    assert parsed['steps'] == 20 and parsed['n'] == 48 and parsed['ids'] == record['dev_ids_blake2b_128']
    assert parsed['selected']['step'] == record['selected_step'] and parsed['checkpoint'] == record['selected_checkpoint_blake2b_128']
    assert parsed['training']['clipped_fraction'] == record['training']['clipped_fraction']
