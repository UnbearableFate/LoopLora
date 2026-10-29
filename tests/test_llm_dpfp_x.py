"""WP-L8 DPFP-X: the two extra arms (X256, X17) and the proof that the DPFP-off path and the DPFP configs are unchanged.

Compute only (imports torch; tiny models). The registered rule is docs/prereg/DPFP-X.md.
"""
import copy
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm import dpfp_run, inject, read_fix
from looplora.llm.dpfp import DPFPExperiment, configuration
from looplora.llm.dpfp_release import (ARMS, BACKBONE_17B, XARMS, XSPECS, arm_spec, prereg_digest, scheduled_steps,
                                       study_of)
from looplora.llm.dpfp_run import choose_x, load_inference
from looplora.llm.inject import MODEL_ID, MODEL_REVISION, load_tokenizer
from looplora.llm.read_fix_run import fit, validate

ROOT = Path(__file__).resolve().parents[1]
REVISION = '0123456789abcdef0123456789abcdef01234567'


def recorded(path):
    return json.loads((ROOT / path).read_text())


def as_json(value):
    return json.loads(json.dumps(value))


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'dpfp-test-model'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield model
    assert not hasattr(model, 'looplora_adapters')


# ------------------------------------------------------- the DPFP-off path and the three DPFP configs are unchanged

@pytest.mark.parametrize('arm', ARMS)
def test_the_three_dpfp_configs_still_build_the_recorded_configs_and_hashes(arm):
    record = recorded(f'reports/p1l-dpfp/arms/{arm}/record.json')
    cfg = configuration(arm, record['config']['train']['steps'])
    assert 'backbone' not in cfg and 'backbone' not in cfg['dpfp']
    assert as_json(cfg) == record['config']
    assert stable_hash(cfg) == record['config_hash']


def test_the_e8_retrain_config_and_the_wp_l4_references_keep_their_hashes():
    record = recorded('reports/p1l-dpfp/arms-e8/D128-3/record.json')
    cfg = configuration('D128-3', record['config']['train']['steps'], E=8)
    assert as_json(cfg) == record['config'] and stable_hash(cfg) == record['config_hash']
    for rank, label in ((32, 'R-r32'), (128, 'R-r128')):  # the DPFP-off reader: WP-L4's R, no dpfp key, default backbone
        record = recorded(f'reports/p1l-dpfp/refs/{label}/record.json')
        cfg = read_fix.configuration('R', record['config']['train']['steps'], E=16, rank=rank)
        assert 'backbone' not in cfg and 'dpfp' not in cfg and 'dpfp_nu' not in cfg['model']
        assert as_json(cfg) == record['config'] and stable_hash(cfg) == record['config_hash']


def test_the_default_backbone_stays_the_pinned_06b_and_the_rank_guard_is_lifted_only_for_x256():
    assert inject.backbone_pin() == (MODEL_ID, MODEL_REVISION) == (
        'Qwen/Qwen3-0.6B-Base', 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
    assert (256, 'R') not in read_fix.RUNS and (256, 'R') not in read_fix.CANDIDATES
    with pytest.raises(ValueError):
        read_fix.configuration('R', 100, rank=256)
    with pytest.raises(ValueError):
        read_fix.configuration('W5', 100, rank=256, extra_runs=((256, 'R'),))
    assert read_fix.configuration('R', 100, rank=256, extra_runs=((256, 'R'),))['model']['r'] == 256


# --------------------------------------------------------------------------------------------- the DPFP-X configs

def test_registered_x_arms():
    assert XARMS == ('X256', 'X17') and not set(XARMS) & set(ARMS)
    assert arm_spec('X256') == (256, 3, None)
    assert arm_spec('X17') == (128, 3, {'id': 'Qwen/Qwen3-1.7B-Base', 'revision': 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'})
    assert arm_spec('D128-3') == (128, 3, None)
    assert [study_of(arm) for arm in (*ARMS, *XARMS)] == ['DPFP'] * 3 + ['DPFP-X'] * 2
    with pytest.raises(ValueError):
        arm_spec('X32')


def test_x_configs_are_d128_3_with_only_the_registered_changes():
    steps = 17000
    d = configuration('D128-3', steps)
    x256, x17 = configuration('X256', steps), configuration('X17', steps)
    assert (x256['model']['r'], x256['model']['dpfp_nu'], x256['dpfp']['key_dim']) == (256, 3, 1536)
    assert (x17['model']['r'], x17['model']['dpfp_nu'], x17['dpfp']['key_dim']) == (128, 3, 768)
    assert 'backbone' not in x256 and x17['backbone'] == BACKBONE_17B == x17['dpfp']['backbone']
    assert (x256['name'], x17['name']) == ('dpfp-X256-E16', 'dpfp-X17-E16')

    def strip(cfg, *keys):
        cfg = copy.deepcopy(cfg)
        for key in ('name', 'backbone'):
            cfg.pop(key, None)
        cfg['dpfp'] = {k: v for k, v in cfg['dpfp'].items() if k not in ('arm', 'backbone', 'rank', 'key_dim')}
        for key in keys:
            del cfg['model'][key]
        return cfg
    assert strip(x17) == strip(d)                      # X17: the backbone and the arm label
    assert strip(x256, 'r') == strip(d, 'r')           # X256: the rank (and the arm label)
    base = read_fix.configuration('R', steps, rank=256, extra_runs=((256, 'R'),))
    assert x256['train'] == base['train'] and x256['task'] == base['task'] and x256['eval'] == base['eval']
    assert (x256['seed'], x256['model']['layer'], x256['train']['prefetch_workers'], x256['task']['E']) == (17, 14, 2, 16)
    assert (x256['read_fix']['arm'], x256['read_fix']['read_weight']) == ('R', 1)
    assert x256['eval']['episodes'] == 1000 == x17['eval']['episodes']
    assert configuration('X256', 20, smoke=True)['eval']['episodes'] == 100
    assert stable_hash(x256) != stable_hash(x17) != stable_hash(d)
    for arm in XARMS:                                  # E16 only
        with pytest.raises(ValueError):
            configuration(arm, 100, E=8)


def test_x_experiment_rejects_inconsistent_arm_config_and_backbone(model, tokenizer):
    cfg = configuration('X17', 10)
    cfg['model'].update(layer=1)
    for change in (lambda c: c.pop('backbone'), lambda c: c['dpfp'].pop('backbone'),
                   lambda c: c.update(backbone={**BACKBONE_17B, 'revision': 'x'})):
        broken = copy.deepcopy(cfg)
        change(broken)
        with pytest.raises(ValueError, match='mismatch'):
            DPFPExperiment(broken, 'X17', model=model, tokenizer=tokenizer)
    cfg = configuration('X256', 10)
    cfg['model'].update(layer=1, r=128)
    with pytest.raises(ValueError, match='mismatch'):
        DPFPExperiment(cfg, 'X256', model=model, tokenizer=tokenizer)
    cfg = configuration('D128-3', 10)
    cfg['model'].update(layer=1)
    cfg['backbone'] = BACKBONE_17B
    with pytest.raises(ValueError, match='mismatch'):
        DPFPExperiment(cfg, 'D128-3', model=model, tokenizer=tokenizer)


# ---------------------------------------------------------------------------- the rank 256 path on a tiny model

def test_x256_trains_exports_and_reloads_with_the_rank_256_memory(model, tokenizer, tmp_path):
    cfg = configuration('X256', 10)
    cfg['model'].update(layer=1)                       # rank stays 256: a [256, 1536] memory per episode
    experiment = DPFPExperiment(cfg, 'X256', model=model, tokenizer=tokenizer)
    try:
        adapter = experiment.adapter
        assert (adapter.cfg.r, adapter.cfg.r_mem, adapter.cfg.dpfp_nu, adapter.cfg.key_dim) == (256, 256, 3, 1536)
        assert (adapter.cfg.d_in, adapter.cfg.d_out) == (32, 16)   # from the loaded model: rank above d_in is allowed here
        experiment.standardize()
        result = fit(experiment, tmp_path, 5, evaluate=False)
        assert result['actual_steps'] == 5
        rows = [json.loads(line) for line in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
        assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in rows)
        assert rows[0]['read_loss'] > 0 and rows[0]['address_loss'] > 0
        exported = torch.load(tmp_path / 'writer.pt', weights_only=True)
        assert exported['arm'] == 'X256' and exported['signature']['dpfp_nu'] == 3 and exported['signature']['r_mem'] == 256
        assert 'backbone' not in exported['signature']
        with torch.no_grad():
            batch = experiment.prepare(read_fix_data(cfg))
            assert batch.snapshot.states[1].shape == (2, 256, 1536) and batch.keys.shape == (2, 16, 1536)
            expected, _ = experiment.score(batch, 1)
        metrics = validate(experiment, tmp_path / 'val', 16)
        assert metrics['n'] == 16 and math.isfinite(metrics['loss']) and metrics['gram_relative_error'] < 1e-3
    finally:
        experiment.close()
    loaded = load_inference(tmp_path / 'writer.pt', model=model, tokenizer=tokenizer)
    try:
        assert loaded.read_arm == 'X256' and loaded.value_head is None
        with torch.no_grad():
            actual, _ = loaded.score(loaded.prepare(read_fix_data(cfg)), 1)
        assert torch.equal(actual, expected)
    finally:
        loaded.close()


def read_fix_data(cfg):
    from looplora.data.relations import generate
    from looplora.llm.train import make_task_spec
    return generate(make_task_spec(cfg), 'train', 0, 2, derive_seed(17, 'dpfp-fixture'))


# ---------------------------------------------------------- the backbone option through the DPFP runner (tiny model)

TINY = dict(vocab_size=151936, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, attention_dropout=0.)  # o_proj 128 -> 64, as nothing in the 0.6B (2048 -> 1024)


@pytest.fixture()
def cache(tmp_path, monkeypatch, tokenizer):
    """A cache directory with a tiny Qwen3 (and the real tokenizer files) found through the backbone option only."""
    snapshot = tmp_path / 'cache' / 'models--tiny--qwen' / 'snapshots' / REVISION
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        Qwen3ForCausalLM(Qwen3Config(**TINY)).save_pretrained(snapshot)
    real = Path(inject.CACHE_DIR) / f'models--{MODEL_ID.replace("/", "--")}' / 'snapshots' / MODEL_REVISION
    for name in ('tokenizer.json', 'vocab.json', 'merges.txt', 'tokenizer_config.json'):
        shutil.copy(real / name, snapshot / name)
    monkeypatch.setattr(inject, 'CACHE_DIR', str(tmp_path / 'cache'))
    return {'id': 'tiny/qwen', 'revision': REVISION}


def test_x17_runner_loads_the_configured_backbone_and_sizes_shapes_from_it(cache, monkeypatch, tmp_path):
    real = dpfp_run.configuration

    def tiny_x17(arm, steps, *, E=16, smoke=False):
        cfg = real(arm, steps, E=E, smoke=smoke)
        cfg['model']['layer'] = 1
        cfg['eval']['episodes'] = 16
        cfg['backbone'] = cfg['dpfp']['backbone'] = dict(cache)  # the tiny model stands in for Qwen3-1.7B-Base
        return cfg
    monkeypatch.setattr(dpfp_run, 'configuration', tiny_x17)
    monkeypatch.setattr(dpfp_run, 'source_metadata', lambda: dict(git_commit='a' * 40, source_hash='0' * 32, source_dirty=False))
    expected_hash = stable_hash(tiny_x17('X17', 10))
    with pytest.raises(ValueError, match='frozen release'):
        dpfp_run.run_one('X17', tmp_path / 'wrong', 10, 16, False, expected_hash='0' * 32)
    record = dpfp_run.run_one('X17', tmp_path / 'run', 10, 16, False, expected_hash=expected_hash)
    assert record['status'] == 'complete' and record['config_hash'] == expected_hash
    assert (record['arm'], record['rank'], record['dpfp_nu'], record['key_dim']) == ('X17', 128, 3, 768)
    assert record['config']['backbone'] == cache and record['memory']['shape'] == [128, 768]
    assert record['memory']['state_bytes_per_episode'] == 4 * 128 * 768
    assert record['inference_parameters'] > 0 and set(record['training']['metrics']) == {'2', '4', '6', '8', '10'}
    experiment = load_inference(tmp_path / 'run' / 'writer.pt')      # no model passed: it comes from the backbone option
    try:
        assert experiment.model.config.hidden_size == 64
        assert (experiment.adapter.cfg.d_in, experiment.adapter.cfg.d_out) == (128, 64)
        assert experiment.writer_signature()['backbone'] == cache
        assert experiment.read_arm == 'X17'
    finally:
        experiment.close()


# ------------------------------------------------------------------------- the registered decision and the release

def x_records(scores):
    return [dict(arm=arm, status='complete', smoke=False, config=dict(task=dict(E=16)),
                 training=dict(actual_steps=10, metrics={'10': dict(correct=scores[arm])})) for arm in XARMS]


def test_choose_x_applies_the_registered_rule_and_choose_is_untouched():
    result = choose_x(x_records({'X256': 800, 'X17': 608}))
    assert (result['x256'], result['x17_vs_d128_3'], result['x17_candidate'], result['line']) == (
        'flagged_candidate', 'not_distinguished', False, 'flagged_candidate')
    assert result['candidates'] == ['X256'] and result['e8_retrain'] is False
    result = choose_x(x_records({'X256': 799, 'X17': 800}))
    assert (result['x256'], result['x17_vs_d128_3'], result['x17_candidate'], result['candidates']) == (
        'no_writer', 'higher', True, ['X17'])
    for scores, label in (({'X256': 600, 'X17': 658}, 'higher'), ({'X256': 600, 'X17': 657}, 'not_distinguished'),
                          ({'X256': 600, 'X17': 558}, 'lower'), ({'X256': 600, 'X17': 559}, 'not_distinguished')):
        result = choose_x(x_records(scores))
        assert (result['x17_vs_d128_3'], result['line']) == (label, 'stopped'), scores   # exactly 5 pp counts
    assert choose_x(x_records({'X256': 600, 'X17': 700}))['x17_gap_pp'] == 9.2
    for bad in (x_records({'X256': 1, 'X17': 1})[:1], x_records({'X256': 1, 'X17': 1}) * 2,
                [{**r, 'smoke': True} for r in x_records({'X256': 1, 'X17': 1})],
                [{**r, 'status': 'failed'} for r in x_records({'X256': 1, 'X17': 1})],
                [{**r, 'config': dict(task=dict(E=8))} for r in x_records({'X256': 1, 'X17': 1})]):
        with pytest.raises(ValueError):
            choose_x(bad)
    with pytest.raises(ValueError):                    # the DPFP rule still needs its own three arms
        dpfp_run.choose(x_records({'X256': 900, 'X17': 900}))


def test_x_step_counts_come_from_the_release_and_the_dpfp_release_cannot_name_them():
    release = dict(study='DPFP-X', E=16, steps_by_arm={'X256': 17000, 'X17': 21000})
    assert [scheduled_steps(arm, None, release) for arm in XARMS] == [17000, 21000]
    assert scheduled_steps('X17', 1800, None, smoke=True) == 1800
    dpfp = dict(study='DPFP', E=16, steps_by_arm={'D32-3': 20000, 'D128-1': 19000, 'D128-3': 18000})
    for call in (lambda: scheduled_steps('X256', None, dpfp),
                 lambda: scheduled_steps('D32-3', None, release),
                 lambda: scheduled_steps('X256', None, release, E=8),
                 lambda: scheduled_steps('X256', None, {**release, 'E': 8}, E=8),
                 lambda: scheduled_steps('X256', None, {**release, 'steps_by_arm': {'X256': 17000}}),
                 lambda: scheduled_steps('X256', None, {**release, 'steps_by_arm': {'X256': 17001, 'X17': 21000}}),
                 lambda: scheduled_steps('X256', 17000, release)):
        with pytest.raises(ValueError):
            call()
    assert scheduled_steps('D128-3', None, dpfp) == 18000


def test_prereg_digest_is_a_blake2b_128_of_the_committed_registration():
    import hashlib
    expected = hashlib.blake2b((ROOT / 'docs/prereg/DPFP-X.md').read_bytes(), digest_size=16).hexdigest()
    assert prereg_digest() == prereg_digest(ROOT) == expected and len(expected) == 32
    assert XSPECS['X17'][2]['id'] == 'Qwen/Qwen3-1.7B-Base'
