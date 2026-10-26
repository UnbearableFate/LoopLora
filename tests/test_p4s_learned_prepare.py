"""Preparation seals, inherited thresholds and pre-head validation barrier."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('prepare_p4s_learned', ROOT / 'scripts/prepare_p4s_learned.py')
prepare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_module)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def released(tmp_path):
    manifest = json.loads((ROOT / 'reports/p4s-r32/manifest.json').read_text())
    original_runs = manifest['runs']
    def make_checkpoint(run, folder):
        path = folder / run['name'] / 'checkpoint.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(('sealed reader ' + run['name']).encode())
        config = dict(name=run['name'], writer=run['writer'], variant=run['variant'], seed=run['seed'],
                      model=dict(r=32, r_mem=32))
        _save(path.with_name('config.json'), dict(config=config))
        _save(path.with_name('record.json'), dict(status='complete', experiment_id=run['name'],
              writer=run['writer'], variant=run['variant'], rank=32, train_seed=run['seed']))
        return path
    for run in original_runs:
        path = make_checkpoint(run, tmp_path / 'c4-readers')
        run['checkpoint'] = str(path)
        run['checkpoint_blake2b'] = prepare_module.digest_blake2b(path)
    h_root = tmp_path / 'h'
    for variant in prepare_module.VARIANTS:
        for seed in prepare_module.SEEDS:
            name = f'primary-oracle_random-{variant}-r32-s{seed}'
            make_checkpoint(dict(name=name, writer='oracle_random', variant=variant, seed=seed), h_root / 'runs')
    manifest_path = tmp_path / 'c4-manifest.json'
    _save(manifest_path, manifest)
    return manifest, manifest_path, h_root


def test_manifest_preserves_c4_seals_and_adds_exact_random_r32_readers(released):
    manifest, _, h_root = released
    before = copy.deepcopy(manifest)
    runs = prepare_module.build_runs(manifest, h_root)
    assert len(runs) == 24
    assert [run for run in runs if run['writer'] != 'oracle_random'] == sorted(manifest['runs'], key=lambda r: r['name'])
    assert manifest == before
    assert {prepare_module.identity(run) for run in runs if run['writer'] == 'oracle_random'} == {
        ('oracle_random', 32, variant, seed) for variant in prepare_module.VARIANTS for seed in prepare_module.SEEDS}
    assert all(prepare_module.digest_blake2b(run['checkpoint']) == run['checkpoint_blake2b'] for run in runs)


@pytest.mark.parametrize('damage', ['seal', 'record', 'config', 'payload_missing', 'identity_duplicate'])
def test_manifest_rejects_changed_or_incomplete_readers(released, damage):
    manifest, _, h_root = released
    run = manifest['runs'][0]
    path = Path(run['checkpoint'])
    if damage == 'seal':
        path.write_bytes(b'changed checkpoint')
    elif damage == 'record':
        record = json.loads(path.with_name('record.json').read_text())
        record['status'] = 'running'
        _save(path.with_name('record.json'), record)
    elif damage == 'config':
        envelope = json.loads(path.with_name('config.json').read_text())
        envelope['config']['seed'] = 999
        _save(path.with_name('config.json'), envelope)
    elif damage == 'payload_missing':
        run['checkpoint'] = str(path.with_name('absent.pt'))
    else:
        manifest['runs'][1] = copy.deepcopy(run)
    with pytest.raises(ValueError):
        prepare_module.build_runs(manifest, h_root)


def test_c4_rows_are_reused_and_missing_random_rows_are_not_inferred():
    manifest = json.loads((ROOT / 'reports/p4s-r32/manifest.json').read_text())
    design = prepare_module.collection_design(manifest['design'])
    rows = prepare_module.read_c4_thresholds(ROOT / 'reports/p4s-r32/thresholds.csv', design)
    assert len(rows) == 18
    with (ROOT / 'reports/p4s-r32/thresholds.csv').open(newline='') as stream:
        original = list(csv.DictReader(stream))
    for row, source in zip(rows, original):
        assert row['family'] == source['family']
        assert row['criterion'] == source['criterion']
        for key in ('threshold', 'validation_accuracy', 'validation_mean_steps'):
            assert row[key] == float(source[key])
    assert all(not row['family'].startswith('oracle_random/') for row in rows)
    assert design['collect_certificates'] is False and design['latency'] is False


def test_best_reference_ties_use_criterion_order_without_reusing_threshold_step_tiebreak():
    rows = [dict(family='oracle_random/r32/M1', criterion=criterion, threshold=1,
                 validation_accuracy=.5, validation_mean_steps=16 - index * 5)
            for index, criterion in enumerate(prepare_module.CRITERIA)]
    best = prepare_module.best_label_free(rows)
    assert best['oracle_random/r32/M1']['criterion'] == 'read_norm'
    rows[1]['validation_accuracy'] = .6
    assert prepare_module.best_label_free(rows)['oracle_random/r32/M1']['criterion'] == 'relative_change'
    with pytest.raises(ValueError, match='Incomplete'):
        prepare_module.best_label_free(rows[:2])


def _fake_runtime(monkeypatch, events, mutate=None):
    import audit_p4s_stopping
    import looplora.sym.run
    monkeypatch.setattr(audit_p4s_stopping, 'require_compute', lambda: None)
    monkeypatch.setattr(looplora.sym.run, 'source_metadata', lambda: dict(git_commit='a' * 40,
                                                                      source_hash='b' * 32, source_dirty=False))
    fake_cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda index: None,
                                get_device_name=lambda index: 'test GPU')
    fake_torch = SimpleNamespace(cuda=fake_cuda, set_num_threads=lambda count: None, __version__='test',
        load=lambda path, **kwargs: {'config': json.loads(Path(path).with_name('config.json').read_text())['config']})
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    def collect(model, split, design, out):
        assert split == 'val' and design['episodes_per_split'] == 2000
        assert design['evaluation_seed'] == 4242 and design['collect_certificates'] is False
        events.append(('collect', model['name'], split))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b'validation trace placeholder')
        if mutate is not None:
            mutate()
        return dict(episodes=2000, queries=16000, total_seconds=0.), None
    def select(root, runs, design):
        assert len(events) == 6
        assert all(run['writer'] == 'oracle_random' for run in runs)
        events.append(('select',))
        rows = [dict(family=family, criterion=criterion, threshold=1,
                     validation_accuracy=.5, validation_mean_steps=4.)
                for family in sorted({run['family'] for run in runs}) for criterion in prepare_module.CRITERIA]
        return dict(selected=rows, grid=copy.deepcopy(rows), metadata={'fake': True})
    monkeypatch.setitem(sys.modules, 'run_p4s_stopping', SimpleNamespace(collect=collect, load_model=lambda run: run))
    monkeypatch.setitem(sys.modules, 'analyze_p4s_stopping', SimpleNamespace(select_thresholds=select))
    monkeypatch.setenv('PBS_JOBID', 'fixture.opbs')


def test_prepare_locks_all_constants_after_only_six_random_validation_collections(released, tmp_path, monkeypatch):
    _, manifest_path, h_root = released
    events = []
    _fake_runtime(monkeypatch, events)
    raw, reports = tmp_path / 'raw', tmp_path / 'reports'
    audit = prepare_module.prepare(raw, reports, manifest_path, ROOT / 'reports/p4s-r32/thresholds.csv', h_root)
    design = json.loads((reports / 'design.json').read_text())
    assert audit['head_training_performed'] is False and audit['confirmation_inputs_opened'] is False
    assert [event[0] for event in events] == ['collect'] * 6 + ['select']
    assert len(design['runs']) == 24 and len(design['label_free_thresholds']) == 24
    assert len(design['primary_families']) == 6 and len(design['descriptive_families']) == 2
    assert design['theta_grid'] == [round(index / 20, 2) for index in range(1, 20)]
    assert design['normalization_epsilon'] == 1e-12
    assert design['head']['input_dim'] == sum(item['width'] for item in design['head']['features']) == 51
    assert design['head']['hidden_dim'] == 64
    assert design['training'] == dict(split='train', generator_seed=31337, episodes=20000,
        steps=2000, optimizer='adam', lr=.001, batch_episodes=256, hops_per_episode=8,
        steps_per_query=16, loss='binary_cross_entropy',
        minibatch='all eight queries and all sixteen post-update steps for each sampled episode')
    assert design['validation'] == dict(split='val', generator_seed=4242, episodes=2000)
    assert design['confirmation'] == dict(split='test', generator_seed=999002, episodes=2000)
    assert design['bootstrap'] == dict(resamples=2000, confidence=1 - .05 / 6, seed=2718,
        resampling='paired episode bootstrap, shared episode draws across three seeds')
    assert design['latency'] == dict(batch_size=64, warmup=5, repeats=20)
    assert design['seed_derivation']['head_init'] == "derive_seed(seed,'p4s-l',family,'head_init')"
    assert design['seed_derivation']['episode_sampler'] == "derive_seed(seed,'p4s-l',family,'episode_sampler')"
    assert 'design_hash' not in design
    from looplora.infra.seeds import stable_hash
    assert audit['design_hash'] == stable_hash(design)
    assert len(list((raw / 'traces').glob('*/*'))) == 6
    assert {row['source']['kind'] for row in design['label_free_thresholds']} == {
        'reused_C4', 'new_random_r32_validation_before_head_training'}
    assert all(row['source']['selected_before_any_head_training'] for row in design['label_free_thresholds'])
    with pytest.raises(ValueError, match='must be new'):
        prepare_module.prepare(raw, reports, manifest_path, ROOT / 'reports/p4s-r32/thresholds.csv', h_root)


def test_checkpoint_seal_change_during_validation_prevents_design_lock(released, tmp_path, monkeypatch):
    manifest, manifest_path, h_root = released
    path = Path(manifest['runs'][0]['checkpoint'])
    _fake_runtime(monkeypatch, [], mutate=lambda: path.write_bytes(b'changed while preparing'))
    raw, reports = tmp_path / 'changed-raw', tmp_path / 'changed-reports'
    with pytest.raises(ValueError, match='Checkpoint changed during'):
        prepare_module.prepare(raw, reports, manifest_path, ROOT / 'reports/p4s-r32/thresholds.csv', h_root)
    assert not (reports / 'design.json').exists()
    assert json.loads((raw / 'manifest.json').read_text())['status'] == 'failed'
