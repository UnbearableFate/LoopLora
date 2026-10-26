"""WP-C4 paired endpoints, certificate-free traces, and released matrix checks."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_p4s_r32 as analysis
import analyze_p4s_stopping as shared
from run_p4s_r32 import checkpoint_manifest


def trace(episodes=2):
    shape = (episodes, 8)
    return dict(predictions=np.zeros((*shape, 17), dtype=np.int64),
                answer=np.ones(shape, dtype=np.int64),
                hop=np.broadcast_to(np.arange(1, 9), shape).copy(),
                episode_id=np.broadcast_to(np.arange(episodes)[:, None], shape).copy(),
                read_norm=np.ones((*shape, 17)), relative_change=np.ones((*shape, 17)))


def test_g1_differences_pair_queries_and_specificity_subtracts_far_gain():
    data = trace()
    data['predictions'][:, [0, 6], 1] = 1
    data['predictions'][:, 1, 4] = 1
    data['predictions'][0, [0, 2, 5, 6], 4] = 1
    endpoints = analysis.g1_estimates(data, np.full((2, 8), 4))
    assert endpoints == pytest.approx(dict(hop1_delta=-.5, reach_delta=.75,
                                          far_delta=0, walk_specificity=.75))
    assert analysis.g1_estimates(data, np.ones((2, 8), dtype=int)) == {
        k: 0 for k in endpoints}


@pytest.mark.parametrize('h1,reach,specificity,expected', [
    (-.01, .03, .02, True), (.02, .03, .02, True),
    (-.0101, .03, .02, False), (0, 0, .02, False), (0, .03, 0, False),
    (0, .03, -.01, False),
])
def test_candidate_means_hop1_preservation_and_strictly_positive_reach_specificity(
        h1, reach, specificity, expected):
    assert analysis.candidate(dict(hop1_delta=h1, reach_delta=reach,
                                   walk_specificity=specificity)) is expected


@pytest.fixture
def study(tmp_path):
    design = json.loads((Path(__file__).resolve().parents[1] / 'reports/p4s-r32/design.json').read_text())
    runs = [dict(name=f'{w}-{v}-{s}', writer=w, r=32, variant=v, seed=s,
                 family=f'{w}/r32/{v}', checkpoint=f'/checkpoint/{w}-{v}-{s}.pt')
            for w in design['families']['writer'] for v in design['families']['variant']
            for s in design['training_seeds']]
    for run in runs:
        folder = tmp_path / 'traces' / run['name']
        folder.mkdir(parents=True)
        for split in ('val', 'test'):
            data = trace()
            data['predictions'][..., 1] = 1
            data['read_norm'][..., 1] = 0
            data['relative_change'][..., 1] = 0
            np.savez(folder / f'{split}.npz', **data)
    return tmp_path, runs, design


def test_certificate_free_selection_and_summary_cover_six_families(study):
    root, runs, design = study
    selection = shared.select_thresholds(root, runs, design)
    assert len(selection['selected']) == 18
    endpoints = analysis.summarize(root, runs, design, selection, root / 'out')
    assert len(endpoints) == 6 * 8
    assert all(r['hop1_delta'] == 0 for r in endpoints if r['method'] == 'relative_change')
    assert all(not r['screening_candidate'] for r in endpoints)
    with (root / 'out/accuracy.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 * 2 * 8 * 9
    assert {r['method'] for r in rows} == {
        'fixed_1', 'fixed_4', 'fixed_8', 'fixed_16', *shared.CRITERIA, 'oracle_true_hop'}
    assert not (root / 'out/contraction.csv').exists()
    assert not (root / 'out/latency.csv').exists()
    assert (root / 'out/tables.md').read_text().count('## ') == 6
    # The default WP-C3 contract still requires real certificates.
    design.pop('collect_certificates')
    with pytest.raises(KeyError, match='certified_c'):
        shared._load_trace(root, runs[0], 'val', design)


def test_summary_rejects_frozen_selection_drift_before_opening_traces(study, monkeypatch):
    root, runs, design = study
    selection = shared.select_thresholds(root, runs, design)
    selection['selected'][0]['threshold'] = 123

    def unexpected(*args, **kwargs):
        pytest.fail('Trace opened after frozen selection drift')

    monkeypatch.setattr(analysis, '_load_trace', unexpected)
    with pytest.raises(ValueError):
        analysis.summarize(root, runs, design, selection, root / 'out')


def test_family_endpoints_average_training_seeds_equally(study):
    root, runs, design = study
    family = runs[0]['family']
    for run in [r for r in runs if r['family'] == family]:
        data = trace(8 if run['seed'] == 17 else 2)
        data['predictions'][..., 1] = 0
        data['predictions'][..., 4] = int(run['seed'] == 17)
        np.savez(root / 'traces' / run['name'] / 'test.npz', **data)
    selection = shared.select_thresholds(root, runs, design)
    endpoints = analysis.summarize(root, runs, design, selection, root / 'out')
    row = next(r for r in endpoints if r['family'] == family and r['method'] == 'fixed_4')
    assert row['reach_delta'] == pytest.approx(1 / 3)
    assert row['far_delta'] == pytest.approx(1 / 3)
    assert row['walk_specificity'] == 0


@pytest.mark.parametrize('mutation', [None, 'missing', 'duplicate'])
def test_released_manifest_requires_exactly_18_identities_and_complete_checkpoints(tmp_path, mutation):
    rows = []
    for w in ('joint_fact', 't1_frozen_fact'):
        for v in ('M1', 'M1-lin'):
            for s in (17, 29, 43):
                name = f'f2-main-{w}-{v}-s{s}'
                folder = tmp_path / 'f2' / name
                rows.append(dict(name=name, group='primary', writer=w, variant=v, seed=s,
                                 training_dir=str(folder)))
    for v in ('M1', 'M1-lin'):
        for s in (17, 29, 43):
            folder = tmp_path / 'h/runs' / f'primary-joint-{v}-r32-s{s}'
            folder.mkdir(parents=True)
            (folder / 'checkpoint.pt').write_bytes(b'checkpoint')
            (folder / 'record.json').write_text('{"status":"complete"}')
    for row in rows:
        folder = Path(row['training_dir'])
        folder.mkdir(parents=True)
        (folder / 'checkpoint.pt').write_bytes(b'checkpoint')
        (folder / 'record.json').write_text('{"status":"complete"}')
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1] = rows[0]
    manifest = tmp_path / 'manifest.csv'
    analysis.write_csv(manifest, rows)
    if mutation:
        with pytest.raises(ValueError, match='exactly the 18'):
            checkpoint_manifest(manifest, tmp_path / 'h', tmp_path / 'f2')
    else:
        runs = checkpoint_manifest(manifest, tmp_path / 'h', tmp_path / 'f2')
        assert len(runs) == 18 and len({r['family'] for r in runs}) == 6
        assert all(len(r['checkpoint_blake2b']) == 32 for r in runs)
        Path(runs[0]['checkpoint']).with_name('record.json').write_text('{"status":"running"}')
        with pytest.raises(ValueError, match='Incomplete training'):
            checkpoint_manifest(manifest, tmp_path / 'h', tmp_path / 'f2')
