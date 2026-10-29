import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from looplora.llm.s1b_analysis import validate_samples

root = Path('/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587')
digest = lambda p: hashlib.blake2b(Path(p).read_bytes(), digest_size=16).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
tick = time.perf_counter()
ledger = load(root / 'main-jobs.json')
lock = load(root / 'manifest.lock.json')
release = load(root / 'release.json')
assert ledger['status'] == 'complete'
assert digest(root / 'release.json') == 'd1ccbb7ff4c31fa53abfd7a3cc9ad572'
assert digest(root / 'approval.json') == '27f254b499b302ed932e5f3260e08034'
assert digest('scripts/analyze_s1b.py') == '78f001a232595acce6d38091f0e4895b'
assert not (root / 'dev-selection').exists()
assert not (root / 'dev-audit/main-audit.json').exists()
receipts = []
for job in ledger['jobs']:
    path = Path(job['out'])
    record = load(path / 'record.json')
    assert digest(path / 'record.json') == job['record_blake2b_128']
    assert record['stream_seed'] == lock['manifest']['train_streams'][str(job['seed'])]['base_seed']
    with np.load(path / 'train_steps.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'loss', 'grad_norm', 'step_seconds'}
        arrays = {name: saved[name] for name in saved.files}
    for name, value in arrays.items():
        assert value.shape == (18000,) and np.isfinite(value).all(), (job['label'], name)
    assert (arrays['grad_norm'] >= 0).all() and (arrays['step_seconds'] > 0).all()
    assert float(np.mean(arrays['grad_norm'] > 1.)) == record['clipped_fraction']
    assert float(np.mean(arrays['step_seconds'][10:])) == record['step_seconds']
    logs = [json.loads(line) for line in (path / 'train_log.jsonl').read_text().splitlines()]
    assert [row['step'] for row in logs] == [1, *range(50, 18001, 50)]
    for row in logs:
        i = row['step'] - 1
        assert row['loss'] == float(arrays['loss'][i])
        assert row['grad_norm'] == float(arrays['grad_norm'][i])
        assert row['seconds'] == float(arrays['step_seconds'][i])
        assert row['clipped'] == bool(arrays['grad_norm'][i] > 1.)
        assert row['lr'] == record['lr'] * min(1., row['step'] / 20)
    for row in record['dev']:
        with np.load(path / row['checkpoint']['samples'], allow_pickle=False) as samples:
            validate_samples(samples, lock['manifest']['ids']['dev'], hops=(1, 2, 3))
            correct = samples['correct']
            nll = samples['nll'].astype(np.float64)
            assert np.array_equal(correct, samples['prediction'] == samples['answer'])
            assert row['em'] == float(correct.mean())
            assert row['nll'] == float(nll.mean())
            for i, hop in enumerate((1, 2, 3)):
                h = row['per_hop'][str(hop)]
                assert h['correct'] == int(correct[:, i].sum())
                assert h['em'] == float(correct[:, i].mean())
                assert h['nll'] == float(nll[:, i].mean())
    receipts.append(dict(label=job['label'], status='passed', updates=18000,
        train_log_rows=len(logs), dev_sample_sets=11,
        record_blake2b_128=digest(path / 'record.json'),
        train_steps_blake2b_128=digest(path / 'train_steps.npz'),
        train_log_blake2b_128=digest(path / 'train_log.jsonl')))
    print(json.dumps(receipts[-1]), flush=True)
audit = dict(stage='S1B-A1-DEV-AUDIT', status='complete', job_id=os.environ['PBS_JOBID'],
    host=os.uname().nodename, source_commit=ledger['source_commit'],
    release_blake2b_128=digest(root / 'release.json'),
    manifest_blake2b_128=digest(root / 'manifest.lock.json'),
    approval_blake2b_128=digest(root / 'approval.json'),
    payload_blake2b_128=digest(root / 'dev-audit/payload.py'),
    elapsed_seconds=time.perf_counter() - tick, runs=receipts,
    finite_updates=216000, dev_sample_sets=132)
(root / 'dev-audit/main-audit.json').write_text(json.dumps(audit, indent=1, sort_keys=True, allow_nan=False) + '\n')
subprocess.run([sys.executable, 'scripts/analyze_s1b.py', '--stage', 'DEV',
    '--manifest', str(root / 'manifest.lock.json'), '--release', str(root / 'release.json'),
    '--records', str(root / 'main'), '--out', str(root / 'dev-selection')], check=True)
selection = load(root / 'dev-selection/selection.json')
assert selection['status'] == 'selected' and len(selection['selected_checkpoints']) == 12
assert selection['release_blake2b_128'] == digest(root / 'release.json')
print(json.dumps(dict(status='complete', stage='S1B-A1-DEV',
    audit_blake2b_128=digest(root / 'dev-audit/main-audit.json'),
    selection_blake2b_128=digest(root / 'dev-selection/selection.json'),
    total_seconds=time.perf_counter() - tick)), flush=True)
