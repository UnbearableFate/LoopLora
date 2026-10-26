#!/usr/bin/python3
"""Authorized one-time, stdlib-only packing change before the dedicated boundary."""
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import subprocess

from run_p1l_main import ROOT, save


BASE = Path('/work/xg24i002/x10041/LoopLora/runs/p1l/main')
CUTOFF = datetime(2026, 10, 26, 22, 58, tzinfo=timezone.utc)


def main():
    changes = []
    for rank, seeds in [(32, [43]), (128, [17, 29, 43])]:
        root = BASE if rank == 32 else BASE / 'r128'
        report = ROOT / 'reports/p1l' / ('' if rank == 32 else 'r128')
        sizing = json.loads((report / 'smoke-sizing.json').read_text())
        times = {row['label']: row for row in sizing['timing_rows']}
        state_path, manifest_path = root / 'main-jobs.json', root / 'training-manifest.json'
        state, manifest = json.loads(state_path.read_text()), json.loads(manifest_path.read_text())
        for seed in seeds:
            tag = f'B4-B3-s{seed}'
            if tag not in state['jobs']:
                raise ValueError(f'packed receipt missing or already split: {rank}/{tag}')
            packed = state['jobs'][tag]
            detail = subprocess.check_output(['qstat', '-f', packed['job_id']], text=True)
            if not re.search(r'\n\s+job_state = Q\s*\n', detail):
                raise ValueError(f'will only replace a queued job: {packed["job_id"]}')
            old_out = Path(packed['argv'][packed['argv'].index('--out') + 1])
            if old_out.exists():
                raise FileExistsError(f'packed job has runtime output: {old_out}')
            plan = []
            for label in ['B4', 'B3-L4']:
                row = times[label]
                expected = manifest['S'] * row['step_seconds'] + row['overhead_seconds'] + 30.
                seconds = math.ceil(1.5 * expected)
                h, rem = divmod(seconds, 3600)
                m, sec = divmod(rem, 60)
                wall = f'{h:02d}:{m:02d}:{sec:02d}'
                out = root / 'main' / f'{label}-s{seed}'
                if out.exists():
                    raise FileExistsError(out)
                plan.append(dict(label=label, expected_seconds=expected, margin_seconds=seconds,
                                 walltime=wall, out=str(out)))
            now = datetime.now(timezone.utc)
            if any(now + timedelta(seconds=p['margin_seconds']) >= CUTOFF for p in plan):
                changes.append(dict(rank=rank, packed_job=packed['job_id'], status='left-queued-no-fit'))
                continue
            subprocess.run(['qdel', packed['job_id']], check=True)
            change = dict(rank=rank, seed=seed, packed_job=packed['job_id'], source=packed['source'],
                          reason='2026-10-27 orchestrator: same design, split to fit by07:58JST',
                          cancelled_at=now.isoformat(), cutoff=CUTOFF.isoformat(), singles=[])
            state.setdefault('superseded_jobs', {})[tag] = dict(packed, replacement=change)
            del state['jobs'][tag]
            save(state_path, state)
            for single in plan:
                label, out = single['label'], Path(single['out'])
                argv = ['scripts/run_p1l_main.py', 'main', '--out', str(out), '--seed', str(seed),
                        '--rank', str(rank), '--labels', label]
                name = f'p1l-r{rank}-one-{label}-s{seed}'
                command = [str(ROOT / 'scripts/submit.py'), '--name', name, '--queue', 'regular-g',
                           '--walltime', single['walltime'], '--commit', packed['source']['ref'], '--', *argv]
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                receipt = json.loads(result.stdout.strip())
                receipt.update(name=name, requested_walltime=single['walltime'], nodes=1,
                               expected_seconds=single['expected_seconds'], queue='regular-g', argv=argv)
                state['jobs'][f'{label}-s{seed}'] = receipt
                manifest['runs'][f'{label}-s{seed}'] = str(out / f'{label}-s{seed}')
                change['singles'].append(dict(single, job_id=receipt['job_id']))
                save(state_path, state)
                save(manifest_path, manifest)
                print(json.dumps(dict(rank=rank, seed=seed, **single, job_id=receipt['job_id'])), flush=True)
            manifest.setdefault('packing_changes', []).append(change)
            save(manifest_path, manifest)
            save(state_path, state)
            changes.append(change)
            save(ROOT / 'reports/p1l/dedicated-boundary-splits.json', changes)


if __name__ == '__main__':
    main()
