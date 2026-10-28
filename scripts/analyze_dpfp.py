"""Markdown tables of the DPFP study (docs/prereg/DPFP.md) from the committed records. Standard library only.

usage: python3 scripts/analyze_dpfp.py [reports/p1l-dpfp]
Reads arms/ARM/record.json, arms/ARM/val-STEP/metrics.json and refs/R-r32, refs/R-r128 (WP-L4 R records, not rerun) and prints
markdown. The decision is the registered rule (the same as `dpfp_run.choose`, restated here because that module imports torch).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ('D32-3', 'D128-1', 'D128-3')  # candidate order of the registered rule
DIAGNOSTICS = [
    ('em', 'val EM %', 100, 1), ('nll', 'NLL', 1, 3), ('relative_read_error', 'normalized read error', 1, 3),
    ('clean_value_correct', 'clean-value head acc %', 100, 1), ('actual_read_correct', 'read-value head acc %', 100, 1),
    ('value_SIR', 'value SIR', 1, 2), ('key_top1', 'key top-1 %', 100, 1), ('key_margin', 'key margin', 1, 3),
    ('key_gram_rms', 'key Gram RMS (phi space for DPFP)', 1, 3), ('key_gram_rms_base', 'key Gram RMS (original space)', 1, 3),
    ('key_gram_max_abs', 'mean max |cos| (phi space for DPFP)', 1, 3), ('key_gram_max_abs_base', 'mean max |cos| (original space)', 1, 3),
    ('key_pairwise_cosine', 'mean pairwise cosine (as logged)', 1, 3), ('parallel_interference', 'parallel interference', 1, 3),
    ('q_norm', 'query norm', 1, 1),
]


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def load(folder):
    folder = Path(folder)
    record = json.loads((folder / 'record.json').read_text())
    steps = sorted(int(p.name.split('-')[1]) for p in folder.glob('val-*'))
    metrics = {s: json.loads((folder / f'val-{s}' / 'metrics.json').read_text()) for s in steps}
    return record, metrics


def decide(correct):
    """docs/prereg/DPFP.md: first candidate with 800 of 1,000; else the best at 500 or more retries at E = 8; else stop."""
    passed = [a for a in ARMS if correct[a] >= 800]
    if passed:
        return f'select {passed[0]} at E = 16'
    best = max(ARMS, key=lambda a: (correct[a], -ARMS.index(a)))
    return f'retry {best} at E = 8 (flagged)' if correct[best] >= 500 else 'no writer; stop'


def fmt(value, scale, digits):
    return '' if value is None else f'{scale * value:.{digits}f}'


def main(folder):
    folder = Path(folder)
    runs = {a: load(folder / 'arms' / a) for a in ARMS}
    refs = {r: load(folder / 'refs' / r) for r in ('R-r32', 'R-r128')}
    out = ['### Decision table (final checkpoint, 1,000 validation episodes, eval seed 4242)\n']
    rows, correct = [], {}
    for a, (r, m) in runs.items():
        last = max(m)
        assert last == r['training']['actual_steps'] and r['status'] == 'complete' and not r['smoke'] and r['config']['task']['E'] == 16
        correct[a] = m[last]['correct']
        rows.append([a, r['rank'], r['dpfp_nu'], r['key_dim'], last, m[last]['correct'], f'{100 * m[last]["em"]:.1f}', 'no' if m[last]['correct'] < 800 else 'yes'])
    for name, (r, m) in refs.items():
        last = max(m)
        rows.append([f'{name} (WP-L4, not rerun)', r['rank'], '', '', last, m[last]['correct'], f'{100 * m[last]["em"]:.1f}', ''])
    out.append(table(['arm', 'rank', 'nu', 'key dim', 'steps', 'correct of 1,000', 'val EM %', '>= 0.80'], rows))
    out.append(f'\nRegistered decision: **{decide(correct)}** (best {max(correct, key=correct.get)} with {max(correct.values())} of 1,000; thresholds 800 and 500).')
    out.append('\n### Validation EM % at the five evenly spaced checkpoints\n')
    rows = []
    for a, (r, m) in runs.items():
        rows.append([a] + [f'{s}: {100 * m[s]["em"]:.1f}' for s in sorted(m)])
    for name, (r, m) in refs.items():
        rows.append([f'{name} (WP-L4)'] + [f'{s}: {100 * m[s]["em"]:.1f}' for s in sorted(m)])
    out.append(table(['arm', 'checkpoint 1', '2', '3', '4', '5 (final)'], rows))
    out.append('\n### Diagnostics at the final checkpoint (WP-L4 audit columns; one seed)\n')
    cols = list(runs) + list(refs)
    finals = {a: runs[a][1][max(runs[a][1])] for a in runs} | {n: refs[n][1][max(refs[n][1])] for n in refs}
    out.append(table(['metric'] + cols, [[label] + [fmt(finals[c].get(key), scale, digits) for c in cols] for key, label, scale, digits in DIAGNOSTICS]))
    out.append('\n### Read error and clean/read head accuracy over training (normalized read error / read-value head %)\n')
    rows = [[a] + [f'{s}: {m[s]["relative_read_error"]:.3f} / {100 * m[s]["actual_read_correct"]:.1f}' for s in sorted(m)] for a, (r, m) in runs.items()]
    rows += [[f'{n} (WP-L4)'] + [f'{s}: {m[s]["relative_read_error"]:.3f} / {100 * m[s]["actual_read_correct"]:.1f}' for s in sorted(m)] for n, (r, m) in refs.items()]
    out.append(table(['arm', 'checkpoint 1', '2', '3', '4', '5 (final)'], rows))
    out.append('\n### Jobs, time and memory\n')
    rows = []
    for a, (r, m) in runs.items():
        t, mem = r['training'], r['memory']
        rows.append([a, r['job_id'], r['git_commit'][:8], t['actual_steps'], f'{t["seconds"] / 60:.1f}', f'{t["steady_step_mean_seconds"]:.3f}',
                     f'{max(v["seconds"] for v in t["metrics"].values()):.1f}', f'{mem["peak_cuda_allocated_bytes"] / 2**30:.2f}',
                     mem['state_bytes_per_episode'], r['inference_parameters']])
    out.append(table(['arm', 'job', 'source', 'steps', 'training min', 'steady step s', 'slowest 1,000-episode eval s', 'peak CUDA GiB',
                      'state bytes per episode', 'inference parameters'], rows))
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/p1l-dpfp')
