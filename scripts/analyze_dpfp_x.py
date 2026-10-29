"""Markdown tables of the DPFP-X study (docs/prereg/DPFP-X.md) from the committed records. Standard library only.

usage: python3 scripts/analyze_dpfp_x.py [reports/p1l-dpfp-x] [reports/p1l-dpfp]
Reads arms/X256 and arms/X17 (record.json, val-STEP/metrics.json) of the first folder and D128-3 of the second (DPFP, the reference
of the registered comparison), and prints the decision (the rule of `dpfp_run.choose_x`, restated because that module imports
torch), the curves at the five evenly spaced checkpoints (= 20, 40, 60, 80 and 100 % of each arm's steps, so they are matched
fractions), the diagnostics at the final checkpoint and the compute record of each run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from analyze_dpfp import DIAGNOSTICS, fmt, load, table

D128_3_CORRECT = 608


def decide(correct):
    gap = (correct['X17'] - D128_3_CORRECT) / 10
    versus = 'higher' if gap >= 5 else 'lower' if gap <= -5 else 'not distinguished'
    candidates = [a for a in ('X256', 'X17') if correct[a] >= 800]
    return dict(x256='flagged writer candidate' if correct['X256'] >= 800 else 'no writer at r = 256', x17_vs_d128_3=versus,
                x17_gap_pp=gap, candidates=candidates, line='flagged candidate' if candidates else 'learned-writer line stays stopped')


def main(root='reports/p1l-dpfp-x', dpfp='reports/p1l-dpfp'):
    release = json.loads((Path(root) / 'release.json').read_text())
    runs = {a: load(Path(root) / 'arms' / a) for a in ('X256', 'X17')}
    runs['D128-3'] = load(Path(dpfp) / 'arms' / 'D128-3')
    out, correct = [], {}
    rows = []
    for a, (r, m) in runs.items():
        last = max(m)
        assert last == r['training']['actual_steps'] and r['status'] == 'complete' and not r['smoke'] and r['config']['task']['E'] == 16
        assert r['source_dirty'] is False and r['training']['matched_eval_steps'] == []
        if a != 'D128-3':
            assert last == release['steps_by_arm'][a] and r['config_hash'] == release['config_hashes'][a]
            assert r['git_commit'] == release['science_commit']
        correct[a] = m[last]['correct']
        rows.append([a, r['rank'], r['dpfp_nu'], r['key_dim'], 'default 0.6B' if 'backbone' not in r['config'] else 'Qwen3-1.7B-Base',
                     last, m[last]['correct'], f'{100 * m[last]["em"]:.1f}', 'no' if m[last]['correct'] < 800 else 'yes'])
    out.append(table(['arm', 'rank', 'nu', 'key dim', 'backbone', 'steps', 'correct of 1,000', 'val EM %', '>= 0.80'], rows))
    d = decide(correct)
    out.append('\nRegistered decision: ' + '; '.join(f'{k} = {v}' for k, v in d.items()))
    out.append('\n### Validation EM % at the five evenly spaced checkpoints (the same fractions of training for every arm)\n')
    rows = []
    for a, (r, m) in runs.items():
        rows.append([a] + [f'{s}: {100 * m[s]["em"]:.1f}' for s in sorted(m)])
    out.append(table(['arm', '20 %', '40 %', '60 %', '80 %', '100 % (final)'], rows))
    out.append('\n### Difference to D128-3 at the same fraction of training (pp of val EM; standard error of a difference about 2.2 pp)\n')
    base = runs['D128-3'][1]
    rows = []
    for a in ('X256', 'X17'):
        m = runs[a][1]
        rows.append([a] + [f'{100 * (m[s]["em"] - base[t]["em"]):+.1f}' for s, t in zip(sorted(m), sorted(base))])
    out.append(table(['arm', '20 %', '40 %', '60 %', '80 %', '100 % (final)'], rows))
    out.append('\n### Diagnostics at the final checkpoint (one seed)\n')
    cols = list(runs)
    finals = {a: runs[a][1][max(runs[a][1])] for a in cols}
    out.append(table(['metric'] + cols, [[label] + [fmt(finals[c].get(key), scale, digits) for c in cols] for key, label, scale, digits in DIAGNOSTICS]))
    out.append('\n### Read error and read-value head accuracy over training (normalized read error / read-value head %)\n')
    rows = [[a] + [f'{s}: {m[s]["relative_read_error"]:.3f} / {100 * m[s]["actual_read_correct"]:.1f}' for s in sorted(m)] for a, (r, m) in runs.items()]
    out.append(table(['arm', '20 %', '40 %', '60 %', '80 %', '100 % (final)'], rows))
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
    main(*sys.argv[1:3])
