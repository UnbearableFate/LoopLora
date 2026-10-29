"""Markdown tables and the registered decision of LSQT (docs/prereg/LSQT.md) from the committed records. Standard library only.

usage: python3 scripts/analyze_lsqt.py [reports/p1l-lsqt] [reports/p1l-dpfp] [--decision-out FILE]
Reads release.json, arms/ARM/record.json, arms/ARM/val-STEP/metrics.json and arms/ARM/train_log.jsonl of the first folder, and R-r128
(refs/R-r128, WP-L4, not rerun) and D128-3 (arms/D128-3) of the second as the descriptive references. It first verifies every arm record
against the release (complete, released steps, all marks, config hash, science commit, finite training), then prints the decision
(the rule of `lsqt_run.choose_t`, restated because that module imports torch), the curves, the diagnostics and the registered predictions.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from analyze_dpfp import DIAGNOSTICS, fmt, load, table

ARMS = ('LT-K4', 'LT-RG001', 'LT-RG1')  # candidate order
RIDGE = ('LT-RG001', 'LT-RG1')
R_R128 = 335
D128_3 = 608
BAND = dict(w3=R_R128 + 150, closed=400)


def decide(correct):
    """docs/prereg/LSQT.md: outcome 1 (any arm >= 800: the highest, ties in candidate order), outcome 2 by the best RG arm."""
    passed = [a for a in ARMS if correct[a] >= 800]
    best_rg = max(RIDGE, key=lambda a: (correct[a], -ARMS.index(a)))
    band = 'w3_trigger' if correct[best_rg] >= BAND['w3'] else 'closed' if correct[best_rg] < BAND['closed'] else 'open'
    best = max(ARMS, key=lambda a: (correct[a], -ARMS.index(a)))
    return dict(study='LSQT', outcome='flagged_candidate' if passed else 'no_candidate',
                candidate=max(passed, key=lambda a: (correct[a], -ARMS.index(a))) if passed else None, e8_retrain=False,
                best_rg=best_rg, best_rg_band=band, trained_lsq_line='closed' if band == 'closed' else 'open',
                u_control_proposed=correct[best] >= BAND['closed'], best_arm=best,
                vs_r_r128_pp={a: (correct[a] - R_R128) / 10 for a in ARMS}, vs_d128_3_pp={a: (correct[a] - D128_3) / 10 for a in ARMS},
                correct=dict(correct))


def verify(root, release, runs):
    facts = {}
    for a, (r, m) in runs.items():
        t = r['training']
        marks = release['primary_eval_steps_by_arm'][a]
        assert r['status'] == 'complete' and not r['smoke'] and r['source_dirty'] is False and r['config']['task']['E'] == 16, a
        assert t['actual_steps'] == release['steps_by_arm'][a] and sorted(m) == marks and sorted(map(int, t['metrics'])) == marks, a
        assert r['config_hash'] == release['config_hashes'][a] and r['git_commit'] == release['science_commit'], a
        assert all(v['n'] == 1000 for v in m.values()) and r['rank'] == 128 and r['store_eta'] == release['etas'][a], a
        rows = [json.loads(line) for line in (root / 'arms' / a / 'train_log.jsonl').read_text().splitlines()]
        finite_metrics = all(math.isfinite(x) for v in m.values() for x in v.values() if isinstance(x, (int, float)))
        facts[a] = dict(finite=all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in rows) and finite_metrics, rows=rows,
                        clipped=sum(x['clipped'] for x in rows) / len(rows))
    return facts


def main(folder='reports/p1l-lsqt', dpfp='reports/p1l-dpfp', decision_out=None):
    root = Path(folder)
    release = json.loads((root / 'release.json').read_text())
    runs = {a: load(root / 'arms' / a) for a in ARMS}
    facts = verify(root, release, runs)
    refs = {'R-r128 (WP-L4, not rerun)': load(Path(dpfp) / 'refs' / 'R-r128'), 'D128-3 (DPFP)': load(Path(dpfp) / 'arms' / 'D128-3')}
    final = {a: m[max(m)] for a, (r, m) in runs.items()}
    correct = {a: final[a]['correct'] for a in ARMS}
    d = decide(correct)
    out = ['### Decision table (each arm at its own final checkpoint, 1,000 validation episodes, eval seed 4242)\n']
    rows = [[a, runs[a][0]['job_id'], runs[a][0]['store'] + ('' if runs[a][0]['store_eta'] is None else f' (eta {runs[a][0]["store_eta"]:g})'),
             max(runs[a][1]), correct[a], f'{100 * final[a]["em"]:.1f}', f'{final[a]["relative_read_error"]:.3f}',
             f'{100 * final[a]["clean_value_correct"]:.1f}', f'{100 * final[a]["actual_read_correct"]:.1f}', 'yes' if correct[a] >= 800 else 'no'] for a in ARMS]
    for name, (r, m) in refs.items():
        f = m[max(m)]
        rows.append([name, '', 'delta', max(m), f['correct'], f'{100 * f["em"]:.1f}', f'{f["relative_read_error"]:.3f}',
                     f'{100 * f["clean_value_correct"]:.1f}', f'{100 * f["actual_read_correct"]:.1f}', 'no'])
    out.append(table(['arm', 'job', 'store', 'steps', 'correct of 1,000', 'val EM %', 'normalized read error', 'clean-value head %', 'read-value head %', '>= 0.80'], rows))
    out.append(f'\nRegistered decision: **{"flagged writer candidate " + d["candidate"] if d["candidate"] else "no arm reaches 0.80, no writer candidate"}**. '
               f'Best RG arm {d["best_rg"]} with {correct[d["best_rg"]]} of 1,000 ({correct[d["best_rg"]] / 10:.1f} %): band **{d["best_rg_band"]}** '
               f'(>= 48.5 % W3 trigger; 40 % to < 48.5 % neither; < 40 % the trained-LSQ line closes). Trained-LSQ line: **{d["trained_lsq_line"]}**. '
               f'Best arm {d["best_arm"]} ({correct[d["best_arm"]] / 10:.1f} %): the conditional U control is {"proposed (best arm >= 40 %)" if d["u_control_proposed"] else "not proposed (best arm < 40 %)"}.')
    out.append('\nDifference to the references in pp of val EM (descriptive; standard error of a difference about 2.2 pp):\n')
    out.append(table(['arm', 'minus R-r128 (33.5 %)', 'minus D128-3 (60.8 %)'], [[a, f'{d["vs_r_r128_pp"][a]:+.1f}', f'{d["vs_d128_3_pp"][a]:+.1f}'] for a in ARMS]))
    out.append('\n### Validation EM % at the five evenly spaced checkpoints (20, 40, 60, 80 and 100 % of each arm\'s steps: matched fractions of training)\n')
    rows = [[a] + [f'{s}: {100 * m[s]["em"]:.1f}' for s in sorted(m)] for a, (r, m) in runs.items()]
    rows += [[name] + [f'{s}: {100 * m[s]["em"]:.1f}' for s in sorted(m)] for name, (r, m) in refs.items()]
    out.append(table(['arm', '20 %', '40 %', '60 %', '80 %', '100 % (final)'], rows))
    out.append('\n### Normalized read error at the five checkpoints\n')
    out.append(table(['arm', '20 %', '40 %', '60 %', '80 %', '100 % (final)'],
                     [[a] + [f'{m[s]["relative_read_error"]:.3f}' for s in sorted(m)] for a, (r, m) in list(runs.items()) + list(refs.items())]))
    out.append('\n### Read-value head accuracy % (actual read) and clean-value head accuracy % at the five checkpoints\n')
    out.append(table(['arm'] + ['20 %', '40 %', '60 %', '80 %', '100 % (final)'], [[a] + [f'{100 * m[s]["actual_read_correct"]:.1f} / {100 * m[s]["clean_value_correct"]:.1f}' for s in sorted(m)] for a, (r, m) in runs.items()]))
    out.append('\n### Diagnostics at the final checkpoint (one seed)\n')
    cols = list(ARMS)
    out.append(table(['metric'] + cols + ['D128-3'], [[label] + [fmt(final[a].get(key), scale, digits) for a in cols] +
                      [fmt(refs['D128-3 (DPFP)'][1][max(refs['D128-3 (DPFP)'][1])].get(key), scale, digits)] for key, label, scale, digits in DIAGNOSTICS]))
    extra = [('gram_relative_error', 'Gram reconstruction relative error', 1, 6), ('key_gram_log10_cond', 'log10 cond of the key Gram', 1, 2), ('ridge_log10_cond', 'log10 cond of the ridge system', 1, 2)]
    out.append('\n' + table(['metric'] + cols, [[label] + [fmt(final[a].get(key), scale, digits) for a in cols] for key, label, scale, digits in extra]))
    out.append('\n### Training-log diagnostics (mean of the last 10 logged steps; one batch each)\n')
    def tail(a, key):
        vals = [x[key] for x in facts[a]['rows'][-10:] if key in x]
        return f'{sum(vals) / len(vals):.3f}' if vals else ''
    keys = [('read_loss', 'raw-read loss'), ('main_loss', 'main loss'), ('key_gram_log10_cond_median', 'log10 cond of the key Gram (median)'),
            ('ridge_log10_cond_median', 'log10 cond of the ridge system (median)'), ('grad_norm', 'gradient norm before clipping')]
    out.append(table(['metric'] + cols, [[label] + [tail(a, key) for a in cols] for key, label in keys]))
    out.append('\nFraction of logged steps with the gradient clipped: ' + ', '.join(f'{a} {100 * facts[a]["clipped"]:.0f} %' for a in ARMS) + '.')
    rg = d['best_rg']
    out.append('\n### Registered predictions (fable W2; reported, they decide nothing)\n')
    out.append(table(['prediction', 'observed', 'held'], [
        ['RG >= 50 % with read error <= 0.35', f'best RG {rg} {correct[rg] / 10:.1f} %, read error {final[rg]["relative_read_error"]:.3f}',
         'yes' if correct[rg] >= 500 and final[rg]['relative_read_error'] <= .35 else 'no'],
        ['K4 between R-r128 and RG', f'K4 {correct["LT-K4"] / 10:.1f} %, R-r128 {R_R128 / 10:.1f} %, RG {" / ".join(f"{correct[a] / 10:.1f}" for a in RIDGE)} %',
         'yes' if R_R128 <= correct['LT-K4'] <= min(correct[a] for a in RIDGE) else 'no'],
        ['read error < 0.1 and read-value head within 5 pp of the clean head (at r = 128)',
         'read errors ' + ' / '.join(f'{final[a]["relative_read_error"]:.3f}' for a in ARMS) + '; read-value minus clean-value head ' +
         ' / '.join(f'{100 * (final[a]["actual_read_correct"] - final[a]["clean_value_correct"]):.1f} pp' for a in ARMS),
         'yes' if all(final[a]['relative_read_error'] < .1 for a in RIDGE) else 'no'],
        ['if read error < 0.1 and EM < 0.50, the fault is in the query path', 'not applicable: no arm has read error below 0.1' if not any(final[a]['relative_read_error'] < .1 for a in ARMS) else 'see table', 'n/a']]))
    out.append('\n### Jobs, time and memory\n')
    rows = []
    for a, (r, m) in runs.items():
        t, mem = r['training'], r['memory']
        rows.append([a, r['job_id'], r['git_commit'][:8], t['actual_steps'], f'{t["seconds"] / 60:.1f}', f'{t["steady_step_mean_seconds"]:.3f}',
                     f'{max(v["seconds"] for v in t["metrics"].values()):.1f}', f'{mem["peak_cuda_allocated_bytes"] / 2**30:.2f}', mem['state_bytes_per_episode'], r['inference_parameters']])
    out.append(table(['arm', 'job', 'source', 'steps', 'training min', 'steady step s', 'slowest 1,000-episode eval s', 'peak CUDA GiB', 'state bytes per episode', 'inference parameters'], rows))
    print('\n'.join(out))
    if decision_out:
        with open(decision_out, 'x') as handle:
            handle.write(json.dumps(d, indent=2, sort_keys=True) + '\n')
    return d


if __name__ == '__main__':
    args = sys.argv[1:]
    target = None
    if '--decision-out' in args:
        i = args.index('--decision-out')
        target = args[i + 1]
        del args[i:i + 2]
    main(*(args[:2] or []), decision_out=target)
