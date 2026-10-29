"""Markdown tables and the registered decision of SQ2 (docs/prereg/SQ2.md, with Clarification C1) from the committed records.
Standard library only.

usage: python3 scripts/analyze_sq2.py [reports/p1l-sq2] [--decision-out FILE]
Reads release.json, numerics.json, step0.json (when present), arms/ARM/record.json, arms/ARM/val-STEP/metrics.json (STEP 0 is the
step-0 evaluation) and arms/ARM/train_log.jsonl. It first verifies every arm record against the release (complete, N steps, all
evaluation marks, config hash, science commit, numerics digest, parent), then prints the decision of the registered primary
contrast S-RC against S-DE, the descriptive cells and the interaction, the registered prediction, and the diagnostics.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from analyze_dpfp import DIAGNOSTICS, fmt, load, table

ARMS = ('S-DE', 'S-DC', 'S-RE', 'S-RC')
D128_3 = dict(correct=608, read_error=0.363)
EM_GAP_PP = 10
ERROR_RATIO = 0.5
CLEAN_SLACK = 0.05
CANDIDATE = 800
SE_PP = 2.2


def decide(correct, read_error, clean, finite):
    """The registered rule. correct: arm -> final correct of 1,000; read_error: arm -> final mean relative_read_error; clean: arm -> the
    five checkpoints' clean-value head accuracies (fractions); finite: arm -> no nonfinite loss or gradient in training."""
    gap = correct['S-RC'] - correct['S-DE']
    floor = clean['S-DE'][-1] - CLEAN_SLACK
    conditions = dict(
        em_gain=gap >= 10 * EM_GAP_PP,
        read_error_halved=read_error['S-RC'] <= ERROR_RATIO * read_error['S-DE'],
        stable_decoding=bool(finite['S-RC']) and all(v >= floor for v in clean['S-RC']))
    advances = all(conditions.values())
    interaction = ((correct['S-RC'] - correct['S-RE']) - (correct['S-DC'] - correct['S-DE'])) / 10
    alone = {a: dict(em_gain_pp=(correct[a] - correct['S-DE']) / 10, read_error_ratio=read_error[a] / read_error['S-DE'],
                     meets_1_and_2=correct[a] - correct['S-DE'] >= 10 * EM_GAP_PP and read_error[a] <= ERROR_RATIO * read_error['S-DE'])
             for a in ('S-DC', 'S-RE')}
    prediction_holds = conditions['em_gain'] and conditions['read_error_halved'] and not any(v['meets_1_and_2'] for v in alone.values())
    return dict(study='SQ2', primary='S-RC against S-DE', conditions=conditions, advances=advances, em_gap_pp=gap / 10,
                read_error_ratio=read_error['S-RC'] / read_error['S-DE'], clean_floor=floor,
                candidate='S-RC flagged writer candidate (validation was used to select D128-3)' if advances and correct['S-RC'] >= CANDIDATE else None,
                non_primary_at_080=[a for a in ARMS if correct[a] >= CANDIDATE and a != 'S-RC'], interaction_pp=interaction,
                alone=alone, prediction_holds=prediction_holds, correct=dict(correct),
                line='primary contrast advances: flagged for a separate three-seed confirmation registration' if advances
                else 'learned-writer line stays stopped')


def verify(root, release, runs):
    """Every arm record against the release; returns the facts the report states."""
    numerics = json.loads((root / 'numerics.json').read_text())
    facts = {}
    for a, (r, m) in runs.items():
        t = r['training']
        marks = release['primary_eval_steps_by_arm'][a]
        assert r['status'] == 'complete' and not r['smoke'] and r['source_dirty'] is False and r['config']['task']['E'] == 16, a
        assert t['actual_steps'] == release['steps_by_arm'][a] == 6000 and sorted(m) == [0] + marks and sorted(map(int, t['metrics'])) == marks, a
        assert r['config_hash'] == release['config_hashes'][a] and r['git_commit'] == release['science_commit'], a
        assert r['config']['sq2']['numerics'] == release['numerics']['file_blake2b_128'], a
        assert r['parent']['config_hash'] == release['parent']['config_hash'] and r['parent']['steps'] == release['parent']['steps'], a
        assert all(v['n'] == 1000 for v in m.values()) and r['step0']['n'] == 1000 and r['step0']['correct'] == m[0]['correct'], a
        assert r['store_eta'] == (numerics['eta'] if a in ('S-RE', 'S-RC') else None), a
        assert r['coefficient_weight'] == (numerics['w_c'] if a in ('S-DC', 'S-RC') else None), a
        rows = [json.loads(line) for line in (root / 'arms' / a / 'train_log.jsonl').read_text().splitlines()]
        facts[a] = dict(finite=all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in rows), rows=rows,
                        clipped=sum(x['clipped'] for x in rows) / len(rows))
    return facts


def main(folder='reports/p1l-sq2', decision_out=None):
    root = Path(folder)
    release = json.loads((root / 'release.json').read_text())
    runs = {a: load(root / 'arms' / a) for a in ARMS}
    facts = verify(root, release, runs)
    final = {a: m[6000] for a, (r, m) in runs.items()}
    correct = {a: final[a]['correct'] for a in ARMS}
    read_error = {a: final[a]['relative_read_error'] for a in ARMS}
    clean = {a: [m[s]['clean_value_correct'] for s in sorted(m) if s] for a, (r, m) in runs.items()}
    d = decide(correct, read_error, clean, {a: facts[a]['finite'] for a in ARMS})
    out = ['### Decision table (final checkpoint, 6,000 steps, 1,000 validation episodes, eval seed 4242)\n']
    rows = [[a, runs[a][0]['job_id'], 'delta' if a[2] == 'D' else 'ridge (eta 1e-4, DPFP features)', 'yes' if a[3] == 'C' else 'no',
             correct[a], f'{100 * final[a]["em"]:.1f}', f'{read_error[a]:.3f}', f'{100 * final[a]["clean_value_correct"]:.1f}',
             'yes' if correct[a] >= CANDIDATE else 'no'] for a in ARMS]
    rows.append(['D128-3 (parent, 22,370 steps)', '', 'delta', 'no', D128_3['correct'], '60.8', f'{D128_3["read_error"]:.3f}', '', 'no'])
    out.append(table(['arm', 'job', 'storage', 'coefficient loss', 'correct of 1,000', 'val EM %', 'normalized read error',
                      'clean-value head %', '>= 0.80'], rows))
    c = d['conditions']
    out.append(f'\nRegistered primary contrast S-RC against S-DE: EM gap {d["em_gap_pp"]:+.1f} pp (needs >= +10.0): **{"met" if c["em_gain"] else "not met"}**; '
               f'read error ratio {d["read_error_ratio"]:.3f} (needs <= 0.500): **{"met" if c["read_error_halved"] else "not met"}**; '
               f'stable decoding (finite training, clean-value head >= {100 * d["clean_floor"]:.1f} % at all five checkpoints; '
               f'S-RC minimum {100 * min(clean["S-RC"]):.1f} %): **{"met" if c["stable_decoding"] else "not met"}**.')
    out.append(f'\nDecision: **{"advances" if d["advances"] else "does not advance"}**; {d["line"]}.')
    out.append(f'\nInteraction (S-RC - S-RE) - (S-DC - S-DE): {d["interaction_pp"]:+.1f} pp of EM (descriptive; standard error of a difference of two EMs about {SE_PP} pp, '
               f'of this double difference about {2 * SE_PP:.1f} pp). Cells at 0.80 other than S-RC: {d["non_primary_at_080"] or "none"}.')
    out.append(f'\nRegistered prediction (S-RC gains >= 10 pp and halves the read error; neither S-RE nor S-DC alone meets both): **{"held" if d["prediction_holds"] else "did not hold"}**.')
    out.append('\n' + table(['single change vs S-DE', 'EM gain pp', 'read error ratio', 'meets conditions 1 and 2'],
        [[a, f'{v["em_gain_pp"]:+.1f}', f'{v["read_error_ratio"]:.3f}', 'yes' if v['meets_1_and_2'] else 'no'] for a, v in d['alone'].items()]))
    out.append('\n### Validation EM % at step 0 and at the five checkpoints (N/5 ... N)\n')
    out.append(table(['arm'] + [f'{s}' for s in sorted(runs['S-DE'][1])],
                     [[a] + [f'{100 * m[s]["em"]:.1f}' for s in sorted(m)] for a, (r, m) in runs.items()]))
    out.append('\n### Normalized read error at step 0 and the checkpoints\n')
    out.append(table(['arm'] + [f'{s}' for s in sorted(runs['S-DE'][1])],
                     [[a] + [f'{m[s]["relative_read_error"]:.3f}' for s in sorted(m)] for a, (r, m) in runs.items()]))
    out.append('\n### Clean-value head accuracy % (condition 3 reads the five checkpoints)\n')
    out.append(table(['arm'] + [f'{s}' for s in sorted(runs['S-DE'][1])],
                     [[a] + [f'{100 * m[s]["clean_value_correct"]:.1f}' for s in sorted(m)] for a, (r, m) in runs.items()]))
    out.append('\n### Read-value head accuracy % (actual read) at step 0 and the checkpoints\n')
    out.append(table(['arm'] + [f'{s}' for s in sorted(runs['S-DE'][1])],
                     [[a] + [f'{100 * m[s]["actual_read_correct"]:.1f}' for s in sorted(m)] for a, (r, m) in runs.items()]))
    out.append('\n### Diagnostics at the final checkpoint (one seed)\n')
    out.append(table(['metric'] + list(ARMS), [[label] + [fmt(final[a].get(key), scale, digits) for a in ARMS]
                                              for key, label, scale, digits in DIAGNOSTICS]))
    out.append('\n### Training-log diagnostics (mean of the last 10 logged steps, 5,550 to 6,000; one batch each)\n')
    keys = [('coefficient_loss', 'coefficient loss (weighted, w_c = 1)'), ('read_loss', 'raw-read loss'), ('direction_cosine', 'direction cosine d_t / ||d||'),
            ('coefficient_cancellation_median', 'coefficient cancellation (median)'), ('key_gram_log10_cond_median', 'log10 cond of the key Gram (median)'),
            ('ridge_log10_cond_median', 'log10 cond of the ridge system (median)'), ('query_norm', 'query norm'), ('grad_norm', 'gradient norm before clipping')]
    def tail(a, key):
        vals = [x[key] for x in facts[a]['rows'][-10:] if key in x]
        return f'{sum(vals) / len(vals):.3f}' if vals else ''
    out.append(table(['metric'] + list(ARMS), [[label] + [tail(a, key) for a in ARMS] for key, label in keys]))
    out.append('\nFraction of logged steps with the gradient clipped: ' + ', '.join(f'{a} {100 * facts[a]["clipped"]:.0f} %' for a in ARMS) + '.')
    n = json.loads((root / 'numerics.json').read_text())
    out.append(f'\n### Numerics (job {n["job_id"]}, {n["batches"]} train batches, {n["episodes"]} train episodes)\n')
    out.append(table(['eta', *[f'{c["eta"]:g}' for c in n['eta_curve']]], [['mean normalized read error', *[f'{c["error"]:.4f}' for c in n['eta_curve']]]]))
    out.append(f'\neta = {n["eta"]:g} (the lowest error is at 1e-6; the registered clip to [1e-4, 1] sets 1e-4). Gradient-norm ratio ||grad read|| / ||grad coef|| (delta store): '
               f'median {n["gradient_ratio"]["median"]:.3f}, min {n["gradient_ratio"]["min"]:.3f}, max {n["gradient_ratio"]["max"]:.3f}, so w_c = {n["w_c"]:g} (log2 rule, Clarification C1); '
               f'under the ridge store the median is {n["ridge_store_gradient_ratio_median"]:.3f} (reported, not used).')
    out.append(f'Deployed delta read error on these batches {n["deployed_read_error_delta_store"]:.3f}; ridge at eta 1e-4 on the same keys and queries {n["eta_curve"][2]["error"]:.3f}. '
               f'Direction cosine {n["statistics_delta_store"]["direction_cosine"]:.3f} (delta) / {n["statistics_ridge_store"]["direction_cosine"]:.3f} (ridge); coefficient cancellation '
               f'{n["statistics_delta_store"]["coefficient_cancellation_median"]:.2f} / {n["statistics_ridge_store"]["coefficient_cancellation_median"]:.2f}.')
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
    main(*(args[:1] or ['reports/p1l-sq2']), decision_out=target)
