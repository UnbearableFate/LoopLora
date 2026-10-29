"""Markdown tables of the LSQT-U control (docs/prereg/LSQT.md, decision 2) and the split of the ridge gain over R-r128. Standard library only.

usage: python3 scripts/analyze_lsqt_u.py [reports/p1l-lsqt-u] [reports/p1l-lsqt] [reports/p1l-dpfp] [--split-out FILE]
Reads release.json, arms/LT-U/ (record.json, val-STEP/metrics.json, train_log.jsonl) and audit.json (when present) of the first folder, the three
registered arms of the second (their decision.json is not changed) and R-r128 (refs/R-r128, WP-L4, not rerun) and D128-3 of the third. It first verifies the
U record against the U release (complete, released steps, all five marks, config hash, science commit, every number finite), then prints the contrasts.

The contrasts are descriptive differences of validation EM at the final checkpoints (one seed; each arm is judged at its own final checkpoint, as registered).
The standard error of a difference is the unpaired binomial one, sqrt(p1 (1 - p1) / n1 + p2 (1 - p2) / n2); the arms were scored on the same 1,000 validation
episodes, which the unpaired figure does not use. Nothing here is a registered test and U enters neither outcome of the registration.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from analyze_dpfp import DIAGNOSTICS, fmt, load, table
from analyze_lsqt import ARMS, D128_3, R_R128, verify

CONTROL = 'LT-U'
N = 1000


def strict(text):
    """json.loads that refuses the NaN, Infinity and -Infinity tokens Python would otherwise accept."""
    def reject(token):
        raise ValueError(f'non-finite token {token}')
    return json.loads(text, parse_constant=reject)


def non_finite(obj, where='', found=None):
    """Paths of every number that is not finite (also an overflow such as 1e999, which parses to inf)."""
    found = [] if found is None else found
    if isinstance(obj, dict):
        for key, value in obj.items():
            non_finite(value, f'{where}.{key}', found)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            non_finite(value, f'{where}[{index}]', found)
    elif isinstance(obj, float) and not math.isfinite(obj):
        found.append(where)
    return found


def check_all_finite(root):
    """Every json file and every train-log row under the U arm folder parses strictly and holds finite numbers only; returns how many numbers were scanned."""
    scanned = 0

    def count(obj):
        return sum(count(v) for v in obj.values()) if isinstance(obj, dict) else sum(count(v) for v in obj) if isinstance(obj, list) else \
            int(isinstance(obj, (int, float)) and not isinstance(obj, bool))
    for path in sorted(Path(root).rglob('*.json')):
        obj = strict(path.read_text())
        assert not non_finite(obj, str(path)), path
        scanned += count(obj)
    for line in (Path(root) / 'train_log.jsonl').read_text().splitlines():
        row = strict(line)
        assert not non_finite(row, 'train_log'), row['step']
        scanned += count(row)
    return scanned


def se_diff(a, b, n=N):
    """Unpaired binomial standard error of the difference of two proportions (correct counts out of n episodes each), in pp."""
    p, q = a / n, b / n
    return 100 * math.sqrt(p * (1 - p) / n + q * (1 - q) / n)


def se_one(a, n=N):
    return 100 * math.sqrt(a / n * (1 - a / n) / n)


def split(correct, n=N):
    """The registered ridge gain over R-r128 (best ridge arm minus R-r128) as the unit-query part (U minus R-r128) and the store part (arm minus U).
    Integer counts: the parts add up to the total exactly. Differences in pp of val EM with their unpaired standard errors."""
    def part(label, a, b, name_a, name_b):
        return dict(part=label, contrast=f'{name_a} - {name_b}', pp=(a - b) / n * 100, se_pp=se_diff(a, b, n))
    u, r = correct[CONTROL], R_R128
    rows = {'unit_query': part('unit query', u, r, CONTROL, 'R-r128')}
    for arm in ARMS:
        rows[f'store_{arm}'] = part(f'store ({arm})', correct[arm], u, arm, CONTROL)
        rows[f'total_{arm}'] = part(f'total ({arm})', correct[arm], r, arm, 'R-r128')
    for arm in ARMS:   # the two parts are the total, by construction
        assert abs(rows['unit_query']['pp'] + rows[f'store_{arm}']['pp'] - rows[f'total_{arm}']['pp']) < 1e-9
    return rows


def tail_mean(rows, key):
    values = [x[key] for x in rows[-10:] if key in x]
    return f'{sum(values) / len(values):.3f}' if values else ''


def main(folder='reports/p1l-lsqt-u', lsqt='reports/p1l-lsqt', dpfp='reports/p1l-dpfp', split_out=None):
    root, base = Path(folder), Path(lsqt)
    release = strict((root / 'release.json').read_text())
    assert release['study'] == 'LSQT-U' and list(release['steps_by_arm']) == [CONTROL]
    scanned = check_all_finite(root / 'arms' / CONTROL)
    run = load(root / 'arms' / CONTROL)
    facts = verify(root, release, {CONTROL: run})[CONTROL]
    assert facts['finite']
    record, metrics = run
    steps_u = sorted(metrics)
    final_u = metrics[steps_u[-1]]
    arms = {a: load(base / 'arms' / a) for a in ARMS}
    refs = {'R-r128 (WP-L4, not rerun)': load(Path(dpfp) / 'refs' / 'R-r128'), 'D128-3 (DPFP)': load(Path(dpfp) / 'arms' / 'D128-3')}
    final = {a: m[max(m)] for a, (r, m) in arms.items()}
    final[CONTROL] = final_u
    correct = {a: final[a]['correct'] for a in final}
    assert all(final[a]['n'] == N for a in final) and refs['R-r128 (WP-L4, not rerun)'][1][max(refs['R-r128 (WP-L4, not rerun)'][1])]['correct'] == R_R128
    assert refs['D128-3 (DPFP)'][1][max(refs['D128-3 (DPFP)'][1])]['correct'] == D128_3
    decision = strict((base / 'decision.json').read_text())    # the registered LSQT decision, which U does not change
    assert decision['correct'] == {a: correct[a] for a in ARMS} and decision['outcome'] == 'no_candidate'
    parts = split(correct)
    out = ['### Record check (U control; run against `release.json` and the committed copies)\n',
           f'- {record["arm"]}, job {record["job_id"]}, status {record["status"]}, smoke {record["smoke"]}, source dirty {record["source_dirty"]}, host {record["host"]}.',
           f'- steps {record["training"]["actual_steps"]} (released {release["steps_by_arm"][CONTROL]}); evaluations at {", ".join(map(str, steps_u))} (released {", ".join(map(str, release["primary_eval_steps_by_arm"][CONTROL]))}); 1,000 validation episodes each.',
           f'- config hash {record["config_hash"]} (released {release["config_hashes"][CONTROL]}); commit {record["git_commit"]} (released science commit {release["science_commit"]}).',
           f'- {scanned} numbers in the record, the training summary, the five metric files and {len(facts["rows"])} train-log rows: all finite, no NaN or infinity token; every logged loss and gradient norm finite.']
    out.append('\n### U control result (final checkpoint, 1,000 validation episodes, eval seed 4242; the standard error of one EM is the binomial one)\n')
    rows = []
    for a in (CONTROL, *ARMS):
        r, m = run if a == CONTROL else arms[a]
        f = final[a]
        rows.append([a, r['job_id'], r['store'] + ('' if r['store_eta'] is None else f' (eta {r["store_eta"]:g})'), max(m), correct[a], f'{100 * f["em"]:.1f}', f'{se_one(correct[a]):.1f}',
                     f'{f["relative_read_error"]:.3f}', f'{100 * f["clean_value_correct"]:.1f}', f'{100 * f["actual_read_correct"]:.1f}', 'yes' if correct[a] >= 800 else 'no'])
    for name, (r, m) in refs.items():
        f = m[max(m)]
        rows.append([name, '', 'delta', max(m), f['correct'], f'{100 * f["em"]:.1f}', f'{se_one(f["correct"]):.1f}', f'{f["relative_read_error"]:.3f}',
                     f'{100 * f["clean_value_correct"]:.1f}', f'{100 * f["actual_read_correct"]:.1f}', 'no'])
    out.append(table(['arm', 'job', 'store', 'steps', 'correct of 1,000', 'val EM %', 'standard error pp', 'normalized read error', 'clean-value head %', 'read-value head %', '>= 0.80'], rows))
    curve = ', '.join('{:.1f}'.format(100 * metrics[s]['em']) for s in steps_u)
    reaches = 'yes' if correct[CONTROL] >= 800 else 'no'
    highest = 'also the highest' if final_u['correct'] == max(m['correct'] for m in metrics.values()) else 'not the highest'
    out.append(f'\nU reaches 0.80: **{reaches}** ({correct[CONTROL]} of 1,000). Judged at its own final checkpoint (step {steps_u[-1]}), as every arm; '
               f'the five U evaluations are {curve} %, so the final checkpoint is {highest} and no selection among the checkpoints is involved.')
    out.append(f'\n### Split of the ridge gain over R-r128 (descriptive; one seed; not a registered test; pp of val EM at the final checkpoints; unpaired binomial standard error of the difference)\n')
    rows = []
    for arm in ARMS:
        total, query, store = parts[f'total_{arm}'], parts['unit_query'], parts[f'store_{arm}']
        rows.append([arm, f'{total["pp"]:+.1f} ({total["se_pp"]:.1f})', f'{query["pp"]:+.1f} ({query["se_pp"]:.1f})', f'{store["pp"]:+.1f} ({store["se_pp"]:.1f})'])
    out.append(table(['arm', 'total: arm - R-r128 (SE)', 'unit-query part: U - R-r128 (SE)', 'store part: arm - U (SE)'], rows))
    out.append(f'\nThe parts add up to the total exactly (same counts). The two parts of one arm share the sampling error of U, so their errors are not independent. The standard error of a difference of two EMs at 1,000 episodes is about 2.2 pp (registered convention).')
    out.append('\n### Validation EM % at the five evenly spaced checkpoints (20, 40, 60, 80 and 100 % of each arm\'s steps)\n')
    rows = [[a] + [f'{s}: {100 * m[s]["em"]:.1f}' for s in sorted(m)] for a, (r, m) in [(CONTROL, run)] + list(arms.items())]
    rows += [[name] + [f'{s}: {100 * m[s]["em"]:.1f}' for s in sorted(m)] for name, (r, m) in refs.items()]
    out.append(table(['arm', '20 %', '40 %', '60 %', '80 %', '100 % (final)'], rows))
    r128 = refs['R-r128 (WP-L4, not rerun)'][1]
    out.append('\nU minus R-r128 at the matched fractions (pp): ' + ', '.join(f'{100 * (metrics[s]["em"] - r128[t]["em"]):+.1f}' for s, t in zip(steps_u, sorted(r128))) + '.')
    out.append('\n### Normalized read error at the five checkpoints\n')
    out.append(table(['arm', '20 %', '40 %', '60 %', '80 %', '100 % (final)'],
                     [[a] + [f'{m[s]["relative_read_error"]:.3f}' for s in sorted(m)] for a, (r, m) in [(CONTROL, run)] + list(arms.items()) + list(refs.items())]))
    out.append('\n### Read-value head accuracy % (actual read) / clean-value head accuracy % at the five checkpoints\n')
    out.append(table(['arm', '20 %', '40 %', '60 %', '80 %', '100 % (final)'],
                     [[a] + [f'{100 * m[s]["actual_read_correct"]:.1f} / {100 * m[s]["clean_value_correct"]:.1f}' for s in sorted(m)] for a, (r, m) in [(CONTROL, run)] + list(arms.items())]))
    out.append('\n### Diagnostics at the final checkpoint (one seed)\n')
    cols = [CONTROL, *ARMS]
    out.append(table(['metric'] + cols + ['R-r128', 'D128-3'], [[label] + [fmt(final[a].get(key), scale, digits) for a in cols] +
                      [fmt(r128[max(r128)].get(key), scale, digits)] + [fmt(refs['D128-3 (DPFP)'][1][max(refs['D128-3 (DPFP)'][1])].get(key), scale, digits)] for key, label, scale, digits in DIAGNOSTICS]))
    extra = [('key_gram_log10_cond', 'log10 cond of the key Gram', '{:.2f}'), ('ridge_log10_cond', 'log10 cond of the ridge system', '{:.2f}'),
             ('gram_relative_error', 'Gram reconstruction relative error', '{:.1e}'), ('gram_absolute_error', 'Gram reconstruction absolute error', '{:.1e}')]
    out.append('\n' + table(['metric'] + cols, [[label] + ['' if final[a].get(key) is None else form.format(final[a][key]) for a in cols] for key, label, form in extra]))
    out.append('\n### Training-log diagnostics (mean of the last 10 logged steps; one batch each)\n')
    logs = {CONTROL: facts['rows'], **{a: [json.loads(x) for x in (base / 'arms' / a / 'train_log.jsonl').read_text().splitlines()] for a in ARMS}}
    keys = [('read_loss', 'raw-read loss'), ('main_loss', 'main loss'), ('key_gram_log10_cond_median', 'log10 cond of the key Gram (median)'),
            ('ridge_log10_cond_median', 'log10 cond of the ridge system (median)'), ('grad_norm', 'gradient norm before clipping')]
    out.append(table(['metric'] + cols, [[label] + [tail_mean(logs[a], key) for a in cols] for key, label in keys]))
    out.append('\nFraction of logged steps with the gradient clipped: ' + ', '.join(f'{a} {100 * sum(x["clipped"] for x in logs[a]) / len(logs[a]):.0f} %' for a in cols) + '.')
    out.append('\n### Job, time and memory\n')
    t, mem = record['training'], record['memory']
    out.append(table(['arm', 'job', 'source', 'steps', 'training min', 'steady step s', 'slowest 1,000-episode eval s', 'peak CUDA GiB', 'state bytes per episode', 'inference parameters'],
                     [[CONTROL, record['job_id'], record['git_commit'][:8], t['actual_steps'], f'{t["seconds"] / 60:.1f}', f'{t["steady_step_mean_seconds"]:.3f}',
                       f'{max(v["seconds"] for v in t["metrics"].values()):.1f}', f'{mem["peak_cuda_allocated_bytes"] / 2**30:.2f}', mem['state_bytes_per_episode'], record['inference_parameters']]]))
    audit_path = root / 'audit.json'
    if audit_path.exists():
        run_audit = strict(audit_path.read_text())['runs'][f'{CONTROL}-E16']
        out.append('\n### Memory interventions at the final checkpoint (audit)\n')
        out.append(table(['arm', 'original correct / NLL', 'zero memory correct / NLL', 'shuffled memory correct / NLL', 'original equals the record', 'max repeated-logit drift'],
                         [[CONTROL] + [f'{run_audit[c]["correct"]} / {run_audit[c]["nll"]:.3f}' for c in ('original', 'zero', 'shuffle')] +
                          ['yes' if run_audit['original_matches_record']['correct'] and run_audit['original']['correct'] == final_u['correct'] else 'no', run_audit['max_repeated_logit_drift']]]))
    print('\n'.join(out))
    result = dict(study='LSQT-U', arm=CONTROL, steps=steps_u[-1], correct={a: correct[a] for a in (CONTROL, *ARMS)}, r_r128_correct=R_R128, n=N,
                  u_reaches_0_80=correct[CONTROL] >= 800, enters_registered_outcomes=False, best_rg=decision['best_rg'],
                  split={k: dict(contrast=v['contrast'], pp=round(v['pp'], 6), se_pp=round(v['se_pp'], 6)) for k, v in parts.items()})
    if split_out:
        with open(split_out, 'x') as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    args = sys.argv[1:]
    target = None
    if '--split-out' in args:
        i = args.index('--split-out')
        target = args[i + 1]
        del args[i:i + 2]
    main(*(args[:3] or []), split_out=target)
