"""Markdown tables of the RFX science run (docs/prereg/RFX.md) from the committed records. Standard library only.

usage: python3 scripts/analyze_rfx.py [reports/p1l-rfx]   (reads summary.json and LABEL.json; prints markdown to stdout)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS = ['clean', 'zero'] + [f'{s}/{q}/{g}' for s in ('delta', 'ridge') for q in ('raw', 'unit', 'true') for g in ('asis', 'fit')]
AGE_CELLS = ['delta/raw/asis', 'delta/raw/fit', 'ridge/raw/fit', 'delta/true/fit', 'ridge/true/fit']


def pct(x, digits=1):
    return f'{100 * x:.{digits}f}'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def refit_key(cell):
    """D3 is fitted for the six unscaled cells, clean and zero."""
    parts = cell.split('/')
    return cell if len(parts) == 1 else '/'.join(parts[:2]) if parts[2] == 'asis' else None


def main(folder):
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    records = {label: json.loads((folder / f'{label}.json').read_text()) for label in summary['labels']}
    out = []
    out.append('### Verdicts and baselines\n')
    rows = []
    for label, r in records.items():
        c, v = r['checkpoint'], r['verdict']
        base = v['baseline']
        refit = r['refit']['delta/raw']['acc']
        rows.append([label, c['rank'], c['steps'], f'`{v["verdict"]}`', f'{base["err_fit"]:.4f}', pct(base['lm_em']),
                     pct(base['lm_em_entity']), pct(refit)])
    out.append(table(['label', 'rank', 'steps', 'verdict', 'e_b = err delta/raw/fit', 'EM_b (%)', 'EMe_b (%)', 'D3 delta/raw (%)'], rows))
    out.append('\nTriggered LSQ arms (`lsq_arms`): ' + (', '.join(summary['lsq_arms']) or 'none') + '.')
    out.append('\n### Rescue flags\n')
    labels = list(records)
    flags = summary['rescue']
    names = CELLS + ['decoder_refit']
    out.append(table(['cell'] + labels, [[f'`{n}`'] + ['**yes**' if flags[l][n] else 'no' for l in labels] for n in names]))
    for label, r in records.items():
        v = r['verdict']
        out.append(f'\n### Cells: {label} (rank {r["rank"]}, {r["checkpoint"]["steps"]} steps)\n')
        rows = []
        for cell in CELLS:
            m = r['cells'][cell]
            key = refit_key(cell)
            d3 = r['refit'].get(key) if key else None
            rows.append([f'`{cell}`', f'{m["err"]:.4g} ({m["err_se"]:.2g})', f'{pct(m["lm_em"])} ({pct(m["lm_em_se"])})',
                         f'{pct(m["lm_em_entity"])} ({pct(m["lm_em_entity_se"])})', f'{m["lm_nll"]:.3f}', pct(m['head_acc']),
                         pct(d3['acc']) if d3 else '', f'{d3["nll"]:.3f}' if d3 else '', 'yes' if flags[label][cell] else ''])
        out.append(table(['cell', 'err (SE)', 'D1 EM % (SE)', 'D1 EMe % (SE)', 'D1 NLL', 'D2 head acc %', 'D3 acc %', 'D3 NLL', 'rescues'], rows))
        p = r['parameters']
        out.append('\nSelected eta (fitting set) and scalar gain g, by query: ' + '; '.join(
            f'{q}: eta {p["eta"][q]:g}, gain ridge {p["gain"]["ridge"][q]:.4g}, gain delta {p["gain"]["delta"][q]:.4g}' for q in ('raw', 'unit', 'true')) + '.')
        out.append('\nFitting-set mean normalized error after the gain fit, by eta:\n')
        etas = [e['eta'] for e in p['eta_curves']['raw']]
        out.append(table(['query'] + [f'{e:g}' for e in etas], [[q] + [f'{e["error"]:.4g}' for e in p['eta_curves'][q]] for q in ('raw', 'unit', 'true')]))
    out.append('\n### Write age (0 = most recent of the 16 writes): error / EM %\n')
    for label, r in records.items():
        out.append(f'\n{label}\n')
        rows = []
        for i in range(len(r['age'][AGE_CELLS[0]])):
            row = [r['age'][AGE_CELLS[0]][i]['age'], r['age'][AGE_CELLS[0]][i]['n']]
            row += [f'{r["age"][c][i]["err"]:.3g} / {pct(r["age"][c][i]["lm_em"])}' for c in AGE_CELLS]
            rows.append(row)
        out.append(table(['age', 'n'] + [f'`{c}`' for c in AGE_CELLS], rows))
    out.append('\n### Write order (four per-episode random permutations, deployed q through the LM path)\n')
    rows = []
    for label, r in records.items():
        w = r['write_order']
        rows.append([label, 'natural', f'{w["natural"]["err"]:.4g}', pct(w['natural']['lm_em']), '', ''])
        for i in range(w['permutations']):
            rows.append([label, f'perm {i}', f'{w["err"][i]:.4g}', pct(w['lm_em'][i]), pct(w['agreement_with_natural_order'][i]),
                         f'{w["read_discrepancy"][i]:.3g}'])
    out.append(table(['label', 'order', 'err', 'EM %', 'agreement with natural %', 'normalized sq. diff of read'], rows))
    out.append('\nError and EM % by permuted write age (mean over the four permutations):\n')
    ages = len(next(iter(records.values()))['write_order']['by_permuted_age'])
    rows = [[a] + [f'{r["write_order"]["by_permuted_age"][a]["err"]:.3g} / {pct(r["write_order"]["by_permuted_age"][a]["lm_em"])}'
                   for r in records.values()] for a in range(ages)]
    out.append(table(['permuted age'] + labels, rows))
    out.append('\n### Conditioning (log10 condition number over the 4,000 evaluation episodes)\n')
    rows = []
    for label, r in records.items():
        g = r['conditioning']['gram']
        rows.append([label, 'K^T K', '', f'{g["median"]:.2f}', f'{g["p90"]:.2f}', f'{g["p99"]:.2f}', f'{g["max"]:.2f}'])
        for q, c in r['conditioning']['ridge'].items():
            rows.append([label, f'K^T K + eta I ({q})', f'{c["eta"]:g}', f'{c["median"]:.2f}', f'{c["p90"]:.2f}', f'{c["p99"]:.2f}', f'{c["max"]:.2f}'])
    out.append(table(['label', 'matrix', 'eta', 'median', 'p90', 'p99', 'max'], rows))
    out.append('\n### Normalization and gain effects\n')
    rows = []
    for label, r in records.items():
        n, g = r['normalization_effect'], r['gain_effect']
        rows.append([label, f'{n["delta"]["delta_err"]:+.4f}', f'{100 * n["delta"]["delta_lm_em"]:+.2f}', f'{n["ridge"]["delta_err"]:+.4f}',
                     f'{100 * n["ridge"]["delta_lm_em"]:+.2f}', f'{g["delta_err"]:+.2e}', f'{100 * g["delta_lm_em"]:+.2f}'])
    out.append(table(['label', 'unit - raw, delta: err', 'EM pp', 'unit - raw, ridge: err', 'EM pp', 'fit - asis, delta/raw: err', 'EM pp'], rows))
    out.append('\n### Instrumentation checks and run\n')
    rows = [[label, r['checks']['identity_max_abs_logit_diff'], r['checks']['raw_recompute_max_rel'],
             r['checks']['delta_reconstruction_max_rel'], f'{r["checks"]["ridge_permutation_max_abs"]:.3g}',
             f'{r["seconds"]["fit"]:.0f} / {r["seconds"]["evaluate"]:.0f}'] for label, r in records.items()]
    out.append(table(['label', 'identity max abs logit diff', 'raw recompute max rel', 'delta rebuild max rel', 'ridge permutation max abs',
                      'fit / evaluate seconds'], rows))
    out.append(f'\nEntity-EM guard: {summary["entity_em_guard"]}; `full_hit_entity_miss` over every cell: '
               f'{max(c["full_hit_entity_miss"] for r in records.values() for c in r["cells"].values())}. '
               f'Instrumentation failures: {summary["instrumentation_failures"]}. Job {summary["job_id"]} at `{summary["git_commit"]}`, '
               f'source dirty {summary["source_dirty"]}, {summary["runtime"]["gpu"]}, torch {summary["runtime"]["torch"]}.')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/p1l-rfx')
