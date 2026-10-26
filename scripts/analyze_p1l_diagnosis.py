"""Audit WP-L2 artifacts and write report tables/curves (compute only)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from looplora.infra.seeds import stable_hash
from looplora.llm.diagnosis_probes import SITES,POSITIONS
from looplora.sym.stats import load_samples


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def audit(root: Path,out: Path,jobs: Path | None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    out.mkdir(parents=True,exist_ok=True)
    d1 = json.loads((root/'D1/summary.json').read_text())
    assert d1['seed']==17 and d1['eval_seed']==4242 and d1['val_episodes']==1000
    assert d1['frozen_features'] and not d1['adapter'] and not d1['source_dirty']
    required = {(site,pos,target,rank) for site in SITES for pos in POSITIONS
                for target in (('tail','head','relation') if pos in POSITIONS[:2] else ('start','relation'))
                for rank in (None,32)}
    assert {(r['site'],r['position'],r['target'],r['rank']) for r in d1['rows']}==required
    assert len(d1['rows'])==len(required)==96
    for row in d1['rows']:
        name = f"{row['site']}-{row['position']}-{row['target']}-{row['rank'] or 'full'}.npz"
        with np.load(root/'D1'/name,allow_pickle=False) as sample:
            y,p = sample['answer'],sample['prediction']
            assert len(y)==row['n'] and int((p==y).sum())==row['correct']
            assert row['accuracy']==row['correct']/row['n']
            assert np.all((y>=0)&(y<row['classes']))
    (out/'probes.csv').write_bytes((root/'D1/probes.csv').read_bytes())
    write_json(out/'D1.json',d1)
    rows,loss_rows,run_records = [],[],[]
    reference_labels = None
    for kind,lrs in (('D2',(.0003,.001,.003)),('D3',(.0003,.001)),('D4',(.0003,.001))):
        folder = root/kind
        if kind=='D4' and not folder.exists():
            continue
        assert (folder/'summary.json').is_file(),f'incomplete {kind}'
        for lr in lrs:
            run = folder/f'lr-{lr:g}'
            rec = json.loads((run/'record.json').read_text())
            cfg = rec['config']
            assert rec['status']=='complete' and rec['source_dirty'] is False
            assert rec['diagnostic']==kind and cfg['seed']==17 and cfg['stage']=='writer'
            expected_site = 'residual' if kind=='D4' else 'o_proj'
            assert rec['input_site']==expected_site and cfg['model'].get('input_site','o_proj')==expected_site
            assert cfg['variant']=='M1' and cfg['protocol']=='memory_only' and cfg['write_policy']=='fact_final'
            assert cfg['model']['r']==32 and cfg['model']['layer']==14
            assert cfg['task']['E']==16 and cfg['task']['n_rel']==4 and cfg['task']['head_sharing']==.3
            assert cfg['train']['batch_size']==32 and cfg['train']['lr']==lr
            assert cfg['eval']['splits']==['val'] and cfg['eval']['seed']==4242
            assert cfg['eval']['episodes']==1000 and cfg['eval']['depths']==[1]
            assert cfg['eval']['counterfactuals'] is False and cfg['eval']['allow_confirmation'] is False
            assert rec['config_hash']==stable_hash(cfg)
            assert rec['runtime']['deterministic_algorithms'] is True
            assert rec['runtime']['cublas_workspace_config']==':4096:8'
            training = rec['training']
            if kind=='D2':
                assert training['actual_steps']==1500
            elif not training['time_guard_reached']:
                assert training['actual_steps']==6400
                assert set(training['evaluations'])=={'1500','3200','6400'}
            run_records.append(rec)
            evaluations = dict(training['evaluations'])
            if kind=='D2':
                evaluations['B0']=rec['B0']
            for step,metric in evaluations.items():
                sample = load_samples(run/('B0' if step=='B0' else f'val-{step}')/'samples.npz')
                assert len(sample['correct'])==1000 and len(np.unique(sample['episode_id']))==1000
                assert (sample['split']=='val').all() and (sample['hop']==1).all() and (sample['L']==1).all()
                assert (sample['condition']=='memory_only/original').all()
                assert np.array_equal(sample['correct'],sample['prediction']==sample['answer'])
                assert metric['correct']==int(sample['correct'].sum())
                assert metric['em']==float(sample['correct'].mean())
                assert metric['loss']==float(sample['nll'].mean())
                assert metric['candidate_accuracy']==float(sample['candidate_correct'].mean())
                labels = {key:sample[key] for key in ('episode_id','query_id','answer')}
                if reference_labels is None:
                    reference_labels = labels
                assert all(np.array_equal(labels[key],reference_labels[key]) for key in labels)
                if kind!='D2':
                    assert metric['matching']['chance']==1/16
                    for key in ('key_top1','key_target_cosine','key_margin'):
                        assert metric['matching'][key]==float(sample[key].mean())
                rows.append(dict(diagnostic=kind,input_site=rec['input_site'],lr=lr,step=step,
                    n=metric['n'],correct=metric['correct'],em=metric['em'],loss=metric['loss'],
                    candidate_accuracy=metric['candidate_accuracy'],validation_seconds=metric['seconds'],
                    key_top1=metric.get('matching',{}).get('key_top1'),
                    key_target_cosine=metric.get('matching',{}).get('key_target_cosine'),
                    key_margin=metric.get('matching',{}).get('key_margin'),
                    run_seconds=training['seconds'],job_id=rec['job_id'],host=rec['host'],
                    git_commit=rec['git_commit']))
            for line in (run/'train_log.jsonl').read_text().splitlines():
                logged = json.loads(line)
                loss_rows.append(dict(diagnostic=kind,input_site=rec['input_site'],lr=lr,
                                      step=logged['step'],loss=logged['loss']))
    for name,data in (('runs.csv',rows),('loss_curves.csv',loss_rows)):
        with (out/name).open('w') as file:
            writer = csv.DictWriter(file,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    fig,axes = plt.subplots(1,3,figsize=(13,3.7),sharey=True)
    for ax,kind in zip(axes,('D2','D3','D4')):
        for lr in (.0003,.001,.003):
            line = [r for r in loss_rows if r['diagnostic']==kind and r['lr']==lr]
            if line:
                ax.plot([r['step'] for r in line],[r['loss'] for r in line],label=f'lr={lr:g}',lw=1)
        ax.set(title=kind,xlabel='Training steps')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend()
    axes[0].set_ylabel('Training full-vocabulary CE')
    fig.tight_layout()
    fig.savefig(out/'loss_curves.png',dpi=180)
    plt.close(fig)
    result = dict(all_sample_tables_reproduced=True,D1_probes=96,D1=d1,rows=rows,
        D2_best_em=max(r['em'] for r in rows if r['diagnostic']=='D2' and r['step']!='B0'),
        D3_best_em=max(r['em'] for r in rows if r['diagnostic']=='D3'),
        D4_ran=any(r['diagnostic']=='D4' for r in rows),run_records=run_records)
    result['D4_oracle_budget_trigger'] = result['D2_best_em']>=.95 and result['D3_best_em']<.95
    if result['D4_oracle_budget_trigger']:
        assert result['D4_ran'],'D2 passed while D3 failed: required D4 is missing'
    if result['D4_ran']:
        result['D4_best_em']=max(r['em'] for r in rows if r['diagnostic']=='D4')
    if jobs is not None:
        job_rows = json.loads(jobs.read_text())
        assert len({r['job_id'] for r in job_rows})==len(job_rows)
        assert all(r['nodes']==1 and r['elapsed_seconds']<=3600 for r in job_rows)
        total = sum(r['elapsed_seconds']/3600 for r in job_rows)
        assert total<=4
        result.update(total_node_hours=total,jobs=job_rows)
        with (out/'job_costs.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(job_rows[0]))
            writer.writeheader()
            writer.writerows(job_rows)
    write_json(out/'summary.json',result)
    print(json.dumps({key:result[key] for key in ('all_sample_tables_reproduced','D1_probes','D2_best_em','D3_best_em','D4_ran')}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--jobs',type=Path)
    args = parser.parse_args()
    audit(args.root,args.out,args.jobs)
