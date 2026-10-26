"""WP-L4 paired sample audit, immutable E8 decision and export verification."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from looplora.infra.seeds import derive_seed,stable_hash
from looplora.llm.read_fix import ARMS,LABELS,configuration
from looplora.llm.read_fix_run import choose,load_inference,validate
from looplora.llm.run import _write_json,source_metadata
from looplora.sym.stats import load_samples


def audit_run(folder,arm,release,reference,E=16):
    rec=json.loads((folder/'record.json').read_text())
    assert rec['status']=='complete' and rec['arm']==arm and rec['smoke'] is False
    assert rec['git_commit']==release['science_commit'] and rec['source_dirty'] is False
    steps=release['steps']
    assert rec['config']==configuration(arm,steps,E=E)
    assert rec['config_hash']==stable_hash(rec['config'])
    assert rec['runtime']['deterministic_algorithms'] and rec['runtime']['cublas_workspace_config']==':4096:8'
    training=rec['training']
    assert training['actual_steps']==steps and training['training_episodes']==steps*32
    assert training['training_queries']==steps*128 and training['data_seed']==derive_seed(17,'llm','writer-data')
    assert set(training['metrics'])=={str(steps*k//5) for k in range(1,6)}
    exported=torch.load(folder/'writer.pt',map_location='cpu',weights_only=True)
    assert exported['read_fix_version']==1 and exported['arm']==arm and exported['step']==steps
    assert 'value_head' not in exported and 'optimizer' not in exported
    assert exported['signature']['read_fix']==arm
    assert all(torch.isfinite(t).all() and t.dtype==torch.float32 for t in exported['adapter'].values())
    assert rec['training_only_parameters']==(0 if arm=='OArho' else 33792)
    rows=[]
    for step,metric in training['metrics'].items():
        samples=load_samples(folder/f'val-{step}'/'samples.npz')
        assert len(samples['correct'])==1000 and len(np.unique(samples['episode_id']))==1000
        assert (samples['split']=='val').all() and (samples['hop']==1).all() and (samples['L']==1).all()
        assert (samples['arm']==arm).all() and (samples['query_id']==0).all()
        assert np.array_equal(samples['correct'],samples['prediction']==samples['answer'])
        if not reference: reference.update({k:samples[k] for k in ('episode_id','query_id','answer')})
        assert all(np.array_equal(samples[k],v) for k,v in reference.items())
        assert metric['n']==1000 and metric['correct']==int(samples['correct'].sum())
        assert metric['em']==float(samples['correct'].mean()) and metric['loss']==float(samples['nll'].mean())
        assert samples['gram_relative_error'].max()<1e-4
        # Metrics other than count/time are direct means of saved scalar samples.
        for key,value in metric.items():
            if key in samples and key!='correct': assert value==float(samples[key].mean()),(arm,step,key)
        rows.append(dict(arm=arm,E=E,step=int(step),source_commit=rec['git_commit'],job_id=rec['job_id'],
                         host=rec['host'],**metric))
    logs=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines()]
    assert {r['step'] for r in logs}=={1,steps,*range(50,steps+1,50)}
    assert logs[-1]['step']==steps
    for row in logs:
        fields=('main_loss','address_loss','value_loss','read_loss','decorrelation_loss')
        assert all(np.isfinite(row[k]) and row[k]>=0 for k in fields)
        assert np.isclose(row['loss'],sum(row[k] for k in fields),rtol=1e-6)
        if arm=='OArho': assert all(row[k]==0 for k in fields[1:])
        if arm!='RK': assert row['decorrelation_loss']==0
    return rec,rows,[dict(arm=arm,E=E,**row) for row in logs]


def write_csv(path,data):
    fields=list(dict.fromkeys(k for row in data for k in row))
    with path.open('w') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n')
        writer.writeheader(); writer.writerows(data)


def run(root,out,release_path,*,e16_only=False,verify_export=False):
    out.mkdir(parents=True,exist_ok=True)
    release=json.loads(release_path.read_text())
    records,rows,logs,reference=[],[],[],{}
    for arm in ARMS:
        rec,r,l=audit_run(root/'arms'/arm,arm,release,reference)
        records.append(rec); rows.extend(r); logs.extend(l)
    context=[dict(arm=r['arm'],config_hash=r['config_hash'],git_commit=r['git_commit'],
        correct=r['training']['metrics'][str(release['steps'])]['correct']) for r in records]
    choice=choose(records)
    lock_path=out/'selection.json'
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        assert lock['context_hash']==stable_hash(context) and lock['choice']==choice
    else:
        lock=dict(created_at=datetime.now(timezone.utc).isoformat(),context_hash=stable_hash(context),
                  choice=choice,context=context)
        _write_json(lock_path,lock)
    selected=choice['arm'] if choice['action']=='select' else None
    capacity=16 if selected else None
    if choice['action']=='retry_e8' and not e16_only:
        arm=choice['arm']
        assert {p.name for p in (root/'e8').iterdir() if p.is_dir()}=={arm}
        rec,r,l=audit_run(root/'e8'/arm,arm,release,{},E=8)
        assert datetime.fromisoformat(rec['created_at'])>=datetime.fromisoformat(lock['created_at'])
        records.append(rec); rows.extend(r); logs.extend(l)
        if rec['training']['metrics'][str(release['steps'])]['correct']>=800: selected,capacity=arm,8
    elif choice['action']!='retry_e8': assert not (root/'e8').exists()
    write_csv(out/'runs.csv',rows); write_csv(out/'loss_curves.csv',logs)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,arm in zip(axes.flat,ARMS):
        for E in (16,8):
            subset=[r for r in logs if r['arm']==arm and r['E']==E]
            if not subset: continue
            for field in ('main_loss','address_loss','value_loss','read_loss','decorrelation_loss'):
                if not any(r[field] for r in subset): continue
                ax.plot([r['step'] for r in subset],[r[field] for r in subset],
                        label=field+(f' E{E}' if E==8 else ''),linestyle='--' if E==8 else '-')
        ax.set(title=LABELS[arm],xlabel='Steps',ylabel='Loss'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'loss_curves.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for arm in ARMS:
        for E in (16,8):
            subset=sorted((r for r in rows if r['arm']==arm and r['E']==E),key=lambda r:r['step'])
            if subset: ax.plot([r['step'] for r in subset],[r['em'] for r in subset],marker='o',label=f'{LABELS[arm]} E{E}')
    ax.axhline(.8,color='black',lw=1,label='writer gate'); ax.set(xlabel='Steps',ylabel='Val EM',ylim=(0,1))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'em_curves.png',dpi=150); plt.close(fig)
    checks=[]
    if verify_export and not e16_only:
        for rec in records:
            arm,E=rec['arm'],rec['config']['task']['E']
            folder=root/('e8' if E==8 else 'arms')/arm
            experiment=load_inference(folder/'writer.pt')
            try: metric=validate(experiment,out/f'export-check-{arm}-E{E}')
            finally: experiment.close()
            actual=load_samples(out/f'export-check-{arm}-E{E}'/'samples.npz')
            expected=load_samples(folder/f"val-{release['steps']}"/'samples.npz')
            assert np.array_equal(actual['prediction'],expected['prediction'])
            checks.append(dict(arm=arm,E=E,identical_predictions=True,training_head_absent=True,em=metric['em']))
    result=dict(**source_metadata(),records=records,rows=rows,selection=lock,
        selected_arm=selected,selected_E=capacity,decision_pending_e8=choice['action']=='retry_e8' and e16_only,
        stop_P1L=selected is None and not e16_only,export_checks=checks,
        all_sample_metrics_reproduced=True,paired_labels_verified=True)
    _write_json(out/('e16-summary.json' if e16_only else 'summary.json'),result)
    print(json.dumps(dict(choice=choice,selected_arm=selected,selected_E=capacity,e16_only=e16_only)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--e16-only',action='store_true')
    parser.add_argument('--verify-export',action='store_true')
    args=parser.parse_args()
    run(args.root,args.out,args.release,e16_only=args.e16_only,verify_export=args.verify_export)
