"""Audit and lock WP-L3 screen decisions on compute; never select from test."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from looplora.infra.seeds import stable_hash
from looplora.sym.stats import load_samples
from looplora.llm.run import _write_json,source_metadata
from looplora.llm.writer_fix import ARMS
from looplora.llm.writer_fix_run import configuration,choose,load_inference,validate


def audit_probes(root,out,release):
    folder=root/'pre-tail'
    record=json.loads((folder/'summary.json').read_text())
    assert record['seed']==17 and record['eval_seed']==4242 and record['val_episodes']==1000
    assert record['git_commit']==release['science_commit'] and not record['source_dirty']
    assert record['frozen_features'] and not record['adapter']
    assert {(r['target'],r['rank']) for r in record['rows']}=={
        ('head',None),('head',32),('relation',None),('relation',32)}
    assert len(record['rows'])==4
    for row in record['rows']:
        with np.load(folder/f"{row['target']}-{row['rank'] or 'full'}.npz",allow_pickle=False) as sample:
            assert len(sample['answer'])==row['n']==16000
            assert int((sample['prediction']==sample['answer']).sum())==row['correct']
            assert row['accuracy']==row['correct']/row['n']
    (out/'probes.csv').write_text((folder/'probes.csv').read_text())
    _write_json(out/'probes.json',record)
    return record


def audit_run(folder,arm,release,reference,*,extension=False):
    rec=json.loads((folder/'record.json').read_text())
    assert rec['status']=='complete' and rec['arm']==arm and rec['extension']==extension
    assert rec['git_commit']==release['science_commit'] and rec['source_dirty'] is False
    assert rec['backend']=='core.scan_final' and rec['seed']==17 and rec['eval_seed']==4242
    steps=release['steps'] if extension else 4800
    expected_config=configuration(arm,steps,prefetch_workers=2 if extension else 0)
    if not extension:
        # The frozen screen predates WP-B4's explicit zero-worker default.
        expected_config['train'].pop('prefetch_workers')
    assert rec['config']==expected_config
    assert rec['config_hash']==stable_hash(rec['config'])
    assert rec['runtime']['deterministic_algorithms'] is True
    assert rec['runtime']['cublas_workspace_config']==':4096:8'
    training=rec['training']; actual=training['actual_steps']
    assert actual==steps and training['termination']=='steps'
    assert training['training_episodes']==actual*32 and training['training_queries']==actual*128
    expected_steps=([steps*k//5 for k in range(1,6)] if extension else [1200,2400,4800])
    assert set(training['metrics'])=={str(s) for s in expected_steps}
    exported=torch.load(folder/'writer.pt',map_location='cpu',weights_only=True)
    assert exported['writer_fix_version']==1 and exported['arm']==arm and exported['step']==actual
    assert 'optimizer' not in exported and 'value_head' not in exported
    assert exported['signature']['writer_fix']==arm
    assert all(torch.isfinite(t).all() and t.dtype==torch.float32 for t in exported['adapter'].values())
    assert (('A_v' in exported['adapter'])==(arm=='W4'))
    assert rec['training_only_parameters']==(33792 if arm in {'W3','W5'} else 0)
    if arm in {'W1','W5'}:
        assert rec['standardization']['episodes']==256
        assert (exported['adapter']['input_std']>=1e-4).all()
    else:
        assert rec['standardization'] is None
    rows=[]
    for step,metric in training['metrics'].items():
        sample=load_samples(folder/f'val-{step}'/'samples.npz')
        assert len(sample['correct'])==1000 and len(np.unique(sample['episode_id']))==1000
        assert (sample['split']=='val').all() and (sample['hop']==1).all() and (sample['L']==1).all()
        assert (sample['query_id']==0).all() and (sample['arm']==arm).all()
        assert (sample['condition']==('memory_only/oracle_addressing' if arm=='OA' else 'memory_only/original')).all()
        assert np.array_equal(sample['correct'],sample['prediction']==sample['answer'])
        if not reference:
            reference.update({k:sample[k] for k in ('episode_id','query_id','answer')})
        assert all(np.array_equal(reference[k],sample[k]) for k in reference)
        assert metric['n']==1000 and metric['correct']==int(sample['correct'].sum())
        for field,key in (('em','correct'),('loss','nll'),('candidate_accuracy','candidate_correct'),
                ('tail_prediction_fraction','tail_prediction'),('key_top1','key_top1'),
                ('key_margin','key_margin'),('key_pairwise_cosine','key_pairwise_cosine')):
            assert metric[field]==float(sample[key].mean()),(arm,step,field)
        rows.append(dict(arm=arm,extension=extension,step=int(step),**{k:v for k,v in metric.items() if k!='table'},
            run_seconds=training['seconds'],job_id=rec['job_id'],host=rec['host'],
            inference_parameters=rec['inference_parameters'],training_only_parameters=rec['training_only_parameters'],
            source_commit=rec['git_commit']))
    logs=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines()]
    assert logs[0]['step']==1 and logs[-1]['step']==actual
    assert {r['step'] for r in logs}=={1,actual,*range(50,actual+1,50)}
    for row in logs:
        assert all(np.isfinite(row[k]) and row[k]>=0 for k in ('main_loss','address_loss','value_loss'))
        assert np.isclose(row['loss'],row['main_loss']+row['address_loss']+row['value_loss'],rtol=1e-6)
        if arm not in {'W2','W5'}: assert row['address_loss']==0
        if arm not in {'W3','W5'}: assert row['value_loss']==0
    return rec,rows,[dict(arm=arm,extension=extension,**r) for r in logs],exported


def run(root,out,release_path,*,screen_only=False,verify_export=False,rerun_release_path=None):
    out.mkdir(parents=True,exist_ok=True)
    release=json.loads(release_path.read_text())
    probes=audit_probes(root,out,release)
    records,rows,logs,exports,reference=[],[],[],{},{}
    assert {p.name for p in (root/'screen').iterdir() if p.is_dir()}==set(ARMS)
    for arm in ARMS:
        rec,r,l,e=audit_run(root/'screen'/arm,arm,release,reference)
        records.append(rec); rows.extend(r); logs.extend(l); exports[arm]=e
    for key in ('input_mean','input_std'):
        assert torch.equal(exports['W1']['adapter'][key],exports['W5']['adapter'][key])
    final_main_losses={arm:next(r['main_loss'] for r in reversed(logs)
                               if r['arm']==arm and r['step']==4800) for arm in ARMS}
    context=[dict(arm=r['arm'],config_hash=r['config_hash'],git_commit=r['git_commit'],
                  correct=r['training']['metrics']['4800']['correct'],
                  final_main_loss=final_main_losses[r['arm']]) for r in records]
    choice=choose(records,final_main_losses)
    lock_path=out/'selection.json'
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        assert lock['context_hash']==stable_hash(context) and lock['choice']==choice
    else:
        lock=dict(created_at=datetime.now(timezone.utc).isoformat(),context_hash=stable_hash(context),
                  source_commit=release['science_commit'],choice=choice,screen=context)
        _write_json(lock_path,lock)
    selected=choice['arm'] if choice['action']=='select' else None
    if choice['action']=='rerun' and not screen_only:
        assert rerun_release_path is not None,'A1 rerun needs its separate frozen release'
        rerun_release=json.loads(rerun_release_path.read_text())
        assert rerun_release['arms']==choice['arms'] and rerun_release['amendment']=='A1'
        assert {p.name for p in (root/'rerun').iterdir() if p.is_dir()}==set(choice['arms'])
        assert json.loads((root/'rerun/selection.json').read_text())==lock
        rerun_records={}
        for arm in choice['arms']:
            rec,r,l,e=audit_run(root/'rerun'/arm,arm,rerun_release,reference,extension=True)
            assert datetime.fromisoformat(rec['created_at'])>=datetime.fromisoformat(lock['created_at'])
            records.append(rec); rows.extend(r); logs.extend(l); rerun_records[arm]=rec
        from looplora.llm.writer_fix import SIMPLICITY
        selected=next((a for a in SIMPLICITY if a in rerun_records and
            rerun_records[a]['training']['metrics'][str(rerun_release['steps'])]['correct']>=800),None)
    elif choice['action']!='rerun':
        assert not (root/'rerun').exists(),'rerun is not authorized by the completed screen'
    for name,data in (('runs.csv',rows),('loss_curves.csv',logs)):
        with (out/name).open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(data[0]),lineterminator='\n')
            writer.writeheader(); writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    for ax,arm in zip(axes.flat,ARMS):
        for extended in (False,True):
            subset=[r for r in logs if r['arm']==arm and r['extension']==extended]
            if not subset: continue
            for field,label,color in (('main_loss','answer','C0'),('address_loss','address','C1'),('value_loss','value','C2')):
                if field!='main_loss' and not any(r[field] for r in subset): continue
                ax.plot([r['step'] for r in subset],[r[field] for r in subset],
                        label=label+(' A1 rerun' if extended else ''),color=color,
                        linestyle='--' if extended else '-',lw=1)
        ax.set(title=arm,xlabel='Steps',ylabel='CE'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes.flat[-1].axis('off'); fig.tight_layout(); fig.savefig(out/'loss_curves.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for arm in ARMS:
        for rerun in (False,True):
            subset=sorted((r for r in rows if r['arm']==arm and r['extension']==rerun),key=lambda r:r['step'])
            if subset:
                ax.plot([r['step'] for r in subset],[r['em'] for r in subset],marker='o',
                        linestyle='--' if rerun else '-',label=arm+(' A1' if rerun else ' screen'))
    ax.axhline(.8,color='black',lw=1,label='writer gate')
    ax.set(xlabel='Steps',ylabel='Validation EM',ylim=(0,1))
    ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(out/'em_curves.png',dpi=160); plt.close(fig)
    check=None
    if verify_export and not screen_only:
        check=[]
        arms=choice['arms'] if choice['action']=='rerun' else [selected]
        for arm in arms:
            folder=(root/'rerun'/arm if choice['action']=='rerun' else root/'screen'/arm)
            rec=next(r for r in reversed(records) if r['arm']==arm)
            experiment=load_inference(folder/'writer.pt')
            check_out=out/f'export-check-{arm}'
            try:
                metric=validate(experiment,check_out)
            finally:
                experiment.close()
            step=rec['training']['actual_steps']
            expected=load_samples(folder/f'val-{step}'/'samples.npz')
            actual=load_samples(check_out/'samples.npz')
            assert np.array_equal(actual['prediction'],expected['prediction'])
            metric.update(arm=arm,identical_predictions=True,training_head_absent=True)
            check.append(metric)
    result=dict(**source_metadata(),all_sample_tables_reproduced=True,paired_labels_verified=True,
        normalization_buffers_identical=True,probes=probes,records=records,rows=rows,
        selection=lock,selected_arm=selected,screen_only=screen_only,
        decision_pending_rerun=choice['action']=='rerun' and screen_only,
        stop_P1L=selected is None and not screen_only,export_check=check)
    _write_json(out/('screen-summary.json' if screen_only else 'summary.json'),result)
    print(json.dumps(dict(choice=choice,selected_arm=selected,screen_only=screen_only)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--release',required=True,type=Path)
    parser.add_argument('--screen-only',action='store_true')
    parser.add_argument('--verify-export',action='store_true')
    parser.add_argument('--rerun-release',type=Path)
    args=parser.parse_args()
    run(args.root,args.out,args.release,screen_only=args.screen_only,verify_export=args.verify_export,
        rerun_release_path=args.rerun_release)
