"""Compute-only smoke artifact gate and immutable A2 timing release; no scores exposed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from looplora.infra.seeds import stable_hash
from looplora.llm.read_fix import configuration,RUNS
from looplora.llm.read_fix_run import largest_schedule,eligible_e8,load_inference,validate
from looplora.llm.run import _write_json,source_metadata
from looplora.sym.stats import load_samples


def prepare(root,out,*,rank=128,arms=('W5','R','OA'),steps=2400,e8_decision=None):
    E=16
    if e8_decision is not None:
        E=eligible_e8(json.loads(e8_decision.read_text()),list(arms),rank)
    assert all((rank,arm) in RUNS for arm in arms)
    assert E==8 or (rank==128 and tuple(arms)==('W5','R','OA'))
    rows=[]; reference={}; sources=set(); normalization=None
    for arm in arms:
        folder=root/arm
        rec=json.loads((folder/'record.json').read_text())
        assert rec['status']=='complete' and rec['smoke'] is True and rec['arm']==arm and rec['rank']==rank
        assert rec['config']==configuration(arm,steps,E=E,smoke=True,rank=rank)
        assert rec['source_dirty'] is False and rec['config_hash']==stable_hash(rec['config'])
        sources.add(rec['git_commit'])
        training=rec['training']
        assert training['actual_steps']==steps and training['training_episodes']==steps*32
        assert training['eval_steps']==training['primary_eval_steps']==[round(steps*k/5) for k in range(1,6)]
        assert training['matched_eval_steps']==[]
        evaluation_seconds=[]
        for step,metric in training['metrics'].items():
            samples=load_samples(folder/f'val-{step}'/'samples.npz')
            assert len(samples['correct'])==100 and len(np.unique(samples['episode_id']))==100
            assert (samples['split']=='val').all() and (samples['hop']==1).all() and (samples['L']==1).all()
            assert (samples['arm']==arm).all() and samples['gram_relative_error'].max()<1e-4
            assert all(np.isfinite(v).all() for v in samples.values() if v.dtype.kind in 'bif')
            if not reference: reference.update({k:samples[k] for k in ('episode_id','query_id','answer')})
            assert all(np.array_equal(samples[k],v) for k,v in reference.items())
            assert metric['correct']==int(samples['correct'].sum())
            evaluation_seconds.append(metric['seconds'])
        exported=torch.load(folder/'writer.pt',map_location='cpu',weights_only=True)
        assert exported['signature']['r']==exported['signature']['r_mem']==rank
        assert exported['read_fix_version']==1 and exported['arm']==arm
        assert 'value_head' not in exported and 'optimizer' not in exported
        assert all(torch.isfinite(v).all() for v in exported['adapter'].values())
        if arm not in {'OA','OArho'}:
            buffers=[exported['adapter'][k] for k in ('input_mean','input_std')]
            if normalization is None: normalization=buffers
            else: assert all(torch.equal(a,b) for a,b in zip(normalization,buffers))
        logs=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines()]
        for row in logs:
            fields=('main_loss','address_loss','value_loss','read_loss','decorrelation_loss')
            assert all(np.isfinite(row[k]) and row[k]>=0 for k in fields)
            assert np.isfinite(row['grad_norm']) and np.isclose(row['loss'],sum(row[k] for k in fields),rtol=1e-6)
            if arm in {'OA','OArho'}: assert all(row[k]==0 for k in fields[1:])
            if arm=='W5': assert row['read_loss']==row['decorrelation_loss']==0
        experiment=load_inference(folder/'writer.pt')
        try:
            assert experiment.value_head is None
            reload_folder=root/f'export-check-{arm}'
            validate(experiment,reload_folder,100)
        finally: experiment.close()
        reloaded=load_samples(reload_folder/'samples.npz')
        saved=load_samples(folder/f'val-{steps}'/'samples.npz')
        assert np.array_equal(reloaded['prediction'],saved['prediction'])
        rows.append(dict(rank=rank,arm=arm,job_id=rec['job_id'],host=rec['host'],steps=steps,
            steady_step_mean_seconds=training['steady_step_mean_seconds'],
            step_p95_seconds=training['step_p95_seconds'],total_seconds=training['seconds'],
            eval100_seconds=evaluation_seconds,artifact_gate_passed=True,reload_identical=True))
    assert len(sources)==1
    science_commit=sources.pop()
    timing=dict(**source_metadata(),science_commit=science_commit,rows=rows,
        quality_scores_exposed=False,smoke_fraction_evaluation=.1,provisional_steps=24000,
        smoke_fraction_provisional_steps=steps/24000,source_and_sample_artifacts_verified=True)
    release=dict(amendment='A2',E=E,rank=rank,science_commit=science_commit,
        status='released_after_smoke_artifact_and_timing_gate',seed=17,eval_seed=4242,
        prior_report_commit='cd5a45c',max_total_node_hours=None,max_job_seconds=7200,walltime='02:00:00',
        scale_up='docs/tasks/WP-L4_p1l_read_fix.md',lr=.003,lr_search=False,
        lr_note=f'rank{rank} retains lr3e-3 by user request; no new search',
        steps_by_arm={},primary_eval_steps_by_arm={},matched_eval_steps_by_arm={},sizing_by_arm={},
        selection_order=['r32/R','r32/Rrho','r32/RK','r128/W5','r128/R'],
        e8_rule='one best learned arm across both ranks; fresh same seed/lr/objective; own smoke-sized schedule')
    for row in rows:
        arm=row['arm']; full_eval=max(row['eval100_seconds'])*10
        target=largest_schedule(row['steady_step_mean_seconds'],full_eval)
        total=1.5*(target*row['steady_step_mean_seconds']+5*full_eval)+90
        assert total<=7200<1.5*((target+5)*row['steady_step_mean_seconds']+5*full_eval)+90
        release['steps_by_arm'][arm]=target
        release['primary_eval_steps_by_arm'][arm]=[target*k//5 for k in range(1,6)]
        release['matched_eval_steps_by_arm'][arm]=[]
        release['sizing_by_arm'][arm]=dict(step_seconds=row['steady_step_mean_seconds'],
            eval1000_seconds=full_eval,evaluations=5,margin=1.5,setup_reserve_seconds=90,
            expected_seconds_with_margin=total,smoke_fraction_of_fixed_steps=steps/target)
    out.mkdir(parents=True,exist_ok=True)
    timing_path=out/('smoke-e8-timing.json' if E==8 else 'smoke-r128-timing.json')
    release_path=out/('e8-release.json' if E==8 else 'release-r128.json')
    if release_path.exists():
        assert json.loads(release_path.read_text())==release,'do not overwrite a frozen release'
    else: _write_json(release_path,release)
    _write_json(timing_path,timing)
    print(json.dumps(dict(artifact_gate_passed=True,rank=rank,E=E,
                         steps_by_arm=release['steps_by_arm'],quality_scores_exposed=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rank',type=int,choices=(32,128),default=128)
    parser.add_argument('--arms',nargs='+',default=['W5','R','OA'])
    parser.add_argument('--steps',type=int,default=2400)
    parser.add_argument('--e8-decision',type=Path)
    args=parser.parse_args()
    prepare(args.root,args.out,rank=args.rank,arms=args.arms,steps=args.steps,e8_decision=args.e8_decision)
