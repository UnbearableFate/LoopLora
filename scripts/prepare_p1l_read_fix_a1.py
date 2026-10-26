"""Verify the no-training audit and freeze WP-L4 A1 per-arm schedules on compute."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from looplora.llm.read_fix import ARMS
from looplora.llm.read_fix_run import largest_schedule
from looplora.llm.run import _write_json,source_metadata
from looplora.sym.stats import load_samples


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--l3-root',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rows=[]; ages=[]
    audit=json.loads((args.root/'audit/summary.json').read_text())
    for arm in ('OA','W5'):
        folder=args.root/'audit'/arm
        result=audit['runs'][arm]
        samples=load_samples(folder/'samples.npz')
        expected=load_samples(args.l3_root/'rerun'/arm/'val-32900/samples.npz')
        assert all(np.array_equal(samples[k],expected[k]) for k in ('episode_id','query_id','answer','prediction','correct'))
        assert result['optimizer_steps']==result['gradients']['optimizer_steps']==0
        assert result['max_repeated_logit_drift']==0 and result['max_gram_relative_error']<1e-4
        assert result['means']['original_correct']==float(expected['correct'].mean())
        assert result['means']['original_nll']==float(expected['nll'].mean())
        with np.load(folder/'coefficients.npz',allow_pickle=False) as dense:
            reconstructed=np.einsum('bqe,ber->bqr',dense['d'],dense['values'].astype(np.float64))
            np.testing.assert_allclose(reconstructed,dense['reconstructed'],rtol=1e-10,atol=1e-10)
            raw=np.einsum('bij,bqj->bqi',dense['matrix'].astype(np.float64),dense['q'].astype(np.float64))
            relative=np.linalg.norm(reconstructed-raw,axis=-1)/np.maximum(np.linalg.norm(raw,axis=-1),1e-8)
            assert relative.max()<1e-4
            np.testing.assert_allclose(np.einsum('ber,bqr->bqe',dense['keys'].astype(np.float64),
                                                dense['q'].astype(np.float64)),dense['b'],rtol=1e-10,atol=1e-10)
        rows.append(dict(arm=arm,**result['means'],max_gram_relative_error=result['max_gram_relative_error'],
            repeated_logit_drift=result['max_repeated_logit_drift'],would_clip=result['gradients']['would_clip_fraction']))
        ages.extend(dict(arm=arm,**row) for row in result['age_strata'])
    args.out.mkdir(parents=True,exist_ok=True)
    for name,data in (('audit.csv',rows),('audit-age.csv',ages)):
        fields=list(dict.fromkeys(k for row in data for k in row))
        with (args.out/name).open('w') as file:
            writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n')
            writer.writeheader(); writer.writerows(data)
    _write_json(args.out/'audit-summary.json',dict(**audit,baseline_predictions_identical=True,
        coefficient_vectors_independently_reconstructed=True,output_audit_source=source_metadata()))
    timing=json.loads((args.out/'smoke-timing.json').read_text())
    release=json.loads((args.out/'release.json').read_text())
    _write_json(args.out/'release-before-A1.json',release)
    release.pop('steps',None); release.pop('eval_steps',None); release.pop('timing_gate',None)
    release.update(status='released_after_final_negative_L3_and_stage0_audit',amendment='A1',E=16,
        science_commit=source_metadata()['git_commit'],master_merged='ad966aa',
        smoke_source_commit=timing['science_commit'],numeric_change_from_smoke='none; scheduling/timeguard and declared OA matched evaluation only',
        prior_report_commit='cd5a45c',max_total_node_hours=8,max_job_seconds=7200,walltime='02:00:00',
        scale_up='docs/tasks/WP-L4_p1l_read_fix.md',steps_by_arm={},primary_eval_steps_by_arm={},
        matched_eval_steps_by_arm={},sizing_by_arm={},
        e8_rule='one best eligible arm only; own smoke-sized A1 schedule, same seed/lr/objective; threshold unchanged')
    for row in timing['rows']:
        arm=row['arm']; extra=[6580] if arm=='OArho' else []
        eval_seconds=max(row['eval100_seconds'])*10
        count=5+len(extra)
        steps=largest_schedule(row['steady_step_mean_seconds'],eval_seconds,evaluations=count)
        total=1.5*(steps*row['steady_step_mean_seconds']+count*eval_seconds)+90
        assert total<=7200 and 1.5*((steps+5)*row['steady_step_mean_seconds']+count*eval_seconds)+90>7200
        release['steps_by_arm'][arm]=steps
        release['primary_eval_steps_by_arm'][arm]=[steps*k//5 for k in range(1,6)]
        release['matched_eval_steps_by_arm'][arm]=extra
        release['sizing_by_arm'][arm]=dict(step_seconds=row['steady_step_mean_seconds'],
            eval1000_seconds=eval_seconds,evaluations=count,margin=1.5,setup_reserve_seconds=90,
            expected_seconds_with_margin=total)
    _write_json(args.out/'release.json',release)
    print(json.dumps(dict(stage0_audit_verified=True,steps_by_arm=release['steps_by_arm'],
        matched_OArho_step=6580,max_total_node_hours=8)),flush=True)
