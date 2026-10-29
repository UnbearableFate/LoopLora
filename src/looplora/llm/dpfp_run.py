"""WP-L5 DPFP arms and WP-L8 DPFP-X arms, one arm per job; smoke and all runtime execute on compute."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket

import torch

from looplora.infra.seeds import stable_hash
from .dpfp import DPFPExperiment,configuration
from .dpfp_release import ARMS,XARMS,eligible_e8,prereg_digest,scheduled_steps,study_of
from .read_fix_run import fit
from .run import _write_json,source_metadata


def load_inference(path,*,model=None,tokenizer=None):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('read_fix_version')!=1 or 'value_head' in payload or 'optimizer' in payload:
        raise ValueError('expected a DPFP inference export')
    experiment=DPFPExperiment(payload['config'],payload['arm'],model=model,tokenizer=tokenizer)
    try:
        if payload['signature']!=experiment.writer_signature():
            raise ValueError('DPFP signature mismatch')
        experiment.adapter.load_state_dict(payload['adapter'],strict=True)
        experiment.value_head=None
    except BaseException:
        experiment.close()
        raise
    return experiment


def choose(records):
    """The registered rule of docs/prereg/DPFP.md; ties keep the candidate order."""
    by_arm={r['arm']:r for r in records}
    if len(records)!=3 or set(by_arm)!=set(ARMS) or any(
            r['status']!='complete' or r['smoke'] or r['config']['task']['E']!=16 for r in records):
        raise ValueError('the decision requires all three complete scientific E16 arms')
    correct={arm:by_arm[arm]['training']['metrics'][str(by_arm[arm]['training']['actual_steps'])]['correct']
             for arm in ARMS}
    passed=[arm for arm in ARMS if correct[arm]>=800]
    if passed:
        return dict(action='select',arm=passed[0],E=16,correct=correct)
    best=max(ARMS,key=lambda arm:(correct[arm],-ARMS.index(arm)))
    if correct[best]>=500:
        return dict(action='retry_e8',arm=best,E=8,correct=correct)
    return dict(action='stop',arm=None,E=None,correct=correct)


# docs/prereg/DPFP-X.md: D128-3 at E16 (docs/reports/DPFP.md): 608 of 1000 validation episodes.
D128_3_CORRECT=608


def choose_x(records):
    """The registered rule of docs/prereg/DPFP-X.md; choose() is the DPFP rule and is not used here.

    X256 reaching 0.80 at its final checkpoint is a flagged writer candidate (no E8 retrain follows). X17 is a diagnostic against
    D128-3: a val EM difference of at least 5 pp is higher or lower, a smaller one not distinguished; at 0.80 it is also a flagged
    candidate. Neither reaching 0.80 keeps the learned-writer line stopped. Read error and the read-value head are descriptive
    and are compared in the report."""
    by_arm={r['arm']:r for r in records}
    if len(records)!=2 or set(by_arm)!=set(XARMS) or any(
            r['status']!='complete' or r['smoke'] or r['config']['task']['E']!=16 for r in records):
        raise ValueError('the decision requires both complete scientific E16 DPFP-X arms')
    correct={arm:by_arm[arm]['training']['metrics'][str(by_arm[arm]['training']['actual_steps'])]['correct'] for arm in XARMS}
    candidates=[arm for arm in XARMS if correct[arm]>=800]
    gap_pp=(correct['X17']-D128_3_CORRECT)/10
    return dict(study='DPFP-X',x256='flagged_candidate' if correct['X256']>=800 else 'no_writer',
        x17_vs_d128_3='higher' if gap_pp>=5 else 'lower' if gap_pp<=-5 else 'not_distinguished',x17_gap_pp=gap_pp,
        x17_candidate=correct['X17']>=800,candidates=candidates,
        line='flagged_candidate' if candidates else 'stopped',e8_retrain=False,correct=correct)


def run_one(arm,out,steps,E=16,smoke=False,expected_hash=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    cfg=configuration(arm,steps,E=E,smoke=smoke)
    if expected_hash is not None and stable_hash(cfg)!=expected_hash:
        raise ValueError('the config differs from the frozen release')
    rank,nu=cfg['model']['r'],cfg['model']['dpfp_nu']
    record=dict(**source_metadata(),arm=arm,config=cfg,config_hash=stable_hash(cfg),status='running',
        smoke=smoke,seed=17,eval_seed=4242,stage='writer',protocol='memory_only',rank=rank,dpfp_nu=nu,
        key_dim=2*rank*nu,layer=14,L=1,host=socket.gethostname(),job_id=os.environ.get('PBS_JOBID'),
        created_at=datetime.now(timezone.utc).isoformat())
    experiment=None
    try:
        experiment=DPFPExperiment(cfg,arm)
        torch.cuda.reset_peak_memory_stats()
        record['standardization']=experiment.standardize()
        record['training']=fit(experiment,out,steps)
        record['memory']=dict(shape=[rank,2*rank*nu],state_bytes_per_episode=4*rank*2*rank*nu,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        record['inference_parameters']=sum(p.numel() for p in experiment.adapter.parameters())
        record['training_only_parameters']=sum(p.numel() for p in experiment.value_head.parameters())
        record['runtime']=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        record.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        record.update(status='failed',failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        if experiment is not None: experiment.close()
        _write_json(out/'record.json',record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS+XARMS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--steps',type=int)
    budget.add_argument('--schedule',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--e8-decision',type=Path)
    args=parser.parse_args()
    E=8 if args.e8_decision else 16
    if args.e8_decision and study_of(args.arm)!='DPFP':
        raise ValueError('DPFP-X has no E8 retrain')
    if args.e8_decision:
        eligible_e8(json.loads(args.e8_decision.read_text()),args.arm)
    schedule=None if args.schedule is None else json.loads(args.schedule.read_text())
    steps=scheduled_steps(args.arm,args.steps,schedule,smoke=args.smoke,E=E)
    if schedule is not None and schedule['science_commit']!=source_metadata()['git_commit']:
        raise ValueError('scientific code must match its frozen schedule release')
    expected=None
    if schedule is not None and study_of(args.arm)=='DPFP-X':
        if schedule['prereg_blake2b_128']!=prereg_digest():
            raise ValueError('the registration differs from the frozen release')
        expected=schedule['config_hashes'][args.arm]
    args.out.mkdir(parents=True,exist_ok=True)
    record=run_one(args.arm,args.out/args.arm,steps,E,args.smoke,expected)
    _write_json(args.out/f'shard-{args.arm}.json',dict(runs=[record],procs=1))
