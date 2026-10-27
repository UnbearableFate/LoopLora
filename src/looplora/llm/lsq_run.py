"""WP-L7 LSQ arms, one arm per job; smoke and all runtime execute on compute."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket

import torch

from looplora.infra.seeds import stable_hash
from .lsq import ARMS,SPECS,LSQExperiment,configuration,lsq_eta
from .read_fix_run import fit
from .run import _write_json,source_metadata


def load_inference(path,*,model=None,tokenizer=None):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('read_fix_version')!=1 or 'value_head' in payload or 'optimizer' in payload:
        raise ValueError('expected an LSQ inference export')
    experiment=LSQExperiment(payload['config'],payload['arm'],model=model,tokenizer=tokenizer)
    try:
        if payload['signature']!=experiment.writer_signature():
            raise ValueError('LSQ signature mismatch')
        experiment.adapter.load_state_dict(payload['adapter'],strict=True)
        experiment.value_head=None
    except BaseException:
        experiment.close()
        raise
    return experiment


def release_arms(schedule):
    """The candidate arms a release covers: every arm of each rank that RFX triggered, in candidate order."""
    ranks=schedule['ranks']
    if not ranks or any(isinstance(r,bool) or r not in (32,128) for r in ranks) or len(set(ranks))!=len(ranks):
        raise ValueError('a release names its triggered ranks (32 and/or 128)')
    arms=[arm for arm in ARMS if SPECS[arm][0] in ranks]
    if list(schedule['arms'])!=arms:
        raise ValueError('the release arms must be every arm of the triggered ranks, in candidate order')
    return arms


def release_eta(schedule,arm):
    """The registered ridge strength of a ridge arm: lsq_eta of RFX's fitting-set selection for its rank."""
    rank,store=SPECS[arm]
    if store!='ridge':
        return None
    eta=lsq_eta(schedule['rfx_eta_by_rank'][str(rank)])
    if schedule['eta_by_rank'][str(rank)]!=eta:
        raise ValueError('the release eta differs from the clipped RFX selection')
    return eta


def scheduled_steps(arm,steps,schedule,*,smoke=False,E=16):
    """Smoke: the explicit step count. Science: the frozen release's count for this arm."""
    if smoke:
        if schedule is not None or steps is None or steps<5:
            raise ValueError('smoke requires its explicit step count')
        return steps
    if steps is not None or schedule is None:
        raise ValueError('scientific arms require an immutable schedule release')
    if schedule['study']!='LSQ' or schedule['E']!=E:
        raise ValueError('schedule study/capacity mismatch')
    arms=release_arms(schedule)
    by_arm=schedule['steps_by_arm']
    if arm not in by_arm or (E==16 and set(by_arm)!=set(arms)):
        raise ValueError('the E16 release fixes the step counts of every triggered arm at once (E8: the retrained arm)')
    result=by_arm[arm]
    if isinstance(result,bool) or not isinstance(result,int) or result<5 or result%5:
        raise ValueError('scientific schedules require five equal integer intervals')
    return result


def choose(records,arms=ARMS):
    """The registered rule of docs/prereg/LSQ.md over the triggered arms; ties keep the candidate order."""
    arms=tuple(arms)
    if not arms or arms!=tuple(a for a in ARMS if a in arms):
        raise ValueError('triggered arms must be a nonempty subset in candidate order')
    by_arm={r['arm']:r for r in records}
    if len(records)!=len(arms) or set(by_arm)!=set(arms) or any(
            r['status']!='complete' or r['smoke'] or r['config']['task']['E']!=16 for r in records):
        raise ValueError('the decision requires every triggered scientific E16 arm, complete')
    correct={arm:by_arm[arm]['training']['metrics'][str(by_arm[arm]['training']['actual_steps'])]['correct']
             for arm in arms}
    passed=[arm for arm in arms if correct[arm]>=800]
    if passed:
        return dict(action='select',arm=passed[0],E=16,correct=correct)
    best=max(arms,key=lambda arm:(correct[arm],-arms.index(arm)))
    if correct[best]>=500:
        return dict(action='retry_e8',arm=best,E=8,correct=correct)
    return dict(action='stop',arm=None,E=None,correct=correct)


def eligible_e8(lock,arm):
    choice=lock['choice']
    if choice['action']!='retry_e8' or choice['arm']!=arm or choice['arm'] not in ARMS:
        raise ValueError('E8 requires the one registered best arm')
    return 8


def run_one(arm,out,steps,E=16,smoke=False,eta=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    cfg=configuration(arm,steps,E=E,smoke=smoke,eta=eta)
    rank,store=SPECS[arm]
    record=dict(**source_metadata(),arm=arm,config=cfg,config_hash=stable_hash(cfg),status='running',
        smoke=smoke,seed=17,eval_seed=4242,stage='writer',protocol='memory_only',rank=rank,store=store,
        store_eta=eta,store_sweeps=cfg['model'].get('store_sweeps'),layer=14,L=1,host=socket.gethostname(),
        job_id=os.environ.get('PBS_JOBID'),created_at=datetime.now(timezone.utc).isoformat())
    experiment=None
    try:
        experiment=LSQExperiment(cfg,arm)
        torch.cuda.reset_peak_memory_stats()
        record['standardization']=experiment.standardize()
        record['training']=fit(experiment,out,steps)
        record['memory']=dict(shape=[rank,rank],state_bytes_per_episode=4*rank*rank,
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
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--steps',type=int)
    budget.add_argument('--schedule',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--eta',type=float,help='smoke only: the ridge strength of a ridge arm')
    parser.add_argument('--e8-decision',type=Path)
    args=parser.parse_args()
    E=8 if args.e8_decision else 16
    if args.e8_decision:
        eligible_e8(json.loads(args.e8_decision.read_text()),args.arm)
    schedule=None if args.schedule is None else json.loads(args.schedule.read_text())
    steps=scheduled_steps(args.arm,args.steps,schedule,smoke=args.smoke,E=E)
    if schedule is not None and schedule['science_commit']!=source_metadata()['git_commit']:
        raise ValueError('scientific code must match its frozen schedule release')
    if args.smoke:
        eta=args.eta
    else:
        if args.eta is not None:
            raise ValueError('scientific arms take their ridge strength from the release')
        eta=release_eta(schedule,args.arm)
    args.out.mkdir(parents=True,exist_ok=True)
    record=run_one(args.arm,args.out/args.arm,steps,E,args.smoke,eta)
    _write_json(args.out/f'shard-{args.arm}.json',dict(runs=[record],procs=1))
