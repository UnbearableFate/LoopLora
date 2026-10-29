"""WP-L12 LSQT arms, one arm per job; smoke and all runtime execute on compute (docs/prereg/LSQT.md)."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket

import torch

from looplora.infra.seeds import stable_hash
from .lsqt import LSQTExperiment,configuration
from .lsqt_release import ARMS,RANK,SPECS,prereg_digest,scheduled_steps
from .read_fix_run import fit
from .run import _write_json,source_metadata

R_R128_CORRECT=335  # WP-L4's R at r = 128 (docs/reports/P1L-read-fix.md): 33.5 % of 1,000 validation episodes
D128_3_CORRECT=608  # DPFP's D128-3 (docs/reports/DPFP.md)


def load_inference(path,*,model=None,tokenizer=None):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('read_fix_version')!=1 or 'value_head' in payload or 'optimizer' in payload:
        raise ValueError('expected an LSQT inference export')
    experiment=LSQTExperiment(payload['config'],payload['arm'],model=model,tokenizer=tokenizer)
    try:
        if payload['signature']!=experiment.writer_signature():
            raise ValueError('LSQT signature mismatch')
        experiment.adapter.load_state_dict(payload['adapter'],strict=True)
        experiment.value_head=None
    except BaseException:
        experiment.close()
        raise
    return experiment


def choose_t(records):
    """The registered rule of docs/prereg/LSQT.md over the three arms, each at its own final checkpoint (1,000 validation episodes).

    Outcome 1: any arm at 0.80 is a flagged writer candidate (the highest EM; ties in candidate order; no E8 retrain). The band of
    the best ridge arm is reported alongside: >= R-r128 + 15 pp (48.5 %) flags fable's W3 trigger, below 40 % closes the trained-LSQ
    line (the falsifier). The comparisons with R-r128 and D128-3 are descriptive."""
    by_arm={r['arm']:r for r in records}
    if len(records)!=len(ARMS) or set(by_arm)!=set(ARMS) or any(
            r['status']!='complete' or r['smoke'] or r['config']['task']['E']!=16 for r in records):
        raise ValueError('the decision requires the three complete scientific E16 LSQT arms')
    correct={arm:by_arm[arm]['training']['metrics'][str(by_arm[arm]['training']['actual_steps'])]['correct'] for arm in ARMS}
    passed=[arm for arm in ARMS if correct[arm]>=800]
    rg=[arm for arm in ARMS if SPECS[arm][0]=='ridge']
    best_rg=max(rg,key=lambda arm:(correct[arm],-ARMS.index(arm)))
    band='w3_trigger' if correct[best_rg]>=R_R128_CORRECT+150 else 'open' if correct[best_rg]>=400 else 'closed'
    best_arm=max(ARMS,key=lambda arm:(correct[arm],-ARMS.index(arm)))
    return dict(study='LSQT',outcome='flagged_candidate' if passed else 'no_candidate',
        candidate=max(passed,key=lambda arm:(correct[arm],-ARMS.index(arm))) if passed else None,e8_retrain=False,
        best_rg=best_rg,best_rg_band=band,trained_lsq_line='closed' if band=='closed' else 'open',
        u_control_runs=correct[best_arm]>=400,vs_r_r128_pp={arm:(correct[arm]-R_R128_CORRECT)/10 for arm in ARMS},
        vs_d128_3_pp={arm:(correct[arm]-D128_3_CORRECT)/10 for arm in ARMS},correct=correct)


def run_one(arm,out,steps,smoke=False,expected_hash=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    cfg=configuration(arm,steps,smoke=smoke)
    if expected_hash is not None and stable_hash(cfg)!=expected_hash:
        raise ValueError('the config differs from the frozen release')
    store,eta=SPECS[arm]
    record=dict(**source_metadata(),arm=arm,config=cfg,config_hash=stable_hash(cfg),status='running',
        smoke=smoke,seed=17,eval_seed=4242,stage='writer',protocol='memory_only',rank=RANK,store=store,
        store_eta=eta,store_sweeps=cfg['model'].get('store_sweeps'),layer=14,L=1,host=socket.gethostname(),
        job_id=os.environ.get('PBS_JOBID'),created_at=datetime.now(timezone.utc).isoformat())
    experiment=None
    try:
        experiment=LSQTExperiment(cfg,arm)
        torch.cuda.reset_peak_memory_stats()
        record['standardization']=experiment.standardize()
        record['training']=fit(experiment,out,steps)
        record['memory']=dict(shape=[RANK,RANK],state_bytes_per_episode=4*RANK*RANK,
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
    parser.add_argument('--arm',choices=tuple(SPECS),required=True)
    parser.add_argument('--out',type=Path,required=True)
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--steps',type=int)
    budget.add_argument('--schedule',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    schedule=None if args.schedule is None else json.loads(args.schedule.read_text())
    steps=scheduled_steps(args.arm,args.steps,schedule,smoke=args.smoke)
    expected=None
    if schedule is not None:
        if schedule['science_commit']!=source_metadata()['git_commit']:
            raise ValueError('scientific code must match its frozen schedule release')
        if schedule['prereg_blake2b_128']!=prereg_digest():
            raise ValueError('the registration differs from the frozen release')
        expected=schedule['config_hashes'][args.arm]
    args.out.mkdir(parents=True,exist_ok=True)
    record=run_one(args.arm,args.out/args.arm,steps,args.smoke,expected)
    _write_json(args.out/f'shard-{args.arm}.json',dict(runs=[record],procs=1))
