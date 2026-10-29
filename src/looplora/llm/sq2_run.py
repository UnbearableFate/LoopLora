"""WP-L12 SQ2 arms, one arm per job; smoke and all runtime execute on compute (docs/prereg/SQ2.md)."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket

import torch

from looplora.infra.seeds import derive_seed,stable_hash
from .read_fix_run import fit,validate
from .run import _write_json,source_metadata
from .sq2 import SQ2Experiment,configuration,load_continuation,load_inference
from .sq2_release import ARMS,PARENT,SPECS,file_digest,prereg_digest,scheduled_steps

__all__=['DATA_SEED','load_inference','numerics_values','run_one']
DATA_SEED=derive_seed(17,'llm','sq2-continuation')  # one fresh training stream shared by every arm
TIME_LIMIT=3400  # seconds of training: an arm job is one hour


def numerics_values(path):
    """eta and w_c from the committed numerics file, and its digest (frozen in the release)."""
    numbers=json.loads(Path(path).read_text())
    return dict(eta=float(numbers['eta']),w_c=float(numbers['w_c']),digest=file_digest(path))


def run_one(arm,out,steps,numerics,smoke=False,expected_hash=None,parent_folder=None,digests=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    store,coefficient=SPECS[arm]
    cfg=configuration(arm,steps,eta=numerics['eta'] if store=='ridge' else None,
        w_c=numerics['w_c'] if coefficient else None,smoke=smoke,numerics_digest=numerics['digest'])
    if expected_hash is not None and stable_hash(cfg)!=expected_hash:
        raise ValueError('the config differs from the frozen release')
    record=dict(**source_metadata(),arm=arm,config=cfg,config_hash=stable_hash(cfg),status='running',
        smoke=smoke,seed=17,eval_seed=4242,stage='writer',protocol='memory_only',rank=128,dpfp_nu=3,key_dim=768,
        store=store,store_eta=cfg['model'].get('store_eta'),coefficient_weight=cfg['read_fix'].get('coefficient_weight'),
        parent=dict(arm=PARENT['arm'],steps=PARENT['steps'],config_hash=PARENT['config_hash']),data_seed=DATA_SEED,
        layer=14,L=1,host=socket.gethostname(),job_id=os.environ.get('PBS_JOBID'),
        created_at=datetime.now(timezone.utc).isoformat())
    experiment=None
    try:
        experiment=SQ2Experiment(cfg,arm)
        optimizer_state=load_continuation(experiment,parent_folder,digests)
        torch.cuda.reset_peak_memory_stats()
        record['step0']=validate(experiment,out/'val-0',cfg['eval']['episodes'])
        record['training']=fit(experiment,out,steps,data_seed=DATA_SEED,optimizer_state=optimizer_state,warmup=0,
                               time_limit=TIME_LIMIT)
        record['memory']=dict(shape=[128,768],state_bytes_per_episode=4*128*768,
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
    parser.add_argument('--numerics',type=Path,required=True,help='the committed numerics file (eta, w_c)')
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--steps',type=int)
    budget.add_argument('--schedule',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    schedule=None if args.schedule is None else json.loads(args.schedule.read_text())
    steps=scheduled_steps(args.arm,args.steps,schedule,smoke=args.smoke)
    numerics=numerics_values(args.numerics)
    expected=None
    if schedule is not None:
        if schedule['science_commit']!=source_metadata()['git_commit']:
            raise ValueError('scientific code must match its frozen schedule release')
        if schedule['prereg_blake2b_128']!=prereg_digest():
            raise ValueError('the registration differs from the frozen release')
        if schedule['numerics']['file_blake2b_128']!=numerics['digest'] or (schedule['numerics']['eta'],schedule['numerics']['w_c'])!=(
                numerics['eta'],numerics['w_c']):
            raise ValueError('the numerics file differs from the frozen release')
        if schedule['step0']['passed'] is not True:
            raise ValueError('the release does not record a passed step-0 check')
        expected=schedule['config_hashes'][args.arm]
    args.out.mkdir(parents=True,exist_ok=True)
    record=run_one(args.arm,args.out/args.arm,steps,numerics,args.smoke,expected)
    _write_json(args.out/f'shard-{args.arm}.json',dict(runs=[record],procs=1))
