"""WP-L7 RFX: the read-failure factorial on the three frozen WP-L4 checkpoints; one job, evaluation only (compute only)."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

from .lsq import lsq_eta
from .rfx import LABELS,load_checkpoint,lsq_ranks,run_checkpoint
from .run import _load_backbone,_write_json,source_metadata

RUNS=Path('/work/xg24i002/x10041/LoopLora/runs/p1l')
# label -> (folder under RUNS, rank, steps of the final checkpoint), as registered in docs/prereg/RFX.md
CHECKPOINTS={'R-r128':('read-fix-20261027/arms-r128/R',128,24645),'R-r32':('read-fix-20261027/arms/R',32,23590),
             'W5-r32':('writer-fix-20261027/rerun/W5',32,32900)}
# label -> (writer.pt, checkpoint.pt) blake2b-128, as registered in docs/prereg/RFX.md; a mismatch voids the run
DIGESTS={'R-r128':('61911b21bbfdbe48604ad3d5961ef9d3','0f51677aea1db4cbb5f7e9349fceffd1'),
         'R-r32':('dce6e6040122f3956444a81b9bec3bf4','a2e79690fafd64d540f8aef8b081efab'),
         'W5-r32':('979b387f8b04165132b91b4d174f1c80','687b5b46f81f845f0b6d33f2723bfe34')}
SMOKE=dict(fit_episodes=64,eval_episodes=64)
TOLERANCE=dict(raw_recompute_max_rel=1e-4,delta_reconstruction_max_rel=1e-4,ridge_permutation_max_abs=1e-4)


def checkpoint_identity(label,folder):
    """The folder must hold the complete final run named in the registration; its files are fingerprinted (blake2b)."""
    folder=Path(folder)
    record=json.loads((folder/'record.json').read_text())
    _,rank,steps=CHECKPOINTS[label]
    if record['status']!='complete' or record.get('smoke') or record['rank']!=rank or record['training']['actual_steps']!=steps:
        raise ValueError(f'{label}: expected the complete final checkpoint of {steps} steps at rank {rank}')
    digest=lambda name:hashlib.blake2b((folder/name).read_bytes(),digest_size=16).hexdigest()
    return dict(folder=str(folder),arm=record['arm'],rank=rank,steps=steps,config_hash=record['config_hash'],
                git_commit=record['git_commit'],job_id=record.get('job_id'),
                writer_blake2b=digest('writer.pt'),checkpoint_blake2b=digest('checkpoint.pt'))


def verify_digests(label,identity):
    """The folder's files must be the registered ones (docs/prereg/RFX.md): a mismatch voids the run."""
    found=(identity['writer_blake2b'],identity['checkpoint_blake2b'])
    if found!=DIGESTS[label]:
        raise ValueError(f'{label}: (writer.pt, checkpoint.pt) blake2b-128 {found} differ from the registered {DIGESTS[label]}')


def instrumentation_failures(checks):
    """Gross-bug guards, not scientific outcomes: the replacement hook and the cached stores must reproduce the model."""
    bad=[]
    if checks['identity_max_abs_logit_diff']!=0.:
        bad.append(f"re-inserting the recorded read changed the logits ({checks['identity_max_abs_logit_diff']})")
    for name,limit in TOLERANCE.items():
        if not checks[name]<=limit:
            bad.append(f'{name} = {checks[name]} exceeds {limit}')
    return bad


def run(runs,out,labels=LABELS,*,smoke=False):
    if not smoke and tuple(labels)!=LABELS:
        raise ValueError('the registered study evaluates all three checkpoints')
    identities={label:checkpoint_identity(label,Path(runs)/CHECKPOINTS[label][0]) for label in labels}
    for label,identity in identities.items():
        verify_digests(label,identity)  # before anything is created or loaded
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    created=datetime.now(timezone.utc).isoformat()
    model,tokenizer,load_seconds=_load_backbone('cuda')
    records,failures={},{}
    for label in labels:
        identity=identities[label]
        experiment,head=load_checkpoint(identity['folder'],model=model,tokenizer=tokenizer)
        try:
            record,samples=run_checkpoint(label,experiment,head,smoke=smoke,**(SMOKE if smoke else {}))
        finally:
            experiment.close()
        record['checkpoint']=identity
        records[label]=record
        np.savez_compressed(out/f'{label}-samples.npz',**{k.replace('/','__'):v for k,v in samples.items()})
        _write_json(out/f'{label}.json',record)
        failures[label]=instrumentation_failures(record['checks'])
    verdicts={label:record['verdict']['verdict'] for label,record in records.items()}
    ranks=lsq_ranks(verdicts)
    rfx_eta={str(rank):records[f'R-r{rank}']['parameters']['eta']['unit'] for rank in ranks}
    summary=dict(**source_metadata(),smoke=smoke,labels=list(labels),verdicts=verdicts,lsq_ranks=ranks,
        rfx_eta_by_rank=rfx_eta,eta_by_rank={rank:lsq_eta(eta) for rank,eta in rfx_eta.items()},
        instrumentation_failures=failures,backbone_load_seconds=load_seconds,host=socket.gethostname(),
        job_id=os.environ.get('PBS_JOBID'),created_at=created,completed_at=datetime.now(timezone.utc).isoformat(),
        runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                     deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                     cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
    _write_json(out/'summary.json',summary)
    if any(failures.values()):
        raise AssertionError(f'instrumentation checks failed: {failures}')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=RUNS)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',choices=LABELS,default=list(LABELS))
    parser.add_argument('--smoke',action='store_true',help='64+64 episodes; evaluates fresh train-split episodes, not the val range')
    args=parser.parse_args()
    run(args.runs,args.out,tuple(args.labels),smoke=args.smoke)
