"""A1 systems-only equality/timing check of the selected real-model paths."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import socket

import torch

from looplora.llm.inject import load_model,load_tokenizer
from looplora.llm.run import _write_json,source_metadata
from looplora.llm.writer_fix import WriterFixExperiment
from looplora.llm.writer_fix_run import configuration,fit


def check(settings):
    arm,out=settings
    out=Path(out)
    model,tokenizer=load_model(),load_tokenizer()
    baseline=None
    rows=[]
    for workers in (0,2):
        folder=out/f'{arm}-{workers}'
        folder.mkdir(parents=True,exist_ok=False)
        config=configuration(arm,100,prefetch_workers=workers)
        experiment=WriterFixExperiment(config,arm,model=model,tokenizer=tokenizer)
        try:
            experiment.standardize()
            result=fit(experiment,folder,steps=100,max_seconds=None,eval_steps=[])
            state={k:v.detach().cpu().clone() for k,v in experiment.adapter.state_dict().items()}
            if experiment.value_head is not None:
                state.update({'head.'+k:v.detach().cpu().clone()
                              for k,v in experiment.value_head.state_dict().items()})
            logs=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines()]
            compared=[{k:v for k,v in row.items() if k!='step_seconds'} for row in logs]
            if baseline is None:
                baseline=state,compared
            else:
                assert state.keys()==baseline[0].keys()
                assert all(torch.equal(v,baseline[0][k]) for k,v in state.items())
                assert compared==baseline[1]
            rows.append(dict(workers=workers,steps=100,
                steady_step_mean_seconds=result['steady_step_mean_seconds'],
                step_p95_seconds=result['step_p95_seconds'],seconds=result['seconds']))
        finally:
            experiment.close()
    return dict(arm=arm,rows=rows,identical_final_parameters=True,identical_logged_losses=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(check,[(a,str(args.out)) for a in ('OA','W5')]))
    _write_json(args.out/'summary.json',dict(**source_metadata(),records=records,
        host=socket.gethostname(),job_id=os.environ.get('PBS_JOBID'),
        purpose='systems timing/equality only; no validation or selection scores'))
