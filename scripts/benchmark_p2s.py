#!/usr/bin/env python3
"""WP-C3 timing clocks and candidate decoding, extended to all P2-S adapters."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time

import torch

from analyze_p1s_main import file_hash, require
from benchmark_p4s_stopping import _measure
from looplora.core.accounting import macs_per_token, state_bytes
from looplora.core.stopping import fixed_read
from looplora.data.relations import generate
from looplora.sym.evaluate import stratify_queries
from looplora.sym.model import make_task_spec
from looplora.sym.train import load_checkpoint
from run_p4s_stopping import reader_batch
import run_p2s as driver


def cost(model,depth: int,candidates: int) -> dict:
    c=model.adapter.cfg
    read=2*(macs_per_token(c,depth,writes=False)-c.r*(c.d_in+c.d_out))
    return dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        registered_parameters=sum(p.numel() for p in model.parameters()),
        state_bytes_fp32=state_bytes(c,1,1,torch.float32),read_flops_per_query=read,
        output_projection_flops=2*c.r*c.d_out,candidate_decode_flops=2*c.d_out*candidates)


@torch.inference_mode()
def benchmark(run: dict) -> list[dict]:
    path=Path(run['checkpoint'])
    require(file_hash(path)==run['checkpoint_hash'],'checkpoint changed before benchmark')
    model=load_checkpoint(path)
    require(model.checkpoint_config==run['config'],'benchmark checkpoint config differs')
    c=driver.confirmation_config(model.config)
    c['eval'].update(seed=4242,splits=['val'],episodes=64)
    spec=make_task_spec(c,evaluation=True)
    arrays=stratify_queries(generate(spec,'val',0,64,4242),spec,list(range(1,9)),4242)
    prepared=model.prepare(arrays)
    flat=reader_batch(prepared)
    selected=torch.arange(64,device=model.device)*8+torch.arange(64,device=model.device)%8
    batch=flat.take(selected)
    original=tuple(v.clone() for v in (batch.u,batch.memory,batch.codes,batch.candidate_ids))
    expected_depths=[4] if run['variant']=='B8' else [1,2,4,8]
    rows=[]
    for depth in expected_depths:
        logits=model.read(prepared,depth)['logits']
        expected=logits.masked_fill(~prepared.candidate_mask[:,None],-torch.inf).argmax(-1).reshape(-1)[selected]
        require(torch.equal(fixed_read(model.adapter,batch,depth),expected),'candidate reader differs from evaluator')
        inputs=[(batch,0)]+[(batch.take(torch.tensor([h-1],device=model.device)),h) for h in range(1,9)]
        for data,hop in inputs:
            if hop: require(torch.equal(fixed_read(model.adapter,data,depth),expected[hop-1:hop]),
                            'batch-one candidate decision differs')
            measurements=_measure(lambda:fixed_read(model.adapter,data,depth),model.device,5,20)
            rows.append({k:run[k] for k in ('name','source','writer','variant','r','r_mem','seed','steps','descriptive_only')}|
                dict(L=depth,batch_size=len(data.u),latency_hop=hop,warmup=5,repeats=20,
                     **cost(model,depth,data.codes.shape[1]),**measurements))
    require(all(torch.equal(a,b) for a,b in zip(original,(batch.u,batch.memory,batch.codes,batch.candidate_ids))),
            'benchmark mutated prepared inputs')
    require(file_hash(path)==run['checkpoint_hash'],'benchmark mutated checkpoint')
    return rows


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1);torch.cuda.set_device(0)
    protocol=json.loads((driver.ROOT/'reports/p4s/design.json').read_text())['latency']
    require(protocol['warmup']==5 and protocol['repeats']==20,'WP-C3 timing protocol changed')
    if args.smoke:
        runs=[]
        for f in driver.families():
            c=driver.config(f,17,500,.001,smoke=True)
            checkpoint=driver.RUNS/'smoke/shard-00'/c['name']/'checkpoint.pt'
            runs.append(dict(**f,name=c['name'],source='smoke',seed=17,steps=500,descriptive_only=True,
                checkpoint=str(checkpoint),checkpoint_hash=file_hash(checkpoint),config=c))
    else:
        driver.all_training()
        runs=json.loads((driver.RUNS/'evaluation_plan.json').read_text())['runs']
    out=driver.RUNS/('benchmark_smoke' if args.smoke else 'benchmark')
    require(not out.exists(),'benchmark output already exists')
    out.mkdir(parents=True)
    started=time.perf_counter();rows=[]
    for run in runs:
        tick=time.perf_counter()
        rows.extend(benchmark(run))
        print(f"Benchmark {run['name']}: {time.perf_counter()-tick:.2f}s",flush=True)
    driver.csv_table(out/'latency.csv',rows)
    driver.write(out/'audit.json',dict(status='passed',models=len(runs),rows=len(rows),
        seconds=time.perf_counter()-started,warmup=5,repeats=20,dtype='float32',
        context_preparation_timed=False,candidate_decoding_timed=True,checkpoints_unchanged=True,
        all_candidate_decisions_match=True,all_prepared_inputs_unchanged=True,
        host=socket.gethostname(),job_id=os.environ['PBS_JOBID'],torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0)))


if __name__=='__main__': main()
