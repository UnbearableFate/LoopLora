#!/usr/bin/env python3
"""WP-C6 fixed-L4 memory-only training smoke and prefetch-2 timing comparison."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import time

import torch

from audit_p4s_stopping import require_compute
from looplora.core import memory
from looplora.core.adapters import UnsharedAdapter
from looplora.infra.seeds import derive_seed
from looplora.llm.config import resolve_config
from looplora.llm.inject import load_model, load_tokenizer, MODEL_ID, MODEL_REVISION
from looplora.llm.train import train
from looplora.sym.run import source_metadata


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(out, smoke=False, summary_out=None):
    require_compute()
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    warmup, repeats = (1,2) if smoke else (5,20)
    steps = warmup+repeats
    record = dict(task='WP-C6',status='running',smoke=smoke,
        settings=dict(variants=['M1','B8'],depth=4,batches=[32,128],processes=1,
                      prefetch_workers=2,warmup=warmup,repeats=repeats,
                      stage='joint',protocol='memory_only',input_site='o_proj',E=16,Q=4,r=32,layer=14),
        model=MODEL_ID,revision=MODEL_REVISION,dtypes='bf16 backbone, fp32 adapter/state',
        measurement='CUDA-synchronized production train() step durations, including batch wait, '
                    'support forward/write, question forward/backward, clipping and AdamW. '
                    'Discard warmup steps; startup and checkpoint export remain in cell time.',
        started_at=datetime.now(timezone.utc).isoformat(),
        runtime=dict(host=socket.gethostname(),job_id=os.environ['PBS_JOBID'],torch=torch.__version__),
        **source_metadata(),cells=[])
    save(out/'benchmark.json',record)
    started = time.perf_counter()
    model, tokenizer = load_model(), load_tokenizer()
    record['runtime']['gpu'] = torch.cuda.get_device_name()
    original_final, original_reference = memory.scan_final, memory.scan_prewrite
    for batch in (32,128):
        variants = ('M1','B8') if batch==32 else ('B8','M1')
        for variant in variants:
            counts = dict(final=0,reference=0)

            def final(*args,**kwargs):
                counts['final'] += 1
                return original_final(*args,**kwargs)

            def reference(*args,**kwargs):
                counts['reference'] += 1
                return original_reference(*args,**kwargs)

            memory.scan_final, memory.scan_prewrite = final, reference
            cfg = resolve_config(dict(name=f'c6-{variant}-b{batch}',variant=variant,stage='joint',
                seed=derive_seed(20261027,'c6-smoke') if smoke else 17,
                model=dict(r=32,layer=14,input_site='o_proj'),
                task=dict(E=16,Q=4,chain_lengths=[8,4,2,2]),
                train=dict(steps=steps,batch_size=batch,depths=[4],log_every=1,
                           prefetch_workers=2,warmup_steps=1),
                eval=dict(episodes=2,counterfactuals=False)))
            experiment = None
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            try:
                folder = out/f'{variant}-b{batch}'
                experiment, summary = train(cfg,folder,model=model,tokenizer=tokenizer)
                logs = [json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines()]
                assert len(logs)==steps and {r['L'] for r in logs}=={4} and summary['mean_L']==4
                assert counts==dict(final=steps,reference=0), counts
                gradients = {}
                if variant=='B8':
                    assert isinstance(experiment.adapter,UnsharedAdapter)
                    for name in ('R','U','Q'):
                        for s,p in enumerate(getattr(experiment.adapter,name)):
                            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                            gradients[f'{name}.{s}'] = float(p.grad.norm())
                durations = [r['step_seconds'] for r in logs[warmup:]]
                assert len(durations)==repeats and all(torch.isfinite(torch.tensor(durations)))
                row = dict(variant=variant,batch=batch,steps=steps,L=4,prefetch_workers=2,
                    mean_step_ms=1000*statistics.mean(durations),median_step_ms=1000*statistics.median(durations),
                    episodes_per_second=batch/statistics.mean(durations),
                    scan_final_calls=counts['final'],scan_prewrite_calls=counts['reference'],
                    step_seconds=durations,all_logged_depths=[r['L'] for r in logs],
                    final_loss=summary['final_loss'],per_step_gradient_norms=gradients,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cell_seconds=time.perf_counter()-tick,train_seconds=summary['train_seconds'])
                record['cells'].append(row)
                save(out/'benchmark.json',record)
                print(json.dumps({k: row[k] for k in ('variant','batch','mean_step_ms','episodes_per_second',
                                                     'scan_final_calls','scan_prewrite_calls')}),flush=True)
            finally:
                memory.scan_final, memory.scan_prewrite = original_final, original_reference
                if experiment is not None:
                    experiment.close()
    assert {(r['variant'],r['batch']) for r in record['cells']}=={(v,b) for v in ('M1','B8') for b in (32,128)}
    record.update(status='complete',seconds=time.perf_counter()-started,
                  finished_at=datetime.now(timezone.utc).isoformat())
    save(out/'benchmark.json',record)
    columns = ['variant','batch','L','prefetch_workers','mean_step_ms','median_step_ms',
               'episodes_per_second','scan_final_calls','scan_prewrite_calls','peak_allocated_bytes','cell_seconds']
    with (out/'timings.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,lineterminator='\n',extrasaction='ignore')
        writer.writeheader()
        writer.writerows(record['cells'])
    lines = ['# WP-C6 B8 versus M1 at fixed L=4', '',
        f'One process, two prefetch workers, {warmup} discarded warmups and {repeats} measured steps per cell. '
        'Pinned Qwen3-0.6B-Base, layer14/o_proj, r32, E16/Q4 T2 train-only data, fact_final writes. '
        'CUDA-synchronized production training steps; process/prefetch startup and export excluded from steady ms. '
        'These are short systems measurements, not quality comparisons.', '',
        '| Variant | Batch | Mean ms/step | Median ms/step | Episodes/s | Peak GiB | Fast / reference scan calls |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in sorted(record['cells'],key=lambda r:(r['batch'],r['variant'])):
        lines.append(f"| {r['variant']} | {r['batch']} | {r['mean_step_ms']:.2f} | {r['median_step_ms']:.2f} | "
                     f"{r['episodes_per_second']:.2f} | {r['peak_allocated_bytes']/2**30:.3f} | "
                     f"{r['scan_final_calls']} / {r['scan_prewrite_calls']} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    if summary_out is not None:
        summary_out.mkdir(parents=True,exist_ok=True)
        for name in ('benchmark.json','timings.csv','tables.md'):
            shutil.copy2(out/name,summary_out/name)
    print(json.dumps(dict(status='complete',seconds=record['seconds'],out=str(out))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--summary-out',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run(args.out,args.smoke,args.summary_out)
