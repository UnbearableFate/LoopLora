#!/usr/bin/env python3
"""Audit the complete WP-C5 benchmark and export compact compute-side tables."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics

from audit_p4s_stopping import markdown_table, require, require_compute


def report(root, out):
    record = json.loads((root/'benchmark.json').read_text())
    require(record['status'] == 'complete' and record['smoke'] is False, 'Full benchmark is incomplete')
    settings = record['settings']
    require(settings == dict(batches=[32,64,128],warmup=5,repeats=20,profile_repeats=3), 'Benchmark design changed')
    cells = record['cells']
    expected = {(p,b,w,m) for p in (1,2) for b in (32,64,128)
                for w in ('fact_final','all_tokens') for m in ('reference','fast')}
    keys = [(r['procs'],r['batch'],r['policy'],r['method']) for r in cells]
    require(len(keys) == len(expected) and set(keys) == expected, 'Benchmark matrix is incomplete')
    sources = {}
    equivalent = []
    for procs in (1,2):
        for index in range(procs):
            name = f'worker-{procs}-{index}.json'
            worker = json.loads((root/name).read_text())
            require(worker['status'] == 'complete' and len(worker['rows']) == 12, 'Incomplete worker')
            equivalent.extend(worker['loss_equivalence'])
            for row in worker['rows']:
                durations = row['durations_seconds']
                require(len(durations) == 20 and all(math.isfinite(v) and v > 0 for v in durations), 'Invalid durations')
                require(abs(statistics.mean(durations)*1000-row['worker_mean_ms']) < 1e-7, 'Worker mean differs')
                require(len(row['profiled_steps']) == 3, 'Missing profiling repeats')
                for phases in row['profiled_steps']:
                    require(all(math.isfinite(v) and v >= 0 for v in phases.values()), 'Invalid profile timing')
                    total = sum(phases[k] for k in ('data_ms','prepare_other_ms','support_forward_ms',
                        'write_ms','question_forward_backward_ms','optimizer_ms'))
                    require(abs(total-phases['total_ms']) < 1e-6, 'Phase times do not partition total')
                key = (procs,row['batch'],row['policy'],row['method'])
                sources.setdefault(key,[]).append(row)
    require(len(equivalent) == 4 and {(r['input_site'],r['policy']) for r in equivalent} == {
        (s,w) for s in ('o_proj','residual') for w in ('fact_final','all_tokens')}, 'Missing real-model loss checks')
    require(all(abs(r['fast_loss']-r['reference_loss']) <= 1e-5*abs(r['reference_loss']) for r in equivalent),
            'Fixed P1-L loss equivalence failed')
    for row, key in zip(cells,keys,strict=True):
        members = sources[key]
        require(len(members) == row['procs'], 'Missing concurrent worker')
        require(abs(row['episodes_per_second']-row['procs']*row['batch']*1000/row['ms_per_step']) < 1e-7,
                'Achieved throughput and step time differ')
        require(row['sum_peak_allocated_bytes'] == sum(r['peak_allocated_bytes'] for r in members),
                'Peak allocation accounting differs')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('benchmark.json','timings.csv','worker-1-0.json','worker-2-0.json','worker-2-1.json'):
        shutil.copy2(root/name,out/name)
    lines = ['# WP-C5 full P1-L T1 benchmark', '',
        'Pinned Qwen3-0.6B-Base, E=16, Q=4, r=32, layer 14, o_proj input. '
        'Frozen BF16 backbone, FP32 adapter/state, AdamW 3e-4. Five warmups and 20 measured steps. '
        'Both methods start from identical parameters, optimizer and episode streams. '
        'Each step generates and renders fresh training episodes; no accuracy selection.', '',
        'Wall ms/step includes synchronization barriers in two-process achieved throughput. '
        'Worker durations without barriers remain in the JSON files. Memory is the sum of per-worker '
        'peak allocated GiB, not an observed simultaneous peak. Every requested cell fit.', '',
        '## Before / after', '']
    lookup = dict(zip(keys,cells))
    rows = []
    for p in (1,2):
        for w in ('fact_final','all_tokens'):
            for b in (32,64,128):
                before,after = lookup[p,b,w,'reference'],lookup[p,b,w,'fast']
                rows.append([p,w,b,f"{before['ms_per_step']:.1f} / {after['ms_per_step']:.1f}",
                    f"{before['episodes_per_second']:.1f} / {after['episodes_per_second']:.1f}",
                    f"{before['sum_peak_allocated_bytes']/2**30:.3f} / {after['sum_peak_allocated_bytes']/2**30:.3f}",
                    f"{before['ms_per_step']/after['ms_per_step']:.2f}×"])
    lines += markdown_table(['Processes','Policy','Batch/worker','ms/step before / after',
        'Episodes/s before / after','Sum peak GiB before / after','Speedup'],rows)
    lines += ['', '## Separate profiling steps', '',
        'Three extra steps synchronize at phase boundaries. They diagnose remaining cost and do '
        'not replace the unprofiled timings above. Support forward includes the frozen prefix, '
        'A projection and support bookkeeping, excluding gates/K/V/state updates in Write. '
        'Question forward/backward includes the backward pass through the writer; optimizer includes '
        'gradient clipping. Render means support tokenization and remaining preparation.', '']
    fields = ('data_ms','prepare_other_ms','support_forward_ms','write_ms',
              'question_forward_backward_ms','optimizer_ms','total_ms')
    lines += markdown_table(['Processes','Policy','Batch','Method','Data','Render','Support fwd','Write',
        'Question fwd/bwd','Optimizer','Total'],
        [[p,w,b,m,*[f"{lookup[p,b,w,m][k]:.2f}" for k in fields]]
         for p in (1,2) for w in ('fact_final','all_tokens') for b in (32,64,128) for m in ('reference','fast')])
    lines += ['', '## Fixed-batch real-model equivalence', '',
        'Nonzero B, batch 2, same episode data. Required loss relative tolerance 1e-5; '
        'state checks use rtol 1e-5, atol 1e-6. Core tests also compare u/K/V/gate gradients.', '']
    lines += markdown_table(['Input','Policy','Reference CE','Fast CE','Maximum state error'],
        [[r['input_site'],r['policy'],f"{r['reference_loss']:.9f}",f"{r['fast_loss']:.9f}",
          f"{r['state_max_absolute_error']:.3g}"] for r in equivalent])
    (out/'tables.md').write_text('\n'.join(lines).rstrip()+'\n')
    result = dict(status='passed',cells=24,workers=3,measured_worker_steps=720,
                  fixed_loss_checks=equivalent,phase_partition_checks='passed',
                  all_cells_fit=True,raw_root=str(root),seconds=record['seconds'])
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    require_compute()
    report(args.root,args.out)
