#!/usr/bin/env python3
"""WP-C5 P1-L T1 step timings, reference versus final-only writes (compute only)."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time
import traceback
from types import MethodType


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(index, procs, settings, out, barrier):
    """One persistent model per process; barriers keep matching cells concurrent."""
    try:
        import numpy as np
        import torch
        from torch.nn import functional as F
        from looplora.data.relations import generate
        from looplora.infra.seeds import derive_seed
        from looplora.llm.inject import load_model, load_tokenizer
        from looplora.llm.pilot1_select import writer_config
        from looplora.llm.train import LLMExperiment, make_task_spec
        import looplora.llm.train as training

        torch.set_num_threads(1)
        model, tokenizer = load_model(), load_tokenizer()
        rows, equivalent = [], []
        original_encode = training.encode_support

        def config(batch, policy, site='o_proj'):
            value = writer_config('fast-write-benchmark', 16, policy, .0003, 32, 1)
            value['train']['batch_size'] = batch
            value['model']['input_site'] = site
            return value

        if procs == 1:
            for site in ('o_proj', 'residual'):
                for policy in ('fact_final', 'all_tokens'):
                    cfg = config(2, policy, site)
                    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
                    try:
                        g = torch.Generator(device='cuda').manual_seed(derive_seed(17, 'c5-real-B'))
                        with torch.no_grad():
                            experiment.adapter.B.normal_(0, .03, generator=g)
                        fast_scan = experiment.adapter.scan_final
                        arrays = generate(make_task_spec(cfg), 'train', 0, 2,
                                          derive_seed(17, 'c5-fixed-equivalence'))
                        outcomes = []
                        for fast in (False, True):
                            experiment.adapter.scan_final = (fast_scan if fast else MethodType(
                                lambda self, *args, **kw: self.scan_prewrite(*args, **kw)[1],
                                experiment.adapter))
                            with torch.no_grad():
                                prepared = experiment.prepare(arrays)
                                logits, _ = experiment.score(prepared, 1)
                                loss = F.cross_entropy(logits.float().flatten(0, 1), prepared.answer_ids.flatten())
                                outcomes.append((loss.detach(), prepared.snapshot.states[14].detach().clone()))
                        torch.testing.assert_close(outcomes[1][0], outcomes[0][0], rtol=1e-5, atol=0)
                        torch.testing.assert_close(outcomes[1][1], outcomes[0][1], rtol=1e-5, atol=1e-6)
                        equivalent.append(dict(input_site=site, policy=policy,
                            reference_loss=float(outcomes[0][0]), fast_loss=float(outcomes[1][0]),
                            state_max_absolute_error=float((outcomes[1][1]-outcomes[0][1]).abs().max())))
                    finally:
                        experiment.close()

        for policy_index, policy in enumerate(('fact_final', 'all_tokens')):
            for batch_index, batch in enumerate(settings['batches']):
                cfg = config(batch, policy)
                experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
                try:
                    adapter = experiment.adapter
                    parameters = experiment.trainable_parameters()
                    baseline = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
                    fast_scan = adapter.scan_final
                    phases, profiling = {}, False

                    def scan(self, *args, **kwargs):
                        if profiling:
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                        value = (fast_scan(*args, **kwargs) if method == 'fast' else
                                 self.scan_prewrite(*args, **kwargs)[1])
                        if profiling:
                            torch.cuda.synchronize()
                            phases['write_ms'] = 1000 * (time.perf_counter()-tick)
                        return value

                    def encode(*args, **kwargs):
                        if profiling:
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                        value = original_encode(*args, **kwargs)
                        if profiling:
                            torch.cuda.synchronize()
                            phases['support_total_ms'] = 1000 * (time.perf_counter()-tick)
                        return value

                    adapter.scan_final = MethodType(scan, adapter)
                    training.encode_support = encode
                    # Both modes and both concurrent workers receive identical episode streams.
                    seed = derive_seed(17, 'llm', 'writer-data')
                    spec = make_task_spec(cfg)
                    order = ('reference', 'fast') if (policy_index+batch_index) % 2 == 0 else ('fast', 'reference')
                    for method in order:
                        adapter.load_state_dict(baseline)
                        optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=.01)
                        experiment.bank._last_states.clear()
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        phase_rows = []

                        def step(number):
                            nonlocal phases
                            phases = {}
                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            arrays = generate(spec, 'train', number*batch, batch, seed)
                            optimizer.zero_grad(set_to_none=True)
                            data_done = time.perf_counter()
                            prepared = experiment.prepare(arrays)
                            if profiling:
                                torch.cuda.synchronize()
                            prepare_done = time.perf_counter()
                            logits, _ = experiment.score(prepared, 1)
                            loss = F.cross_entropy(logits.float().flatten(0, 1), prepared.answer_ids.flatten())
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite benchmark loss')
                            loss.backward()
                            if profiling:
                                torch.cuda.synchronize()
                            backward_done = time.perf_counter()
                            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                            optimizer.step()
                            torch.cuda.synchronize()
                            end = time.perf_counter()
                            if profiling:
                                phases.update(data_ms=1000*(data_done-begin),
                                    prepare_other_ms=1000*(prepare_done-data_done)-phases['support_total_ms'],
                                    support_forward_ms=phases['support_total_ms']-phases['write_ms'],
                                    question_forward_backward_ms=1000*(backward_done-prepare_done),
                                    optimizer_ms=1000*(end-backward_done), total_ms=1000*(end-begin))
                            return end-begin

                        barrier.wait(timeout=120)
                        for number in range(settings['warmup']):
                            step(number)
                            barrier.wait(timeout=120)
                        torch.cuda.reset_peak_memory_stats()
                        barrier.wait(timeout=120)
                        clock = time.perf_counter()
                        durations = []
                        for number in range(settings['repeats']):
                            barrier.wait(timeout=120)
                            durations.append(step(settings['warmup']+number))
                            barrier.wait(timeout=120)
                        wall = time.perf_counter()-clock
                        peak = torch.cuda.max_memory_allocated()
                        profiling = True
                        for number in range(settings['profile_repeats']):
                            barrier.wait(timeout=120)
                            step(settings['warmup']+settings['repeats']+number)
                            phase_rows.append(dict(phases))
                            barrier.wait(timeout=120)
                        profiling = False
                        row = dict(worker=index, procs=procs, batch=batch, policy=policy, method=method,
                            worker_mean_ms=1000*float(np.mean(durations)),
                            worker_p95_ms=1000*float(np.quantile(durations, .95)),
                            synchronized_measured_seconds=wall, peak_allocated_bytes=peak,
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            durations_seconds=durations,
                            phases_ms={k: float(np.mean([r[k] for r in phase_rows])) for k in phase_rows[0]},
                            profiled_steps=phase_rows)
                        rows.append(row)
                        save(Path(out)/f'worker-{procs}-{index}.json', dict(status='running', rows=rows,
                                                                                 loss_equivalence=equivalent))
                        if index == 0:
                            print(json.dumps({k: row[k] for k in ('procs','batch','policy','method','worker_mean_ms')}), flush=True)
                finally:
                    training.encode_support = original_encode
                    experiment.close()
        save(Path(out)/f'worker-{procs}-{index}.json', dict(status='complete', rows=rows,
            loss_equivalence=equivalent, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            host=socket.gethostname(), job_id=os.environ.get('PBS_JOBID')))
    except BaseException:
        barrier.abort()
        save(Path(out)/f'worker-{procs}-{index}.json', dict(status='failed', error=traceback.format_exc()))
        raise


def run(out, smoke):
    from audit_p4s_stopping import require_compute
    from looplora.sym.run import source_metadata
    require_compute()
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(batches=[32] if smoke else [32, 64, 128],
                    warmup=1 if smoke else 5, repeats=3 if smoke else 20, profile_repeats=1 if smoke else 3)
    record = dict(task='WP-C5', status='running', smoke=smoke, settings=settings,
        E=16, rank=32, layer=14, input_site='o_proj', policies=['fact_final','all_tokens'],
        model='Qwen/Qwen3-0.6B-Base', revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
        dtypes='bf16 frozen backbone; fp32 adapter and state',
        measurement='Unprofiled full training steps include fresh generation/rendering, support, '
                    'question forward/backward, grad clipping and AdamW; barriers excluded from worker '
                    'step times, included in aggregate achieved throughput. Separate synchronized '
                    'profile steps partition time. Question backward includes writer backward.',
        warmup_lr='constant 3e-4; same initial parameters, episode order and optimizer per method',
        started_at=datetime.now(timezone.utc).isoformat(), **source_metadata())
    save(out/'benchmark.json', record)
    context = multiprocessing.get_context('spawn')
    started = time.perf_counter()
    rows = []
    for procs in ([1] if smoke else [1, 2]):
        barrier = context.Barrier(procs)
        processes = [context.Process(target=worker,args=(i,procs,settings,str(out),barrier)) for i in range(procs)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=1000)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.exitcode != 0:
                raise RuntimeError(f'Benchmark worker failed, see {out}')
        workers = [json.loads((out/f'worker-{procs}-{i}.json').read_text()) for i in range(procs)]
        groups = {}
        for w in workers:
            for row in w['rows']:
                groups.setdefault((row['batch'],row['policy'],row['method']), []).append(row)
        for (batch,policy,method), members in groups.items():
            wall = max(r['synchronized_measured_seconds'] for r in members)
            row = dict(procs=procs,batch=batch,policy=policy,method=method,
                ms_per_step=1000*wall/settings['repeats'],
                episodes_per_second=procs*batch*settings['repeats']/wall,
                sum_peak_allocated_bytes=sum(r['peak_allocated_bytes'] for r in members),
                sum_peak_reserved_bytes=sum(r['peak_reserved_bytes'] for r in members))
            row.update({k: sum(r['phases_ms'][k] for r in members)/procs for k in members[0]['phases_ms']})
            rows.append(row)
    columns = list(rows[0])
    with (out/'timings.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    record.update(status='complete',seconds=time.perf_counter()-started,
                  finished_at=datetime.now(timezone.utc).isoformat(),cells=rows)
    save(out/'benchmark.json',record)
    print(json.dumps(dict(status='complete', seconds=record['seconds'], out=str(out))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    run(args.out,args.smoke)
