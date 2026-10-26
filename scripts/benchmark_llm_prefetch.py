#!/usr/bin/env python3
"""WP-B4 training throughput sweep; run via submit.py on Miyabi-G compute."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import time
import traceback


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def worker(index, procs, settings, out, barrier, launched):
    """Keep one backbone resident while comparing fresh, identical optimizers."""
    os.setsid()  # The controller can stop this process and its prefetch children.
    path = Path(out) / f'worker-{procs}-{index}.json'
    record = dict(status='starting', worker=index, procs=procs, rows=[])
    try:
        from audit_p4s_stopping import require_compute
        require_compute()
        import numpy as np
        import torch
        from torch.nn import functional as F
        from looplora.infra.seeds import derive_seed
        from looplora.llm.inject import load_model, load_tokenizer
        from looplora.llm.pilot1_select import writer_config
        from looplora.llm.prefetch import BatchStream
        from looplora.llm.train import LLMExperiment, make_task_spec

        torch.set_num_threads(1)
        tick = time.perf_counter()
        model, tokenizer = load_model(), load_tokenizer()
        torch.cuda.synchronize()
        record.update(status='running', model_load_seconds=time.perf_counter()-tick,
                      process_ready_seconds=time.perf_counter()-launched,
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      host=socket.gethostname(), job_id=os.environ.get('PBS_JOBID'))
        save(path, record)
        seed = derive_seed(17, 'llm', 'writer-data')

        for policy_index, policy in enumerate(settings['policies']):
            for batch_index, batch in enumerate(settings['batches']):
                cfg = writer_config('llm-prefetch-benchmark', 16, policy, .0003, 32, 1)
                cfg['train']['batch_size'] = batch
                cfg['model']['input_site'] = 'o_proj'
                experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
                try:
                    parameters = experiment.trainable_parameters()
                    baseline = {k: v.detach().clone() for k, v in experiment.adapter.state_dict().items()}
                    spec = make_task_spec(cfg)
                    # Alternate order without changing the data or parameter initialization.
                    order = settings['workers']
                    if (policy_index + batch_index) % 2:
                        order = list(reversed(order))
                    for workers in order:
                        experiment.adapter.load_state_dict(baseline)
                        experiment.bank._last_states.clear()
                        optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=.01)
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        total = settings['warmup'] + settings['repeats']
                        stream = BatchStream(spec, 'train', batch, seed, total,
                                             tokenizer=tokenizer, workers=workers)

                        def step(item=None):
                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            arrays, tokens = next(stream) if item is None else item
                            data_done = time.perf_counter()
                            optimizer.zero_grad(set_to_none=True)
                            prepared = experiment.prepare(arrays, tokens=tokens)
                            logits, _ = experiment.score(prepared, 1)
                            loss = F.cross_entropy(logits.float().flatten(0, 1),
                                                   prepared.answer_ids.flatten())
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite benchmark loss')
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                            optimizer.step()
                            torch.cuda.synchronize()
                            duration = time.perf_counter()-begin
                            return duration, data_done-begin, float(loss.detach())

                        barrier.wait(timeout=120)
                        startup = time.perf_counter()
                        with stream:
                            entered = time.perf_counter()
                            first = next(stream)
                            first_ready = time.perf_counter()
                            first_step_seconds, _, first_loss = step(first)
                            losses = [first_loss]
                            del first
                            barrier.wait(timeout=120)
                            for _ in range(1, settings['warmup']):
                                _, _, loss_value = step()
                                losses.append(loss_value)
                                barrier.wait(timeout=120)
                            torch.cuda.reset_peak_memory_stats()
                            barrier.wait(timeout=120)
                            measured = time.perf_counter()
                            durations, waits = [], []
                            for _ in range(settings['repeats']):
                                barrier.wait(timeout=120)
                                duration, waiting, loss_value = step()
                                durations.append(duration)
                                waits.append(waiting)
                                losses.append(loss_value)
                                barrier.wait(timeout=120)
                            wall = time.perf_counter()-measured
                            peak_allocated = torch.cuda.max_memory_allocated()
                            peak_reserved = torch.cuda.max_memory_reserved()
                            closing = time.perf_counter()
                        close_seconds = time.perf_counter()-closing
                        row = dict(worker=index, procs=procs, batch=batch, policy=policy,
                                   workers=workers, worker_mean_ms=1000*float(np.mean(durations)),
                                   worker_p95_ms=1000*float(np.quantile(durations, .95)),
                                   worker_data_wait_mean_ms=1000*float(np.mean(waits)),
                                   stream_enter_seconds=entered-startup,
                                   first_batch_wait_seconds=first_ready-entered,
                                   time_to_first_batch_seconds=first_ready-startup,
                                   first_step_seconds=first_step_seconds,
                                   time_to_first_step_seconds=first_ready-startup+first_step_seconds,
                                   close_seconds=close_seconds,
                                   cell_seconds=time.perf_counter()-startup,
                                   synchronized_measured_seconds=wall,
                                   peak_allocated_bytes=peak_allocated,
                                   peak_reserved_bytes=peak_reserved,
                                   durations_seconds=durations, data_wait_seconds=waits,
                                   losses=losses)
                        record['rows'].append(row)
                        save(path, record)
                        if index == 0:
                            print(json.dumps({k: row[k] for k in
                                  ('procs', 'batch', 'policy', 'workers', 'worker_mean_ms',
                                   'time_to_first_batch_seconds', 'close_seconds')}), flush=True)
                        # Close has completed before a new cell can create another pool.
                        barrier.wait(timeout=120)
                finally:
                    experiment.close()
        baselines = {(r['batch'], r['policy']): r for r in record['rows'] if r['workers'] == 0}
        for row in record['rows']:
            reference = baselines.get((row['batch'], row['policy']))
            row['loss_max_absolute_error_vs_workers0'] = None
            row['loss_max_relative_error_vs_workers0'] = None
            if reference is not None:
                observed = np.asarray(row['losses'])
                expected = np.asarray(reference['losses'])
                row['loss_max_absolute_error_vs_workers0'] = float(np.max(np.abs(observed-expected)))
                row['loss_max_relative_error_vs_workers0'] = float(np.max(np.abs(observed-expected)/np.abs(expected)))
                np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=0,
                                           err_msg='Prefetch changed the training loss trajectory')
        record['status'] = 'complete'
        save(path, record)
    except BaseException:
        barrier.abort()
        record.update(status='failed', error=traceback.format_exc())
        save(path, record)
        raise


def stop_processes(processes):
    """Stop entire training process groups, including their prefetch workers."""
    for process in processes:
        if process.pid is None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
    deadline = time.perf_counter() + 3
    for process in processes:
        if process.pid is not None:
            process.join(timeout=max(0, deadline-time.perf_counter()))
    for process in processes:
        if process.pid is None:
            continue
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
        process.join(timeout=3)


def aggregate(workers, repeats):
    groups = {}
    for worker_record in workers:
        for row in worker_record['rows']:
            groups.setdefault((row['procs'], row['batch'], row['policy'], row['workers']), []).append(row)
    rows = []
    for (procs, batch, policy, count), members in groups.items():
        if len(members) != procs or {r['worker'] for r in members} != set(range(procs)):
            raise ValueError('Missing training worker in benchmark cell')
        if any(len(r['durations_seconds']) != repeats or len(r['data_wait_seconds']) != repeats
               or any(not math.isfinite(v) or v <= 0 for v in r['durations_seconds'])
               for r in members):
            raise ValueError('Missing or invalid measured benchmark steps')
        wall = max(row['synchronized_measured_seconds'] for row in members)
        row = dict(procs=procs, batch=batch, policy=policy, workers=count,
                   ms_per_step=1000*wall/repeats,
                   episodes_per_second=procs*batch*repeats/wall,
                   worker_mean_ms=sum(r['worker_mean_ms'] for r in members)/procs,
                   worst_worker_p95_ms=max(r['worker_p95_ms'] for r in members),
                   worker_data_wait_mean_ms=sum(r['worker_data_wait_mean_ms'] for r in members)/procs,
                   sum_peak_allocated_bytes=sum(r['peak_allocated_bytes'] for r in members),
                   sum_peak_reserved_bytes=sum(r['peak_reserved_bytes'] for r in members))
        for key in ('stream_enter_seconds', 'first_batch_wait_seconds',
                    'time_to_first_batch_seconds', 'first_step_seconds',
                    'time_to_first_step_seconds', 'close_seconds', 'cell_seconds'):
            row['max_' + key] = max(r[key] for r in members)
        for key in ('loss_max_absolute_error_vs_workers0', 'loss_max_relative_error_vs_workers0'):
            row[key] = (max(r[key] for r in members)
                        if all(r[key] is not None for r in members) else None)
        if any(not math.isfinite(value) or value < 0
               for value in row.values() if isinstance(value, (int, float))):
            raise ValueError('Nonfinite or negative benchmark summary metric')
        rows.append(row)
    baselines = {(r['procs'], r['batch'], r['policy']): r for r in rows if r['workers'] == 0}
    for row in rows:
        baseline = baselines.get((row['procs'], row['batch'], row['policy']))
        row['speedup_vs_workers0'] = (baseline['ms_per_step']/row['ms_per_step']
                                     if baseline else None)
    return rows


def write_tables(path, rows):
    lines = ['# WP-B4 prefetch benchmark', '',
             'Batch and prefetch worker counts are per training process. '
             'Throughput includes process barriers; startup and close are separate. '
             'GPU memory is the sum of per-process peaks.', '',
             '| Processes | Batch | Policy | Workers | ms/step | episodes/s | Speedup | '
             'First batch (s) | Close (s) | Peak allocated (GiB) |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in sorted(rows, key=lambda r: (r['procs'], r['batch'], r['policy'], r['workers'])):
        speedup = row['speedup_vs_workers0']
        lines.append(f"| {row['procs']} | {row['batch']} | {row['policy']} | {row['workers']} | "
                     f"{row['ms_per_step']:.2f} | {row['episodes_per_second']:.2f} | "
                     f"{format(speedup, '.3f') if speedup is not None else 'n/a'} | "
                     f"{row['max_time_to_first_batch_seconds']:.3f} | "
                     f"{row['max_close_seconds']:.3f} | "
                     f"{row['sum_peak_allocated_bytes']/2**30:.3f} |")
    path.write_text('\n'.join(lines) + '\n')


def estimate_full(rows, groups):
    """Use smoke's slowest batch/pool per policy and process count as a bound.

    This is a planning estimate, not a measured full-sweep result. It assumes
    smaller batches and worker counts do not exceed the sampled worst case.
    """
    seconds = sum(group['seconds']-group['cells_seconds'] for group in groups)
    for procs in (1, 2):
        for policy in ('fact_final', 'all_tokens'):
            matching = [r for r in rows if r['procs'] == procs and r['policy'] == policy]
            if not matching:
                return None
            for enabled, multiplicity in ((False, 1), (True, 3)):
                samples = [r for r in matching if bool(r['workers']) == enabled]
                if not samples:
                    return None
                step_seconds = max(r['ms_per_step'] for r in samples)/1000
                startup = max(r['max_time_to_first_batch_seconds'] for r in samples)
                close = max(r['max_close_seconds'] for r in samples)
                seconds += 3*multiplicity*(25*step_seconds+startup+close)
    return dict(estimated_seconds=seconds, estimated_node_hours=seconds/3600,
                with_25_percent_margin_seconds=1.25*seconds,
                assumption='Smaller batches and worker counts cost no more than sampled maxima; '
                           'model/process startup retained once per process-count group. '
                           'Confirm against remaining cumulative node-hour budget before submission.')


def run(args):
    from audit_p4s_stopping import require_compute
    require_compute()
    from looplora.sym.run import source_metadata

    started = time.perf_counter()
    settings = dict(batches=args.batches or ([128] if args.smoke else [32, 64, 128]),
                    procs=args.procs or [1, 2],
                    workers=args.workers or ([0, 8] if args.smoke else [0, 2, 4, 8]),
                    policies=['fact_final', 'all_tokens'],
                    # Eight total smoke batches ensure workers=8 really starts eight workers.
                    warmup=5, repeats=3 if args.smoke else 20)
    args.out.mkdir(parents=True, exist_ok=False)
    record = dict(task='WP-B4', status='running', smoke=args.smoke, settings=settings,
                  E=16, Q=4, rank=32, layer=14, input_site='o_proj', seed=17,
                  model='Qwen/Qwen3-0.6B-Base',
                  revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
                  dtypes='bf16 frozen backbone; fp32 adapter and state',
                  measurement='Fresh train episodes in exact step order. Worker times include batch '
                              'wait, preparation, support, question forward/backward, gradient clipping '
                              'and AdamW. Barriers are excluded from worker times and included in '
                              'aggregate throughput. Startup, first batch/step and close are separate. '
                              'Memory sums are sums of per-process peaks, not simultaneous readings.',
                  comparison='Same initial parameters, AdamW, constant lr=3e-4 and episode stream '
                             'for every worker setting; no evaluation or checkpoints.',
                  max_seconds=args.max_seconds,
                  started_at=datetime.now(timezone.utc).isoformat(), **source_metadata(),
                  groups=[], cells=[])
    save(args.out/'benchmark.json', record)
    context = multiprocessing.get_context('spawn')
    try:
        for procs in settings['procs']:
            group_start = time.perf_counter()
            barrier = context.Barrier(procs)
            processes = [context.Process(target=worker,
                         args=(i, procs, settings, str(args.out), barrier, group_start))
                         for i in range(procs)]
            try:
                for process in processes:
                    process.start()
                while any(process.is_alive() for process in processes):
                    if time.perf_counter()-started > args.max_seconds:
                        raise TimeoutError('Benchmark runtime budget reached')
                    if any(process.exitcode not in (None, 0) for process in processes):
                        raise RuntimeError(f'Benchmark worker failed; see {args.out}')
                    for process in processes:
                        process.join(timeout=.1)
                if any(process.exitcode != 0 for process in processes):
                    raise RuntimeError(f'Benchmark worker failed; see {args.out}')
            except BaseException:
                barrier.abort()
                stop_processes(processes)
                raise
            workers = [json.loads((args.out/f'worker-{procs}-{i}.json').read_text())
                       for i in range(procs)]
            if any(w['status'] != 'complete' for w in workers):
                raise ValueError('A training worker did not complete its sweep')
            rows = aggregate(workers, settings['repeats'])
            expected = {(procs, b, p, n) for b in settings['batches']
                        for p in settings['policies'] for n in settings['workers']}
            if {(r['procs'], r['batch'], r['policy'], r['workers']) for r in rows} != expected:
                raise ValueError('Benchmark cells do not match requested matrix')
            group_seconds = time.perf_counter()-group_start
            cells_seconds = sum(r['max_cell_seconds'] for r in rows)
            record['groups'].append(dict(procs=procs, seconds=group_seconds,
                                         cells_seconds=min(cells_seconds, group_seconds),
                                         process_ready_seconds=max(w['process_ready_seconds'] for w in workers),
                                         model_load_seconds=[w['model_load_seconds'] for w in workers]))
            record['cells'].extend(rows)
            save(args.out/'benchmark.json', record)
        write_tables(args.out/'tables.md', record['cells'])
        with (args.out/'timings.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record['cells'][0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(record['cells'])
        record.update(status='complete', seconds=time.perf_counter()-started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        record['node_hours'] = record['seconds']/3600
        if args.smoke:
            record['full_sweep_estimate'] = estimate_full(record['cells'], record['groups'])
        save(args.out/'benchmark.json', record)
        print(json.dumps(dict(status='complete', seconds=record['seconds'], out=str(args.out),
                              full_sweep_estimate=record.get('full_sweep_estimate'))), flush=True)
    except BaseException:
        record.update(status='failed', seconds=time.perf_counter()-started,
                      finished_at=datetime.now(timezone.utc).isoformat(), error=traceback.format_exc())
        save(args.out/'benchmark.json', record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batches', nargs='+', type=int, choices=(32, 64, 128))
    parser.add_argument('--procs', nargs='+', type=int, choices=(1, 2))
    parser.add_argument('--workers', nargs='+', type=int, choices=(0, 2, 4, 8))
    parser.add_argument('--max-seconds', type=float, default=1000,
                        help='Whole benchmark runtime cap, excluding final process cleanup (default: 1000)')
    options = parser.parse_args()
    if not 0 < options.max_seconds <= 1800:
        parser.error('--max-seconds must be in (0, 1800]')
    for name in ('batches', 'procs', 'workers'):
        values = getattr(options, name)
        if values is not None and len(set(values)) != len(values):
            parser.error(f'--{name} must not repeat values')
    run(options)
