"""WP-L1 writer throughput grid; execute only in a Miyabi-G allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import socket
import time
import traceback


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _worker(index: int, batch: int, out: str, warmup: int, measured: int, ready, queue) -> None:
    experiment = None
    try:
        import numpy as np
        import torch
        from looplora.llm.pilot1_select import writer_config
        from looplora.llm.train import LLMExperiment, _fit

        config = writer_config(f'throughput-{index}', 24, 'all_tokens', .001, batch, warmup + measured)
        config['train']['log_every'] = 1
        output = Path(out)
        output.mkdir()
        experiment = LLMExperiment(config)
        torch.cuda.reset_peak_memory_stats()
        ready.wait(timeout=120)
        start = time.perf_counter()
        summary = _fit(experiment, output)
        elapsed = time.perf_counter() - start
        rows = [json.loads(line) for line in (output/'train_log.jsonl').read_text().splitlines()]
        durations = [row['step_seconds'] for row in rows[warmup:]]
        if len(durations) != measured:
            raise RuntimeError('benchmark did not complete the requested steps')
        free, total = torch.cuda.mem_get_info()
        queue.put(dict(status='complete', index=index, measured_steps=measured,
            measured_seconds=sum(durations), p95_step_seconds=float(np.quantile(durations, .95)),
            mean_step_seconds=float(np.mean(durations)),
            episodes_per_second=batch*measured/sum(durations),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), device_free_bytes=free,
            device_total_bytes=total, training_seconds=elapsed, train_tokens=summary['train_tokens'],
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            threads=torch.get_num_threads(), pid=os.getpid()))
    except BaseException as error:
        try:
            import torch
            oom = isinstance(error, torch.cuda.OutOfMemoryError)
        except ImportError:
            oom = False
        ready.abort()
        queue.put(dict(status='oom' if oom else 'failed', index=index, error=traceback.format_exc()))
    finally:
        if experiment is not None:
            experiment.close()


def _cell(batch: int, procs: int, out: Path, warmup: int, measured: int) -> dict:
    out.mkdir()
    context = multiprocessing.get_context('spawn')
    ready, queue = context.Barrier(procs), context.Queue()
    processes = [context.Process(target=_worker,
        args=(i,batch,str(out/f'worker-{i}'),warmup,measured,ready,queue)) for i in range(procs)]
    start = time.perf_counter()
    for process in processes:
        process.start()
    workers = []
    try:
        for _ in processes:
            try:
                workers.append(queue.get(timeout=240))
            except Empty:
                raise RuntimeError('throughput worker timed out') from None
    finally:
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        queue.close()
    workers.sort(key=lambda row: row['index'])
    row = dict(E=24, write_policy='all_tokens', batch_size=batch, procs=procs,
               workers=workers, total_seconds=time.perf_counter()-start)
    if any(worker['status']=='failed' for worker in workers) and not any(
        worker['status']=='oom' for worker in workers
    ):
        _write(out/'cell.json', row)
        raise RuntimeError(f'non-memory benchmark failure: {workers}')
    row['status'] = 'oom' if any(worker['status']=='oom' for worker in workers) else 'complete'
    if row['status']=='complete':
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError('throughput worker exited unsuccessfully after reporting')
        row.update(episodes_per_second=sum(worker['episodes_per_second'] for worker in workers),
            worst_worker_p95_step_seconds=max(worker['p95_step_seconds'] for worker in workers),
            sum_peak_allocated_bytes=sum(worker['peak_allocated_bytes'] for worker in workers),
            sum_peak_reserved_bytes=sum(worker['peak_reserved_bytes'] for worker in workers))
    _write(out/'cell.json', row)
    return row


def run(out: Path, warmup: int = 5, measured: int = 25) -> dict:
    from looplora.llm.run import source_metadata
    from looplora.llm.pilot1_select import choose_steps

    out.mkdir(parents=True, exist_ok=False)
    record = dict(seed=17, warmup_steps=warmup, measured_steps=measured, cells=[],
        selection_rule='highest sum of simultaneous worker training episodes/s among fitting cells',
        scope='writer only; E24/all_tokens worst pilot load; no accuracy-based choice',
        started_at=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
        job_id=os.environ.get('PBS_JOBID'), **source_metadata())
    start = time.perf_counter()
    for procs in (1,2):
        for batch in (8,16,32):
            row = _cell(batch,procs,out/f'batch-{batch}-procs-{procs}',warmup,measured)
            record['cells'].append(row)
            _write(out/'throughput.json',record)
            print(json.dumps({key:row[key] for key in ('batch_size','procs','status','total_seconds')}),flush=True)
    fitting = [row for row in record['cells'] if row['status']=='complete']
    if not fitting:
        raise RuntimeError('no writer batch/process setting fits memory')
    record['selected'] = max(fitting,key=lambda row:(row['episodes_per_second'],-row['procs'],-row['batch_size']))
    record['steps'] = choose_steps(record['selected'])
    record['training_episodes_per_run'] = record['steps']*record['selected']['batch_size']
    record['seconds'] = time.perf_counter()-start
    record['finished_at'] = datetime.now(timezone.utc).isoformat()
    _write(out/'throughput.json',record)
    print(json.dumps(dict(selected_batch=record['selected']['batch_size'],
                         selected_procs=record['selected']['procs'],steps=record['steps'])),flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    run(args.out)


if __name__=='__main__':
    main()
