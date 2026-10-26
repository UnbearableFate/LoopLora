#!/usr/bin/env python3
"""WP-C7 train-only oracle acceptance smoke and batch-64 step timings."""
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
from unittest.mock import patch

import torch

from audit_p4s_stopping import require_compute
from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm.config import resolve_config
from looplora.llm.inject import load_model, load_tokenizer, MODEL_ID, MODEL_REVISION
from looplora.llm.oracle import OracleRandomWriter
from looplora.llm.train import train
from looplora.sym.evaluate import tensor_hash
from looplora.sym.run import source_metadata


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def run(out, trial=False, summary_out=None):
    require_compute()
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    steps=10 if trial else 100
    warmup=5
    record=dict(task="WP-C7",status="running",trial=trial,
        settings=dict(writer="oracle_random",oracle_seed=17,variants=["M1-L4","B8","B4"],
            depths=[4,4,1],batch_size=64,steps=steps,warmup=warmup,processes=1,prefetch_workers=2,
            stage="reader",protocol="memory_only",r=32,layer=14,input_site="o_proj",E=12,Q=4,
            n_rel=4,head_sharing=.3,chain_lengths=[8,4],train_hops=[1,2,3]),
        model=MODEL_ID,revision=MODEL_REVISION,dtypes="bf16 backbone, fp32 adapter/state/codes",
        measurement="CUDA-synchronized production train() steps, including batch wait, fact-only oracle "
                    "write, question forward/backward, clipping and AdamW. Five warmups discarded. "
                    "Cell time also includes prefetch startup/shutdown and checkpoint export.",
        science_scope="Acceptance smoke only, train split and development seed; no evaluation or selection.",
        started_at=datetime.now(timezone.utc).isoformat(),**source_metadata(),
        runtime=dict(host=socket.gethostname(),job_id=os.environ["PBS_JOBID"],torch=torch.__version__),cells=[])
    save(out/"benchmark.json",record)
    started=time.perf_counter()
    model,tokenizer=load_model(),load_tokenizer()
    record["runtime"]["gpu"]=torch.cuda.get_device_name()
    original=OracleRandomWriter.forward
    for label,variant,depth in (("M1-L4","M1",4),("B8","B8",4),("B4","B4",1)):
        cfg=resolve_config(dict(name=f"c7-{label}",writer="oracle_random",oracle_seed=17,
            stage="reader",variant=variant,smoke=True,seed=derive_seed(20261027,"wp-c7-acceptance"),
            model=dict(r=32,layer=14,input_site="o_proj"),
            task=dict(name="t2_chain_end",E=12,Q=4,n_rel=4,head_sharing=.3,chain_lengths=[8,4]),
            train=dict(steps=steps,batch_size=64,depths=[depth],log_every=1,prefetch_workers=2),
            eval=dict(episodes=2,counterfactuals=False)))
        count=0
        initial_hash=None

        def writer_forward(writer,*args,**kwargs):
            nonlocal count,initial_hash
            count+=1
            if initial_hash is None:
                initial_hash=stable_hash({k:tensor_hash(v) for k,v in writer.state_dict().items()})
            state=original(writer,*args,**kwargs)
            assert not state.requires_grad
            return state

        experiment=None
        cell_start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            with patch.object(OracleRandomWriter,"forward",writer_forward), \
                 patch("looplora.llm.train.encode_support",side_effect=AssertionError("support LM used")), \
                 patch("looplora.core.memory.scan_final",side_effect=AssertionError("learned writer used")):
                experiment,summary=train(cfg,out/label,model=model,tokenizer=tokenizer)
            logs=[json.loads(line) for line in (out/label/"train_log.jsonl").read_text().splitlines()]
            assert len(logs)==steps and {row["L"] for row in logs}=={depth} and count==steps
            assert summary["writer_frozen"] and summary["train_tokens"]>0
            code_hash=stable_hash({k:tensor_hash(v) for k,v in experiment.writer_state().items()})
            assert initial_hash==code_hash and all(b.grad is None for b in experiment.writer.buffers())
            gradients={}
            for name,p in experiment.adapter.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None
                    continue
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0, name
                gradients[name]=float(p.grad.norm())
            durations=[row["step_seconds"] for row in logs[warmup:]]
            row=dict(variant=label,L=depth,batch_size=64,steps=steps,measured_steps=len(durations),
                mean_step_ms=1000*statistics.mean(durations),median_step_ms=1000*statistics.median(durations),
                p95_step_ms=1000*sorted(durations)[int(.95*(len(durations)-1))],
                episodes_per_second=64/statistics.mean(durations),step_seconds=durations,
                writer_signature=experiment.writer_signature(),code_hash=code_hash,
                oracle_writes=count,learned_scan_final_calls=0,support_lm_calls=0,
                final_loss=summary["final_loss"],gradient_norms=gradients,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                train_seconds=summary["train_seconds"],cell_seconds=time.perf_counter()-cell_start)
            record["cells"].append(row)
            save(out/"benchmark.json",record)
            print(json.dumps({k:row[k] for k in ("variant","steps","mean_step_ms","code_hash")}),flush=True)
        finally:
            if experiment is not None: experiment.close()
    assert len({row["code_hash"] for row in record["cells"]})==1
    record.update(status="complete",seconds=time.perf_counter()-started,
                  finished_at=datetime.now(timezone.utc).isoformat())
    save(out/"benchmark.json",record)
    columns=["variant","L","batch_size","steps","measured_steps","mean_step_ms","median_step_ms",
             "p95_step_ms","episodes_per_second","peak_allocated_bytes","train_seconds","cell_seconds"]
    with (out/"timings.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore",lineterminator="\n")
        writer.writeheader()
        writer.writerows(record["cells"])
    lines=["# WP-C7 oracle writer training smoke", "",
        f"{steps} train-only steps per variant, batch 64, one GPU process, two CPU prefetch workers; "
        "five warmups discarded. Pinned Qwen3-0.6B, layer14/o_proj, r32, E12/Q4, chains (8,4), "
        "four relations, head sharing 0.3. Full-vocabulary CE; no support text/LM forward or evaluation.", "",
        "| Reader | L | Mean ms/step | Median ms/step | Episodes/s | Peak GiB |",
        "| --- | --- | --- | --- | --- | --- |"]
    for row in record["cells"]:
        lines.append(f"| {row['variant']} | {row['L']} | {row['mean_step_ms']:.2f} | "
                     f"{row['median_step_ms']:.2f} | {row['episodes_per_second']:.2f} | "
                     f"{row['peak_allocated_bytes']/2**30:.3f} |")
    (out/"tables.md").write_text("\n".join(lines)+"\n")
    if summary_out is not None:
        summary_out.mkdir(parents=True,exist_ok=True)
        for name in ("benchmark.json","timings.csv","tables.md"):
            shutil.copy2(out/name,summary_out/name)
    print(json.dumps(dict(status="complete",seconds=record["seconds"],out=str(out))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--summary-out",type=Path)
    parser.add_argument("--trial",action="store_true",help="10-step timing preflight before the 100-step acceptance smoke")
    args=parser.parse_args()
    run(args.out,args.trial,args.summary_out)
