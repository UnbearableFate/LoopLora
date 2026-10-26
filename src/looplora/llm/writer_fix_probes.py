"""WP-L3 pre-tail identity probes, before the factorized writer screen."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from looplora.data.relations import generate
from looplora.data.render import render_episodes
from looplora.infra.seeds import derive_seed
from .diagnosis_probes import _pad,fit_probe
from .evaluate import _slice
from .inject import load_model,load_tokenizer
from .run import _write_json,source_metadata
from .train import make_task_spec
from .writer_fix import fact_positions
from .writer_fix_run import configuration


@torch.no_grad()
def extract(model,tokenizer,split,episodes):
    spec=make_task_spec(configuration('W4'))
    seed=derive_seed(17,'llm','writer-data') if split=='train' else 4242
    arrays=generate(spec,split,0,episodes,seed)
    features,labels,captured=[],[],[]
    handle=model.model.layers[14].self_attn.o_proj.register_forward_pre_hook(
        lambda _,args:captured.append(args[0]))
    try:
        for begin in range(0,episodes,64):
            part=_slice(arrays,slice(begin,min(begin+64,episodes)))
            rendered=render_episodes(part,tokenizer=tokenizer)
            positions=fact_positions(part,rendered,model.device)
            ids,mask=_pad([r.context_ids for r in rendered],tokenizer.pad_token_id,model.device)
            model.model(input_ids=ids,attention_mask=mask,use_cache=False)
            x=captured.pop()
            index=positions[:,:,1]
            features.append(x.gather(1,index[...,None].expand(-1,-1,x.shape[-1])).flatten(0,1).cpu())
            labels.append(torch.from_numpy(part.facts[:,:,:2].copy()).flatten(0,1).long())
    finally:
        handle.remove()
    return torch.cat(features),torch.cat(labels)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    model,tokenizer=load_model(),load_tokenizer()
    x,y=extract(model,tokenizer,'train',4096)
    vx,vy=extract(model,tokenizer,'val',1000)
    del model
    torch.cuda.empty_cache()
    rows=[]
    for column,target in enumerate(('head','relation')):
        for rank in (None,32):
            row,prediction=fit_probe(x,y[:,column],vx,vy[:,column],classes=1024 if column==0 else 4,
                rank=rank,purpose=f'o14/pre_tail/{target}',steps=1200)
            row.update(site='o14',position='pre_tail',target=target)
            rows.append(row)
            np.savez_compressed(out/f'{target}-{rank or "full"}.npz',prediction=prediction,answer=vy[:,column].numpy())
    fields=['site','position','target','rank','classes','dimensions','train_n','train_accuracy','n','correct','accuracy']
    with (out/'probes.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n')
        writer.writeheader(); writer.writerows({key:r[key] for key in fields} for r in rows)
    summary=dict(seed=17,eval_seed=4242,train_episodes=4096,val_episodes=1000,
        frozen_features=True,adapter=False,backend='core.scan_final',**source_metadata(),
        host=socket.gethostname(),job_id=os.environ.get('PBS_JOBID'),seconds=time.perf_counter()-started,rows=rows)
    _write_json(out/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
