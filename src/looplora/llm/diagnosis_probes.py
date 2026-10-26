"""WP-L2 frozen-feature linear probes; run only on Miyabi-G compute."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.data.render import load_name_pool, relation_phrase, render_episodes
from looplora.infra.seeds import derive_seed
from looplora.sym.evaluate import stratify_queries
from .inject import load_model, load_tokenizer
from .pilot1_select import writer_config
from .run import _write_json, source_metadata
from .train import make_task_spec, synchronize

SITES = ("o14", "res14", "res7", "res21")
POSITIONS = ("fact_final", "tail_token", "question_final", "question_start", "question_relation")


def span_token(offsets, start: int, end: int, *, single=False) -> int:
    """Last token overlapping a known character span, excluding surrounding punctuation."""
    hits = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not hits or (single and len(hits) != 1):
        raise ValueError(f"expected {'one' if single else 'some'} token(s) at {start}:{end}")
    return hits[-1]


def positions(arrays, rendered, tokenizer):
    pool = load_name_pool()
    support, questions, write_labels, read_labels = [], [], [], []
    for b, row in enumerate(rendered):
        encoded = tokenizer(row.context_text, add_special_tokens=False, return_offsets_mapping=True)
        if tuple(encoded['input_ids']) != row.context_ids:
            raise AssertionError('support offsets must use the actual rendered tokens')
        for span in row.fact_spans:
            head, rel, tail = map(int, arrays.facts[b, span.fact_index])
            tail_end = span.char_end - 2  # every fact ends in period/newline
            tail_position = span_token(encoded['offset_mapping'], tail_end-len(pool.names[tail]), tail_end, single=True)
            if row.context_ids[tail_position] != pool.token_ids[tail]:
                raise AssertionError('tail position is not the recorded single name token')
            support.append((b, span.token_end-1, tail_position))
            write_labels.append((tail, head, rel))
        for q, query in enumerate(row.queries):
            encoded = tokenizer(query.question_text, add_special_tokens=False, return_offsets_mapping=True)
            if tuple(encoded['input_ids']) != query.question_ids:
                raise AssertionError('question offsets must use the actual rendered tokens')
            head, rel = int(arrays.q_start[b,q]), int(arrays.q_rel[b,q])
            start = query.question_text.index('from ') + len('from ')
            rstart = query.question_text.index("relation '") + len("relation '")
            hpos = span_token(encoded['offset_mapping'], start, start+len(pool.names[head]), single=True)
            rpos = span_token(encoded['offset_mapping'], rstart, rstart+len(relation_phrase(rel)))
            if query.question_ids[hpos] != pool.token_ids[head]:
                raise AssertionError('question start position is not the recorded name token')
            questions.append((b*len(row.queries)+q, len(query.question_ids)-1, hpos, rpos))
            read_labels.append((head, rel))
    return (torch.tensor(support), torch.tensor(questions),
            torch.tensor(write_labels), torch.tensor(read_labels))


def _pad(rows, pad: int, device):
    length = max(map(len, rows))
    ids = torch.full((len(rows), length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i,:len(row)] = torch.tensor(row, device=device)
        mask[i,:len(row)] = True
    return ids, mask


@torch.no_grad()
def extract(model, tokenizer, split: str, episodes: int, *, batch_size=64):
    if hasattr(model, 'looplora_adapters'):
        raise ValueError('D1 requires frozen features without any adapter')
    config = writer_config('D1',16,'fact_final',.0003,32,1500)
    spec = make_task_spec(config)
    seed = derive_seed(17,'llm','writer-data') if split=='train' else 4242
    arrays = generate(spec,split,0,episodes,seed)
    if split=='val':
        arrays = stratify_queries(arrays,spec,[1],seed)
    chunks = {(site,position):[] for site in SITES for position in POSITIONS}
    labels = {'write':[], 'read':[]}
    captured, hooks = {}, []
    def hook(site):
        return lambda _, args: captured.__setitem__(site,args[0])
    hooks.append(model.model.layers[14].self_attn.o_proj.register_forward_pre_hook(hook('o14')))
    for layer in (7,14,21):
        hooks.append(model.model.layers[layer].register_forward_pre_hook(hook(f'res{layer}')))
    try:
        from .evaluate import _slice
        for begin in range(0,episodes,batch_size):
            part = _slice(arrays,slice(begin,min(begin+batch_size,episodes)))
            rendered = render_episodes(part,tokenizer=tokenizer)
            widx,ridx,wy,ry = positions(part,rendered,tokenizer)
            labels['write'].append(wy)
            labels['read'].append(ry)
            ids,mask = _pad([row.context_ids for row in rendered],tokenizer.pad_token_id,model.device)
            model.model(input_ids=ids,attention_mask=mask,use_cache=False)
            for site in SITES:
                for col,position in enumerate(POSITIONS[:2],1):
                    chunks[site,position].append(captured[site][widx[:,0],widx[:,col]].cpu())
            captured.clear()
            ids,mask = _pad([q.question_ids for row in rendered for q in row.queries],
                           tokenizer.pad_token_id,model.device)
            # Fresh positions and no support KV: exactly memory-only question features.
            model.model(input_ids=ids,attention_mask=mask,use_cache=False)
            for site in SITES:
                for col,position in enumerate(POSITIONS[2:],1):
                    chunks[site,position].append(captured[site][ridx[:,0],ridx[:,col]].cpu())
            captured.clear()
            if begin % 1024 == 0:
                print(f'D1 features {split}: {min(begin+batch_size,episodes)}/{episodes}',flush=True)
    finally:
        for handle in hooks:
            handle.remove()
    return ({key:torch.cat(value) for key,value in chunks.items()},
            {key:torch.cat(value) for key,value in labels.items()})


class LinearProbe(nn.Module):
    def __init__(self, dimensions: int, classes: int, rank: int | None):
        super().__init__()
        self.map = (nn.Linear(dimensions,classes) if rank is None else
                    nn.Sequential(nn.Linear(dimensions,rank,bias=False),nn.Linear(rank,classes)))

    def forward(self, features):
        return self.map(features)


def fit_probe(x, labels, vx, vlabels, *, classes: int, rank: int | None,
              purpose: str, steps=1200, batch_size=512, device='cuda'):
    seed = derive_seed(17,'D1-probe',purpose,'full' if rank is None else rank)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        probe = LinearProbe(x.shape[1],classes,rank).to(device)
    x,labels,vx,vlabels = x.float().to(device),labels.to(device),vx.float().to(device),vlabels.to(device)
    mean,scale = x.mean(0),x.std(0).clamp_min(1e-4)
    x,vx = (x-mean)/scale,(vx-mean)/scale
    optimizer = torch.optim.AdamW(probe.parameters(),lr=.01,weight_decay=0.)
    generator = torch.Generator(device=device).manual_seed(derive_seed(seed,'batches'))
    trace = []
    for step in range(steps):
        indices = torch.randint(len(x),(batch_size,),generator=generator,device=device)
        rate = .01*(.1+.9*(1+math.cos(math.pi*step/steps))/2)
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x[indices])
        loss = F.cross_entropy(logits,labels[indices])
        loss.backward()
        optimizer.step()
        if step==0 or (step+1)%100==0:
            trace.append(dict(step=step+1,loss=float(loss.detach())))
    def accuracy(features,targets):
        predictions = torch.cat([probe(part).argmax(-1) for part in features.split(1024)])
        return int((predictions==targets).sum()),predictions.cpu().numpy()
    with torch.no_grad():
        correct,prediction = accuracy(vx,vlabels)
        train_correct,_ = accuracy(x,labels)
    return dict(correct=correct,n=len(vlabels),accuracy=correct/len(vlabels),
                train_accuracy=train_correct/len(labels),train_n=len(labels),
                rank=rank,steps=steps,classes=classes,dimensions=x.shape[1],trace=trace),prediction


def run(out: Path, train_episodes=4096, probe_steps=1200):
    out.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    model,tokenizer = load_model(),load_tokenizer()
    train,ty = extract(model,tokenizer,'train',train_episodes)
    val,vy = extract(model,tokenizer,'val',1000)
    del model
    torch.cuda.empty_cache()
    rows = []
    with (out/'probes.csv').open('w') as file:
        fields = ['site','position','target','rank','dimensions','classes','train_n','train_accuracy','n','correct','accuracy']
        writer = csv.DictWriter(file,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for site in SITES:
            for position in POSITIONS:
                kind = 'write' if position in POSITIONS[:2] else 'read'
                targets = ('tail','head','relation') if kind=='write' else ('start','relation')
                for column,target in enumerate(targets):
                    for rank in (None,32):
                        purpose = f'{site}/{position}/{target}'
                        row,prediction = fit_probe(train[site,position],ty[kind][:,column],
                            val[site,position],vy[kind][:,column],classes=4 if target=='relation' else 1024,
                            rank=rank,purpose=purpose,steps=probe_steps)
                        row.update(site=site,position=position,target=target)
                        rows.append(row)
                        writer.writerow({key:row[key] for key in fields})
                        file.flush()
                        np.savez_compressed(out/f'{site}-{position}-{target}-{rank or "full"}.npz',
                                            prediction=prediction,answer=vy[kind][:,column].numpy())
                        print(json.dumps({key:row[key] for key in fields}),flush=True)
    result = dict(diagnostic='D1',**source_metadata(),host=socket.gethostname(),job_id=os.environ.get('PBS_JOBID'),
                  seed=17,eval_seed=4242,train_episodes=train_episodes,val_episodes=1000,
                  frozen_features=True,adapter=False,split_train='train',split_eval='val',
                  name_classes=1024,active_name_pool=[0,768],
                  positions='resN = input residual before decoder layer N (zero-based); o14 = attention o_proj input',
                  method='train-standardized multinomial linear CE; full matrix or learned rank-32 factorization; '
                         'AdamW lr .01 cosine to .001; weight_decay 0; batch512; no val-based selection',
                  seconds=time.perf_counter()-started,rows=rows)
    _write_json(out/'summary.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    run(args.out)
