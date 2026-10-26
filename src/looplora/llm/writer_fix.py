"""WP-L3 writer interventions with explicit fact-aligned training signals."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.core import memory
from looplora.core.adapters import IterativeQueryAdapter
from looplora.data.relations import generate
from looplora.data.render import load_name_pool, render_episodes
from looplora.infra.seeds import derive_seed
from .batching import PreparedBatch, tokenize_batch
from .protocols import encode_support
from .train import LLMExperiment, make_task_spec

ARMS = ('OA','W0','W1','W2','W3','W4','W5')
SIMPLICITY = ('W0','W1','W4','W3','W2','W5')


def fact_positions(arrays,rendered,device='cpu') -> Tensor:
    """[B,E,(fact_final,pre_tail,tail)], using actual single-name token IDs."""
    pool = load_name_pool()
    positions = torch.full((*arrays.facts.shape[:2],3),-1,dtype=torch.long)
    for b,row in enumerate(rendered):
        for span in row.fact_spans:
            tail = int(arrays.facts[b,span.fact_index,2])
            hits = [i for i in range(span.token_start,span.token_end)
                    if row.context_ids[i]==pool.token_ids[tail]]
            if not hits or hits[-1]<=span.token_start:
                raise ValueError('fact tail must be a single name token after its prefix')
            t = hits[-1]
            positions[b,span.fact_index] = torch.tensor((span.token_end-1,t-1,t))
    if bool((positions<0).any()):
        raise ValueError('WP-L3 requires all E fact slots to be valid')
    return positions.to(device)


def gather_positions(x: Tensor,indices: Tensor) -> Tensor:
    return x.gather(1,indices[...,None].expand(-1,-1,x.shape[-1]))


def true_fact_indices(arrays,device) -> Tensor:
    facts = torch.as_tensor(arrays.facts,device=device)
    head = torch.as_tensor(arrays.q_start,device=device)
    rel = torch.as_tensor(arrays.q_rel,device=device)
    matches = (facts[:,None,:,0]==head[...,None]) & (facts[:,None,:,1]==rel[...,None])
    if not bool((matches.sum(-1)==1).all()):
        raise ValueError('T1 query must identify exactly one source fact')
    return matches.long().argmax(-1)


class FixAdapter(IterativeQueryAdapter):
    """Same reader/writer numerics except the explicitly named intervention."""
    def __init__(self,initial,arm: str):
        if arm not in ARMS:
            raise ValueError('unknown writer-fix arm')
        # Preserve the caller RNG; shared parameters are copied exactly below.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed(17,'writer-fix-container'))
            super().__init__(initial.cfg)
        self.arm = arm
        self.load_state_dict(initial.state_dict())
        if arm=='W4':
            # A is A_k and remains shared with the reader, as in the symbolic analog.
            self.A_v = nn.Parameter(self.A.detach().clone())
        self.register_buffer('input_mean',torch.zeros(initial.cfg.d_in))
        self.register_buffer('input_std',torch.ones(initial.cfg.d_in))
        self.phase = 'idle'
        self.positions = self.support_x = self.keys = self.values = self.oracle_values = None

    def u(self,x: Tensor) -> Tensor:
        if self.arm in {'W1','W5'}:
            x = (x-self.input_mean)/self.input_std
        return F.linear(x,self.A)

    def _prepare(self,x,*args,**kwargs):
        if self.phase=='support':
            self.support_x = x
        return super()._prepare(x,*args,**kwargs)

    def scan_final(self,M0,u,valid=None,write=None,reset=None):
        if self.phase!='support' or self.positions is None:
            raise RuntimeError('writer-fix support must set fact positions before its write')
        if self.arm=='W4':
            ku = self.u(gather_positions(self.support_x,self.positions[:,:,1]))
            vu = F.linear(gather_positions(self.support_x,self.positions[:,:,2]),self.A_v)
            # The core scan consumes paired key/value inputs, one ordered write/fact.
            packed = torch.cat((ku,vu),-1)
            zeros = torch.zeros_like(self.K)
            K,V = torch.cat((self.K,zeros),-1),torch.cat((zeros,self.V),-1)
            mask = torch.ones(packed.shape[:2],dtype=torch.bool,device=packed.device)
            final = memory.scan_final(M0,packed,mask,mask,K=K,V=V,
                                      lam=self.lam,beta=self.beta,eps=self.cfg.eps)
        else:
            ku = vu = gather_positions(u,self.positions[:,:,0])
            final = super().scan_final(M0,u,valid,write,reset)
        # Match the core scan's per-write projection shape and FP32 rounding.
        inputs,kmap,vmap = ((packed,K,V) if self.arm=='W4' else (ku,self.K,self.V))
        raw = [F.linear(inputs[:,j],kmap) for j in range(inputs.shape[1])]
        self.keys = torch.stack([k/torch.linalg.vector_norm(k,dim=-1,keepdim=True).clamp_min(
            self.cfg.eps) for k in raw],1)
        self.values = torch.stack([F.linear(inputs[:,j],vmap) for j in range(inputs.shape[1])],1)
        return final

    def _masked_branch(self,u,state,loop_steps,valid):
        if self.arm!='OA' or self.phase!='read':
            return super()._masked_branch(u,state,loop_steps,valid)
        if self.oracle_values is None:
            raise RuntimeError('OA requires fact values scoped to the current questions')
        flat = u.reshape(-1,self.cfg.r)
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        oracle = self.oracle_values
        if u.ndim==3:
            oracle = oracle[:,None,:].expand(*u.shape[:-1],self.cfg.r_mem)
        selected = flat.index_select(0,indices)
        value = oracle.reshape(-1,self.cfg.r_mem).index_select(0,indices)
        z = selected
        for _ in range(loop_steps):
            z = selected+self.cfg.rho*self._phi(F.linear(z,self.R)+F.linear(value,self.U))
        output = u.new_zeros((len(flat),self.cfg.d_out)).index_copy(0,indices,self.branch(z))
        return output.reshape(*u.shape[:-1],self.cfg.d_out)


@dataclass(frozen=True)
class FixBatch(PreparedBatch):
    keys: Tensor
    values: Tensor
    positions: Tensor
    raw_support: Tensor


class WriterFixExperiment(LLMExperiment):
    def __init__(self,config,arm: str,*,model=None,tokenizer=None):
        super().__init__(config,model=model,tokenizer=tokenizer)
        self.arm = arm
        adapter = FixAdapter(self.adapter,arm).to(device=self.device,dtype=torch.float32)
        self.adapter = self.writer = self.bank.modules[self.layer].adapter = adapter
        self.value_head = None
        if arm in {'W3','W5'}:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(derive_seed(17,'writer-fix-value-head'))
                self.value_head = nn.Linear(adapter.cfg.r_mem,config['task']['V']).to(self.device)

    def prepare(self,arrays,protocol=None,*,truncate=True):
        if self._closed:
            raise RuntimeError('experiment is closed')
        if protocol not in {None,'memory_only'} or not truncate:
            raise ValueError('WP-L3 only defines truncated memory-only support')
        rendered = render_episodes(arrays,tokenizer=self.tokenizer)
        tokens = tokenize_batch(arrays,self.tokenizer,device=self.device,rendered=rendered)
        positions = fact_positions(arrays,rendered,self.device)
        write_mask = tokens.fact_final
        if self.arm=='W4':
            write_mask = torch.zeros_like(tokens.fact_final).scatter(1,positions[:,:,2],True)
        self.adapter.positions,self.adapter.phase = positions,'support'
        try:
            snapshot = encode_support(self.model,tokens.support_ids,tokens.support_mask,
                'fact_final',fact_final=write_mask,protocol='memory_only',loop_steps=1,truncate=True)
        finally:
            self.adapter.phase='idle'
        return FixBatch(snapshot,tokens.question_ids,tokens.question_mask,tokens.answer_ids,
            tokens.candidate_ids,tokens.candidate_mask,arrays,int(tokens.support_mask.sum()),
            write_mask.sum(-1),tokens.support_ids,tokens.support_mask,write_mask,
            self.adapter.keys,self.adapter.values,positions,self.adapter.support_x)

    def score(self,prepared,L=1,*,capture_inputs=False):
        self.adapter.phase='read'
        if self.arm=='OA':
            target = true_fact_indices(prepared.arrays,self.device)
            self.adapter.oracle_values = prepared.values.gather(
                1,target[...,None].expand(-1,-1,self.adapter.cfg.r_mem)).flatten(0,1)
        try:
            return super().score(prepared,L,capture_inputs=capture_inputs)
        finally:
            self.adapter.phase='idle'
            self.adapter.oracle_values=None

    def trainable_parameters(self):
        params = super().trainable_parameters()
        return params+([] if self.value_head is None else list(self.value_head.parameters()))

    @torch.no_grad()
    def standardize(self):
        if self.arm not in {'W1','W5'}:
            return None
        config = self.config
        spec = make_task_spec(config)
        seed = derive_seed(17,'llm','writer-data')
        count,total,squared = 0,None,None
        for begin in range(0,256,32):
            arrays = generate(spec,'train',begin,32,seed)
            batch = self.prepare(arrays)
            _,read = self.score(batch,1,capture_inputs=True)
            write_x = gather_positions(batch.raw_support,batch.positions[:,:,0]).flatten(0,1)
            read_x = read[batch.question_mask]
            features = torch.cat((write_x,read_x),0).double()
            count += len(features)
            summed,squared_sum = features.sum(0),features.square().sum(0)
            total = summed if total is None else total+summed
            squared = squared_sum if squared is None else squared+squared_sum
        mean = total/count
        std = ((squared-count*mean.square())/(count-1)).clamp_min(0).sqrt().clamp_min(1e-4)
        self.adapter.input_mean.copy_(mean.float())
        self.adapter.input_std.copy_(std.float())
        return dict(episodes=256,positions=count,data_seed=seed,
                    policy='pooled fact_final writes and all valid question reads; sample std clamp1e-4')

    def inference_state(self):
        return {name:value.detach().cpu().clone() for name,value in self.adapter.state_dict().items()}

    def writer_signature(self):
        return {**super().writer_signature(),'writer_fix':self.arm,
                'key_projection':'A_k=reader A' if self.arm=='W4' else 'shared A',
                'value_projection':'A_v' if self.arm=='W4' else 'shared A'}


def query_codes(experiment,batch,question_x):
    last = batch.question_mask.long().sum(-1)-1
    B,Q = last.shape
    x = question_x[torch.arange(B,device=last.device)[:,None],
                   torch.arange(Q,device=last.device)[None,:],last]
    raw = F.linear(experiment.adapter.u(x.float()),experiment.adapter.Q)
    return F.normalize(raw,dim=-1,eps=experiment.adapter.cfg.eps)


def auxiliary_losses(experiment,batch,question_x):
    zero = batch.values.new_zeros(())
    address = value = zero
    if experiment.arm in {'W2','W5'}:
        scores = torch.einsum('bqr,ber->bqe',query_codes(experiment,batch,question_x),batch.keys)/.1
        targets = true_fact_indices(batch.arrays,experiment.device)
        address = F.cross_entropy(scores.flatten(0,1),targets.flatten())
    if experiment.arm in {'W3','W5'}:
        targets = torch.as_tensor(batch.arrays.facts[:,:,2],device=experiment.device,dtype=torch.long)
        value = F.cross_entropy(experiment.value_head(batch.values).flatten(0,1),targets.flatten())
    return address,value


def alignment(experiment,batch,question_x):
    q = query_codes(experiment,batch,question_x)
    scores = torch.einsum('bqr,ber->bqe',q,batch.keys)
    targets = true_fact_indices(batch.arrays,experiment.device)
    mask = F.one_hot(targets,num_classes=batch.keys.shape[1]).bool()
    actual = scores.gather(-1,targets[...,None]).squeeze(-1)
    other = scores.masked_fill(mask,-torch.inf).amax(-1)
    E = batch.keys.shape[1]
    gram = torch.einsum('ber,bfr->bef',batch.keys,batch.keys)
    # Off-diagonal ordered-pair mean; one value per episode, repeated by query.
    diagonal = torch.eye(E,dtype=torch.bool,device=gram.device)
    pairwise = gram[:,~diagonal].mean(-1)[:,None].expand_as(actual)
    return dict(key_top1=scores.argmax(-1)==targets,key_margin=actual-other,
                key_pairwise_cosine=pairwise)
