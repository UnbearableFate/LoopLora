"""WP-L3 writer interventions with explicit fact-aligned training signals."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.core import memory
from looplora.core.adapters import IterativeQueryAdapter, UnsharedAdapter
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


def token_fact_positions(arrays,tokens,device='cpu') -> Tensor:
    """Recover the same spans from prefetched CPU tokens, without retokenizing."""
    if tokens.support_ids.device.type!='cpu':
        raise ValueError('prefetched position alignment requires CPU tokens')
    pool=load_name_pool()
    positions=torch.full((*arrays.facts.shape[:2],3),-1,dtype=torch.long)
    for b in range(len(arrays.facts)):
        finals=tokens.fact_final[b].nonzero(as_tuple=True)[0].tolist()
        if len(finals)!=arrays.facts.shape[1] or not arrays.fact_valid[b].all():
            raise ValueError('WP-L3 requires all E fact slots to be valid')
        begin=0
        for j,final in enumerate(finals):
            tail_id=pool.token_ids[int(arrays.facts[b,j,2])]
            hits=(tokens.support_ids[b,begin:final+1]==tail_id).nonzero(as_tuple=True)[0]
            if not len(hits) or int(hits[-1])==0:
                raise ValueError('fact tail must follow its prefix')
            tail=begin+int(hits[-1])
            positions[b,j]=torch.tensor((final,tail-1,tail))
            begin=final+1
    return positions.to(device)


def true_fact_indices(arrays,device) -> Tensor:
    facts = torch.as_tensor(arrays.facts,device=device)
    head = torch.as_tensor(arrays.q_start,device=device)
    rel = torch.as_tensor(arrays.q_rel,device=device)
    matches = (facts[:,None,:,0]==head[...,None]) & (facts[:,None,:,1]==rel[...,None])
    if not bool((matches.sum(-1)==1).all()):
        raise ValueError('T1 query must identify exactly one source fact')
    return matches.long().argmax(-1)


class _WriterFix:
    """Same reader/writer numerics except the explicitly named intervention."""
    def __init__(self,initial,arm: str):
        if arm not in ARMS:
            raise ValueError('unknown writer-fix arm')
        if arm=='W4' and initial.cfg.dpfp_nu:
            raise ValueError('DPFP features apply to the fact_final writers, not the paired W4 write')
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
        self.base_keys = None

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
        unit = lambda k: k/torch.linalg.vector_norm(k,dim=-1,keepdim=True).clamp_min(self.cfg.eps)
        self.keys = torch.stack([unit(k) for k in raw],1)
        if self.cfg.dpfp_nu:
            # Keys are the unit DPFP features actually written; base_keys keep the projected space.
            self.base_keys = self.keys
            self.keys = torch.stack([unit(memory.dpfp_features(k,self.cfg.dpfp_nu)) for k in raw],1)
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
        for step in range(loop_steps):
            R = self.R[step] if self.cfg.variant=='unshared' else self.R
            U = self.U[step] if self.cfg.variant=='unshared' else self.U
            z = selected+self.cfg.rho*self._phi(F.linear(z,R)+F.linear(value,U))
        output = u.new_zeros((len(flat),self.cfg.d_out)).index_copy(0,indices,self.branch(z))
        return output.reshape(*u.shape[:-1],self.cfg.d_out)


class FixAdapter(_WriterFix,IterativeQueryAdapter):
    """WP-L3 interventions on the original shared reader."""


class UnsharedFixAdapter(_WriterFix,UnsharedAdapter):
    """The same writer interventions with B8's four step-specific reads."""


@dataclass(frozen=True)
class FixBatch(PreparedBatch):
    keys: Tensor
    values: Tensor
    positions: Tensor
    raw_support: Tensor
    base_keys: Tensor | None = None


class WriterFixExperiment(LLMExperiment):
    def __init__(self,config,arm: str,*,model=None,tokenizer=None):
        super().__init__(config,model=model,tokenizer=tokenizer)
        self.arm = arm
        container = UnsharedFixAdapter if self.adapter.cfg.variant=='unshared' else FixAdapter
        adapter = container(self.adapter,arm).to(device=self.device,dtype=torch.float32)
        self.adapter = self.writer = self.bank.modules[self.layer].adapter = adapter
        self.value_head = None
        if arm in {'W3','W5'}:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(derive_seed(17,'writer-fix-value-head'))
                self.value_head = nn.Linear(adapter.cfg.r_mem,config['task']['V']).to(self.device)

    def prepare(self,arrays,protocol=None,*,truncate=True,tokens=None):
        if self._closed:
            raise RuntimeError('experiment is closed')
        if protocol not in {None,'memory_only'} or not truncate:
            raise ValueError('WP-L3 only defines truncated memory-only support')
        if tokens is None:
            rendered = render_episodes(arrays,tokenizer=self.tokenizer)
            tokens = tokenize_batch(arrays,self.tokenizer,device=self.device,rendered=rendered)
            positions = fact_positions(arrays,rendered,self.device)
        else:
            positions = token_fact_positions(arrays,tokens,self.device)
            tokens = tokens.to(self.device)
        write_mask = tokens.fact_final
        if self.arm=='W4':
            write_mask = torch.zeros_like(tokens.fact_final).scatter(1,positions[:,:,2],True)
        self.adapter.positions,self.adapter.phase = positions,'support'
        try:
            snapshot = encode_support(self.model,tokens.support_ids,tokens.support_mask,
                'fact_final',fact_final=write_mask,protocol='memory_only',
                loop_steps=4 if self.config['variant']=='B8' else 1,truncate=True)
        finally:
            self.adapter.phase='idle'
        return FixBatch(snapshot,tokens.question_ids,tokens.question_mask,tokens.answer_ids,
            tokens.candidate_ids,tokens.candidate_mask,arrays,int(tokens.support_mask.sum()),
            write_mask.sum(-1),tokens.support_ids,tokens.support_mask,write_mask,
            self.adapter.keys,self.adapter.values,positions,self.adapter.support_x,self.adapter.base_keys)

    def score(self,prepared,L=None,*,capture_inputs=False):
        if L is None:
            L = 4 if self.config['variant']=='B8' else 1
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
            _,read = self.score(batch,4 if self.config['variant']=='B8' else 1,capture_inputs=True)
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
    # The addressing target supervises the first read, whose state is z^0 = u.
    Q = experiment.adapter.Q[0] if experiment.adapter.cfg.variant=='unshared' else experiment.adapter.Q
    raw = F.linear(experiment.adapter.u(x.float()),Q)
    if experiment.adapter.cfg.dpfp_nu:
        # DPFP: the address is the unit feature code that the deployed read uses.
        return experiment.adapter.query_code(raw)
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
