"""WP-L7 RFX: read-failure factorial on frozen WP-L4 checkpoints (evaluation only; compute only).

The deployed read at the final question token, m = M q(z) in R^{r_mem}, is replaced and nothing else changes: every
other token keeps its deployed read and the LM path after the read is the unmodified one. Cells: storage {the deployed
delta store, the ridge store  M = V K^T (K K^T + eta I)^-1  over the same written unit keys and values} x query {the raw
learned q, the unit q, the true source key of the queried fact}, each as is and with a scalar gain fitted on a fitting
set, plus the clean value and a zero read. Decoders: the deployed LM path, the training-only clean-value head, and a
linear decoder refit on fitting-set reads. Docs: docs/prereg/RFX.md.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from looplora.core.memory import delta_update,gram_conditions
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.sym.evaluate import stratify_queries
from .evaluate import _slice
from .train import make_task_spec
from .writer_fix import true_fact_indices

LABELS=('R-r128','R-r32','W5-r32')
STORES=('delta','ridge')
QUERIES=('raw','unit','true')
CELLS=tuple((s,q) for s in STORES for q in QUERIES)
VARIANTS=tuple(f'{s}/{q}/{g}' for s,q in CELLS for g in ('asis','fit'))+('clean','zero')
REFIT_CELLS=tuple(f'{s}/{q}' for s,q in CELLS)+('clean','zero')
ETA_GRID=(1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.)
FIT_EPISODES,EVAL_EPISODES,EVAL_START,EVAL_SEED=4096,4000,200000,4242
FIT_SEED=derive_seed(17,'rfx','fit')
SMOKE_START,SMOKE_SEED=10_000_000,derive_seed(17,'rfx','smoke-eval')
PERMUTATIONS=4
DECODER_DECAYS=(1e-4,1e-3,1e-2,1e-1)
EPS=1e-6
ERROR_RATIO,EM_GAIN,TOLERANCE=.5,.10,1e-9  # TOLERANCE: differences of two k/n rates may miss exactly 10 pp by rounding


class FinalRead:
    """Record, and optionally replace, the read m = M q(z) of an installed shared reader at each row's last question token.

    last [N]: position of the last real token of every flattened (episode-major) question row. replacement [N, r_mem] or
    None. Inside the context the adapter instance shadows process_sequence, _masked_branch and _query; the reads are
    computed by the unchanged expressions, so a replacement equal to the recorded read leaves the logits bitwise unchanged.
    """
    def __init__(self,adapter,last,replacement=None):
        cfg=adapter.cfg
        if cfg.variant!='iterq' or cfg.dpfp_nu or cfg.store!='delta' or cfg.unit_query or cfg.eps!=EPS:
            raise ValueError('RFX reads shared delta-store WP-L4 readers with raw queries')
        if replacement is not None and tuple(replacement.shape)!=(len(last),cfg.r_mem):
            raise ValueError('replacement must be [rows, r_mem]')
        self.adapter,self.last,self.last_cpu,self.replacement=adapter,last,last.cpu(),replacement
        self.q=torch.zeros(len(last),cfg.r_mem,device=adapter.A.device)
        self.m=torch.zeros_like(self.q)
        self.seen=torch.zeros(len(last),dtype=torch.bool,device=adapter.A.device)
        self._t,self._final=0,None

    def __enter__(self):
        adapter=self.adapter
        if any(name in adapter.__dict__ for name in ('process_sequence','_masked_branch','_query')):
            raise RuntimeError('the adapter is already intercepted')
        self._process,self._masked=adapter.process_sequence,adapter._masked_branch
        adapter.process_sequence,adapter._masked_branch,adapter._query=(
            self._process_sequence,self._masked_branch,self._query)
        return self

    def __exit__(self,*exc):
        for name in ('process_sequence','_masked_branch','_query'):
            delattr(self.adapter,name)

    def check(self):
        if not bool(self.seen.all()):
            raise RuntimeError('a question row was not read at its last token')

    def _process_sequence(self,*args,**kwargs):
        self._t,self._final=0,None
        return self._process(*args,**kwargs)

    def _masked_branch(self,u,state,loop_steps,valid):
        if loop_steps!=1:
            raise ValueError('RFX replaces the read of a one-step reader')
        t=self._t
        self._t+=1
        self._final=None
        if bool((self.last_cpu==t).any()):
            rows=valid.nonzero(as_tuple=True)[0]
            final=self.last[rows]==t
            if bool(final.any()):
                self._final=(rows,final)
        return self._masked(u,state,loop_steps,valid)

    def _query(self,u,M,Q,U):
        q=self.adapter.query_code(F.linear(u,Q))
        m=torch.matmul(M,q.to(M.dtype).unsqueeze(-1)).squeeze(-1)
        if self._final is not None:
            rows,final=self._final
            picked=rows[final]
            self.q[picked]=q[final].detach().float()
            self.m[picked]=m[final].detach().float()
            self.seen[picked]=True
            if self.replacement is not None:
                m=torch.where(final[:,None],self.replacement[rows].to(m.dtype),m)
        return F.linear(m.to(u.dtype),U)


def last_positions(batch):
    return batch.question_mask.long().sum(-1).reshape(-1)-1


@torch.no_grad()
def observe(experiment,batch):
    """One deployed forward with the final-question-token query and read recorded, plus the cached store inputs."""
    B,Q,_=batch.question_ids.shape
    adapter,r=experiment.adapter,experiment.adapter.cfg.r_mem
    with FinalRead(adapter,last_positions(batch)) as probe:
        logits,_=experiment.score(batch,1)
    probe.check()
    q,deployed=probe.q.reshape(B,Q,r),probe.m.reshape(B,Q,r)
    state=batch.snapshot.states[experiment.layer]
    target=true_fact_indices(batch.arrays,state.device)
    pick=lambda x:x.gather(1,target[...,None].expand(-1,-1,r))
    read=lambda code:torch.einsum('bij,bqj->bqi',state,code)
    return dict(logits=logits,keys=batch.keys,values=batch.values,q=q,true_key=pick(batch.keys),clean=pick(batch.values),
                delta=dict(raw=deployed,unit=read(F.normalize(q,dim=-1,eps=EPS)),true=read(pick(batch.keys))),
                recomputed_raw=read(q),target=target,answer=batch.answer_ids,
                answer_entity=torch.as_tensor(batch.arrays.answer,device=state.device,dtype=torch.long))


@torch.no_grad()
def replaced_logits(experiment,batch,replacement):
    """Logits with the final-token read replaced by replacement [B*Q, r_mem]; nothing else changes."""
    with FinalRead(experiment.adapter,last_positions(batch),replacement):
        logits,_=experiment.score(batch,1)
    return logits


def normalized_error(read,target):
    """|read - target|^2 / (|target|^2 + 1e-6), the WP-L4 relative read error."""
    return (read-target).square().sum(-1)/(target.square().sum(-1)+EPS)


def ridge_reads(keys,values,q,eta):
    """Float64 reads V (K^T K + eta I)^-1 K^T q of the ridge store over n written unit keys/values: [B,n,r] x [B,Q,r]."""
    keys,values,q=keys.double(),values.double(),q.double()
    system=keys@keys.transpose(1,2)+eta*torch.eye(keys.shape[1],dtype=keys.dtype,device=keys.device)
    weights=torch.linalg.solve(system,torch.einsum('bnr,bqr->bnq',keys,q))
    return torch.einsum('bnr,bnq->bqr',values,weights)


def query_vectors(obs,query,sl=slice(None)):
    q=obs['q'][sl]
    if query=='raw':
        return q
    return F.normalize(q,dim=-1,eps=EPS) if query=='unit' else obs['true_key'][sl]


class Moments:
    """Sufficient statistics of the scalar gain that minimizes the mean normalized read error."""
    def __init__(self):
        self.mt=self.mm=self.tt=0.
        self.n=0

    def add(self,read,target):
        weight=1/(target.square().sum(-1)+EPS)
        self.mt+=float(((read*target).sum(-1)*weight).sum())
        self.mm+=float((read.square().sum(-1)*weight).sum())
        self.tt+=float((target.square().sum(-1)*weight).sum())
        self.n+=weight.numel()

    def gain_error(self):
        """(gain, mean normalized error at that gain); a zero read has gain one."""
        gain=self.mt/self.mm if self.mm>0 else 1.
        return gain,(self.tt-2*gain*self.mt+gain*gain*self.mm)/self.n


def fit_gain(read,target):
    """The scalar g minimizing the mean normalized error |g read - target|^2 / (|target|^2 + 1e-6)."""
    moments=Moments()
    moments.add(read.double(),target.double())
    return moments.gain_error()[0]


def slice_obs(obs,sl):
    return dict(keys=obs['keys'][sl],values=obs['values'][sl],q=obs['q'][sl],true_key=obs['true_key'][sl],
                clean=obs['clean'][sl],delta={k:v[sl] for k,v in obs['delta'].items()})


def fit_parameters(fit,chunk=256):
    """Everything fitted on the fitting set only: every cell's gain and every ridge cell's eta (the fitted-gain argmin)."""
    target=fit['clean'].double()
    params=dict(eta={},gain=dict(delta={},ridge={}),curves={})
    for query in QUERIES:
        moments=Moments()
        moments.add(fit['delta'][query].double(),target)
        params['gain']['delta'][query]=moments.gain_error()[0]
        curve=[]
        for eta in ETA_GRID:
            moments=Moments()
            for begin in range(0,len(target),chunk):
                sl=slice(begin,begin+chunk)
                moments.add(ridge_reads(fit['keys'][sl],fit['values'][sl],query_vectors(fit,query,sl),eta),target[sl])
            gain,error=moments.gain_error()
            curve.append(dict(eta=eta,gain=gain,error=error))
        best=min(curve,key=lambda row:(row['error'],-row['eta']))
        params['eta'][query],params['gain']['ridge'][query],params['curves'][query]=best['eta'],best['gain'],curve
    return params


def variant_reads(obs,params):
    """Every variant's replacement read [B,Q,r] (float64) for one batch of cached observations."""
    reads={}
    for store,query in CELLS:
        if store=='delta':
            base=obs['delta'][query].double()
        else:
            base=ridge_reads(obs['keys'],obs['values'],query_vectors(obs,query),params['eta'][query])
        reads[f'{store}/{query}/asis']=base
        reads[f'{store}/{query}/fit']=params['gain'][store][query]*base
    reads['clean']=obs['clean'].double()
    reads['zero']=torch.zeros_like(reads['clean'])
    return reads


def cell_reads(obs,params,chunk=256):
    """Unscaled reads [N*Q, r] of the refit-decoder cells over a cached set, in chunks."""
    parts={cell:[] for cell in REFIT_CELLS}
    for begin in range(0,len(obs['clean']),chunk):
        reads=variant_reads(slice_obs(obs,slice(begin,begin+chunk)),params)
        for cell in REFIT_CELLS:
            parts[cell].append(reads[cell if cell in ('clean','zero') else cell+'/asis'].flatten(0,1))
    return {cell:torch.cat(parts[cell]) for cell in REFIT_CELLS}


@torch.enable_grad()
def _logistic(x,y,classes,decay,iterations):
    weight=torch.zeros(x.shape[1],classes,dtype=x.dtype,device=x.device,requires_grad=True)
    bias=torch.zeros(classes,dtype=x.dtype,device=x.device,requires_grad=True)
    optimizer=torch.optim.LBFGS([weight,bias],lr=1.,max_iter=iterations,max_eval=10*iterations,history_size=20,
                                line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss=F.cross_entropy(x@weight+bias,y)+.5*decay*weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach(),bias.detach()


def fit_decoder(x,y,classes,*,decays=DECODER_DECAYS,holdout=.2,iterations=100):
    """Affine multinomial logistic decoder on standardized reads; its L2 weight is chosen on the last 20% of the rows."""
    x=x.double()
    mean,std=x.mean(0),x.std(0).clamp_min(1e-8)
    z=(x-mean)/std
    split=len(z)-int(len(z)*holdout)
    decay=decays[len(decays)//2]
    if split<len(z):
        scores=[]
        for candidate in decays:
            weight,bias=_logistic(z[:split],y[:split],classes,candidate,iterations)
            with torch.no_grad():
                scores.append(float(F.cross_entropy(z[split:]@weight+bias,y[split:])))
        decay=decays[int(np.argmin(scores))]
    weight,bias=_logistic(z,y,classes,decay,iterations)
    return dict(mean=mean,std=std,weight=weight,bias=bias,decay=decay)


@torch.no_grad()
def decoder_scores(decoder,x,y):
    logits=((x.double()-decoder['mean'])/decoder['std'])@decoder['weight']+decoder['bias']
    return dict(acc=float((logits.argmax(-1)==y).double().mean()),nll=float(F.cross_entropy(logits,y)))


@torch.no_grad()
def collect(experiment,arrays,batch_size):
    """Cached observations of a whole set (on the device), as observe() returns them per batch."""
    parts=[]
    for begin in range(0,len(arrays.answer),batch_size):
        batch=experiment.prepare(_slice(arrays,slice(begin,min(begin+batch_size,len(arrays.answer)))))
        obs=observe(experiment,batch)
        del obs['logits']  # [rows, vocabulary] per batch; the fitting set never needs them
        parts.append(obs)
    cat=lambda name:torch.cat([p[name] for p in parts])
    return dict(keys=cat('keys'),values=cat('values'),q=cat('q'),true_key=cat('true_key'),clean=cat('clean'),
                target=cat('target'),answer_entity=cat('answer_entity'),
                delta={k:torch.cat([p['delta'][k] for p in parts]) for k in QUERIES})


def mean_se(values):
    values=np.asarray(values,dtype=np.float64)
    return float(values.mean()),(float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.)


def age_table(age,E,**columns):
    """Rows per write age (0 = the most recent write) with the mean of every column over its queries."""
    rows=[]
    for a in range(E):
        mask=age==a
        rows.append(dict(age=a,n=int(mask.sum()),**{k:(float(v[mask].mean()) if mask.any() else None)
                                                    for k,v in columns.items()}))
    return rows


def episode_permutations(episode_seeds,E,index):
    """[B,E] write orders: every episode's own permutation, a pure function of its episode seed and the index."""
    return np.stack([np.random.default_rng(derive_seed(17,'rfx','permutation',index,int(seed))).permutation(E)
                     for seed in episode_seeds])


def _worst(a,b):
    """max(a, b) that keeps a NaN: Python's max(0., nan) is 0., which would hide it from the instrumentation guards."""
    return b if (b!=b or b>a) else a


def delta_state(adapter,keys,values):
    """The deployed decay-first delta rule applied to cached unit keys and values, in the given order."""
    B,E,r=keys.shape
    state=torch.zeros(B,r,r,dtype=keys.dtype,device=keys.device)
    for j in range(E):
        state=delta_update(state,keys[:,j],values[:,j],None,adapter.lam,adapter.beta)
    return state


def classify(cells,refit):
    """The registered verdict of docs/prereg/RFX.md for one checkpoint from its evaluation-set tables."""
    base=cells['delta/raw/fit']
    base_em=max(cells['delta/raw/asis']['lm_em'],base['lm_em'])

    def rescues(name):
        cell=cells[name]
        return bool(cell['err']<=ERROR_RATIO*base['err'] and cell['lm_em']-base_em>=EM_GAIN-TOLERANCE)

    storage,query_delta,query_ridge=rescues('ridge/raw/fit'),rescues('delta/true/fit'),rescues('ridge/true/fit')
    decoder=bool(refit['delta/raw']['acc']-base_em>=EM_GAIN-TOLERANCE)
    verdict='storage' if storage else 'query' if (query_delta or query_ridge) else 'decoder' if decoder else 'none'
    return dict(verdict=verdict,rescue=dict(storage_ridge_raw=storage,query_delta_true=query_delta,
        query_ridge_true=query_ridge,decoder_refit=decoder),baseline=dict(lm_em=base_em,err_fit=base['err']),
        thresholds=dict(error_ratio=ERROR_RATIO,em_gain=EM_GAIN))


def lsq_ranks(verdicts):
    """Ranks whose R checkpoint is classified 'storage' (they trigger the LSQ arms of that rank)."""
    return [rank for rank,label in ((32,'R-r32'),(128,'R-r128')) if verdicts.get(label)=='storage']


def eval_arrays(spec,episodes,smoke):
    """The evaluation set and its registered identity; smoke runs use fresh train-split episodes, never the val range."""
    if smoke:
        arrays=stratify_queries(generate(spec,'train',SMOKE_START,episodes,SMOKE_SEED),spec,[1],SMOKE_SEED)
        return arrays,dict(split='train',seed=SMOKE_SEED,start=SMOKE_START,episodes=episodes)
    arrays=stratify_queries(generate(spec,'val',EVAL_START,episodes,EVAL_SEED),spec,[1],EVAL_SEED)
    return arrays,dict(split='val',seed=EVAL_SEED,start=EVAL_START,episodes=episodes)


@torch.no_grad()
def evaluate(experiment,head,arrays,params,*,batch_size=64,permutations=PERMUTATIONS):
    """Every variant through the deployed LM path on the evaluation set, plus write-order sensitivity and conditioning."""
    adapter,r=experiment.adapter,experiment.adapter.cfg.r_mem
    pieces={v:dict(correct=[],nll=[],err=[],head=[]) for v in VARIANTS}
    kept={cell:[] for cell in REFIT_CELLS}
    perm=[dict(err=[],correct=[],discrepancy=[],agree=[],age=[]) for _ in range(permutations)]
    side=dict(age=[],entity=[],cond_gram=[],cond_ridge={q:[] for q in QUERIES})
    checks=dict(identity_max_abs_logit_diff=None,raw_recompute_max_rel=0.,delta_reconstruction_max_rel=0.,
                ridge_permutation_max_abs=0.)
    relative=lambda a,b:float(((a-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-8)).max())
    E=None
    for begin in range(0,len(arrays.answer),batch_size):
        part=_slice(arrays,slice(begin,min(begin+batch_size,len(arrays.answer))))
        batch=experiment.prepare(part)
        obs=observe(experiment,batch)
        deployed=obs['delta']['raw']
        E=obs['keys'].shape[1]
        reads=variant_reads(obs,params)
        if begin==0:
            again=replaced_logits(experiment,batch,deployed.reshape(-1,r))
            checks['identity_max_abs_logit_diff']=float((again.float()-obs['logits'].float()).abs().max())
        checks['raw_recompute_max_rel']=_worst(checks['raw_recompute_max_rel'],relative(obs['recomputed_raw'],deployed))
        for name,read in reads.items():
            logits=obs['logits'] if name=='delta/raw/asis' else replaced_logits(experiment,batch,read.reshape(-1,r).float())
            piece=pieces[name]
            piece['correct'].append((logits.argmax(-1)==obs['answer']).flatten().cpu())
            piece['nll'].append(F.cross_entropy(logits.float().flatten(0,1),obs['answer'].flatten(),reduction='none').cpu())
            piece['err'].append(normalized_error(read,obs['clean'].double()).flatten().cpu())
            if head is not None:
                piece['head'].append((head(read.float()).argmax(-1)==obs['answer_entity']).flatten().cpu())
            key=name[:-5] if name.endswith('/asis') else name
            if key in kept:
                kept[key].append(read.flatten(0,1).float().cpu())
        # Write order: the same cached writes, re-ordered, through the deployed decay-first delta rule.
        natural=delta_state(adapter,obs['keys'],obs['values'])
        checks['delta_reconstruction_max_rel']=_worst(checks['delta_reconstruction_max_rel'],
            relative(torch.einsum('bij,bqj->bqi',natural,obs['q']),deployed))
        for p in range(permutations):
            order=torch.as_tensor(episode_permutations(part.episode_seed,E,p),device=obs['keys'].device)
            expand=order[...,None].expand(-1,-1,r)
            keys,values=obs['keys'].gather(1,expand),obs['values'].gather(1,expand)
            read=torch.einsum('bij,bqj->bqi',delta_state(adapter,keys,values),obs['q'])
            logits=replaced_logits(experiment,batch,read.reshape(-1,r))
            position=(order[:,None,:]==obs['target'][...,None]).long().argmax(-1)
            perm[p]['age'].append((E-1-position).flatten().cpu())
            perm[p]['err'].append(normalized_error(read,obs['clean']).flatten().cpu())
            perm[p]['correct'].append((logits.argmax(-1)==obs['answer']).flatten().cpu())
            perm[p]['agree'].append((logits.argmax(-1)==obs['logits'].argmax(-1)).flatten().cpu())
            perm[p]['discrepancy'].append(((read-deployed).square().sum(-1)/(deployed.square().sum(-1)+EPS)).flatten().cpu())
            if p==0:
                one=ridge_reads(obs['keys'],obs['values'],obs['q'],params['eta']['raw'])
                two=ridge_reads(keys,values,obs['q'],params['eta']['raw'])
                checks['ridge_permutation_max_abs']=_worst(checks['ridge_permutation_max_abs'],float((one-two).abs().max()))
        side['age'].append((E-1-obs['target']).flatten().cpu())
        side['entity'].append(obs['answer_entity'].flatten().cpu())
        side['cond_gram'].append(gram_conditions(obs['keys'])['gram'].cpu())
        for query in QUERIES:
            side['cond_ridge'][query].append(gram_conditions(obs['keys'],None,params['eta'][query])['ridge'].cpu())
    return dict(pieces={v:{k:torch.cat(x).numpy() for k,x in p.items() if x} for v,p in pieces.items()},
                reads={c:torch.cat(x) for c,x in kept.items()},
                perm=[{k:torch.cat(x).numpy() for k,x in p.items()} for p in perm],
                side=dict(age=torch.cat(side['age']).numpy(),entity=torch.cat(side['entity']),
                          cond_gram=torch.cat(side['cond_gram']).numpy(),
                          cond_ridge={q:torch.cat(x).numpy() for q,x in side['cond_ridge'].items()}),
                checks=checks,E=E)


def cond_summary(values):
    log=np.log10(values)
    return dict(median=float(np.median(log)),p90=float(np.quantile(log,.9)),p99=float(np.quantile(log,.99)),
                max=float(log.max()),unit='log10 condition number')


@torch.no_grad()
def run_checkpoint(label,experiment,head,*,smoke=False,fit_episodes=FIT_EPISODES,eval_episodes=EVAL_EPISODES):
    """Fit on the fitting set, evaluate every variant on the evaluation set; returns (JSON record, per-sample arrays)."""
    started=time.perf_counter()
    spec=make_task_spec(experiment.config)
    fit=collect(experiment,generate(spec,'train',0,fit_episodes,FIT_SEED),32)
    params=fit_parameters(fit)
    fitted=time.perf_counter()
    arrays,eval_info=eval_arrays(spec,eval_episodes,smoke)
    out=evaluate(experiment,head,arrays,params)
    pieces,E=out['pieces'],out['E']
    cells={}
    for name,piece in pieces.items():
        err,err_se=mean_se(piece['err'])
        em,em_se=mean_se(piece['correct'].astype(np.float64))
        cells[name]=dict(err=err,err_se=err_se,lm_em=em,lm_em_se=em_se,lm_nll=float(piece['nll'].mean()))
        if 'head' in piece:
            cells[name]['head_acc']=float(piece['head'].mean())
    fit_reads=cell_reads(fit,params)
    answers=fit['answer_entity'].flatten()
    refit={}
    for cell in REFIT_CELLS:
        decoder=fit_decoder(fit_reads[cell],answers,experiment.config['task']['V'])
        score=decoder_scores(decoder,out['reads'][cell].to(answers.device),out['side']['entity'].to(answers.device))
        refit[cell]=dict(acc=score['acc'],nll=score['nll'],decay=decoder['decay'])
    age=out['side']['age']
    columns=lambda v:dict(err=pieces[v]['err'],lm_em=pieces[v]['correct'].astype(np.float64))
    permuted=dict(err=np.concatenate([p['err'] for p in out['perm']]),
                  lm_em=np.concatenate([p['correct'] for p in out['perm']]).astype(np.float64))
    base=pieces['delta/raw/asis']
    write_order=dict(permutations=len(out['perm']),
        natural=dict(err=float(base['err'].mean()),lm_em=float(base['correct'].mean())),
        err=[float(p['err'].mean()) for p in out['perm']],lm_em=[float(p['correct'].mean()) for p in out['perm']],
        agreement_with_natural_order=[float(p['agree'].mean()) for p in out['perm']],
        read_discrepancy=[float(p['discrepancy'].mean()) for p in out['perm']],
        by_permuted_age=age_table(np.concatenate([p['age'] for p in out['perm']]),E,**permuted))
    result=dict(label=label,smoke=smoke,rank=experiment.adapter.cfg.r_mem,
        data=dict(fit=dict(split='train',seed=FIT_SEED,start=0,episodes=fit_episodes,
                           queries_per_episode=int(fit['q'].shape[1])),
                  eval=dict(**eval_info,queries_per_episode=1,query='one hop-1 query per episode (stratify_queries)')),
        parameters=dict(eta=params['eta'],gain=params['gain'],eta_curves=params['curves']),
        cells=cells,refit=refit,verdict=classify(cells,refit),
        normalization_effect={s:dict(delta_lm_em=cells[f'{s}/unit/fit']['lm_em']-cells[f'{s}/raw/fit']['lm_em'],
                                     delta_err=cells[f'{s}/unit/fit']['err']-cells[f'{s}/raw/fit']['err'])
                              for s in STORES},
        gain_effect=dict(delta_lm_em=cells['delta/raw/fit']['lm_em']-cells['delta/raw/asis']['lm_em'],
                         delta_err=cells['delta/raw/fit']['err']-cells['delta/raw/asis']['err']),
        age={v:age_table(age,E,**columns(v)) for v in
             ('delta/raw/asis','delta/raw/fit','ridge/raw/fit','delta/true/fit','ridge/true/fit')},
        write_order=write_order,
        conditioning=dict(gram=cond_summary(out['side']['cond_gram']),
                          ridge={q:dict(eta=params['eta'][q],**cond_summary(c)) for q,c in out['side']['cond_ridge'].items()}),
        checks=out['checks'],seconds=dict(fit=fitted-started,evaluate=time.perf_counter()-fitted))
    return result,{f'{v}/{k}':x for v,p in pieces.items() for k,x in p.items()}


def load_checkpoint(folder,*,model=None,tokenizer=None):
    """A frozen WP-L3/WP-L4 checkpoint folder: the head-free inference export plus the training-only clean-value head."""
    folder=Path(folder)
    payload=torch.load(folder/'writer.pt',map_location='cpu',weights_only=True)
    if 'read_fix_version' in payload:
        from .read_fix_run import load_inference
    elif 'writer_fix_version' in payload:
        from .writer_fix_run import load_inference
    else:
        raise ValueError('expected a writer-fix or read-fix inference export')
    experiment=load_inference(folder/'writer.pt',model=model,tokenizer=tokenizer)
    try:
        state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)['value_head']
        if state is None:
            raise ValueError('RFX needs the training-only clean-value head of the checkpoint')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed(17,'rfx','head-container'))
            head=nn.Linear(experiment.adapter.cfg.r_mem,experiment.config['task']['V']).to(experiment.device)
        head.load_state_dict(state,strict=True)
    except BaseException:
        experiment.close()
        raise
    return experiment,head
