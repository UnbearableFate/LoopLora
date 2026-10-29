"""WP-L4 raw deployed-read supervision and exact delta-read diagnostics."""
from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from looplora.core.memory import gram_conditions,ridge_coefficients,sweep_coefficients
from .train import LLMExperiment
from .writer_fix import WriterFixExperiment,true_fact_indices
from .writer_fix_run import configuration as writer_configuration

ARMS=('R','Rrho','RK','OArho','W5','OA')
SIMPLICITY=('R','Rrho','RK')
CANDIDATES=((32,'R'),(32,'Rrho'),(32,'RK'),(128,'W5'),(128,'R'))
RUNS=CANDIDATES+((32,'OArho'),(128,'OA'))
LABELS=dict(R='R',Rrho='Rρ',RK='RK',OArho='OAρ',W5='W5',OA='OA')


def configuration(arm,steps,*,E=16,smoke=False,rank=32,extra_runs=()):
    # extra_runs: (rank, arm) pairs that are not WP-L4 runs but are registered elsewhere (DPFP-X's rank 256); empty for WP-L4.
    if (rank,arm) not in RUNS+tuple(extra_runs) or E not in (8,16):
        raise ValueError('WP-L4 requires a registered arm at E16 or the conditional E8')
    base='OA' if arm in {'OArho','OA'} else 'W5'
    cfg=writer_configuration(base,steps,prefetch_workers=2)
    cfg['name']=f'read-fix-{arm}-E{E}' if rank==32 else f'read-fix-{arm}-r128-E{E}'
    cfg['model']['r']=rank
    if rank!=32 and base=='W5':
        cfg['writer_fix']['value_head']=f'training_only_{rank}_to_1024_affine'
    cfg['task']['E']=E
    cfg['model']['rho']=.5 if arm in {'Rrho','OArho'} else .25
    cfg['smoke']=smoke
    cfg['eval']['episodes']=100 if smoke else 1000
    cfg['read_fix']=dict(arm=arm,base=base,read_weight=1 if arm in {'R','Rrho','RK'} else 0,
                         decorrelation_weight=1 if arm=='RK' else 0,
                         read_target='stop-gradient raw written true-fact value',epsilon=1e-6)
    return cfg


class ReadFixExperiment(WriterFixExperiment):
    def __init__(self,config,arm,**kwargs):
        if config['read_fix']['arm']!=arm or arm not in ARMS:
            raise ValueError('read-fix arm/config mismatch')
        super().__init__(config,config['read_fix']['base'],**kwargs)
        self.read_arm=arm

    def writer_signature(self):
        return {**super().writer_signature(),'read_fix':self.read_arm}

    def coefficient_loss(self,batch,x):
        """SQ2's weighted coefficient loss, or None when `read_fix.coefficient_weight` is absent or zero (every other study)."""
        weight=self.config['read_fix'].get('coefficient_weight')
        return None if not weight else float(weight)*coefficient_loss(self,batch,x)


def question_terms(experiment,batch,x):
    """Raw linear query/read at the final question token, preserving gradients."""
    last=batch.question_mask.long().sum(-1)-1
    B,Q=last.shape
    final=x[torch.arange(B,device=x.device)[:,None],torch.arange(Q,device=x.device)[None,:],last]
    u=experiment.adapter.u(final.float())
    q=experiment.adapter.query_code(F.linear(u,experiment.adapter.Q))
    matrix=batch.snapshot.states[experiment.layer]
    stored=torch.einsum('bij,bqj->bqi',matrix,q)
    target=true_fact_indices(batch.arrays,x.device)
    clean=batch.values.gather(1,target[...,None].expand(-1,-1,batch.values.shape[-1]))
    deployed=clean if experiment.arm=='OA' else stored
    return dict(u=u,q=q,stored=stored,m=deployed,clean=clean,target=target)


def raw_read_loss(read,target,eps=1e-6):
    target=target.detach()
    return ((read-target).square().sum(-1)/(target.square().sum(-1)+eps)).mean()


def key_decorrelation(keys,facts,eps=1e-6):
    unit=F.normalize(keys,dim=-1,eps=eps)
    gram=unit@unit.transpose(-1,-2)
    addresses=torch.as_tensor(facts[:,:,:2],device=keys.device)
    distinct=(addresses[:,:,None,:]!=addresses[:,None,:,:]).any(-1)
    return gram.square()[distinct].mean() if distinct.any() else keys.sum()*0


def extra_losses(experiment,batch,x):
    zero=batch.values.new_zeros(())
    if experiment.read_arm in {'OArho','OA','W5'}:
        return zero,zero
    terms=question_terms(experiment,batch,x)
    read=raw_read_loss(terms['m'],terms['clean'])
    decor=key_decorrelation(batch.keys,batch.arrays.facts) if experiment.read_arm=='RK' else zero
    return read,decor


COEFFICIENT_EPS=1e-6  # docs/prereg/SQ2.md: the epsilon of the directional coefficient loss


def coefficient_loss(experiment,batch,x):
    """SQ2: mean over queries of || d/(||d|| + eps) - e_t ||^2, with d the exact read coefficients of the deployed store
    (Mq = sum_i d_i v_i, float64, graph kept) and e_t the basis vector of the true fact among the written facts."""
    terms=question_terms(experiment,batch,x)
    d=store_read(experiment.adapter,batch.keys,batch.values,terms['q'])['d']
    target=F.one_hot(terms['target'],d.shape[-1]).to(d.dtype)
    return (d/(d.norm(dim=-1,keepdim=True)+COEFFICIENT_EPS)-target).square().sum(-1).mean()


@torch.no_grad()
def coefficient_statistics(experiment,batch,x):
    """SQ2 logging (means over the batch's queries): query norm, direction cosine d_t/||d||, coefficient cancellation
    sum_i |d_i| ||v_i|| / ||sum_i d_i v_i|| (large when the coefficients cancel), and the log10 conditioning of the written keys."""
    terms=question_terms(experiment,batch,x)
    adapter=experiment.adapter
    exact=store_read(adapter,batch.keys,batch.values,terms['q'])
    d=exact['d']
    target=terms['target']
    cosine=d.gather(-1,target[...,None]).squeeze(-1)/d.norm(dim=-1).clamp_min(1e-12)
    mass=(d.abs()*batch.values.double().norm(dim=-1)[:,None,:]).sum(-1)
    cancellation=mass/exact['reconstructed'].norm(dim=-1).clamp_min(1e-12)
    ridge=adapter.cfg.store=='ridge'
    cond=gram_conditions(batch.keys.detach(),adapter.lam.detach() if ridge else None,adapter.cfg.store_eta if ridge else None)
    result=dict(query_norm=float(terms['q'].norm(dim=-1).mean()),direction_cosine=float(cosine.mean()),
        coefficient_cancellation_median=float(cancellation.median()),key_gram_log10_cond_median=float(cond['gram'].log10().median()))
    if ridge:
        result['ridge_log10_cond_median']=float(cond['ridge'].log10().median())
    return result


def gram_read(keys,values,q,lam,beta):
    """Prop11.2 exact triangular coefficients; no softmax or normalized query."""
    keys,values,q=keys.double(),values.double(),q.double()
    lam,beta=lam.double(),beta.double()
    gram=keys@keys.transpose(-1,-2)
    b=torch.einsum('ber,bqr->bqe',keys,q)
    E=keys.shape[1]
    system=torch.eye(E,device=keys.device,dtype=keys.dtype)+beta*torch.triu(gram,diagonal=1)
    a=torch.linalg.solve_triangular(system[:,None],b[...,None],upper=True).squeeze(-1)
    ages=torch.arange(E-1,-1,-1,device=keys.device)
    d=beta*lam.pow(ages)*a
    reconstructed=torch.einsum('bqe,ber->bqr',d,values)
    return dict(b=b,a=a,d=d,gram=gram,reconstructed=reconstructed)


def store_read(adapter,keys,values,q):
    """Exact coefficients of the deployed read for the adapter's store (WP-L7 adds the two least-squares stores)."""
    store=adapter.cfg.store
    if store=='delta':
        return gram_read(keys,values,q,adapter.lam,adapter.beta)
    if store=='ridge':
        result=ridge_coefficients(keys,q,adapter.lam,adapter.cfg.store_eta)
    else:
        result=sweep_coefficients(keys,q,adapter.lam,adapter.cfg.store_sweeps)
    result['reconstructed']=torch.einsum('bqe,ber->bqr',result['d'],values.double())
    return result


def diagnostics(experiment,batch,x):
    terms=question_terms(experiment,batch,x)
    adapter=experiment.adapter
    exact=store_read(adapter,batch.keys,batch.values,terms['q'])
    target=terms['target']
    coeff=exact['d'].gather(-1,target[...,None]).squeeze(-1)
    signal=coeff[...,None]*terms['clean'].double()
    interference=exact['reconstructed']-signal
    Ru=F.linear(terms['u'],adapter.R)
    Um=F.linear(terms['m'],adapter.U)
    g=adapter.cfg.rho*(adapter._phi(Ru+Um)-adapter._phi(Ru))
    Bg=F.linear(g,adapter.B)*adapter.cfg.alpha
    E=batch.keys.shape[1]
    eye=torch.eye(E,device=x.device,dtype=torch.bool)
    cos=F.normalize(batch.keys,dim=-1)@F.normalize(batch.keys,dim=-1).transpose(-1,-2)
    pair=cos[:,~eye]
    norm=lambda a:a.norm(dim=-1)
    result=dict(q_norm=norm(terms['q']),true_value_norm=norm(terms['clean']),
        read_norm=norm(terms['m']),u_norm=norm(terms['u']),Ru_norm=norm(Ru),Um_norm=norm(Um),
        gM_norm=norm(g),BgM_norm=norm(Bg),
        relative_read_error=(terms['m']-terms['clean']).square().sum(-1)/(terms['clean'].square().sum(-1)+1e-6),
        gram_relative_error=norm(exact['reconstructed']-terms['stored'])/norm(terms['stored']).clamp_min(1e-8),
        gram_absolute_error=(exact['reconstructed']-terms['stored']).abs().amax(-1),
        target_b=exact['b'].gather(-1,target[...,None]).squeeze(-1),
        target_a=exact['a'].gather(-1,target[...,None]).squeeze(-1),target_d=coeff,
        target_age=E-1-target,
        value_SIR=signal.square().sum(-1)/interference.square().sum(-1).clamp_min(1e-12),
        parallel_interference=(signal*interference).sum(-1)/signal.square().sum(-1).clamp_min(1e-12),
        key_gram_rms=pair.square().mean(-1).sqrt()[:,None].expand_as(target),
        key_gram_max_abs=pair.abs().amax(-1)[:,None].expand_as(target))
    if batch.base_keys is not None:
        # DPFP: the metrics above are in the feature space; also report the projected key space.
        base=F.normalize(batch.base_keys,dim=-1)
        base_pair=(base@base.transpose(-1,-2))[:,~eye]
        result.update(key_gram_rms_base=base_pair.square().mean(-1).sqrt()[:,None].expand_as(target),
                      key_gram_max_abs_base=base_pair.abs().amax(-1)[:,None].expand_as(target))
    if adapter.cfg.store!='delta' or adapter.cfg.unit_query:
        # LSQ arms (U, K4, RG): conditioning of the key Gram (and of the weighted ridge system for RG), as log10.
        ridge=adapter.cfg.store=='ridge'
        cond=gram_conditions(batch.keys.detach(),adapter.lam.detach() if ridge else None,
                             adapter.cfg.store_eta if ridge else None)
        result.update(key_gram_log10_cond=cond['gram'].log10()[:,None].expand_as(target))
        if ridge:
            result.update(ridge_log10_cond=cond['ridge'].log10()[:,None].expand_as(target))
    if experiment.value_head is not None:
        answers=torch.as_tensor(batch.arrays.answer,device=x.device)
        result.update(clean_value_correct=experiment.value_head(terms['clean']).argmax(-1)==answers,
                      actual_read_correct=experiment.value_head(terms['m']).argmax(-1)==answers)
    return result,exact


def controlled_score(experiment,batch,condition='original',donor=None):
    """Intervene on actual deployed memory, including OA's bypassed value."""
    if condition not in {'original','zero','shuffle'} or (condition=='shuffle' and donor is None):
        raise ValueError('memory condition requires its explicit donor')
    if experiment.arm=='OA':
        source=batch if donor is None else donor
        target=true_fact_indices(source.arrays,experiment.device)
        value=source.values.gather(1,target[...,None].expand(-1,-1,source.values.shape[-1]))
        experiment.adapter.oracle_values=(torch.zeros_like(value) if condition=='zero' else value).flatten(0,1)
    elif condition!='original':
        state=batch.snapshot.states[experiment.layer]
        memory=torch.zeros_like(state) if condition=='zero' else donor.snapshot.states[experiment.layer]
        batch=replace(batch,snapshot=replace(batch.snapshot,states={experiment.layer:memory}))
    experiment.adapter.phase='read'
    try:
        return LLMExperiment.score(experiment,batch,1,capture_inputs=True)
    finally:
        experiment.adapter.phase='idle'
        experiment.adapter.oracle_values=None
