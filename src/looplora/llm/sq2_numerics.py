"""WP-L12 SQ2 numerics (docs/prereg/SQ2.md "Fixed before any validation result"): eta of the ridge arms and the weight w_c of the
coefficient loss, from training-only numerics at the parent checkpoint. Train split only; no validation or test episode. Compute only."""
from __future__ import annotations

import numpy as np
import torch

from looplora.core.memory import ridge_coefficients
from looplora.infra.seeds import derive_seed
from .prefetch import BatchStream
from .read_fix import coefficient_loss,coefficient_statistics,extra_losses,question_terms
from .sq2 import SQ2Experiment,configuration,load_continuation
from .sq2_release import ETA_GRID,N_MAX,NUMERICS_BATCHES,choose_eta,round_weight
from .train import make_task_spec

SEED_LABELS=('sq2','numerics')  # derive_seed(17, 'sq2', 'numerics'): a train stream separate from the continuation's


def stream_batches(experiment,count):
    spec=make_task_spec(experiment.config)
    stream=BatchStream(spec,'train',32,derive_seed(17,*SEED_LABELS),count,tokenizer=experiment.tokenizer,workers=2)
    with stream:
        for _ in range(count):
            arrays,tokens=next(stream)
            yield experiment.prepare(arrays,tokens=tokens)


def gradient_norm(loss,params):
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return float(torch.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None)))


def read_error(experiment,batch,terms,eta):
    """Normalized read error of the ridge store over the batch's written keys and values with the checkpoint's own queries."""
    adapter=experiment.adapter
    d=ridge_coefficients(batch.keys.detach(),terms['q'].detach(),adapter.lam.detach(),eta)['d']
    read=torch.einsum('bqe,ber->bqr',d,batch.values.detach().double())
    clean=terms['clean'].detach().double()
    return ((read-clean).square().sum(-1)/(clean.square().sum(-1)+1e-6))


def measure(experiment,count,*,etas=()):
    """Per training batch: the gradient-norm ratio ||grad L_read|| / ||grad L_coef|| and the coefficient statistics, and, for each
    eta, the mean normalized read error of the ridge store (eta-curve sums)."""
    params=[p for p in experiment.adapter.parameters() if p.requires_grad]
    ratios,stats,errors={eta:[] for eta in etas},[]
    deployed=[]
    for batch in stream_batches(experiment,count):
        logits,x=experiment.score(batch,1,capture_inputs=True)
        read,_=extra_losses(experiment,batch,x)
        coefficient=coefficient_loss(experiment,batch,x)
        ratios.append(gradient_norm(read,params)/gradient_norm(coefficient,params))
        with torch.no_grad():
            terms=question_terms(experiment,batch,x)
            for eta in etas:
                errors[eta].append(float(read_error(experiment,batch,terms,eta).mean()))
            stats.append(coefficient_statistics(experiment,batch,x))
            deployed.append(float(((terms['m']-terms['clean']).square().sum(-1)/(terms['clean'].square().sum(-1)+1e-6)).mean()))
    keys=stats[0]
    return dict(ratios=ratios,statistics={k:float(np.mean([s[k] for s in stats])) for k in keys},
                eta_errors={eta:float(np.mean(v)) for eta,v in errors.items()},deployed_read_error=float(np.mean(deployed)),
                batches=count,episodes=32*count)


def numerics(*,count=NUMERICS_BATCHES,model=None,tokenizer=None,folder=None,digests=None):
    """The whole job: eta (delta-trained queries, ridge store, RFX's grid), w_c (delta store), the ridge store's ratio (reported)."""
    shared=dict(model=model,tokenizer=tokenizer)
    delta=SQ2Experiment(configuration('S-DE',N_MAX),'S-DE',**shared)
    try:
        load_continuation(delta,folder,digests)
        first=measure(delta,count,etas=ETA_GRID)
    finally:
        delta.close()
    curve=[dict(eta=eta,error=first['eta_errors'][eta]) for eta in ETA_GRID]
    eta=choose_eta(curve)
    w_c=round_weight(first['ratios'])
    ridge=SQ2Experiment(configuration('S-RC',N_MAX,eta=eta,w_c=w_c),'S-RC',**shared)
    try:
        load_continuation(ridge,folder,digests)
        second=measure(ridge,count)
    finally:
        ridge.close()
    ordered=sorted(first['ratios'])
    return dict(study='SQ2',seed_labels=list(SEED_LABELS),split='train',batches=count,episodes=first['episodes'],
        eta_curve=curve,eta=eta,eta_rule='lowest mean normalized read error of the ridge store with the checkpoint queries; ties larger; clip [1e-4, 1]',
        w_c=w_c,w_c_rule='median of ||grad read|| / ||grad coefficient|| over the batches (delta store), nearest power of two, clip [1/16, 16]',
        gradient_ratio=dict(median=float(np.median(first['ratios'])),min=ordered[0],max=ordered[-1],delta_store=first['ratios']),
        ridge_store_gradient_ratio_median=float(np.median(second['ratios'])),
        deployed_read_error_delta_store=first['deployed_read_error'],
        statistics_delta_store=first['statistics'],statistics_ridge_store=second['statistics'])
