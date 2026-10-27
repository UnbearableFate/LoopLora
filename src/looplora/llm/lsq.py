"""WP-L7 LSQ arms: WP-L4's R objective with unit-normalized queries, on the deployed delta store (U) or on a
least-squares store (K4, RG)."""
from __future__ import annotations

import math

import numpy as np
import torch

from looplora.core.memory import gram_conditions
from .read_fix import ReadFixExperiment,configuration as read_fix_configuration

# Candidate order of docs/prereg/LSQ.md: rank 32 first, then 128. U (the deployed delta store with the unit query; it
# runs under RFX's normalization verdict) and K4 and RG (the least-squares stores; they run under the storage verdict) are
# mutually exclusive per rank; K4 (no new operator, no regularizer) comes before RG.
ARMS=('L32-U','L32-K4','L32-RG','L128-U','L128-K4','L128-RG')
SPECS={'L32-U':(32,'delta'),'L32-K4':(32,'sweep'),'L32-RG':(32,'ridge'),
       'L128-U':(128,'delta'),'L128-K4':(128,'sweep'),'L128-RG':(128,'ridge')}
SWEEPS=4
ETA_BOUNDS=(1e-4,1e-1)


def lsq_eta(rfx_eta):
    """Ridge strength of an LSQ arm: the RFX fitting-set selection for the arm's rank, clipped to [1e-4, 1e-1]."""
    if isinstance(rfx_eta,bool) or not math.isfinite(rfx_eta) or rfx_eta<=0:
        raise ValueError('the RFX-selected eta must be positive and finite')
    return min(max(float(rfx_eta),ETA_BOUNDS[0]),ETA_BOUNDS[1])


def check_arms(arms):
    """A set of triggered arms: each arm once, in candidate order, and per rank U alone or K4 together with RG."""
    arms=list(arms)
    if not arms or any(arm not in SPECS for arm in arms) or arms!=[a for a in ARMS if a in arms]:
        raise ValueError('triggered arms must be registered LSQ arms, each once, in candidate order')
    for rank in (32,128):
        stores=[SPECS[arm][1] for arm in arms if SPECS[arm][0]==rank]
        if stores not in ([],['delta'],['sweep','ridge']):
            raise ValueError(f'rank {rank}: the arms are U alone (normalization verdict) or K4 with RG (storage verdict)')
    return arms


def configuration(arm,steps,*,E=16,smoke=False,eta=None):
    """R at rank r = r_mem with unit queries, on the deployed delta store (U) or a least-squares store (K4, RG).

    Every other field is read_fix.configuration('R', ...). eta is the registered ridge strength (lsq_eta of RFX's
    selection; smoke runs use any value in the bounds) and must be given exactly for the ridge arms.
    """
    if arm not in SPECS or E not in (8,16):
        raise ValueError('WP-L7 requires a registered LSQ arm at E16 or the conditional E8')
    rank,store=SPECS[arm]
    if (store=='ridge')!=(eta is not None):
        raise ValueError('the RG arms need their registered eta; U and K4 have none')
    if eta is not None and not (ETA_BOUNDS[0]<=eta<=ETA_BOUNDS[1]):
        raise ValueError(f'eta must lie in {ETA_BOUNDS}')
    cfg=read_fix_configuration('R',steps,E=E,smoke=smoke,rank=rank)
    cfg['name']=f'lsq-{arm}-E{E}'
    cfg['model']['unit_query']=True
    if store=='ridge':
        cfg['model'].update(store='ridge',store_eta=float(eta))
    elif store=='sweep':
        cfg['model'].update(store='sweep',store_sweeps=SWEEPS)
    cfg['lsq']=dict(arm=arm,base_arm='R',rank=rank,store=store,eta=None if eta is None else float(eta),
        sweeps=SWEEPS if store=='sweep' else None,
        store_rule={'ridge':'ridge least squares, weights lam^(n-1-j) of the learned decay gate, float64 Cholesky solve',
                    'sweep':'the beta = 1 delta scan_final repeated 4 times (Kaczmarz), learned decay gate',
                    'delta':'the deployed decay-first delta scan_final of R, learned decay and write-strength gates, unchanged'}[store],
        query='unit L2 normalization of Q z at read (eps 1e-6)',parameter_free=True)
    return cfg


class LSQExperiment(ReadFixExperiment):
    """R training and diagnostics; read_arm carries the LSQ arm label into every record."""
    def __init__(self,config,arm,**kwargs):
        spec,model=config.get('lsq'),config['model']
        if arm not in SPECS or spec is None or spec['arm']!=arm:
            raise ValueError('LSQ arm/config mismatch')
        rank,store=SPECS[arm]
        if (spec['rank'],spec['store'],model.get('store','delta'),model.get('unit_query'),model.get('dpfp_nu',0))!=(
                rank,store,store,True,0):
            raise ValueError('LSQ arm/config mismatch')
        registered={'ridge':{'store_eta':spec['eta']},'sweep':{'store_sweeps':SWEEPS},'delta':{}}[store]
        if {k:model[k] for k in ('store_eta','store_sweeps') if k in model}!=registered or (
                store=='ridge' and spec['eta'] is None):
            raise ValueError('LSQ arm/config mismatch')
        super().__init__(config,'R',**kwargs)
        self.read_arm=arm

    def log_extras(self):
        """Conditioning of this step's written keys (log10), added to every training-log row by read_fix_run.fit."""
        keys=self.adapter.keys
        if keys is None:
            return {}
        ridge=self.adapter.cfg.store=='ridge'
        with torch.no_grad():
            cond=gram_conditions(keys.detach(),self.adapter.lam.detach() if ridge else None,
                                 self.adapter.cfg.store_eta if ridge else None)
        extras={}
        for name,label in (('gram','key_gram'),('ridge','ridge')):
            if name in cond:
                log=np.log10(cond[name].cpu().numpy())
                extras[f'{label}_log10_cond_median']=float(np.median(log))
                extras[f'{label}_log10_cond_max']=float(log.max())
        return extras
