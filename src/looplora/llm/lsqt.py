"""WP-L12 LSQT arms (docs/prereg/LSQT.md): WP-L4's R at r = 128 with unit queries on a least-squares store, trained from scratch.

The configuration is lsq.configuration's (R + unit query + the K4 or ridge store of core/memory.py) with the registered arm names
and etas; the old LSQ registry, its etas and its config hashes are untouched.
"""
from __future__ import annotations

from .lsq import LSQExperiment,SWEEPS,STORE_RULES
from .lsqt_release import ETA_BOUNDS,RANK,SPECS
from .read_fix import ReadFixExperiment,configuration as read_fix_configuration


def configuration(arm,steps,*,E=16,smoke=False):
    if arm not in SPECS or E!=16:
        raise ValueError('LSQT requires a registered arm at E16')
    store,eta=SPECS[arm]
    if eta is not None and not (ETA_BOUNDS[0]<=eta<=ETA_BOUNDS[1]):
        raise ValueError(f'eta must lie in {ETA_BOUNDS}')
    cfg=read_fix_configuration('R',steps,E=E,smoke=smoke,rank=RANK)
    cfg['name']=f'lsqt-{arm}-E{E}'
    cfg['model']['unit_query']=True
    if store=='ridge':
        cfg['model'].update(store='ridge',store_eta=float(eta))
    elif store=='sweep':
        cfg['model'].update(store='sweep',store_sweeps=SWEEPS)
    cfg['lsqt']=dict(arm=arm,base_arm='R',rank=RANK,store=store,eta=None if eta is None else float(eta),
        sweeps=SWEEPS if store=='sweep' else None,store_rule=STORE_RULES[store],
        query='unit L2 normalization of Q z at read (eps 1e-6)',parameter_free=True)
    return cfg


class LSQTExperiment(LSQExperiment):
    """R training and diagnostics (LSQExperiment's conditioning log); read_arm carries the LSQT arm label into every record."""
    def __init__(self,config,arm,**kwargs):
        spec,model=config.get('lsqt'),config['model']
        if arm not in SPECS or spec is None or spec['arm']!=arm or 'lsq' in config:
            raise ValueError('LSQT arm/config mismatch')
        store,eta=SPECS[arm]
        if (spec['rank'],spec['store'],spec['eta'],model['r'],model.get('store','delta'),model.get('unit_query'),
                model.get('dpfp_nu',0))!=(RANK,store,eta,RANK,store,True,0):
            raise ValueError('LSQT arm/config mismatch')
        registered={'ridge':{'store_eta':eta},'sweep':{'store_sweeps':SWEEPS},'delta':{}}[store]
        if {k:model[k] for k in ('store_eta','store_sweeps') if k in model}!=registered:
            raise ValueError('LSQT arm/config mismatch')
        ReadFixExperiment.__init__(self,config,'R',**kwargs)
        self.read_arm=arm
