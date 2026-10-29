"""WP-L5 DPFP arms (and WP-L8's DPFP-X arms): WP-L4's R on a memory addressed by DPFP key and query features."""
from __future__ import annotations

from .dpfp_release import SPECS,XSPECS,arm_spec
from .read_fix import ReadFixExperiment,configuration as read_fix_configuration


def configuration(arm,steps,*,E=16,smoke=False):
    """R at rank r = r_mem with DPFP nu; every other field is read_fix.configuration('R', ...). DPFP-X arms (E16 only): X256 is
    rank 256 on the default backbone, X17 is rank 128 with the optional `backbone` (P5L's option, Qwen3-1.7B-Base)."""
    if arm not in SPECS and arm not in XSPECS:
        raise ValueError('WP-L5 requires a registered DPFP arm at E16 or the conditional E8; WP-L8 a registered DPFP-X arm at E16')
    if E not in ((8,16) if arm in SPECS else (16,)):
        raise ValueError('the DPFP-X arms run at E16 only')
    rank,nu,backbone=arm_spec(arm)
    cfg=read_fix_configuration('R',steps,E=E,smoke=smoke,rank=rank,extra_runs=((256,'R'),) if arm=='X256' else ())
    cfg['name']=f'dpfp-{arm}-E{E}'
    cfg['model']['dpfp_nu']=nu
    cfg['dpfp']=dict(arm=arm,base_arm='R',nu=nu,rank=rank,key_dim=2*rank*nu,
        features='phi_nu(k)=[relu([k;-k])*roll(relu([k;-k]),j)]_{j=1..nu}',
        normalization='L2 of phi(k) at write and phi(q) at read',
        address_space='phi: InfoNCE, key metrics, raw-read loss on M q_hat',parameter_free=True)
    if backbone is not None:
        cfg['backbone']=dict(backbone)
        cfg['dpfp']['backbone']=dict(backbone)
    return cfg


class DPFPExperiment(ReadFixExperiment):
    """R training and diagnostics; read_arm carries the DPFP arm label into every record."""
    def __init__(self,config,arm,**kwargs):
        spec=config.get('dpfp')
        if arm not in SPECS and arm not in XSPECS:
            raise ValueError('DPFP arm/config mismatch')
        if (spec is None or spec['arm']!=arm or spec['nu']!=arm_spec(arm)[1]
                or config['model'].get('dpfp_nu')!=spec['nu']
                or config.get('backbone')!=spec.get('backbone')
                or (arm in XSPECS and config['model']['r']!=arm_spec(arm)[0])):
            raise ValueError('DPFP arm/config mismatch')
        super().__init__(config,'R',**kwargs)
        self.read_arm=arm
