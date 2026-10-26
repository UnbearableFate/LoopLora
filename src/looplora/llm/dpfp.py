"""WP-L5 DPFP arms: WP-L4's R on a memory addressed by DPFP key and query features."""
from __future__ import annotations

from .read_fix import ReadFixExperiment,configuration as read_fix_configuration

# Candidate order of docs/prereg/DPFP.md: r = 32 first, then r = 128 with the smaller feature map first.
ARMS=('D32-3','D128-1','D128-3')
SPECS={'D32-3':(32,3),'D128-1':(128,1),'D128-3':(128,3)}


def configuration(arm,steps,*,E=16,smoke=False):
    """R at rank r = r_mem with DPFP nu; every other field is read_fix.configuration('R', ...)."""
    if arm not in SPECS or E not in (8,16):
        raise ValueError('WP-L5 requires a registered DPFP arm at E16 or the conditional E8')
    rank,nu=SPECS[arm]
    cfg=read_fix_configuration('R',steps,E=E,smoke=smoke,rank=rank)
    cfg['name']=f'dpfp-{arm}-E{E}'
    cfg['model']['dpfp_nu']=nu
    cfg['dpfp']=dict(arm=arm,base_arm='R',nu=nu,rank=rank,key_dim=2*rank*nu,
        features='phi_nu(k)=[relu([k;-k])*roll(relu([k;-k]),j)]_{j=1..nu}',
        normalization='L2 of phi(k) at write and phi(q) at read',
        address_space='phi: InfoNCE, key metrics, raw-read loss on M q_hat',parameter_free=True)
    return cfg


class DPFPExperiment(ReadFixExperiment):
    """R training and diagnostics; read_arm carries the DPFP arm label into every record."""
    def __init__(self,config,arm,**kwargs):
        spec=config.get('dpfp')
        if (arm not in SPECS or spec is None or spec['arm']!=arm or spec['nu']!=SPECS[arm][1]
                or config['model'].get('dpfp_nu')!=spec['nu']):
            raise ValueError('DPFP arm/config mismatch')
        super().__init__(config,'R',**kwargs)
        self.read_arm=arm
