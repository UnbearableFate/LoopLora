"""WP-L12 SQ2 arms (docs/prereg/SQ2.md): D128-3 continued with delta or ridge storage on the DPFP features, with or without the
directional read-coefficient loss. Every arm starts from the parent checkpoint (adapter, training-only value head, AdamW state)."""
from __future__ import annotations

from pathlib import Path

import torch

from looplora.infra.seeds import stable_hash
from .dpfp import DPFPExperiment,configuration as dpfp_configuration
from .read_fix import COEFFICIENT_EPS,coefficient_statistics
from .sq2_release import ETA_BOUNDS,PARENT,SPECS,WEIGHT_BOUNDS,file_digest


def configuration(arm,steps,*,eta=None,w_c=None,smoke=False,numerics_digest=None):
    """D128-3's config at `steps` (the continuation length) with the registered changes: the ridge store (eta) and/or the
    coefficient loss (weight w_c). eta and w_c are given exactly for the arms that use them."""
    if arm not in SPECS:
        raise ValueError('SQ2 requires a registered arm')
    store,coefficient=SPECS[arm]
    if (store=='ridge')!=(eta is not None) or coefficient!=(w_c is not None):
        raise ValueError('eta belongs to the ridge arms and w_c to the coefficient arms, and to no other')
    if eta is not None and not (ETA_BOUNDS[0]<=eta<=ETA_BOUNDS[1]):
        raise ValueError(f'eta must lie in {ETA_BOUNDS}')
    if w_c is not None and not (WEIGHT_BOUNDS[0]<=w_c<=WEIGHT_BOUNDS[1]):
        raise ValueError(f'w_c must lie in {WEIGHT_BOUNDS}')
    cfg=dpfp_configuration(PARENT['arm'],steps,E=16,smoke=smoke)
    cfg['name']=f'sq2-{arm}'
    if store=='ridge':
        cfg['model'].update(store='ridge',store_eta=float(eta))
    if coefficient:
        cfg['read_fix'].update(coefficient_weight=float(w_c),coefficient_eps=COEFFICIENT_EPS)
    cfg['sq2']=dict(arm=arm,store=store,coefficient=coefficient,eta=None if eta is None else float(eta),
        coefficient_weight=None if w_c is None else float(w_c),steps=steps,
        parent=dict(arm=PARENT['arm'],steps=PARENT['steps'],config_hash=PARENT['config_hash'],digests=dict(PARENT['digests'])),
        start='adapter, training-only value head and AdamW state of the parent checkpoint',
        data="fresh train stream derive_seed(17, 'llm', 'sq2-continuation'), batch 32, the same for every arm",
        lr='constant 3e-3 (the parent lr after its warmup; no new warmup)',numerics=numerics_digest)
    return cfg


class SQ2Experiment(DPFPExperiment):
    """The continued D128-3 writer; read_arm carries the SQ2 arm label into every record."""
    def __init__(self,config,arm,**kwargs):
        spec,model,read=config.get('sq2'),config['model'],config['read_fix']
        if arm not in SPECS or spec is None or spec['arm']!=arm:
            raise ValueError('SQ2 arm/config mismatch')
        store,coefficient=SPECS[arm]
        registered_eta=spec['eta'] if store=='ridge' else None
        if (spec['store'],spec['coefficient'],model.get('store','delta'),model.get('store_eta'),read.get('coefficient_weight'),
                config['dpfp']['arm'],model.get('unit_query',False))!=(
                store,coefficient,store,registered_eta,spec['coefficient_weight'] if coefficient else None,PARENT['arm'],False):
            raise ValueError('SQ2 arm/config mismatch')
        if (store=='ridge')!=(spec['eta'] is not None) or coefficient!=(spec['coefficient_weight'] is not None):
            raise ValueError('SQ2 arm/config mismatch')
        super().__init__(config,PARENT['arm'],**kwargs)
        self.read_arm=arm
        self._observed={}

    def observe(self,batch,x):
        """Called by read_fix_run.fit at every logged step (no grad): query norm, direction cosine, cancellation, conditioning."""
        self._observed=coefficient_statistics(self,batch,x)

    def log_extras(self):
        return dict(self._observed)


def load_continuation(experiment,folder=None,digests=None):
    """Load the parent checkpoint into the experiment (adapter, training-only value head) and return its AdamW state.

    The files' blake2b-128 digests and the checkpoint's config hash and step must be the registered parent's."""
    folder=Path(PARENT['folder'] if folder is None else folder)
    digests=PARENT['digests'] if digests is None else digests
    for name,expected in digests.items():
        found=file_digest(folder/name)
        if found!=expected:
            raise ValueError(f'{name}: blake2b-128 {found} differs from the registered {expected}')
    payload=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    if payload['arm']!=PARENT['arm'] or payload['step']!=PARENT['steps'] or stable_hash(payload['config'])!=PARENT['config_hash']:
        raise ValueError('the checkpoint is not the registered D128-3 final checkpoint')
    if payload['value_head'] is None or experiment.value_head is None:
        raise ValueError('the continuation needs the training-only value head')
    experiment.adapter.load_state_dict(payload['adapter'],strict=True)
    experiment.value_head.load_state_dict(payload['value_head'],strict=True)
    return payload['optimizer']


def load_inference(path,*,model=None,tokenizer=None):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('read_fix_version')!=1 or 'value_head' in payload or 'optimizer' in payload:
        raise ValueError('expected an SQ2 inference export')
    experiment=SQ2Experiment(payload['config'],payload['arm'],model=model,tokenizer=tokenizer)
    try:
        if payload['signature']!=experiment.writer_signature():
            raise ValueError('SQ2 signature mismatch')
        experiment.adapter.load_state_dict(payload['adapter'],strict=True)
        experiment.value_head=None
    except BaseException:
        experiment.close()
        raise
    return experiment
