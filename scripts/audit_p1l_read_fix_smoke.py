"""Timing/artifact-only WP-L4 smoke gate; do not expose accuracy before schedule lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from looplora.llm.read_fix import ARMS,configuration
from looplora.llm.run import _write_json,source_metadata
from looplora.sym.stats import load_samples


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rows=[]; reference={}; sources=set(); normalization=None
    for arm in ARMS:
        folder=args.root/arm
        rec=json.loads((folder/'record.json').read_text())
        assert rec['status']=='complete' and rec['smoke'] is True and rec['arm']==arm
        assert rec['config']==configuration(arm,823,smoke=True)
        assert rec['source_dirty'] is False
        sources.add(rec['git_commit'])
        training=rec['training']
        assert training['actual_steps']==823 and training['training_episodes']==823*32
        assert set(training['metrics'])=={str(round(823*k/5)) for k in range(1,6)}
        evaluation_seconds=[]
        for step,metric in training['metrics'].items():
            samples=load_samples(folder/f'val-{step}'/'samples.npz')
            assert len(samples['correct'])==100 and len(np.unique(samples['episode_id']))==100
            assert (samples['split']=='val').all() and (samples['hop']==1).all() and (samples['L']==1).all()
            assert (samples['arm']==arm).all()
            assert samples['gram_relative_error'].max()<1e-4
            assert all(np.isfinite(v).all() for v in samples.values() if v.dtype.kind in 'bif')
            if not reference: reference.update({k:samples[k] for k in ('episode_id','query_id','answer')})
            assert all(np.array_equal(samples[k],v) for k,v in reference.items())
            assert metric['correct']==int(samples['correct'].sum())
            evaluation_seconds.append(metric['seconds'])
        exported=torch.load(folder/'writer.pt',map_location='cpu',weights_only=True)
        assert exported['read_fix_version']==1 and exported['arm']==arm
        assert 'value_head' not in exported and 'optimizer' not in exported
        assert all(torch.isfinite(v).all() for v in exported['adapter'].values())
        if arm!='OArho':
            buffers=[exported['adapter'][k] for k in ('input_mean','input_std')]
            if normalization is None: normalization=buffers
            else: assert all(torch.equal(a,b) for a,b in zip(normalization,buffers))
        rows.append(dict(arm=arm,job_id=rec['job_id'],host=rec['host'],steps=823,
            steady_step_mean_seconds=training['steady_step_mean_seconds'],
            step_p95_seconds=training['step_p95_seconds'],total_seconds=training['seconds'],
            eval100_seconds=evaluation_seconds,artifact_gate_passed=True))
    assert len(sources)==1
    slow=max(row['steady_step_mean_seconds'] for row in rows)
    full_eval=max(max(row['eval100_seconds']) for row in rows)*10
    predicted=8225*slow*1.5+5*full_eval+90
    if predicted>3600: raise RuntimeError(f'provisional8225-step schedule fails1h gate:{predicted}')
    args.out.mkdir(parents=True,exist_ok=True)
    result=dict(**source_metadata(),science_commit=sources.pop(),rows=rows,quality_scores_exposed=False,
        smoke_fraction_steps=823/8225,smoke_fraction_evaluation=.1,
        scientific_steps=8225,scientific_eval_steps=[1645,3290,4935,6580,8225],
        sizing=dict(training_margin=1.5,setup_reserve_seconds=90,
                    conservative_eval1000_seconds=full_eval,expected_seconds_with_margin=predicted),
        source_and_sample_artifacts_verified=True)
    _write_json(args.out/'smoke-timing.json',result)
    print(json.dumps(dict(artifact_gate_passed=True,scientific_steps=8225,
                         expected_seconds_with_margin=predicted)),flush=True)
