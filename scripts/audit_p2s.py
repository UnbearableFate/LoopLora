"""P2-S evaluation integrity checks; no confirmation access before release."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_p1s_main import file_hash, require
from looplora.infra.seeds import stable_hash


def audit_evaluation(folder: Path,expected_config: dict,*,samples_only: bool=True) -> tuple[list[dict],dict]:
    from looplora.sym.stats import load_samples
    metadata=json.loads((folder/('evaluation.json' if samples_only else 'metrics.json')).read_text())
    c=metadata.get('config') or json.loads((folder/'config.json').read_text())['config']
    require(c==expected_config and metadata['config_hash']==stable_hash(c),'evaluation configuration changed')
    if samples_only:
        require(metadata['aggregation_deferred'] and 'table' not in metadata
                and not (folder/'metrics.json').exists(),'confirmation already aggregated')
    path=folder/'samples.npz'
    require(file_hash(path)==metadata['samples_hash'],'sample seal changed')
    with np.load(path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
    a=load_samples(a)
    settings=c['eval'];split=settings['splits'][0];depths=settings['depths'];hops=settings['hops']
    n=settings['episodes']
    require(settings['splits']==[split] and not settings['counterfactuals'],'only original single split expected')
    require(set(a['split'])=={split} and set(a['condition'])=={'original'},'unexpected sample groups')
    require(len(a['correct'])==n*len(hops)*len(depths),'sample count changed')
    require(np.array_equal(a['correct'],a['prediction']==a['answer']) and
            np.array_equal(a['full_correct'],a['full_prediction']==a['answer']),'saved correctness differs')
    require(np.array_equal(a['original_answer'],a['answer']) and
            np.array_equal(a['original_prediction'],a['prediction']),'original source fields differ')
    require(set(a['L'])==set(depths) and bool(np.all(a['candidate_size']>0)),'depth or candidates differ')
    reference=None;table=[]
    for d in depths:
        i=np.flatnonzero(a['L']==d)
        i=i[np.lexsort((a['query_id'][i],a['episode_id'][i]))]
        require(len(np.unique(a['episode_id'][i]))==n,'episode count differs')
        require(np.array_equal(a['query_id'][i],np.tile(np.arange(len(hops)),n)) and
                np.array_equal(a['hop'][i],np.tile(hops,n)),'hop/query coverage differs')
        labels=np.stack([a[k][i] for k in ('episode_id','query_id','hop','answer','candidate_size')],axis=1)
        require(reference is None or np.array_equal(reference,labels),'depth-dependent query labels')
        reference=labels
        for h in hops:
            take=i[a['hop'][i]==h]
            table.append(dict(hop=int(h),L=int(d),n=len(take),accuracy=float(a['correct'][take].mean()),
                              full_accuracy=float(a['full_correct'][take].mean()),
                              chance_accuracy=float((1./a['candidate_size'][take]).mean())))
    seen=set();snapshots={}
    require(metadata['gate0_passed'] is True,'immutable-depth gate failed')
    for row in metadata['memory_hashes']:
        key=row['batch_start'],row['L']
        require(row['split']==split and row['condition']=='original' and key not in seen,'snapshot scope differs')
        seen.add(key)
        values=tuple(row[k] for k in ('x','u','memory','write_count'))
        require(all(isinstance(v,str) and len(v)==32 for v in values),'invalid snapshot digest')
        require(row['batch_start'] not in snapshots or snapshots[row['batch_start']]==values,
                'prepared x/u/memory/write-count changes across depths')
        snapshots[row['batch_start']]=values
    require(seen=={(b,d) for b in range(0,n,settings['batch_size']) for d in depths},'snapshot coverage differs')
    if not samples_only:
        saved={(r['hop'],r['L']):r for r in metadata['table']}
        for row in table:
            require(all(saved[row['hop'],row['L']][k]==v for k,v in row.items()),'samples do not reproduce saved table')
    return table,dict(arrays=a,labels=reference,sample_rows=len(a['correct']),snapshots=len(snapshots))
