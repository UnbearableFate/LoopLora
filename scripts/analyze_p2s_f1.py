#!/usr/bin/env python3
"""Release-gated sharing, training-depth and shortcut-free budget contrasts."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import fmean

import numpy as np

import analyze_p1s_main as shared
from audit_p2s import audit_evaluation
from looplora.infra.seeds import stable_hash
import run_p2s_f1 as driver


def verify_release() -> tuple[dict,dict]:
    trained=driver.all_training()
    plan=shared.read_json(driver.RUNS/'evaluation_plan.json')
    barrier=shared.read_json(driver.RUNS/'barrier.json')
    receipt=shared.read_json(driver.RUNS/'training_barrier.json')
    shared.require(len(plan['runs'])==42 and plan['training_plan_hash']==stable_hash(trained)
        and receipt['status']=='passed' and receipt['training_stages']==24
        and receipt['evaluation_plan_hash']==stable_hash(plan),'training barrier differs')
    names={r['name'] for r in plan['runs']}
    shared.require(barrier['status']=='passed' and barrier['evaluation_plan_hash']==stable_hash(plan)
        and len(barrier['runs'])==42 and set(barrier['runs'])==names,'global sample barrier differs')
    paths=[]
    for r in plan['runs']:
        folder=Path(r['run_dir']);rec=shared.read_json(folder/'record.json')
        meta=shared.read_json(folder/'confirmation/evaluation.json')
        shared.require(rec['status']=='complete' and rec['checkpoint_unchanged']
            and meta['config']==driver.confirmation_config(r['config'])
            and meta['checkpoint_config']==r['config']
            and meta['aggregation_deferred'] and 'table' not in meta
            and not (folder/'confirmation/metrics.json').exists(),'confirmation scope/state differs')
        path=folder/'confirmation/samples.npz'
        shared.require(path.is_file(),'missing blind samples')
        paths.append(str(path))
    shared.require(len(barrier['confirmation_samples'])==42
        and set(barrier['confirmation_samples'])==set(paths),'barrier sample membership differs')
    return plan,barrier


def endpoints(plan: dict,cache) -> list[dict]:
    def select(source,w,v,r=32,m=32):
        rows=[a for a in plan['runs'] if (a['source'],a['writer'],a['variant'],a['r'],a['r_mem'])==(source,w,v,r,m)]
        shared.require(len(rows)==3 and {a['seed'] for a in rows}==set(driver.SEEDS),'endpoint seed matrix differs')
        return rows
    result=[]
    comparisons=[]
    for w in driver.WRITERS:
        comparisons.append(('F-E1',w,select('p2s-f1' if w=='joint_fact' else 'p2s',w,'B8'),
            select('p2s-f1',w,'M1'),4,4,('reach','hop1','long')))
        comparisons.append(('F-E2',w,select('p2s-f1',w,'M1'),
            select('f2' if w=='joint_fact' else 'wph',w,'M1'),4,4,('reach','hop1')))
    for v,r,m in (('B6',64,64),('B6',128,128),('B7',32,64),('B7',32,128)):
        comparisons.append(('F-E3','joint_fact',select('f2','joint_fact','M1'),
            select('p2s-f1','joint_fact',v,r,m),4,1,('reach','hop1','long')))
    subsets=dict(reach=(2,3),hop1=(1,),long=(4,5,6,7,8))
    for endpoint,w,a,b,La,Lb,names in comparisons:
        for subset in names:
            result.append(dict(endpoint=endpoint,writer=w,variant_a=a[0]['variant'],variant_b=b[0]['variant'],
                source_a=a[0]['source'],source_b=b[0]['source'],r_b=b[0]['r'],r_mem_b=b[0]['r_mem'],
                depth_a=La,depth_b=Lb,subset=subset,hops=list(subsets[subset]),
                decision_endpoint=subset=='reach',
                statistics=shared.paired(cache,a,b,'confirmation',La,Lb,subsets[subset])))
    return result


def decisions(rows: list[dict]) -> dict:
    def low(endpoint,writer):
        found=[r for r in rows if r['endpoint']==endpoint and r['writer']==writer and r['subset']=='reach']
        shared.require(len(found)==1,'decision endpoint missing or duplicated')
        return found[0]['statistics']['gate']['ci_low']
    sharing={w:low('F-E1',w)>0 for w in driver.WRITERS}
    reader='B8' if sharing['joint'] and sharing['joint_fact'] else 'M1'
    depth={w:('fixed L=4' if low('F-E2',w)>0 else 'mixed L=1/2/4') for w in driver.WRITERS}
    capability=[r for r in rows if r['endpoint']=='F-E3' and r['subset']=='reach']
    shared.require(len(capability)==4 and
        {(r['variant_b'],r['r_b'],r['r_mem_b']) for r in capability}==
        {('B6',64,64),('B6',128,128),('B7',32,64),('B7',32,128)},'capability matrix differs')
    passed=all(r['statistics']['gate']['ci_low']>0 for r in capability)
    return dict(reader_for_P1L=reader,sharing_CI_positive=sharing,
        training_depth_by_writer=depth,
        training_depth_rule='Use the listed F-E2 outcome for the writer carried into P1-L; writer choice is not selected by this follow-up',
        shortcut_free_capability_passed=passed,
        capability_scope='oracle and joint_fact at tested budgets' if passed else 'restricted to oracle codes',
        failing_capability_rows=[{k:r[k] for k in ('variant_b','r_b','r_mem_b')} for r in capability
            if r['statistics']['gate']['ci_low']<=0])


def analyze() -> None:
    plan,barrier=verify_release()
    raw=driver.RUNS/'analysis';out=driver.REPORTS
    shared.require(not (raw/'started.json').exists(),'P2-S-F1 analysis already started')
    driver.write(raw/'started.json',dict(created_at=datetime.now(timezone.utc).isoformat(),plan_hash=stable_hash(plan)))
    cache=shared.CompactCache(driver.RUNS/'analysis_cache',limit=12)
    reference=None;accuracy=[];sample_rows=0
    for r in plan['runs']:
        shared.require(shared.file_hash(Path(r['checkpoint']))==r['checkpoint_hash'],'checkpoint changed')
        folder=Path(r['run_dir'])/'confirmation'
        table,audit=audit_evaluation(folder,driver.confirmation_config(r['config']))
        shared.require(reference is None or np.array_equal(reference,audit['labels']),'models evaluate different episode/query/hop/answer/candidates')
        reference=audit['labels'];sample_rows+=audit['sample_rows']
        cache.save(r['name'],'confirmation',audit['arrays'],'test')
        accuracy.extend({k:r[k] for k in ('name','source','writer','variant','r','r_mem','seed','steps','descriptive_only')}|row for row in table)
        driver.write(folder/'metrics.json',dict(table=table,finalized_after_global_barrier=True,
            barrier_created_at=barrier['created_at']))
        del audit
    stats=endpoints(plan,cache)
    driver.write(raw/'endpoints.json',stats)
    driver.csv_table(out/'endpoints.csv',[
        {k:v for k,v in r.items() if k not in ('statistics','hops')}|
        dict(confidence=value['confidence'],difference=value['difference'],ci_low=value['ci_low'],ci_high=value['ci_high'],
             per_seed=json.dumps(value['per_seed'])) for r in stats for value in r['statistics'].values()])
    pooled=defaultdict(list)
    keys=('source','writer','variant','r','r_mem','steps','descriptive_only','hop','L')
    for r in accuracy: pooled[tuple(r[k] for k in keys)].append(r)
    driver.csv_table(out/'accuracy.csv',[dict(zip(keys,key))|dict(n=sum(r['n'] for r in rows),
        accuracy=fmean(r['accuracy'] for r in rows),full_accuracy=fmean(r['full_accuracy'] for r in rows),
        chance_accuracy=fmean(r['chance_accuracy'] for r in rows)) for key,rows in sorted(pooled.items())])
    decision=decisions(stats)
    driver.write(out/'decision.json',decision)
    driver.write(out/'audit.json',dict(status='passed',training_stages=24,reference_checkpoints=18,
        confirmation_sets=42,sample_rows=sample_rows,all_samples_sealed=True,
        all_tables_reproduced=True,all_depth_snapshots_identical=True,all_query_labels_paired=True,
        checkpoints_unchanged=True,confirmation_look_count_this_followup=1,
        confirmation_set_reused_from_P2S=True,
        uncertainty='Episode bootstrap conditions on the three fixed training seeds; confirmation set was already used in P2-S',
        training_barrier=shared.read_json(driver.RUNS/'training_barrier.json'),
        sample_barrier_created_at=barrier['created_at'],completed_at=datetime.now(timezone.utc).isoformat()))
    total=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    shared.require(total<=1_000_000,f'committed P2-S-F1 outputs exceed 1 MB: {total}')
    print(f'P2-S-F1 analysis complete: 42 models, {sample_rows} samples, decision={decision}, compact bytes={total}',flush=True)


if __name__=='__main__': analyze()
