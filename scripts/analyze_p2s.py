#!/usr/bin/env python3
"""Release-gated P2-S paired endpoints and primary/descriptive cost planes."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np

import analyze_p1s_main as shared
from audit_p2s import audit_evaluation
from looplora.infra.seeds import stable_hash
import run_p2s as driver


def verify_release() -> tuple[dict,dict]:
    trained=driver.all_training()
    plan=shared.read_json(driver.RUNS/'evaluation_plan.json')
    barrier=shared.read_json(driver.RUNS/'barrier.json')
    receipt=shared.read_json(driver.RUNS/'training_barrier.json')
    shared.require(len(plan['runs'])==84 and plan['training_plan_hash']==stable_hash(trained)
        and receipt['status']=='passed' and receipt['training_stages']==24
        and receipt['evaluation_plan_hash']==stable_hash(plan),'training barrier differs')
    names={r['name'] for r in plan['runs']}
    shared.require(barrier['status']=='passed' and barrier['evaluation_plan_hash']==stable_hash(plan)
        and len(barrier['runs'])==84 and set(barrier['runs'])==names,'global sample barrier differs')
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
    shared.require(len(barrier['confirmation_samples'])==84
        and set(barrier['confirmation_samples'])==set(paths),'barrier sample membership differs')
    return plan,barrier


def endpoints(plan: dict,cache) -> list[dict]:
    def select(source,w,v,r,m):
        rows=[a for a in plan['runs'] if (a['source'],a['writer'],a['variant'],a['r'],a['r_mem'])==(source,w,v,r,m)]
        shared.require(len(rows)==3 and {a['seed'] for a in rows}==set(driver.SEEDS),'endpoint seed matrix differs')
        return rows
    result=[]
    comparisons=[('E1',w,'B6',r,r,1) for w in driver.WRITERS for r in (64,128)]
    comparisons += [('E2','joint','B7',32,m,1) for m in (64,128)]
    comparisons += [('E3',w,'B8',32,32,4) for w in driver.WRITERS]
    for endpoint,w,v,r,m,L in comparisons:
        a=select('wph',w,'M1',32,32);b=select('p2s',w,v,r,m)
        for subset,hops in (('reach',(2,3)),('hop1',(1,)),('long',(4,5,6,7,8))):
            result.append(dict(endpoint=endpoint,writer=w,comparator=v,r=r,r_mem=m,depth_a=4,depth_b=L,
                subset=subset,hops=list(hops),statistics=shared.paired(cache,a,b,'confirmation',4,L,hops)))
    return result


def capability(endpoints: list[dict]) -> bool:
    chosen=[r for r in endpoints if r['endpoint'] in ('E1','E2') and r['subset']=='reach']
    shared.require(len(chosen)==6,'capability requires all six rank/state contrasts')
    return all(r['statistics']['gate']['ci_low']>0 for r in chosen)


def practical_rows(costs: list[dict]) -> list[dict]:
    # F1's 20000-step models contribute only descriptive curves. A single-step
    # point has L=1; extrapolated extra reads of B4/B6/B7 are not single-step.
    result=[]
    for a in costs:
        if a['source']!='wph' or a['variant']!='M1' or a['r']!=32: continue
        competitors=[b for b in costs if b['writer']==a['writer'] and b['batch_size']==a['batch_size']
                     and b['steps']==15000 and b['L']==1 and b['variant']!='B8']
        matches=[b for b in competitors if b['cuda_ms_mean']<=a['cuda_ms_mean']
                 and b['accuracy_hops1to3']>=a['accuracy_hops1to3']]
        result.append(dict(writer=a['writer'],batch_size=a['batch_size'],M1_L=a['L'],
            accuracy_hops1to3=a['accuracy_hops1to3'],cuda_ms_mean=a['cuda_ms_mean'],
            unmatched_by_equal_or_faster_single_step=not matches,
            matching_points=json.dumps([dict(variant=b['variant'],r=b['r'],r_mem=b['r_mem'],L=b['L']) for b in matches])))
    shared.require(len(result)==16,'practical check requires two writers, four depths, two batch sizes')
    return result


def cost_tables(plan: dict,accuracy: list[dict]) -> tuple[list[dict],list[dict]]:
    with (driver.RUNS/'benchmark/latency.csv').open() as f: raw=list(csv.DictReader(f))
    index={r['name']:r for r in plan['runs']}
    accum=defaultdict(list)
    for r in raw:
        run=index[r['name']]
        key=(run['name'],int(r['L']),int(r['batch_size']))
        accum[key].append(r)
    fields=('cuda_ms_mean','cuda_ms_median','cuda_ms_p95','wall_ms_mean','wall_ms_median','wall_ms_p95')
    counts=('trainable_parameters','registered_parameters','state_bytes_fp32','read_flops_per_query',
            'output_projection_flops','candidate_decode_flops')
    per_seed=[]
    cells=defaultdict(dict)
    for r in accuracy: cells[r['name'],r['L']][r['hop']]=r
    for (name,L,batch),rows in sorted(accum.items()):
        run=index[name]
        shared.require(len(rows)==(1 if batch==64 else 8) and
                       {int(r['latency_hop']) for r in rows}==({0} if batch==64 else set(range(1,9))),
                       'latency must cover batch64 and all eight batch-one hops')
        shared.require(all(int(r['warmup'])==5 and int(r['repeats'])==20 for r in rows),'timing protocol changed')
        for k in counts: shared.require(len({r[k] for r in rows})==1,'cost differs by timed query')
        values=cells[name,L]
        shared.require(set(values)==set(range(1,9)),'E4 accuracy hop coverage differs')
        totals={}
        for label,hops in (('hops1to3',(1,2,3)),('hops1to8',tuple(range(1,9)))):
            totals['n_queries_'+label]=sum(values[h]['n'] for h in hops)
            totals['n_correct_'+label]=sum(round(values[h]['n']*values[h]['accuracy']) for h in hops)
        per_seed.append({k:run[k] for k in ('name','source','writer','variant','r','r_mem','seed','steps','descriptive_only')}|
            dict(L=L,batch_size=batch,**totals,
                 accuracy_hops1to3=totals['n_correct_hops1to3']/totals['n_queries_hops1to3'],
                 accuracy_hops1to8=totals['n_correct_hops1to8']/totals['n_queries_hops1to8'],
                 **{k:int(rows[0][k]) for k in counts},
                 **{k:fmean(float(r[k]) for r in rows) for k in fields}))
    expected={(r['name'],d,b) for r in plan['runs'] for d in ([4] if r['variant']=='B8' else [1,2,4,8]) for b in (1,64)}
    shared.require(set(accum)==expected,'cost point matrix differs')
    groups=defaultdict(list)
    identity=('source','writer','variant','r','r_mem','steps','descriptive_only','L','batch_size')
    for r in per_seed: groups[tuple(r[k] for k in identity)].append(r)
    pooled=[]
    for key,rows in sorted(groups.items()):
        shared.require(len(rows)==3 and {r['seed'] for r in rows}==set(driver.SEEDS),'cost requires three seeds')
        for k in counts: shared.require(len({r[k] for r in rows})==1,'parameter/state/FLOP cost varies by seed')
        totals={k:sum(r[k] for r in rows) for k in
                ('n_correct_hops1to3','n_queries_hops1to3','n_correct_hops1to8','n_queries_hops1to8')}
        pooled.append(dict(zip(identity,key))|dict(n_seeds=3,**{k:rows[0][k] for k in counts},**totals,
            accuracy_hops1to3=totals['n_correct_hops1to3']/totals['n_queries_hops1to3'],
            accuracy_hops1to8=totals['n_correct_hops1to8']/totals['n_queries_hops1to8'],
            **{k:fmean(r[k] for r in rows) for k in fields}))
    return pooled,per_seed


def plot_cost(costs: list[dict],out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'M1':'#1765ad','B3':'#888888','B4':'#ba792b','B5':'#77732a','B6':'#ca4d31','B7':'#794d9c','B8':'#218343'}
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharey=True)
    for i,w in enumerate(driver.WRITERS):
        for j,batch in enumerate((64,1)):
            ax=axes[i,j]
            for variant,color in colors.items():
                rows=[r for r in costs if r['writer']==w and r['batch_size']==batch and r['variant']==variant]
                for descriptive,marker in ((False,'o'),(True,'x')):
                    chosen=[r for r in rows if r['descriptive_only']==descriptive]
                    if not chosen: continue
                    ax.scatter([r['cuda_ms_mean'] for r in chosen],[r['accuracy_hops1to3'] for r in chosen],
                               c=color,marker=marker,s=28,label=variant+(' (20k, descriptive)' if descriptive else ' (15k)'))
            primary=[r for r in costs if r['source']=='wph' and r['writer']==w and r['variant']=='M1' and r['batch_size']==batch]
            primary.sort(key=lambda r:r['L'])
            ax.plot([r['cuda_ms_mean'] for r in primary],[r['accuracy_hops1to3'] for r in primary],color=colors['M1'],linewidth=.8)
            for r in primary: ax.annotate(str(r['L']),(r['cuda_ms_mean'],r['accuracy_hops1to3']),fontsize=7,xytext=(3,3),textcoords='offset points')
            for r in costs:
                if r['writer']==w and r['batch_size']==batch and r['variant'] in ('B6','B7') and r['L']==1:
                    label=f"{r['variant']}/{'r' if r['variant']=='B6' else 'm'}{r['r'] if r['variant']=='B6' else r['r_mem']}"
                    ax.annotate(label,(r['cuda_ms_mean'],r['accuracy_hops1to3']),fontsize=6,
                                xytext=(3,-9 if r['r_mem']==64 else 5),textcoords='offset points')
            ax.set_title(f'{w}, batch {batch}')
            ax.set_xscale('log');ax.set_ylim(0,1);ax.grid(alpha=.2)
            ax.set_xlabel('GPU milliseconds / prepared query batch')
            if j==0: ax.set_ylabel('Accuracy, hops 1–3')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7)
    fig.suptitle('P2-S cost plane; numbers label M1/r32 read depth')
    fig.tight_layout(rect=(0,.14,1,.95))
    fig.savefig(out/'cost_plane.pdf');fig.savefig(out/'cost_plane.png',dpi=150);plt.close(fig)


def analyze() -> None:
    plan,barrier=verify_release()
    raw=driver.RUNS/'analysis';out=driver.REPORTS
    shared.require(not (raw/'started.json').exists(),'P2-S analysis already started')
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
        {k:r[k] for k in ('endpoint','writer','comparator','r','r_mem','depth_a','depth_b','subset')}|
        dict(confidence=s['confidence'],difference=s['difference'],ci_low=s['ci_low'],ci_high=s['ci_high'],
             per_seed=json.dumps(s['per_seed'])) for r in stats for s in r['statistics'].values()])
    pooled=defaultdict(list)
    keys=('source','writer','variant','r','r_mem','steps','descriptive_only','hop','L')
    for r in accuracy: pooled[tuple(r[k] for k in keys)].append(r)
    driver.csv_table(out/'accuracy.csv',[dict(zip(keys,key))|dict(n=sum(r['n'] for r in rows),
        accuracy=fmean(r['accuracy'] for r in rows),full_accuracy=fmean(r['full_accuracy'] for r in rows),
        chance_accuracy=fmean(r['chance_accuracy'] for r in rows)) for key,rows in sorted(pooled.items())])
    costs,per_seed=cost_tables(plan,accuracy)
    driver.csv_table(out/'cost.csv',costs);driver.csv_table(out/'cost_per_seed.csv',per_seed)
    practical=practical_rows(costs);driver.csv_table(out/'practical.csv',practical)
    cap=capability(stats)
    by_setting=[dict(writer=w,batch_size=b,any_primary_M1_point_unmatched=any(
        r['unmatched_by_equal_or_faster_single_step'] for r in practical if r['writer']==w and r['batch_size']==b))
        for w in driver.WRITERS for b in (64,1)]
    driver.write(out/'decision.json',dict(capability_passed=cap,practical_settings=by_setting,
        practical_passed=cap and all(r['any_primary_M1_point_unmatched'] for r in by_setting),
        primary_cost_training_steps=15000,F1_only_descriptive=True,
        single_step_definition='L=1 within writer; includes all 15000-step variants and budgets',
        practical_latency_statistic='three-seed mean CUDA-event milliseconds per prepared batch',
        failing_capability_rows=[{k:r[k] for k in ('endpoint','writer','comparator','r','r_mem')}
            for r in stats if r['endpoint'] in ('E1','E2') and r['subset']=='reach' and r['statistics']['gate']['ci_low']<=0]))
    benchmark=shared.read_json(driver.RUNS/'benchmark/audit.json')
    shared.require(benchmark['status']=='passed' and benchmark['models']==84,'benchmark audit incomplete')
    driver.write(out/'audit.json',dict(status='passed',training_stages=24,reference_checkpoints=60,
        confirmation_sets=84,sample_rows=sample_rows,all_samples_sealed=True,
        all_tables_reproduced=True,all_depth_snapshots_identical=True,all_query_labels_paired=True,
        checkpoints_unchanged=True,benchmark=benchmark,confirmation_look_count=1,
        training_barrier=shared.read_json(driver.RUNS/'training_barrier.json'),
        sample_barrier_created_at=barrier['created_at'],completed_at=datetime.now(timezone.utc).isoformat()))
    plot_cost(costs,out)
    total=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    shared.require(total<=1_000_000,f'committed P2-S outputs exceed 1 MB: {total}')
    print(f'P2-S analysis complete: 84 models, {sample_rows} samples, capability={cap}, compact bytes={total}',flush=True)


if __name__=='__main__': analyze()
