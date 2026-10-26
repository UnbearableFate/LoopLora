#!/usr/bin/env python3
"""JS trace audits, validation-only stopping selection and paired endpoints."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
import torch

import analyze_p1s_main as shared
from looplora.infra.seeds import derive_seed,stable_hash
from looplora.core.stopping import ReaderTrace,halting_steps
from looplora.sym.stats import pooled_compare_runs
import run_js as driver


class Cache(shared.CompactCache):
    def save(self,name,arrays):
        path=self.root/'confirmation'/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**arrays,split=np.asarray('test'))


def verify_set(stage):
    trained=driver.all_training();plan=shared.read_json(driver.RUNS/'evaluation_plan.json');barrier=shared.read_json(driver.RUNS/f'{stage}_barrier.json')
    receipt=shared.read_json(driver.RUNS/'training_barrier.json')
    shared.require(len(plan['runs'])==24 and len({r['name'] for r in plan['runs']})==24
        and plan['training_plan_hash']==stable_hash(trained) and receipt['status']=='passed' and receipt['training_stages']==12
        and receipt['evaluation_plan_hash']==stable_hash(plan),'JS training receipt differs')
    shared.require(barrier['status']=='passed' and barrier['evaluation_plan_hash']==stable_hash(plan)
        and len(barrier['runs'])==24 and set(barrier['runs'])=={r['name'] for r in plan['runs']},'JS global trace barrier differs')
    paths=[]
    for r in plan['runs']:
        folder=driver.RUNS/stage/r['name'];meta=shared.read_json(folder/'evaluation.json')
        shared.require(meta['status']=='complete' and meta['aggregation_deferred'] and meta['gate0_passed']
            and meta['checkpoint_config']==r['config'] and meta['checkpoint_hash']==r['checkpoint_hash']
            and meta['episodes']==2000 and meta['seed']==(4242 if stage=='validation' else 999003)
            and meta['split']==('val' if stage=='validation' else 'test')
            and meta['radius_depths']==([] if stage=='validation' else [4,8,16]),'JS trace scope differs')
        path=folder/'trace.npz';shared.require(path.is_file(),'JS missing blind trace');paths.append(str(path))
    shared.require(len(barrier['traces'])==24 and set(barrier['traces'])==set(paths),'JS trace membership differs')
    if stage=='confirmation':
        selection=shared.read_json(driver.REPORTS/'thresholds.json')
        archive=Path(os.environ.get('LL_SOURCE_ROOT',driver.ROOT))/'reports/js/thresholds.json'
        shared.require(archive.is_file() and shared.read_json(archive)==selection,'JS thresholds differ from source commit')
        shared.require(barrier['threshold_hash']==stable_hash(selection),'JS threshold receipt differs')
    return plan,barrier


def audit_trace(row,stage):
    folder=driver.RUNS/stage/row['name'];meta=shared.read_json(folder/'evaluation.json');path=folder/'trace.npz'
    shared.require(shared.file_hash(path)==meta['trace_hash'],'JS trace seal changed')
    shared.require(shared.file_hash(Path(row['checkpoint']))==row['checkpoint_hash'],'JS checkpoint changed')
    with np.load(path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
    n=16000
    shared.require(set(a)=={'episode_id','query_id','hop','answer','candidate_size','predictions','relative_change','radii'},'JS trace fields differ')
    for k in ('episode_id','query_id','hop','answer','candidate_size'):
        shared.require(a[k].shape==(n,) and a[k].dtype.kind in 'iu','JS query-label shape/dtype differs')
    shared.require(a['predictions'].shape==(n,17) and a['predictions'].dtype.kind in 'iu','JS prediction shape differs')
    shared.require(a['relative_change'].shape==(n,17) and np.isnan(a['relative_change'][:,0]).all()
        and np.isfinite(a['relative_change'][:,1:]).all() and (a['relative_change'][:,1:]>=0).all(),'JS relative change differs')
    shared.require(a['radii'].shape==(n,0 if stage=='validation' else 3) and np.isfinite(a['radii']).all()
        and (a['radii']>=0).all(),'JS radius diagnostics differ')
    i=np.lexsort((a['query_id'],a['episode_id']))
    shared.require(len(np.unique(a['episode_id']))==2000 and np.array_equal(a['query_id'][i],np.tile(np.arange(8),2000))
        and np.array_equal(a['hop'][i],np.tile(np.arange(1,9),2000)) and (a['candidate_size']==14).all(),'JS query coverage differs')
    labels=np.stack([a[k][i] for k in ('episode_id','query_id','hop','answer','candidate_size')],axis=1)
    seen=set();snapshots={}
    for r in meta['memory_hashes']:
        key=r['batch_start'],r['L'];shared.require(key not in seen,'duplicate JS memory snapshot');seen.add(key)
        values=tuple(r[k] for k in ('x','u','memory','write_count'))
        shared.require(all(isinstance(x,str) and len(x)==32 for x in values),'JS snapshot digest differs')
        shared.require(r['batch_start'] not in snapshots or snapshots[r['batch_start']]==values,'JS immutable snapshot changed')
        snapshots[r['batch_start']]=values
    shared.require(seen=={(b,L) for b in range(0,2000,64) for L in range(1,17)},'JS snapshot coverage differs')
    return a,labels


def stopped(a,threshold):
    trace=ReaderTrace(torch.from_numpy(a['predictions']),torch.zeros(a['predictions'].shape),torch.from_numpy(a['relative_change']))
    steps=halting_steps(trace,'relative_change',threshold).numpy()
    return a['predictions'][np.arange(len(steps)),steps],steps


def best_threshold(cells):
    return max(cells,key=lambda r:(r['correct'],-r['steps_total'],-r['threshold']))


def select_thresholds():
    plan,barrier=verify_set('validation')
    shared.require(not (driver.REPORTS/'thresholds.json').exists(),'JS stopping thresholds already selected')
    torch.set_num_threads(1);traces={};reference=None
    for r in plan['runs']:
        a,labels=audit_trace(r,'validation');shared.require(reference is None or np.array_equal(reference,labels),'JS validation labels not paired')
        reference=labels;traces[r['name']]=a
    grid=[];selected=[]
    for w in driver.WRITERS:
        for condition in (*driver.CONDITIONS,'M1-L4','M1-mixed'):
            runs=[r for r in plan['runs'] if r['writer']==w and r['condition']==condition]
            shared.require(len(runs)==3 and {r['seed'] for r in runs}==set(driver.SEEDS),'JS threshold seed matrix differs')
            cells=[]
            for threshold in driver.THRESHOLDS:
                correct=0;steps_total=0;n=0;per_seed=[]
                for r in runs:
                    a=traces[r['name']];pred,steps=stopped(a,threshold);count=int((pred==a['answer']).sum())
                    correct+=count;steps_total+=int(steps.sum());n+=len(pred)
                    per_seed.append(dict(seed=r['seed'],n=len(pred),correct=count,accuracy=count/len(pred),mean_steps=float(steps.mean())))
                cell=dict(writer=w,condition=condition,threshold=threshold,n=n,correct=correct,steps_total=steps_total,
                    accuracy=correct/n,mean_steps=steps_total/n,per_seed=per_seed);cells.append(cell);grid.append(cell)
            selected.append(best_threshold(cells))
    selection=dict(status='locked',evaluation_plan_hash=stable_hash(plan),validation_barrier_hash=stable_hash(barrier),
        selected=selected,created_at=datetime.now(timezone.utc).isoformat(),seed=4242,episodes=2000,rule=driver.design()['stopping'])
    driver.write(driver.REPORTS/'thresholds.json',selection)
    driver.csv_table(driver.REPORTS/'threshold_grid.csv',[{k:v for k,v in r.items() if k!='per_seed'} for r in grid])
    driver.csv_table(driver.REPORTS/'thresholds.csv',[{k:v for k,v in r.items() if k!='per_seed'} for r in selected])
    print('JS stopping thresholds selected on validation only; commit eight settings before confirmation',flush=True)


def rows_for(plan,w,condition):
    rows=[r for r in plan['runs'] if r['writer']==w and r['condition']==condition]
    shared.require(len(rows)==3 and {r['seed'] for r in rows}==set(driver.SEEDS),'JS endpoint seed matrix differs')
    return rows


def paired(cache,a,b,La,Lb,hops):
    return pooled_compare_runs({r['seed']:cache.get(r['name'],'confirmation') for r in a},
        {r['seed']:cache.get(r['name'],'confirmation') for r in b},La,Lb,hops=list(hops),split='test',condition='original',
        resamples=2000,seed=2718,confidence=.975)


def delta_overshoot(cache,a,b,writer):
    differences=[];per_seed=[];reference=None;episode_ids=None
    for seed in driver.SEEDS:
        values=[]
        for rows,L in ((a,16),(a,4),(b,16),(b,4)):
            run=next(r for r in rows if r['seed']==seed);x=cache.get(run['name'],'confirmation')
            i=np.flatnonzero((x['L']==L)&np.isin(x['hop'],(1,2,3)));i=i[np.lexsort((x['query_id'][i],x['episode_id'][i]))]
            shared.require(set(x['hop'][i])=={1,2,3},'JS DeltaS hop coverage differs')
            labels=np.stack([x[k][i] for k in ('episode_id','query_id','hop','answer')],axis=1)
            shared.require(reference is None or np.array_equal(reference,labels),'JS DeltaS episode/query/answer pairing differs')
            reference=labels;episode_ids=x['episode_id'][i];values.append(x['correct'][i].astype(float))
        j16,j4,r16,r4=values;effect=(j16-j4)-(r16-r4);differences.append(effect)
        per_seed.append(dict(seed=seed,accuracy_J16=float(j16.mean()),accuracy_J4=float(j4.mean()),
            accuracy_RS16=float(r16.mean()),accuracy_RS4=float(r4.mean()),S_J=float((j16-j4).mean()),
            S_RS=float((r16-r4).mean()),difference=float(effect.mean())))
    _,clusters=np.unique(episode_ids,return_inverse=True);counts=np.bincount(clusters).astype(float)
    totals=np.stack([np.bincount(clusters,weights=values) for values in differences]);boot=np.empty(2000)
    bootstrap_seed=derive_seed(2718,'JS','paired_difference_of_depth_gains',writer)
    rng=np.random.default_rng(bootstrap_seed);size=max(1,min(256,1_000_000//(len(counts)*3)))
    for begin in range(0,2000,size):
        end=min(begin+size,2000);draws=rng.integers(0,len(counts),size=(end-begin,len(counts)))
        boot[begin:end]=(totals[:,draws].sum(-1)/counts[draws].sum(-1)).mean(0)
    low,high=np.quantile(boot,[.0125,.9875])
    return dict(difference=float(np.mean([r['difference'] for r in per_seed])),ci_low=float(low),ci_high=float(high),confidence=.975,
        per_seed=per_seed,n_episodes=len(counts),n_queries_per_seed=len(episode_ids),n_seeds=3,resamples=2000,
        seed=2718,bootstrap_seed=bootstrap_seed,method='pooled paired episode bootstrap of four-term depth-gain difference')


def endpoints(plan,cache):
    rows=[]
    for w in driver.WRITERS:
        rs=rows_for(plan,w,'RS');j=rows_for(plan,w,'RS+J')
        for condition,runs in (('RS',rs),('RS+J',j)):
            rows.append(dict(endpoint='JS-E1-S',writer=w,condition=condition,comparator=condition,L_a=16,L_b=4,hops='1-3',descriptive=False,
                statistics=paired(cache,runs,runs,16,4,(1,2,3))))
        rows.append(dict(endpoint='JS-E1-DeltaS',writer=w,condition='RS+J',comparator='RS',L_a='16-4',L_b='16-4',hops='1-3',descriptive=False,
            statistics=delta_overshoot(cache,j,rs,w)))
        for endpoint,L,hops in (('JS-E2',4,(1,2,3)),('JS-E3',16,(4,5,6,7,8)),('JS-E4',17,tuple(range(1,9)))):
            rows.append(dict(endpoint=endpoint,writer=w,condition='RS+J',comparator='RS',L_a=L,L_b=L,hops='-'.join(map(str,(min(hops),max(hops)))),descriptive=False,
                statistics=paired(cache,j,rs,L,L,hops)))
        for reference in ('M1-L4','M1-mixed'):
            base=rows_for(plan,w,reference)
            for L in (4,16):
                for label,hops in (('1-3',(1,2,3)),('4-8',(4,5,6,7,8)),('1-8',tuple(range(1,9)))):
                    rows.append(dict(endpoint='RS-reference',writer=w,condition='RS',comparator=reference,L_a=L,L_b=L,hops=label,descriptive=True,
                        statistics=paired(cache,rs,base,L,L,hops)))
    return rows


def decision(rows):
    result=[]
    for w in driver.WRITERS:
        delta=next(r['statistics'] for r in rows if r['writer']==w and r['endpoint']=='JS-E1-DeltaS')
        s=next(r['statistics'] for r in rows if r['writer']==w and r['endpoint']=='JS-E1-S' and r['condition']=='RS+J')
        total=s['n_queries_per_seed']*s['n_seeds'];floor=round(s['difference']*total)*100>=-total
        result.append(dict(writer=w,removes_overshoot=delta['difference']>0 and delta['ci_low']>0 and floor,
            DeltaS=delta['difference'],DeltaS_ci_low=delta['ci_low'],DeltaS_ci_high=delta['ci_high'],S_RS_J=s['difference'],
            positive_DeltaS_CI=delta['difference']>0 and delta['ci_low']>0,S_at_least_minus_one_pp=floor))
    return result


def normalized(a,pred):
    n=len(pred);depths=(4,16,17)
    return dict(episode_id=np.tile(a['episode_id'],3),query_id=np.tile(a['query_id'],3),hop=np.tile(a['hop'],3),
        answer=np.tile(a['answer'],3),L=np.repeat(np.array(depths,dtype=np.int16),n),
        correct=np.concatenate([(a['predictions'][:,L] if L!=17 else pred)==a['answer'] for L in depths]))


def plot_accuracy(accuracy,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'RS':'#1765ad','RS+J':'#ca4d31','M1-L4':'#218343','M1-mixed':'#777777'}
    for w in driver.WRITERS:
        fig,axes=plt.subplots(2,4,figsize=(11,5),sharex=True,sharey=True)
        for h,ax in enumerate(axes.flat,1):
            for condition,color in colors.items():
                points=[]
                for L in range(1,17):
                    cells=[r for r in accuracy if r['writer']==w and r['condition']==condition and r['hop']==h and r['L']==L]
                    points.append(sum(r['correct'] for r in cells)/sum(r['n'] for r in cells))
                ax.plot(range(1,17),points,color=color,label=condition,linewidth=1.5)
            ax.axvspan(4,8,color='#bbbbbb',alpha=.12);ax.set_title(f'hop {h}');ax.set_xticks([1,4,8,12,16]);ax.grid(alpha=.2);ax.set_ylim(0,1)
        for ax in axes[1]: ax.set_xlabel('Read depth L')
        for ax in axes[:,0]: ax.set_ylabel('Accuracy')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9)
        fig.suptitle(f'JS: {w}; shaded range is RS training depths4-8')
        fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(out/f'accuracy_depth_{w}.pdf');fig.savefig(out/f'accuracy_depth_{w}.png',dpi=150);plt.close(fig)


def tables(rows,decisions,out):
    lines=['# JS paired contrasts','', 'Effects and 97.5% CIs are percentage points. Per-seed effects are ordered17 /29 /43.',
        'L=17 is the internal stopped-policy sentinel, selecting an actual state between1 and16; it is not a seventeenth read.',
        '', '| Endpoint | Writer | A - B | L_A / L_B | Hops | Estimate [97.5% CI], pp | Per seed, pp |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        v=r['statistics'];effect=f"{100*v['difference']:+.3f} [{100*v['ci_low']:+.3f}, {100*v['ci_high']:+.3f}]"
        seeds=' / '.join(f"{100*p['difference']:+.3f}" for p in v['per_seed'])
        lines.append(f"| {r['endpoint']} | {r['writer']} | {r['condition']} - {r['comparator']} | {r['L_a']} / {r['L_b']} | {r['hops']} | {effect} | {seeds} |")
    lines.extend(['','Registered overshoot decision:',''])
    lines.extend(f"- {r['writer']}: {'passes' if r['removes_overshoot'] else 'does not pass'}; positive DeltaS CI={r['positive_DeltaS_CI']}; S(RS+J)>=-1pp={r['S_at_least_minus_one_pp']}." for r in decisions)
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


def training_table():
    plan=driver.all_training();rows=[]
    for r in plan['runs']:
        folder=Path(r['training_dir']);rec=shared.read_json(folder/'record.json')
        logs=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines()]
        shared.require(logs[0]['step']==1 and logs[-1]['step']==15000 and all(np.isfinite(x['total_loss']) for x in logs),'JS training step/loss trace differs')
        rows.append({k:r[k] for k in ('writer','condition','seed')}|
            {k:rec[k] for k in ('steps','mean_L','train_seconds','peak_memory_bytes','active_batches','active_queries','max_penalty','max_radius')}|
            dict(initial_CE=logs[0]['ce'],final_CE=logs[-1]['ce'],final_penalty=logs[-1]['penalty']))
    driver.csv_table(driver.REPORTS/'training.csv',rows)


def analyze():
    plan,barrier=verify_set('confirmation');out=driver.REPORTS;raw=driver.RUNS/'analysis'
    shared.require(not (raw/'started.json').exists(),'JS confirmation already aggregated')
    driver.write(raw/'started.json',dict(created_at=datetime.now(timezone.utc).isoformat()))
    training_table()
    selection=shared.read_json(out/'thresholds.json');cache=Cache(driver.RUNS/'analysis_cache',limit=12)
    reference=None;accuracy=[];rho_seed=[];rho_groups=defaultdict(list);rows_count=0;stopping=[]
    for r in plan['runs']:
        a,labels=audit_trace(r,'confirmation');shared.require(reference is None or np.array_equal(reference,labels),'JS confirmation query labels not paired')
        reference=labels;threshold=next(x['threshold'] for x in selection['selected'] if (x['writer'],x['condition'])==(r['writer'],r['condition']))
        pred,steps=stopped(a,threshold);cache.save(r['name'],normalized(a,pred));rows_count+=len(pred)*17
        stopping.append(dict(writer=r['writer'],condition=r['condition'],seed=r['seed'],threshold=threshold,n=len(pred),mean_steps=float(steps.mean()),
            stopped_before_cap=float((steps<16).mean()),correct=int((pred==a['answer']).sum())))
        for h in range(1,9):
            take=a['hop']==h
            for L in range(1,18):
                prediction=a['predictions'][:,L] if L!=17 else pred;count=int((prediction[take]==a['answer'][take]).sum());n=int(take.sum())
                accuracy.append(dict(writer=r['writer'],condition=r['condition'],seed=r['seed'],hop=h,L=L if L!=17 else 'stopped',n=n,correct=count,
                    accuracy=count/n,mean_steps=float(steps[take].mean()) if L==17 else L))
        for index,L in enumerate((4,8,16)):
            for h in range(9):
                values=a['radii'][:,index] if h==0 else a['radii'][a['hop']==h,index]
                rho_groups[r['writer'],r['condition'],L,h].append(values)
                rho_seed.append(dict(writer=r['writer'],condition=r['condition'],seed=r['seed'],L=L,hop=h,**radius_summary(values)))
    rows=endpoints(plan,cache);driver.write(raw/'endpoints.json',rows)
    driver.csv_table(out/'accuracy.csv',accuracy);driver.csv_table(out/'stopping.csv',stopping)
    driver.csv_table(out/'rho_per_seed.csv',rho_seed)
    driver.csv_table(out/'rho.csv',[dict(writer=w,condition=c,L=L,hop=h,n_seeds=3,**radius_summary(np.concatenate(values))) for (w,c,L,h),values in sorted(rho_groups.items())])
    driver.csv_table(out/'endpoints.csv',[{k:v for k,v in r.items() if k!='statistics'}|dict(confidence=.975,difference=r['statistics']['difference'],
        ci_low=r['statistics']['ci_low'],ci_high=r['statistics']['ci_high'],n_episodes=r['statistics']['n_episodes'],
        n_queries_per_seed=r['statistics']['n_queries_per_seed'],per_seed=json.dumps(r['statistics']['per_seed'])) for r in rows])
    decisions=decision(rows);driver.write(out/'decision.json',decisions);tables(rows,decisions,out);plot_accuracy(accuracy,out)
    driver.write(out/'audit.json',dict(status='passed',training_stages=12,reference_checkpoints=12,confirmation_traces=24,
        confirmation_query_contexts=2000,normalized_sample_rows=rows_count,spectral_measurements=24*16000*3,
        traces_sealed=True,all_inputs_and_memory_immutable=True,all_query_labels_paired=True,checkpoints_unchanged=True,
        threshold_hash=stable_hash(selection),training_barrier=shared.read_json(driver.RUNS/'training_barrier.json'),
        sample_barrier_created_at=barrier['created_at'],confirmation_look_count=1,completed_at=datetime.now(timezone.utc).isoformat()))
    print(f'JS analysis complete: {rows_count} normalized samples; decisions={decisions}',flush=True)


def radius_summary(values):
    return dict(n=len(values),mean=float(values.mean()),p05=float(np.quantile(values,.05)),p25=float(np.quantile(values,.25)),
        median=float(np.median(values)),p75=float(np.quantile(values,.75)),p95=float(np.quantile(values,.95)),maximum=float(values.max()),
        fraction_le_gamma=float((values<=.9).mean()),fraction_lt_one=float((values<1).mean()))

if __name__=='__main__': analyze()
