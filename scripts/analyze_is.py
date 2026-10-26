#!/usr/bin/env python3
"""IS audits, independent-load depth contrasts, capability tables and update strata."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

import analyze_p1s_main as shared
from audit_p2s import audit_evaluation
from looplora.infra.seeds import derive_seed,stable_hash
from looplora.sym.stats import pooled_compare_runs
import run_is as driver


class Cache(shared.CompactCache):
    def save(self,name,dataset,arrays,split):
        path=self.root/dataset/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        fields=('episode_id','query_id','hop','L','answer','correct','prediction')
        np.savez_compressed(path,**{k:arrays[k] for k in fields},split=np.asarray(split))


def verify_release():
    trained=driver.all_training();plan=shared.read_json(driver.RUNS/'evaluation_plan.json')
    barrier=shared.read_json(driver.RUNS/'barrier.json');receipt=shared.read_json(driver.RUNS/'training_barrier.json')
    shared.require(len(plan['runs'])==360 and plan['training_plan_hash']==stable_hash(trained)
        and receipt['status']=='passed' and receipt['training_stages']==36
        and receipt['evaluation_plan_hash']==stable_hash(plan),'IS training receipt differs')
    names={r['name'] for r in plan['runs']}
    shared.require(len(names)==360 and barrier['status']=='passed' and len(barrier['runs'])==360
        and set(barrier['runs'])==names and barrier['evaluation_plan_hash']==stable_hash(plan),'IS global sample barrier differs')
    paths=[]
    for r in plan['runs']:
        folder=Path(r['run_dir']);rec=shared.read_json(folder/'record.json');meta=shared.read_json(folder/'confirmation/evaluation.json')
        shared.require(rec['status']=='complete' and rec['checkpoint_unchanged'] and meta['aggregation_deferred']
            and meta['config']==driver.evaluation_config(r,r['eval_E'],r['update']) and meta['checkpoint_config']==r['config']
            and not (folder/'confirmation/metrics.json').exists(),'IS evaluation scope/state differs')
        path=folder/'confirmation/samples.npz';shared.require(path.is_file(),'missing IS blind samples');paths.append(str(path))
    shared.require(len(barrier['samples'])==360 and set(barrier['samples'])==set(paths),'IS sample membership differs')
    return plan,barrier


def update_paths(arrays):
    """Read chronological facts independently of the model; first versus latest graph."""
    labels={}
    def walk(graph,start):
        node=int(start);seen=set();keys=[]
        while (node,0) in graph:
            if node in seen: raise ValueError('cycle in update graph')
            seen.add(node);keys.append((node,0));node=graph[node,0]
        return node,keys
    for i,episode in enumerate(arrays.episode_seed):
        first={};latest={}
        for fact,valid in zip(arrays.facts[i],arrays.fact_valid[i]):
            if not valid: continue
            head,rel,tail=map(int,fact);key=head,rel
            first.setdefault(key,tail);latest[key]=tail
        updated={key for key in latest if latest[key]!=first[key]}
        for q,start in enumerate(arrays.q_start[i]):
            answer,path=walk(latest,start);stale,_=walk(first,start)
            changed=bool(updated.intersection(path))
            shared.require(answer==int(arrays.answer[i,q]) and len(path)==int(arrays.q_hop_true[i,q]),'update path labels disagree')
            shared.require(changed==(stale!=answer),'obsolete terminal does not identify update path')
            labels[int(episode),q]=(changed,stale if changed else -1)
    return labels


def select(plan,w,reader,train_E,eval_E,update=False):
    rows=[r for r in plan['runs'] if (r['writer'],r['reader'],r['train_E'],r['eval_E'],r['update'])==(w,reader,train_E,eval_E,update)]
    shared.require(len(rows)==3 and {r['seed'] for r in rows}==set(driver.SEEDS),'IS endpoint seed matrix differs')
    return rows


def paired(cache,a,b,La,Lb,hops,E):
    return pooled_compare_runs({r['seed']:cache.get(r['name'],'confirmation') for r in a},
        {r['seed']:cache.get(r['name'],'confirmation') for r in b},La,Lb,hops=list(hops),split='test',condition='original',
        resamples=2000,seed=derive_seed(2718,'IS','load',E),confidence=.975)


def episode_bootstrap(cache,a,b,La,Lb,hops,stats):
    """Reproduce shared.stats draws exactly to form independent-load differences."""
    values=[];counts=None
    for seed in driver.SEEDS:
        arrays=[]
        for runs,L in ((a,La),(b,Lb)):
            run=next(r for r in runs if r['seed']==seed);x=cache.get(run['name'],'confirmation')
            i=np.flatnonzero((x['L']==L)&np.isin(x['hop'],hops));i=i[np.lexsort((x['query_id'][i],x['episode_id'][i]))]
            arrays.append((x,i))
        x,i=arrays[0];y,j=arrays[1]
        shared.require(np.array_equal(x['episode_id'][i],y['episode_id'][j]) and np.array_equal(x['answer'][i],y['answer'][j]),'bootstrap pairing differs')
        _,clusters=np.unique(x['episode_id'][i],return_inverse=True)
        counts=np.bincount(clusters).astype(float)
        values.append(np.bincount(clusters,weights=x['correct'][i].astype(float)-y['correct'][j].astype(float)))
    totals=np.stack(values);rng=np.random.default_rng(stats['bootstrap_seed']);boot=np.empty(2000)
    size=max(1,min(256,1_000_000//(len(counts)*3)))
    for start in range(0,2000,size):
        stop=min(start+size,2000);draws=rng.integers(0,len(counts),size=(stop-start,len(counts)))
        boot[start:stop]=(totals[:,draws].sum(axis=-1)/counts[draws].sum(axis=-1)).mean(axis=0)
    low,high=np.quantile(boot,[.0125,.9875])
    shared.require(np.allclose([low,high],[stats['ci_low'],stats['ci_high']],rtol=0,atol=1e-14),'shared bootstrap reproduction differs')
    return boot


def endpoints(plan,cache):
    rows=[];depth={};boots={}
    for w in driver.ALL_WRITERS:
        for E in driver.LAYOUTS:
            a=select(plan,w,'M1-mixed',12,E)
            for subset,hops in [('reach',(2,3)),('far',(6,7,8))]:
                if E==6 and subset=='far': continue
                stats=paired(cache,a,a,4,1,hops,E)
                rows.append(dict(endpoint='IS-E1-D',writer=w,reader='M1-mixed',train_E=12,E=E,subset=subset,
                    descriptive=w=='joint',statistics=stats))
                if subset=='reach':
                    depth[w,E]=stats
                    if E in (12,36): boots[w,E]=episode_bootstrap(cache,a,a,4,1,hops,stats)
        a,b=depth[w,36],depth[w,12];draws=boots[w,36]-boots[w,12]
        low,high=np.quantile(draws,[.0125,.9875]);per_seed=[]
        for seed in driver.SEEDS:
            x=next(r for r in a['per_seed'] if r['seed']==seed);y=next(r for r in b['per_seed'] if r['seed']==seed)
            per_seed.append(dict(seed=seed,D36=x['difference'],D12=y['difference'],difference=x['difference']-y['difference']))
        rows.append(dict(endpoint='IS-E1-DeltaD',writer=w,reader='M1-mixed',train_E=12,E=36,subset='reach',descriptive=w=='joint',
            statistics=dict(difference=a['difference']-b['difference'],ci_low=float(low),ci_high=float(high),confidence=.975,
                resamples=2000,per_seed=per_seed,method='independent-load paired-episode bootstrap difference',
                bootstrap_seed_E12=b['bootstrap_seed'],bootstrap_seed_E36=a['bootstrap_seed'])))
    for w in driver.WRITERS:
        for E in (12,24,36):
            base=select(plan,w,'B4',E,E)
            for reader in ('M1-mixed','B8'):
                a=select(plan,w,reader,E,E)
                rows.append(dict(endpoint='IS-E2',writer=w,reader=reader,train_E=E,E=E,subset='reach',descriptive=False,
                    statistics=paired(cache,a,base,4,1,(2,3),E)))
        for E in (24,36):
            for reader in ('M1-mixed','B8'):
                a=select(plan,w,reader,E,E);b=select(plan,w,reader,12,E)
                rows.append(dict(endpoint='IS-E3',writer=w,reader=reader,train_E=E,E=E,subset='reach',descriptive=True,
                    statistics=paired(cache,a,b,4,4,(2,3),E)))
    return rows


def capability_passes(stats):
    total=stats['n_queries_per_seed']*stats['n_seeds']
    return round(stats['difference']*total)*100>=3*total and stats['ci_low']>0


def decision_rows(rows):
    result=[]
    for w in driver.WRITERS:
        d=next(r['statistics'] for r in rows if r['endpoint']=='IS-E1-DeltaD' and r['writer']==w)
        use='consistent with Prop.4' if d['ci_high']<0 else 'contradicts tested depth-amplification prediction' if d['ci_low']>0 else 'no detected load dependence'
        result.append(dict(endpoint='IS-E1',writer=w,reader='M1-mixed',interpretation=use))
        for reader in ('M1-mixed','B8'):
            cells=[r for r in rows if r['endpoint']=='IS-E2' and r['writer']==w and r['reader']==reader]
            shared.require(len(cells)==3,'capability requires E12,24,36')
            qualified=[r['E'] for r in cells if capability_passes(r['statistics'])]
            result.append(dict(endpoint='IS-E2',writer=w,reader=reader,largest_qualifying_trained_E=max(qualified) if qualified else None,
                qualifying_E=qualified,nonqualifying_E=[r['E'] for r in cells if r['E'] not in qualified]))
    return result


def plot_accuracy(accuracy,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(10,6),sharey='row')
    for j,w in enumerate(driver.ALL_WRITERS):
        for i,hops in enumerate(((2,3),(6,7,8))):
            ax=axes[i,j]
            for E in driver.LAYOUTS:
                if E==6 and i==1: continue
                cells=[r for r in accuracy if r['writer']==w and r['reader']=='M1-mixed' and r['train_E']==12 and r['E']==E
                       and not r['update'] and r['hop'] in hops]
                points=[]
                for L in range(1,9):
                    group=[r for r in cells if r['L']==L]
                    points.append(sum(r['correct'] for r in group)/sum(r['n'] for r in group))
                ax.plot(range(1,9),points,marker='o',markersize=3,label=f'E={E}')
            ax.set_title(w+(' (descriptive)' if w=='joint' else ''));ax.set_xticks(range(1,9));ax.set_xlabel('Read depth L');ax.grid(alpha=.2)
            if j==0: ax.set_ylabel('Accuracy, hops '+('/'.join(map(str,hops))))
    axes[0,0].legend(fontsize=8);fig.suptitle('Fixed E12 mixed-depth M1: memory load versus read depth')
    fig.tight_layout();fig.savefig(out/'accuracy_vs_L.png',dpi=150);fig.savefig(out/'accuracy_vs_L.pdf');plt.close(fig)


def update_table(events,labels,out):
    """Cluster all queries of each episode; fixed training seeds are averaged."""
    episodes=sorted({key[0] for key in labels});index={e:i for i,e in enumerate(episodes)}
    counts={};correct={};stale={};seed_rows=[]
    for run,a in events:
        path=np.asarray([labels[int(e),int(q)][0] for e,q in zip(a['episode_id'],a['query_id'])])
        old=np.asarray([labels[int(e),int(q)][1] for e,q in zip(a['episode_id'],a['query_id'])])
        ei=np.asarray([index[int(e)] for e in a['episode_id']]);depths=sorted(set(a['L']))
        for h in (0,1,2,3,4):
            for L in depths:
                for changed in (False,True):
                    mask=(a['L']==L)&(path==changed)&(True if h==0 else a['hop']==h)
                    key=(run['writer'],run['reader'],h,int(L),changed)
                    n=np.bincount(ei[mask],minlength=len(episodes)).astype(float)
                    c=np.bincount(ei[mask],weights=a['correct'][mask],minlength=len(episodes))
                    s=np.bincount(ei[mask],weights=a['prediction'][mask]==old[mask],minlength=len(episodes)) if changed else np.zeros(len(episodes))
                    if key in counts: shared.require(np.array_equal(counts[key],n),'update strata differ across seeds')
                    counts[key]=n;correct[key]=correct.get(key,0)+c;stale[key]=stale.get(key,0)+s
                    seed_rows.append(dict(writer=run['writer'],reader=run['reader'],hop=h,L=int(L),updated_path=changed,seed=run['seed'],
                        n=int(n.sum()),n_correct=int(c.sum()),accuracy=float(c.sum()/n.sum()) if n.sum() else None,
                        stale_n=int(s.sum()) if changed else None,stale_rate=float(s.sum()/n.sum()) if changed and n.sum() else None))
    keys=sorted(counts);n=np.stack([counts[k]*3 for k in keys],axis=1)
    c=np.stack([correct[k] for k in keys],axis=1);s=np.stack([stale[k] for k in keys],axis=1)
    # Integer cluster multiplicities are exactly ordinary episode bootstrap draws.
    rng=np.random.default_rng(derive_seed(2718,'IS','t3_update'))
    weights=rng.multinomial(len(episodes),np.full(len(episodes),1/len(episodes)),size=2000).astype(float)
    den=weights@n
    with np.errstate(divide='ignore',invalid='ignore'):
        cb=(weights@c)/den;sb=(weights@s)/den
    pooled=[]
    for i,key in enumerate(keys):
        w,reader,h,L,changed=key;total=n[:,i].sum();nc=c[:,i].sum();ns=s[:,i].sum()
        acc_ci=np.quantile(cb[:,i],[.0125,.9875]).tolist() if total else [None,None]
        stale_ci=np.quantile(sb[:,i],[.0125,.9875]).tolist() if total and changed else [None,None]
        pooled.append(dict(writer=w,reader=reader,E=14,base_E=12,n_updates=2,hop=h,L=L,updated_path=changed,n=int(total),n_episodes=len(episodes),
            n_correct=int(nc),accuracy=float(nc/total) if total else None,accuracy_ci_low=acc_ci[0],accuracy_ci_high=acc_ci[1],
            stale_n=int(ns) if changed else None,stale_rate=float(ns/total) if changed and total else None,
            stale_ci_low=stale_ci[0],stale_ci_high=stale_ci[1],confidence=.975,
            descriptive=w=='joint',n_eligible_episodes=int(np.count_nonzero(n[:,i])),
            per_seed=json.dumps([{k:r[k] for k in ('seed','n','n_correct','accuracy','stale_n','stale_rate')} for r in seed_rows if (r['writer'],r['reader'],r['hop'],r['L'],r['updated_path'])==key])))
    driver.csv_table(out/'t3_update.csv',pooled)
    return len(pooled)


def tables(rows,decision,out):
    lines=['# IS registered contrasts','',
        'All effects and 97.5% intervals are percentage points. Per-seed effects are ordered 17 / 29 / 43.',
        'Reach: hops2/3; far: hops6/7/8 (unavailable at E6). Joint is descriptive only.',
        'DeltaD uses independently bootstrapped D36 minus D12; all other contrasts pair episodes within load.',
        '', '| Endpoint | Writer | Reader | E | Hops | Estimate [97.5% CI], pp | Per seed, pp |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        value=r['statistics'];effect=f"{100*value['difference']:+.3f} [{100*value['ci_low']:+.3f}, {100*value['ci_high']:+.3f}]"
        seeds=' / '.join(f"{100*p['difference']:+.3f}" for p in value['per_seed'])
        lines.append(f"| {r['endpoint']} | {r['writer']} | {r['reader']} | {r['E']} | {r['subset']} | {effect} | {seeds} |")
    lines.extend(['','Interpretation of each primary writer:',''])
    for r in decision:
        value=r.get('interpretation') or f"largest qualifying trained E={r['largest_qualifying_trained_E']}; qualifying={r['qualifying_E']}"
        lines.append(f"- {r['writer']} / {r['reader']} / {r['endpoint']}: {value}.")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


def training_table(out):
    plan=driver.all_training();rows=[]
    for r in plan['runs']:
        folder=Path(r['training_dir']);rec=shared.read_json(folder/'record.json');metrics=shared.read_json(folder/'metrics.json')
        logs=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines()]
        shared.require(logs[0]['step']==1 and logs[-1]['step']==15000 and all(np.isfinite(x['loss']) for x in logs),'training loss/step trace differs')
        L=1 if r['reader']=='B4' else 4
        cells=[x for x in metrics['table'] if x['split']=='val' and x['condition']=='original' and x['L']==L and x['hop'] in (1,2,3)]
        shared.require(sum(x['n'] for x in cells)==3000,'training validation coverage differs')
        correct=sum(round(x['n']*x['accuracy']) for x in cells)
        rows.append({k:r[k] for k in ('writer','reader','train_E','seed')}|dict(steps=15000,initial_loss=logs[0]['loss'],
            final_loss=logs[-1]['loss'],mean_L=rec['mean_L'],val_n=3000,val_correct=correct,val_accuracy=correct/3000,
            train_seconds=rec['train_seconds'],peak_memory_bytes=rec['peak_memory_bytes'],validation_used_for_selection=False))
    driver.csv_table(out/'training.csv',rows)


def analyze():
    plan,barrier=verify_release();raw=driver.RUNS/'analysis';out=driver.REPORTS
    shared.require(not (raw/'started.json').exists(),'IS analysis already started')
    driver.write(raw/'started.json',dict(created_at=datetime.now(timezone.utc).isoformat()))
    training_table(out)
    cache=Cache(driver.RUNS/'analysis_cache',limit=12);references={};accuracy=[];events=[];labels=None;sample_rows=0
    for r in plan['runs']:
        shared.require(shared.file_hash(Path(r['checkpoint']))==r['checkpoint_hash'],'IS checkpoint changed')
        c=driver.evaluation_config(r,r['eval_E'],r['update']);folder=Path(r['run_dir'])/'confirmation'
        table,audit=audit_evaluation(folder,c);key=(r['eval_E'],r['update'])
        shared.require(key not in references or np.array_equal(references[key],audit['labels']),'IS within-load query labels differ')
        references[key]=audit['labels'];sample_rows+=audit['sample_rows'];a=audit['arrays']
        cache.save(r['name'],'confirmation',a,'test')
        for cell in table:
            accuracy.append({k:r[k] for k in ('writer','reader','variant','seed','train_E','update')}|dict(E=14 if r['update'] else r['eval_E'],base_E=r['eval_E'],hop=cell['hop'],L=cell['L'],n=cell['n'],
                correct=round(cell['n']*cell['accuracy']),accuracy=cell['accuracy'],full_accuracy=cell['full_accuracy'],chance_accuracy=cell['chance_accuracy']))
        if r['update']:
            if labels is None:
                from looplora.data.relations import generate
                from looplora.sym.model import make_task_spec
                from looplora.sym.evaluate import stratify_queries
                spec=make_task_spec(c,evaluation=True);source=stratify_queries(generate(spec,'test',0,2000,999001),spec,c['eval']['hops'],999001)
                labels=update_paths(source)
            events.append((r,{k:a[k].copy() for k in ('episode_id','query_id','hop','L','answer','prediction','correct')}))
        driver.write(folder/'metrics.json',dict(table=table,finalized_after_global_barrier=True,barrier_created_at=barrier['created_at']))
    rows=endpoints(plan,cache);driver.write(raw/'endpoints.json',rows)
    driver.csv_table(out/'accuracy.csv',accuracy)
    driver.csv_table(out/'endpoints.csv',[{k:v for k,v in r.items() if k!='statistics'}|dict(
        difference=r['statistics']['difference'],ci_low=r['statistics']['ci_low'],ci_high=r['statistics']['ci_high'],
        confidence=.975,n_seeds=3,n_episodes_per_load=2000,
        n_queries_per_seed=r['statistics'].get('n_queries_per_seed',4000),
        per_seed=json.dumps(r['statistics']['per_seed'])) for r in rows])
    best=[]
    for w in driver.ALL_WRITERS:
        for E in driver.LAYOUTS:
            cells=[r for r in accuracy if r['writer']==w and r['reader']=='M1-mixed' and r['train_E']==12 and r['E']==E and not r['update'] and r['hop'] in (2,3)]
            points=[dict(L=L,n=sum(r['n'] for r in cells if r['L']==L),correct=sum(r['correct'] for r in cells if r['L']==L)) for L in range(1,9)]
            choice=max(points,key=lambda r:(r['correct']/r['n'],-r['L']))
            best.append(dict(writer=w,E=E,best_L=choice['L'],accuracy=choice['correct']/choice['n'],n=choice['n'],descriptive_only=True))
    driver.csv_table(out/'best_L.csv',best);decision=decision_rows(rows);driver.write(out/'decision.json',decision);tables(rows,decision,out)
    updates=update_table(events,labels,out);plot_accuracy(accuracy,out)
    driver.write(out/'audit.json',dict(status='passed',training_stages=36,reference_checkpoints=54,confirmation_sets=360,sample_rows=sample_rows,
        all_samples_sealed=True,all_depth_snapshots_identical=True,all_within_load_labels_paired=True,checkpoints_unchanged=True,
        update_cells=updates,training_barrier=shared.read_json(driver.RUNS/'training_barrier.json'),
        sample_barrier_created_at=barrier['created_at'],confirmation_look_count=1,completed_at=datetime.now(timezone.utc).isoformat()))
    print(f'IS analysis passed: {sample_rows} rows; decisions={decision}',flush=True)

if __name__=='__main__': analyze()
