#!/usr/bin/env python3
"""F2 analysis: shared F1/WP-H audits, four no-replication gates, compact exports."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import analyze_p1s_main as shared
import analyze_p1s_f1 as f1
import run_p1s_f2 as driver
from looplora.infra.seeds import derive_seed, stable_hash

WRITERS = driver.WRITERS
READERS = ('M1','M1-lin')
LEGACY = {'t1_frozen_fact':'t1_frozen','joint_fact':'joint'}


def evaluate_f2_gates(endpoints: dict, *, evaluation_seed: int) -> list[dict]:
    from looplora.sym.gates import _validate_endpoint
    shared.require(evaluation_seed==777001, 'F2 requires confirmation seed 777001')
    shared.require(set(endpoints)=={(w,p) for w in WRITERS for p in READERS}, 'F2 requires four rows')
    rows=[]
    for (writer,reader),stats in sorted(endpoints.items()):
        shared.require(set(stats)==set(f1.ENDPOINTS),'F2 requires six endpoints')
        for name,pair in stats.items(): _validate_endpoint(name,pair)
        values=[('reach_at_least_3pp',stats['reach']['gate']['difference'],.03,True),
                ('reach_ci_positive',stats['reach']['gate']['ci_low'],0.,False),
                ('reach_each_seed_positive',min(s['difference'] for s in stats['reach']['gate']['per_seed']),0.,False),
                ('specificity_ci_positive',stats['specificity']['gate']['ci_low'],0.,False),
                ('hop1_degradation_at_most_1pp',stats['hop1']['gate']['difference'],-.01,True),
                ('vs_fixed_ci_positive',stats['vs_fixed']['gate']['ci_low'],0.,False),
                ('vs_loop_ci_positive',stats['vs_loop']['gate']['ci_low'],0.,False)]
        criteria={name:dict(value=v,threshold=t,operator='>=' if inc else '>',
                           passed=bool(v>=t if inc else v>t)) for name,v,t,inc in values}
        g1=all(criteria[n]['passed'] for n,*_ in values[:5])
        g2=all(criteria[n]['passed'] for n,*_ in values[5:])
        rows.append(dict(writer=writer,r=32,reader=reader,evaluation_seed=777001,
            gate_confidence=.975,report_confidence=.95,replication_required=False,
            weaker_than_P1S_main=True,fixed_query_comparator='B5' if reader=='M1' else 'B5-lin',
            criteria=criteria,g1_passed=g1,g2_passed=g2,passed=g1 and g2,statistics=stats))
    return rows


def verify_release(root: Path,*,aggregated: bool=False) -> tuple[dict,dict]:
    plan=driver.all_training()
    barrier=shared.read_json(root/'barrier.json')
    expected={(w,v,s) for w in WRITERS for v in driver.VARIANTS for s in driver.SEEDS}
    primary=[r for r in plan['runs'] if r['group']=='primary']
    shared.require(len(primary)==36 and {(r['writer'],r['variant'],r['seed']) for r in primary}==expected,
                   'F2 reader matrix differs')
    shared.require(barrier['status']=='passed' and barrier['plan_hash']==stable_hash(plan)
                   and len(barrier['runs'])==39 and set(barrier['runs'])=={r['name'] for r in plan['runs']},
                   'F2 global release is incomplete')
    samples=[]
    for run in plan['runs']:
        folder=Path(run['run_dir'])
        rec=shared.read_json(folder/'record.json')
        shared.require(rec['status']=='complete' and rec['config']==run['config'],'F2 evaluation incomplete')
        if run['group']=='writer': continue
        meta=shared.read_json(folder/'confirmation/evaluation.json')
        e=meta['config']['eval']
        shared.require(meta['aggregation_deferred'] and 'table' not in meta and 'diagnostics' not in meta
                       and meta['checkpoint_config']==run['config'] and e['seed']==777001
                       and e['episodes']==2000 and e['splits']==['test'] and not e['counterfactuals']
                       and e['depths']==driver.DEPTHS and e['hops']==list(shared.HOPS), 'F2 confirmation scope differs')
        path=folder/'confirmation/samples.npz'
        metrics=folder/'confirmation/metrics.json'
        shared.require(path.is_file() and metrics.exists()==aggregated,
                       'F2 confirmation missing or aggregation state differs')
        if aggregated:
            final=shared.read_json(metrics)
            shared.require(final['finalized_after_global_barrier'] and
                           final['barrier_created_at']==barrier['created_at'],
                           'F2 aggregation does not follow its global sample barrier')
        samples.append(str(path))
    shared.require(len(barrier['confirmation_samples'])==36 and set(samples)==set(barrier['confirmation_samples']),
                   'F2 release does not cover every sample set')
    return plan,barrier


def audit_plan(plan: dict) -> dict:
    # Shared audits use legacy role names to distinguish joint and frozen writers.
    # Actual configurations, checkpoints, sample labels and tensor values stay intact.
    result=copy.deepcopy(plan)
    for run in result['runs']: run['writer']=LEGACY[run['writer']]
    return result


def statistics(plan: dict,cache) -> tuple[list[dict],list[dict]]:
    endpoints,reported={},[]
    def select(w,v,d):
        return [r for r in plan['runs'] if r['group']=='primary' and r['writer']==w and r['variant']==v
                and (d!='screening' or r['seed']==17)]
    for dataset in ('confirmation','screening','unseen'):
        for writer in WRITERS:
            for reader in READERS:
                p=select(writer,reader,dataset)
                fixed=select(writer,'B5' if reader=='M1' else 'B5-lin',dataset)
                loop=select(writer,'B3',dataset)
                pairs={
                    'reach':shared.paired(cache,p,p,dataset,4,1,(2,3)),
                    'far':shared.paired(cache,p,p,dataset,4,1,(6,7,8)),
                    'specificity':shared.paired(cache,p,p,dataset,4,1,(2,3),specificity=True),
                    'hop1':shared.paired(cache,p,p,dataset,4,1,(1,)),
                    'vs_fixed':shared.paired(cache,p,fixed,dataset,4,4,(2,3)),
                    'vs_loop':shared.paired(cache,p,loop,dataset,4,4,(2,3))}
                if dataset=='confirmation': endpoints[writer,reader]=pairs
                else:
                    reported.extend(dict(writer=writer,r=32,reader=reader,dataset=dataset,endpoint=n,
                                         statistics=v,gating=False) for n,v in pairs.items())
                reported.append(dict(writer=writer,r=32,reader=reader,dataset=dataset,
                    endpoint='H6_L8_minus_L4_hops4to8',gating=False,
                    statistics=shared.paired(cache,p,p,dataset,8,4,(4,5,6,7,8))))
            for name,va,vb,da,db in (('M1_minus_M1lin_L4_reach','M1','M1-lin',4,4),
                                     ('B4_L1_minus_M1_L1_reach','B4','M1',1,1)):
                reported.append(dict(writer=writer,r=32,reader='M1',dataset=dataset,endpoint=name,gating=False,
                    statistics=shared.paired(cache,select(writer,va,dataset),select(writer,vb,dataset),dataset,da,db,(2,3))))
    return evaluate_f2_gates(endpoints,evaluation_seed=777001),reported


def depth_effect_difference(fact: dict,unfact: dict,*,resamples: int=2000,seed: int=2718) -> dict:
    """Joint episode bootstrap of (fact4-fact1)-(unfact4-unfact1), hops2/3."""
    from looplora.sym.stats import load_samples
    shared.require(set(fact)==set(unfact) and bool(fact),'paired training seeds differ')
    shared.require(resamples>=2000,'at least 2000 resamples required')
    reference=None
    delta_rows=[]
    per_seed=[]
    for s in sorted(fact):
        views=[]
        for source,depth in ((fact[s],4),(fact[s],1),(unfact[s],4),(unfact[s],1)):
            a=load_samples(source)
            keep=np.flatnonzero((a['split']=='test')&(a['condition']=='original')&np.isin(a['hop'],(2,3))&(a['L']==depth))
            i=keep[np.lexsort((a['query_id'][keep],a['episode_id'][keep]))]
            labels=np.stack([a[k][i] for k in ('episode_id','query_id','hop','answer')],axis=1)
            shared.require(len(i)>0,'empty paired contrast')
            if reference is None: reference=labels
            shared.require(np.array_equal(labels,reference),'paired episode/query/hop/answer labels differ')
            views.append(a['correct'][i].astype(np.float64))
        delta=views[0]-views[1]-views[2]+views[3]
        delta_rows.append(delta)
        per_seed.append(dict(seed=int(s),difference=float(delta.mean())))
    episodes,inverse=np.unique(reference[:,0],return_inverse=True)
    counts=np.bincount(inverse)
    shared.require(np.all(counts==2),'each episode must contain hops2 and3')
    for episode in range(len(episodes)):
        shared.require(set(reference[inverse==episode,2])=={2,3},'episode hop coverage differs')
    totals=np.stack([np.bincount(inverse,weights=d)/counts for d in delta_rows])
    pooled=totals.mean(axis=0)
    boot_seed=derive_seed(seed,'sym','f2_factorized_depth_effect',sorted(fact),[2,3])
    rng=np.random.default_rng(boot_seed)
    boot=np.empty(resamples)
    for start in range(0,resamples,128):
        stop=min(resamples,start+128)
        draws=rng.integers(0,len(episodes),size=(stop-start,len(episodes)))
        boot[start:stop]=pooled[draws].mean(axis=1)
    result={}
    for role,confidence in (('report',.95),('gate',.975)):
        low,high=np.quantile(boot,[(1-confidence)/2,1-(1-confidence)/2])
        result[role]=dict(difference=float(pooled.mean()),ci_low=float(low),ci_high=float(high),
            confidence=confidence,per_seed=per_seed,training_seeds=sorted(fact),n_seeds=len(fact),
            n_episodes=len(episodes),n_queries_per_seed=len(reference),resamples=resamples,seed=seed,
            bootstrap_seed=boot_seed,cluster_unit='episode',joint_episode_resampling=True,
            statistic='mean_per_seed_factorized_minus_unfactorized_depth_effect',split='test',
            condition='original',hops=[2,3],depths=[4,1],gating=False)
    return result


def exact_successor_lengths(plan: dict) -> list[dict]:
    counts=defaultdict(lambda:np.zeros(3,dtype=np.int64))
    for run in plan['runs']:
        if run['group']!='primary': continue
        with np.load(Path(run['run_dir'])/'screening/samples.npz',allow_pickle=False) as z:
            chosen=z['condition']=='swap_successors'
            a={k:z[k][chosen] for k in ('L','counterfactual_hop_true','prediction','answer','original_prediction')}
        for depth in (1,4,8):
            for hop in np.unique(a['counterfactual_hop_true']):
                mask=(a['L']==depth)&(a['counterfactual_hop_true']==hop)
                if not mask.any(): continue
                key=(run['writer'],run['variant'],depth,int(hop))
                counts[key]+=np.array([mask.sum(),np.count_nonzero(a['prediction'][mask]==a['answer'][mask]),
                                      np.count_nonzero(a['original_prediction'][mask]==a['answer'][mask])])
    return [dict(writer=w,variant=v,L=d,new_path_length=h,new_hop_le_L=h<=d,n=int(c[0]),
                 follow_rate=float(c[1]/c[0]),unswapped_follow_rate=float(c[2]/c[0]),
                 paired_follow_difference=float((c[1]-c[2])/c[0]),gating=False)
            for (w,v,d,h),c in sorted(counts.items())]


def compact_exports(out: Path,raw: Path,tables: list[dict],gates: list[dict],reported: list[dict]) -> None:
    driver.csv_table(out/'gates.csv',[dict(writer=g['writer'],reader=g['reader'],g1_passed=g['g1_passed'],
                      g2_passed=g['g2_passed'],passed=g['passed'],replication_required=False) for g in gates])
    driver.csv_table(out/'gate_endpoints.csv',[dict(writer=g['writer'],reader=g['reader'],endpoint=n,
             confidence=s['confidence'],difference=s['difference'],ci_low=s['ci_low'],ci_high=s['ci_high'],
             per_seed=json.dumps(s['per_seed'])) for g in gates for n,pair in g['statistics'].items() for s in pair.values()])
    driver.csv_table(out/'gate_criteria.csv',[dict(writer=g['writer'],reader=g['reader'],criterion=n,**c)
                                            for g in gates for n,c in g['criteria'].items()])
    (out/'gates.md').write_text((raw/'gates.md').read_text().replace('F1','F2').rstrip()+'\n')
    # Aggregate over training seeds; detailed run/set cells remain in runs/.
    accum=defaultdict(lambda:np.zeros(4))
    for row in tables:
        key=tuple(row[k] for k in ('group','writer','variant','dataset','split','condition','hop','L'))
        accum[key]+=np.array([row['n'],row['n']*row['accuracy'],row['n']*row['full_accuracy'],row['n']*row['chance_accuracy']])
    summary=[dict(zip(('group','writer','variant','dataset','split','condition','hop','L'),key),
                  n=int(a[0]),accuracy=a[1]/a[0],full_accuracy=a[2]/a[0],chance_accuracy=a[3]/a[0])
             for key,a in sorted(accum.items())]
    driver.csv_table(out/'accuracy_summary.csv',summary)
    driver.csv_table(out/'reported_contrasts.csv',[{k:r[k] for k in ('writer','reader','dataset','endpoint')}|
             dict(confidence=s['confidence'],difference=s['difference'],ci_low=s['ci_low'],ci_high=s['ci_high'],
                  per_seed=json.dumps(s['per_seed'])) for r in reported for s in r['statistics'].values()])
    a5=json.loads((raw/'swap_successors.json').read_text())
    driver.csv_table(out/'swap_successors.csv',a5)
    driver.csv_table(out/'writer_t1_scores.csv',[{k:r[k] for k in
        ('name','writer','seed','n','accuracy','full_accuracy','chance_accuracy')}
        for r in tables if r['group']=='writer'])


def analyze(root: Path,out: Path) -> None:
    plan,barrier=verify_release(root)
    raw=root/'analysis'
    shared.require(not (raw/'analysis_started.json').exists(),'F2 analysis already started')
    shared.write_json(raw/'analysis_started.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
                       plan_hash=stable_hash(plan),confirmation_metrics_previously_read=False))
    cache=shared.CompactCache(root/'analysis_cache',limit=12)
    tables,logs,audit=shared.collect(audit_plan(plan),barrier,raw,cache)
    reverse={v:k for k,v in LEGACY.items()}
    for row in tables: row['writer']=reverse[row['writer']]
    for name in ('swap_successors','swap_successors_per_seed'):
        rows=json.loads((raw/f'{name}.json').read_text())
        for row in rows: row['writer']=reverse[row['writer']]
        shared.write_json(raw/f'{name}.json',rows)
    shared.require(audit['frozen_writers_checked']==18,'F2 must audit all18 frozen readers')
    audit['role_adapter_plan_hash']=audit['plan_hash']
    audit['plan_hash']=stable_hash(plan)
    audit['audit_role_aliases']=LEGACY
    for group in audit['controlled_writer_groups']: group['writer']=reverse[group['writer']]
    shared.require(len(audit['controlled_writer_groups'])==6 and
                   all(len(g['members'])==6 for g in audit['controlled_writer_groups']),
                   'F2 must compare all six frozen variants in each seed and dataset')
    shared.write_json(raw/'artifact_audit.json',audit)
    gates,reported=statistics(plan,cache)
    f1.tables(raw,tables,gates,reported)
    compact_exports(out,raw,tables,gates,reported)
    plot_training(out,plan,logs)
    driver.csv_table(out/'swap_successors_new_path_length.csv',exact_successor_lengths(plan))
    shared.write_json(out/'artifact_audit_summary.json',dict(status=audit['status'],runs=39,
          reader_runs=36,frozen_writers_checked=18,sample_rows=audit['sample_rows'],
          global_barrier_created_at=barrier['created_at'],all_sample_tables_reproduced=True,
          all_depth_snapshots_identical=True,all_controlled_groups_identical=True,
          all_frozen_writers_match_source=True,full_audit=str(raw/'artifact_audit.json')))
    shared.write_json(out/'analysis_summary.json',dict(status='complete',runs=39,reader_runs=36,
          gate_rows=4,confirmation_look_count=1,replication_required=False,
          passing_rows=[dict(writer=g['writer'],reader=g['reader']) for g in gates if g['passed']],
          completed_at=datetime.now(timezone.utc).isoformat(),cross_family='pending paired WP-H analysis'))
    print('F2 analysis completed: 39 stages,36 readers,4 gate rows',flush=True)


def plot_training(out: Path,plan: dict,logs: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(15,7))
    grid=fig.add_gridspec(3,6,height_ratios=(1,1,.9))
    colors={17:'#1765ad',29:'#c05e13',43:'#29843e'}
    for i,writer in enumerate(WRITERS):
        for j,variant in enumerate(driver.VARIANTS):
            ax=fig.add_subplot(grid[i,j])
            for run in plan['runs']:
                if run['group']!='primary' or (run['writer'],run['variant'])!=(writer,variant): continue
                points=logs[run['name']]
                ax.plot([r['step'] for r in points],[r['loss'] for r in points],
                        color=colors[run['seed']],linewidth=.8,label=str(run['seed']))
            ax.set_title(f'{writer}\n{variant}',fontsize=8)
            ax.set_yscale('log')
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.2)
            if j==0: ax.set_ylabel('Training loss')
    ax=fig.add_subplot(grid[2,:])
    for run in plan['runs']:
        if run['group']!='writer': continue
        points=logs[run['name']]
        ax.plot([r['step'] for r in points],[r['loss'] for r in points],
                color=colors[run['seed']],label=str(run['seed']))
    ax.set_yscale('log')
    ax.set_title('Fresh T1 factorized writers',fontsize=9)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Training loss')
    ax.legend(title='Seed',ncol=3,fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'training_losses.pdf')
    fig.savefig(out/'training_losses.png',dpi=140)
    plt.close(fig)


def cross_family(root: Path,out: Path,reference_plan: Path,reference_barrier: Path) -> None:
    # No reference file is opened before F2's complete training/sample barriers.
    plan,barrier=verify_release(root,aggregated=True)
    checked_at=datetime.now(timezone.utc).isoformat()
    summary=shared.read_json(out/'analysis_summary.json')
    shared.require(summary['status']=='complete','finish F2 analysis before paired reference contrasts')
    released=shared.read_json(reference_plan)
    released_barrier=shared.read_json(reference_barrier)
    actual_plan,actual_barrier=shared.verify_release(Path(released['root'])/'plan.json')
    shared.require(actual_plan==released and actual_barrier==released_barrier,
                   'released reference copies differ from raw run records')
    shared.require(released['steps']==15000,'reference training budget must match F2')
    expected={(LEGACY[w],v,s) for w in WRITERS for v in READERS for s in driver.SEEDS}
    references=[r for r in released['runs'] if r['group']=='primary'
                and (r['writer'],r['variant'],r['seed']) in expected and r['r']==32]
    shared.require(len(references)==12 and
                   {(r['writer'],r['variant'],r['seed']) for r in references}==expected,
                   'reference must cover twelve matched unfactorized readers')
    cache=shared.CompactCache(root/'analysis_cache',limit=12)
    audits=[]
    accuracy=defaultdict(lambda:np.zeros(2))
    for run in references:
        folder=Path(run['run_dir'])/'confirmation'
        metadata=shared.read_json(folder/'evaluation.json')
        snapshot=shared.audit_snapshots(metadata,metadata['config'])
        sample_hash=shared.file_hash(folder/'samples.npz')
        shared.require(sample_hash==metadata['samples_hash'],'released reference sample seal differs')
        with np.load(folder/'samples.npz',allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        table,audit=shared.audit_samples(arrays,metadata['config'])
        for row in table:
            accuracy[run['writer'],run['variant'],row['hop'],row['L']]+=np.array(
                [row['n'],row['n']*row['accuracy']])
        cache.save('wph-'+run['name'],'confirmation',arrays,'test')
        audits.append(dict(name=run['name'],samples=str(folder/'samples.npz'),
                           file_hash=sample_hash,
                           sample_rows=audit['rows'],depth_snapshot_count=len(snapshot)))
        del arrays
    reported=[]
    for writer in WRITERS:
        for reader in READERS:
            fact=[r for r in plan['runs'] if r['group']=='primary'
                  and r['writer']==writer and r['variant']==reader]
            unfact=[dict(r,name='wph-'+r['name']) for r in references
                    if r['writer']==LEGACY[writer] and r['variant']==reader]
            at4=shared.paired(cache,fact,unfact,'confirmation',4,4,(2,3))
            effects=depth_effect_difference(
                {r['seed']:cache.get(r['name'],'confirmation') for r in fact},
                {r['seed']:cache.get(r['name'],'confirmation') for r in unfact})
            for name,pair in (('factorized_minus_unfactorized_L4_reach',at4),
                              ('factorized_minus_unfactorized_depth_effect_reach',effects)):
                reported.append(dict(writer=writer,reader=reader,endpoint=name,
                                     statistics=pair,gating=False))
    shared.write_json(root/'analysis/cross_family_contrasts.json',reported)
    driver.csv_table(out/'cross_family_contrasts.csv',[
        {k:r[k] for k in ('writer','reader','endpoint','gating')}|
        dict(confidence=s['confidence'],difference=s['difference'],ci_low=s['ci_low'],ci_high=s['ci_high'],
             per_seed=json.dumps(s['per_seed'])) for r in reported for s in r['statistics'].values()])
    shared.write_json(root/'analysis/reference_audit.json',dict(runs=audits))
    reference_accuracy=[dict(writer=w,reader=p,hop=h,L=d,n=int(a[0]),accuracy=float(a[1]/a[0]))
                        for (w,p,h,d),a in sorted(accuracy.items())]
    driver.csv_table(out/'reference_accuracy.csv',reference_accuracy)
    plot_depth_profiles(out,reference_accuracy)
    shared.write_json(out/'reference_audit_summary.json',dict(status='passed',reference_commit='758c06a',
        reference_plan=str(reference_plan),reference_barrier=str(reference_barrier),
        reference_plan_hash=stable_hash(released),reference_run_count=12,
        reference_sample_rows=sum(r['sample_rows'] for r in audits),
        f2_plan_hash=stable_hash(plan),f2_sample_barrier_created_at=barrier['created_at'],
        f2_training_barrier_checked_before_reference_at=checked_at,
        all_episode_query_hop_answer_labels_paired=True,depth_effects_jointly_resampled=True,
        gating=False,full_audit=str(root/'analysis/reference_audit.json')))
    summary.update(cross_family='complete',cross_family_completed_at=datetime.now(timezone.utc).isoformat())
    shared.write_json(out/'analysis_summary.json',summary)
    print('F2 paired reference contrasts completed: twelve matched WP-H readers',flush=True)


def plot_depth_profiles(out: Path,reference_accuracy: list[dict]) -> None:
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (out/'accuracy_summary.csv').open() as f:
        fact=[r for r in csv.DictReader(f) if r['group']=='primary' and r['dataset']=='confirmation'
              and r['condition']=='original']
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    for i,writer in enumerate(WRITERS):
        for j,reader in enumerate(READERS):
            ax=axes[i,j]
            for family,source,w,pkey,color in (
                ('F2 factorized',fact,writer,'variant','#1765ad'),
                ('WP-H unfactorized',reference_accuracy,LEGACY[writer],'reader','#555555')):
                for hops,label,style in (((2,3),'hops 2–3','-'),((6,7,8),'hops 6–8','--')):
                    values=[np.mean([float(r['accuracy']) for r in source if r['writer']==w
                                    and r[pkey]==reader and int(r['L'])==d and int(r['hop']) in hops])
                            for d in driver.DEPTHS]
                    ax.plot(driver.DEPTHS,values,style,color=color,marker='.',
                            label=f'{family}, {label}')
            ax.set_title(f'{writer}, {reader}',fontsize=10)
            ax.grid(alpha=.2)
            ax.set_ylim(bottom=0)
            ax.set_xticks([0,1,4,8,12,16])
            if i==1: ax.set_xlabel('Reader depth L')
            if j==0: ax.set_ylabel('Confirmation accuracy')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    fig.suptitle('Three-seed means; F2 has no replication setting',fontsize=11)
    fig.tight_layout(rect=(0,.10,1,.95))
    fig.savefig(out/'depth_profiles.pdf')
    fig.savefig(out/'depth_profiles.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=driver.RUNS)
    parser.add_argument('--out',type=Path,default=driver.REPORTS)
    parser.add_argument('--reference-plan',type=Path)
    parser.add_argument('--reference-barrier',type=Path)
    args=parser.parse_args()
    if args.reference_plan is not None:
        if args.reference_barrier is None: parser.error('--reference-barrier is required with --reference-plan')
        cross_family(args.root,args.out,args.reference_plan,args.reference_barrier)
    else:
        if args.reference_barrier is not None: parser.error('--reference-plan is required with --reference-barrier')
        analyze(args.root,args.out)
