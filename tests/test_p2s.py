"""P2-S budget invariants, embargo boundaries and descriptive-cost exclusions."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_p2s as analysis
import benchmark_p2s as bench
import run_p2s as driver
from audit_p2s import audit_evaluation
from looplora.sym.config import resolve_config
from looplora.sym.model import SymbolicModel
from looplora.infra.seeds import stable_hash


def test_exact_registered_eight_families_and_twenty_four_seeded_stages():
    rows=driver.families()
    assert len(rows)==8 and sum(r['variant']=='B6' for r in rows)==4
    assert sum(r['variant']=='B7' for r in rows)==2
    assert sum(r['variant']=='B8' for r in rows)==2
    for f in rows:
        for seed in driver.SEEDS:
            c=driver.config(f,seed,15000,.001)
            assert c['train']['steps']==15000 and c['train']['batch_size']==64
            assert c['train']['depths']==([4] if f['variant']=='B8' else [1])
            assert c['model']['reference_r']==32 and c['model']['rho']==.25 and c['model']['alpha']==1
            before=copy.deepcopy(c);confirm=driver.confirmation_config(c)
            assert c==before
            assert confirm['eval']['depths']==([4] if f['variant']=='B8' else driver.DEPTHS)
            assert confirm['eval']['seed']==888001 and confirm['eval']['episodes']==2000
            assert confirm['eval']['splits']==['test'] and not confirm['eval']['counterfactuals']
    assert all(r['writer']=='joint' for r in rows if r['variant']=='B7')


def model(variant,r=32,m=32,writer='joint'):
    c=resolve_config(dict(variant=variant,writer=writer,device='cpu',
        model=dict(r=r,r_mem=m,reference_r=32),task=dict(name='t2_chain_end',E=12,Q=4,V=1024,n_rel=4,
        chain_lengths=[8,4],hops=[1,2,3])))
    return SymbolicModel(c)


@pytest.mark.parametrize('writer',['joint','oracle_random'])
def test_rank64_one_read_matches_rank32_four_read_update_flops(writer):
    a=bench.cost(model('M1',writer=writer),4,14)
    b=bench.cost(model('B6',64,64,writer),1,14)
    assert a['read_flops_per_query']==b['read_flops_per_query']==32768
    assert a['state_bytes_fp32']==4096 and b['state_bytes_fp32']==16384
    assert b['trainable_parameters']>a['trainable_parameters']


@pytest.mark.parametrize('memory,byte_count,flops',[(64,16384,18432),(128,65536,51200)])
def test_state_variant_changes_memory_without_changing_core_width(memory,byte_count,flops):
    m=model('B7',32,memory)
    c=bench.cost(m,1,14)
    assert m.r==32 and m.r_mem==memory
    assert c['state_bytes_fp32']==byte_count and c['read_flops_per_query']==flops


@pytest.mark.parametrize('writer',['joint','oracle_random'])
def test_unshared_reader_has_same_work_and_state_but_extra_nine_rank_squared_parameters(writer):
    a=bench.cost(model('M1',writer=writer),4,14)
    unshared=model('B8',writer=writer);b=bench.cost(unshared,4,14)
    assert b['state_bytes_fp32']==a['state_bytes_fp32']
    assert b['read_flops_per_query']==a['read_flops_per_query']
    assert b['trainable_parameters']-a['trainable_parameters']==9*32**2
    with pytest.raises(ValueError,match='unshared'): bench.cost(unshared,1,14)


def test_reference_files_are_not_opened_when_main_training_is_incomplete(monkeypatch):
    def stop(): raise ValueError('P2-S global training barrier')
    def forbidden(*args,**kwargs): pytest.fail('reference/control file opened before training barrier')
    monkeypatch.setattr(driver,'all_training',stop)
    monkeypatch.setattr(Path,'read_text',forbidden)
    with pytest.raises(ValueError,match='training barrier'): driver.evaluation_plan()
    with pytest.raises(ValueError,match='training barrier'): analysis.verify_release()


def cost_fixture():
    rows=[]
    for writer in driver.WRITERS:
        for batch in (1,64):
            for L in (1,2,4,8):
                rows.append(dict(source='wph',writer=writer,batch_size=batch,variant='M1',r=32,r_mem=32,
                    steps=15000,L=L,cuda_ms_mean=1 if L==1 else 2,
                    accuracy_hops1to3=.5 if L==1 else .8))
            rows.append(dict(source='f1',writer=writer,batch_size=batch,variant='B5',r=128,r_mem=128,
                steps=20000,L=1,cuda_ms_mean=.1,accuracy_hops1to3=1.))
    return rows


def test_twenty_thousand_step_f1_points_cannot_change_primary_practical_decision():
    rows=analysis.practical_rows(cost_fixture())
    assert len(rows)==16
    assert all(r['unmatched_by_equal_or_faster_single_step'] for r in rows if r['M1_L']==4)
    assert all(not r['unmatched_by_equal_or_faster_single_step'] for r in rows if r['M1_L']==1)


def test_equal_latency_and_equal_accuracy_single_step_matches_m1():
    costs=cost_fixture()
    costs.extend(dict(source='p2s',writer='joint',batch_size=b,variant='B6',r=64,r_mem=64,steps=15000,
                      L=1,cuda_ms_mean=2.,accuracy_hops1to3=.8) for b in (1,64))
    rows=analysis.practical_rows(costs)
    assert all(not r['unmatched_by_equal_or_faster_single_step'] for r in rows if r['writer']=='joint')
    assert all(r['unmatched_by_equal_or_faster_single_step'] for r in rows
               if r['writer']=='oracle_random' and r['M1_L']==4)


def test_capability_requires_every_rank_state_lower_bound_strictly_positive():
    rows=[dict(endpoint='E1' if i<4 else 'E2',subset='reach',statistics={'gate':{'ci_low':.01}})
          for i in range(6)]
    rows.append(dict(endpoint='E3',subset='reach',statistics={'gate':{'ci_low':-.5}}))
    assert analysis.capability(rows)
    rows[5]['statistics']['gate']['ci_low']=0.
    assert not analysis.capability(rows)
    with pytest.raises(ValueError,match='six'): analysis.capability(rows[:5])


def test_practical_claim_requires_capability_and_at_least_one_named_setting():
    settings=[dict(any_primary_M1_point_unmatched=False),dict(any_primary_M1_point_unmatched=True)]
    assert analysis.practical_claim(True,settings)
    assert not analysis.practical_claim(False,settings)
    assert not analysis.practical_claim(True,settings[:1])


def test_validation_ties_lock_smaller_lr_and_fresh_three_seed_main_runs(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'RUNS',tmp_path/'runs')
    monkeypatch.setattr(driver,'REPORTS',tmp_path/'reports')
    for f in driver.families():
        for i,lr in enumerate(driver.LRS):
            c=driver.config(f,17,5000,lr)
            folder=driver.RUNS/'pilot'/f'shard-{i:02d}'/c['name']
            driver.write(folder/'config.json',dict(config=c))
            driver.write(folder/'record.json',dict(status='complete',config_hash=stable_hash(c),train_seconds=10))
            driver.write(folder/'metrics.json',dict(table=[dict(split='val',condition='original',hop=h,
                L=4 if f['variant']=='B8' else 1,n=1000,accuracy=.1 if i==0 else .2) for h in (1,2,3)]))
    driver.select()
    with (driver.REPORTS/'selected_lrs.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==8 and all(float(r['lr'])==.001 and int(r['val_correct'])==600 for r in rows)
    plan=json.loads((driver.RUNS/'main_plan.json').read_text())
    assert len(plan['runs'])==24
    assert {r['seed'] for r in plan['runs']}==set(driver.SEEDS)
    assert all(r['config']['train']['steps']==15000 and r['config']['train']['lr']==.001 for r in plan['runs'])


def evaluation_fixture(root,change=None):
    c={'eval':dict(seed=900101,episodes=2,batch_size=2,splits=['test'],counterfactuals=False,
                    depths=[0,1,4],hops=list(range(1,9)))}
    rows=[(ep,q,q+1,d,q+7) for d in (0,1,4) for ep in (2**60+1,2**60+7) for q in range(8)]
    a=np.asarray(rows,dtype=np.int64)
    arrays={k:a[:,i] for i,k in enumerate(('episode_id','query_id','hop','L','answer'))}
    arrays.update(split=np.full(len(a),'test'),condition=np.full(len(a),'original'),
        prediction=a[:,4].copy(),full_prediction=a[:,4].copy(),correct=np.ones(len(a),dtype=bool),
        full_correct=np.ones(len(a),dtype=bool),original_answer=a[:,4].copy(),
        original_prediction=a[:,4].copy(),candidate_size=np.full(len(a),14))
    if change: change(arrays)
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'samples.npz',**arrays)
    meta=dict(config=c,config_hash=stable_hash(c),samples_hash=analysis.shared.file_hash(root/'samples.npz'),
        aggregation_deferred=True,gate0_passed=True,memory_hashes=[dict(split='test',condition='original',
        batch_start=0,L=d,x='a'*32,u='b'*32,memory='c'*32,write_count='d'*32) for d in (0,1,4)])
    driver.write(root/'evaluation.json',meta)
    return c,meta


def test_audit_preserves_full_64_bit_episode_ids_and_matches_all_depths(tmp_path):
    c,_=evaluation_fixture(tmp_path)
    table,audit=audit_evaluation(tmp_path,c)
    assert len(table)==24 and audit['sample_rows']==48 and audit['snapshots']==1
    assert set(audit['labels'][:,0])=={2**60+1,2**60+7}


def test_audit_rejects_correctness_that_does_not_match_saved_predictions(tmp_path):
    c,_=evaluation_fixture(tmp_path,lambda a:a['correct'].__setitem__(0,False))
    with pytest.raises(ValueError,match='correctness'): audit_evaluation(tmp_path,c)


def test_audit_rejects_changed_labels_even_with_consistent_predictions(tmp_path):
    def change(a):
        for k in ('answer','original_answer','prediction','original_prediction','full_prediction'):a[k][0]+=1
    c,_=evaluation_fixture(tmp_path,change)
    with pytest.raises(ValueError,match='depth-dependent'): audit_evaluation(tmp_path,c)


def test_audit_rejects_mutated_memory_snapshot(tmp_path):
    c,meta=evaluation_fixture(tmp_path)
    meta['memory_hashes'][1]['memory']='e'*32
    driver.write(tmp_path/'evaluation.json',meta)
    with pytest.raises(ValueError,match='changes across depths'): audit_evaluation(tmp_path,c)


def test_all_84_blind_sets_must_be_released_before_any_archive_read(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'RUNS',tmp_path)
    trained={'runs':[{'name':f'train-{i}'} for i in range(24)]}
    monkeypatch.setattr(driver,'all_training',lambda:trained)
    runs=[];samples=[]
    for i in range(84):
        folder=tmp_path/'evaluation'/str(i)
        c={'variant':'M1','eval':{}}
        runs.append(dict(name=str(i),run_dir=str(folder),config=c))
        driver.write(folder/'record.json',dict(status='complete',checkpoint_unchanged=True))
        driver.write(folder/'confirmation/evaluation.json',dict(config=driver.confirmation_config(c),
            checkpoint_config=c,aggregation_deferred=True))
        path=folder/'confirmation/samples.npz';path.touch();samples.append(str(path))
    plan={'runs':runs,'training_plan_hash':stable_hash(trained)}
    driver.write(tmp_path/'evaluation_plan.json',plan)
    driver.write(tmp_path/'training_barrier.json',dict(status='passed',training_stages=24,
        evaluation_plan_hash=stable_hash(plan)))
    driver.write(tmp_path/'barrier.json',dict(status='passed',runs=[r['name'] for r in runs],
        evaluation_plan_hash=stable_hash(plan),confirmation_samples=samples[:-1]))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('archive opened before full sample barrier'))
    with pytest.raises(ValueError,match='sample membership'): analysis.verify_release()


def test_cost_summary_keeps_integer_accuracy_totals_and_averages_all_eight_batch_one_queries(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'RUNS',tmp_path)
    runs=[];timing=[];accuracy=[]
    metrics=('cuda_ms_mean','cuda_ms_median','cuda_ms_p95','wall_ms_mean','wall_ms_median','wall_ms_p95')
    counts=dict(trainable_parameters=13314,registered_parameters=13314,state_bytes_fp32=4096,
                read_flops_per_query=32768,output_projection_flops=4096,candidate_decode_flops=1792)
    for index,seed in enumerate(driver.SEEDS):
        run=dict(name=f'm{index}',source='wph',writer='joint',variant='M1',r=32,r_mem=32,
                 seed=seed,steps=15000,descriptive_only=False)
        runs.append(run)
        for L in (1,2,4,8):
            accuracy.extend(dict(name=run['name'],L=L,hop=h,n=2000,accuracy=(h+index)/2000) for h in range(1,9))
            for batch in (1,64):
                for hop in (range(1,9) if batch==1 else (0,)):
                    timing.append(dict(name=run['name'],L=L,batch_size=batch,latency_hop=hop,warmup=5,repeats=20,
                        **counts,**{k:hop if batch==1 else .5 for k in metrics}))
    driver.csv_table(tmp_path/'benchmark/latency.csv',timing)
    pooled,per_seed=analysis.cost_tables({'runs':runs},accuracy)
    assert len(pooled)==8 and len(per_seed)==24
    assert all(r['n_correct_hops1to3']==27 and r['n_queries_hops1to3']==18000
               and r['accuracy_hops1to3']==27/18000 for r in pooled)
    assert all(r['n_correct_hops1to8']==132 and r['n_queries_hops1to8']==48000 for r in pooled)
    assert all(r['cuda_ms_mean']==(4.5 if r['batch_size']==1 else .5) for r in pooled)
