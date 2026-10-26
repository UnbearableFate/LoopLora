"""F2-specific selection, gate and paired difference-in-depth regression checks."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_p1s_f2 as f2
import run_p1s_f2 as driver


def endpoint_table():
    spec=importlib.util.spec_from_file_location('f2_gate_fixtures',ROOT/'tests/test_sym_gates.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source=module._table()
    return {(w,p):{n:copy.deepcopy(source[f2.LEGACY[w],p][n]) for n in f2.f1.ENDPOINTS}
            for w in f2.WRITERS for p in f2.READERS}


def test_four_f2_rows_omit_replication_and_apply_the_preregistered_point_boundaries():
    table=endpoint_table()
    for row in table.values():
        for name,value in (('reach',.03),('hop1',-.01)):
            for role in ('report','gate'):
                row[name][role]['difference']=value
                for seed in row[name][role]['per_seed']: seed['difference']=value
    rows=f2.evaluate_f2_gates(table,evaluation_seed=777001)
    assert len(rows)==4 and all(r['passed'] for r in rows)
    assert all(len(r['criteria'])==7 and not r['replication_required'] for r in rows)


@pytest.mark.parametrize('name',['reach','specificity','vs_fixed','vs_loop'])
def test_zero_975_lower_bound_fails_only_the_affected_condition_reader(name):
    table=endpoint_table()
    table['joint_fact','M1-lin'][name]['gate']['ci_low']=0.
    rows=f2.evaluate_f2_gates(table,evaluation_seed=777001)
    assert {(r['writer'],r['reader']) for r in rows if not r['passed']}=={('joint_fact','M1-lin')}


def test_gate_missing_row_and_screening_seed_are_rejected():
    with pytest.raises(ValueError,match='confirmation'):
        f2.evaluate_f2_gates(endpoint_table(),evaluation_seed=4242)
    rows=endpoint_table()
    rows.pop(('t1_frozen_fact','M1'))
    with pytest.raises(ValueError,match='four'):
        f2.evaluate_f2_gates(rows,evaluation_seed=777001)


def sample(high):
    rows=[]
    for L in (1,4):
        for ep in range(4):
            for q,hop in enumerate((2,3)):
                rows.append((ep,q,hop,L,9,high[ep] if L==4 else 0))
    a=np.asarray(rows,dtype=np.int64)
    return {**{k:a[:,i] for i,k in enumerate(('episode_id','query_id','hop','L','answer','correct'))},
            'split':np.full(len(a),'test'),'condition':np.full(len(a),'original')}


def test_common_episode_depth_effects_cancel_before_bootstrap():
    source={17:sample([1,0,1,0]),29:sample([0,1,0,1]),43:sample([1,0,0,1])}
    result=f2.depth_effect_difference(source,copy.deepcopy(source))
    assert all(r['difference']==r['ci_low']==r['ci_high']==0 for r in result.values())
    assert result['gate']['bootstrap_seed']==result['report']['bootstrap_seed']
    assert result['gate']['joint_episode_resampling'] is True


def test_factorized_depth_gain_minus_unfactorized_has_correct_orientation():
    a={s:sample([1,1,1,1]) for s in (17,29,43)}
    b={s:sample([0,0,0,0]) for s in (17,29,43)}
    assert f2.depth_effect_difference(a,b)['gate']['difference']==1
    assert f2.depth_effect_difference(b,a)['gate']['difference']==-1


def test_cross_family_label_mismatch_is_rejected():
    a={s:sample([1,0,1,0]) for s in (17,29,43)}
    b=copy.deepcopy(a)
    b[29]['answer'][0]+=1
    with pytest.raises(ValueError,match='labels differ'): f2.depth_effect_difference(a,b)


def test_legacy_audit_roles_do_not_rewrite_factorized_configurations():
    plan={'runs':[{'writer':w,'config':{'writer':w}} for w in f2.WRITERS]}
    before=copy.deepcopy(plan)
    adapted=f2.audit_plan(plan)
    assert plan==before
    assert [r['writer'] for r in adapted['runs']]==['t1_frozen','joint']
    assert [r['config']['writer'] for r in adapted['runs']]==list(f2.WRITERS)


def test_pilot_and_main_b4_keep_distinct_training_and_evaluation_depths():
    pilot=driver.config('pilot','joint_fact','B4',17,5000,.001)
    main=driver.config('main','joint_fact','B4',17,15000,.001)
    assert pilot['train']['depths']==main['train']['depths']==[1]
    assert pilot['eval']['depths']==[1] and main['eval']['depths']==driver.DEPTHS
    assert pilot['eval']['splits']==main['eval']['splits']==['val']


def test_main_freezes_pilot_lrs_and_uses_fresh_seed_specific_shared_writers(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'RUNS',tmp_path/'runs')
    monkeypatch.setattr(driver,'REPORTS',tmp_path/'reports')
    driver.write(driver.REPORTS/'pilot_writer_selection.json',{'lr':.003})
    rows=[dict(name=f'{w}-{v}-{lr}',writer=w,variant=v,lr=lr,val_accuracy=.7 if lr>=.001 else .4)
          for w in driver.WRITERS for v in driver.VARIANTS for lr in driver.LRS]
    monkeypatch.setattr(driver,'complete_rows',lambda stage: rows)
    driver.select_readers()
    plan=json.loads((driver.RUNS/'main_plan.json').read_text())
    assert len(plan['runs'])==39
    assert sum(r['group']=='writer' for r in plan['runs'])==3
    for seed in driver.SEEDS:
        readers=[r for r in plan['runs'] if r['writer']=='t1_frozen_fact' and r['seed']==seed]
        checkpoints={r['config']['pretrain']['writer_checkpoint'] for r in readers}
        assert len(readers)==6 and len(checkpoints)==1
        assert 'main-writers' in next(iter(checkpoints)) and f's{seed}' in next(iter(checkpoints))
    assert all(r['config']['train']['lr']==.001 for r in plan['runs'] if r['group']=='primary')
    assert all(r['config']['train']['steps']==15000 for r in plan['runs'])
