"""Follow-up depth contracts, signed contrasts and release embargo."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_p2s_f1 as driver
import analyze_p2s_f1 as analysis


def test_registered_matrix_and_fixed_depth_pilot_main_confirmation():
    families=driver.families()
    assert len(families)==8 and len({driver.family_name(f) for f in families})==8
    assert {(f['writer'],f['variant']) for f in families if f['variant']=='M1'}=={(w,'M1') for w in driver.WRITERS}
    assert all(f['writer']=='joint_fact' for f in families if f['variant']!='M1')
    for f in families:
        for seed in driver.SEEDS:
            for steps in (5000,15000):
                c=driver.config(f,seed,steps,.001)
                depth=4 if f['variant'] in ('M1','B8') else 1
                assert c['train']['depths']==[depth] and c['eval']['depths']==[depth]
                assert c['train']['steps']==steps and c['train']['batch_size']==64
                assert c['task']['chain_lengths']==[8,4] and c['task']['hops']==[1,2,3]
                assert c['model']['reference_r']==32 and c['model']['rho']==.25 and c['model']['alpha']==1
                before=copy.deepcopy(c);evaluation=driver.confirmation_config(c)
                assert before==c
                assert evaluation['eval']['seed']==888001 and evaluation['eval']['episodes']==2000
                assert evaluation['eval']['splits']==['test'] and evaluation['eval']['hops']==list(range(1,9))
                assert evaluation['eval']['depths']==([4] if f['variant']=='B8' else driver.DEPTHS)


def test_reference_and_confirmation_access_is_after_all_training(monkeypatch):
    def stop(): raise ValueError('training barrier')
    def forbidden(*args,**kwargs): pytest.fail('opened reference or confirmation before barrier')
    monkeypatch.setattr(driver,'all_training',stop)
    monkeypatch.setattr(Path,'read_text',forbidden)
    with pytest.raises(ValueError,match='training barrier'): driver.evaluation_plan()
    with pytest.raises(ValueError,match='training barrier'): analysis.verify_release()
    with pytest.raises(ValueError,match='training barrier'): driver.evaluate_one('none')


def endpoint_plan():
    rows=[dict(f,source='p2s-f1',seed=s,name=f"new-{driver.family_name(f)}-{s}")
          for f in driver.families() for s in driver.SEEDS]
    for source,w,v in [('p2s','joint','B8'),('p2s','oracle_random','B8'),
                       ('wph','joint','M1'),('wph','oracle_random','M1'),
                       ('f2','joint_fact','M1'),('f2','joint_fact','B4')]:
        rows.extend(dict(source=source,writer=w,variant=v,r=32,r_mem=32,seed=s,
            name=f'{source}-{w}-{v}-{s}') for s in driver.SEEDS)
    return dict(runs=rows)


def test_endpoint_sign_depth_source_and_all_seeds(monkeypatch):
    seen=[]
    def pair(cache,a,b,dataset,La,Lb,hops):
        assert {r['seed'] for r in a}=={17,29,43}=={r['seed'] for r in b}
        assert dataset=='confirmation' and La==4
        seen.append((a[0]['source'],a[0]['variant'],b[0]['source'],b[0]['variant'],Lb,hops))
        return dict(gate=dict(ci_low=.1))
    monkeypatch.setattr(analysis.shared,'paired',pair)
    rows=analysis.endpoints(endpoint_plan(),None)
    assert len(rows)==27 and sum(r['decision_endpoint'] for r in rows)==10
    for r,(sa,va,sb,vb,Lb,hops) in zip(rows,seen):
        if r['endpoint']=='F-E1':
            assert va=='B8' and vb=='M1' and sb=='p2s-f1' and Lb==4
            assert sa==('p2s-f1' if r['writer']=='joint_fact' else 'p2s')
        elif r['endpoint']=='F-E2':
            assert va==vb=='M1' and sa=='p2s-f1' and Lb==4
            assert sb==('f2' if r['writer']=='joint_fact' else 'wph')
            assert hops in ((2,3),(1,))
        else:
            assert sa=='f2' and va=='M1' and sb=='p2s-f1' and vb in ('B6','B7') and Lb==1


def decision_rows():
    rows=[]
    for endpoint in ('F-E1','F-E2'):
        for w in driver.WRITERS:
            rows.append(dict(endpoint=endpoint,writer=w,subset='reach',statistics=dict(gate=dict(ci_low=.01))))
    for v,r,m in (('B6',64,64),('B6',128,128),('B7',32,64),('B7',32,128)):
        rows.append(dict(endpoint='F-E3',writer='joint_fact',subset='reach',variant_b=v,r_b=r,r_mem_b=m,
            statistics=dict(gate=dict(ci_low=.01))))
    return rows


def test_registered_decision_rules():
    rows=decision_rows();result=analysis.decisions(rows)
    assert result['reader_for_P1L']=='B8' and result['shortcut_free_capability_passed']
    assert set(result['training_depth_by_writer'].values())=={'fixed L=4'}
    rows[1]['statistics']['gate']['ci_low']=0 # oracle sharing is not a reader-decision criterion
    assert analysis.decisions(rows)['reader_for_P1L']=='B8'
    rows[2]['statistics']['gate']['ci_low']=0 # joint_fact must be strictly positive
    assert analysis.decisions(rows)['reader_for_P1L']=='M1'
    rows[3]['statistics']['gate']['ci_low']=0
    assert analysis.decisions(rows)['training_depth_by_writer']['joint']=='mixed L=1/2/4'
    rows[-1]['statistics']['gate']['ci_low']=0
    assert not analysis.decisions(rows)['shortcut_free_capability_passed']


def test_capability_rejects_missing_or_duplicate_budget():
    rows=decision_rows();rows[-1]=copy.deepcopy(rows[-2])
    with pytest.raises(ValueError,match='capability matrix'): analysis.decisions(rows)
