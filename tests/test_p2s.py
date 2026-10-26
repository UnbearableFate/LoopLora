"""P2-S budget invariants, embargo boundaries and descriptive-cost exclusions."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_p2s as analysis
import benchmark_p2s as bench
import run_p2s as driver
from looplora.sym.config import resolve_config
from looplora.sym.model import SymbolicModel


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
