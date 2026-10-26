"""Exact Jacobian/eigenvalue gradients and registered reader/stopping contracts."""
from __future__ import annotations
from pathlib import Path
import sys
import pytest
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from js_jacobian import jacobian,spectral_radius


@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_exact_r32_jacobian_radius_and_all_parameter_gradients(backend):
    device='cuda';torch.manual_seed(17);dtype=torch.float64;r=32
    R=(torch.eye(r,device=device,dtype=dtype)*3+torch.randn(r,r,device=device,dtype=dtype)*.02)
    R[0,0]+=2;R.requires_grad_()
    U=(torch.randn(r,r,device=device,dtype=dtype)*.1).requires_grad_()
    Q=(torch.randn(r,r,device=device,dtype=dtype)*.1).requires_grad_()
    M=(torch.randn(1,r,r,device=device,dtype=dtype)*.1).requires_grad_()
    z=(torch.rand(1,1,r,device=device,dtype=dtype)+1).requires_grad_();rho=.4
    def step(v):
        return z.detach().flatten()+rho*F.relu(F.linear(v,R)+F.linear((M[0]@F.linear(v,Q)),U))
    actual=torch.autograd.functional.jacobian(step,z.flatten(),create_graph=True)
    closed=jacobian(R,U,Q,M,z,rho)[0,0]
    torch.testing.assert_close(closed,actual,rtol=1e-12,atol=1e-12)
    a=spectral_radius(closed,backend);b=spectral_radius(actual,backend)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    parameters=(R,U,Q,M,z)
    ga=torch.autograd.grad(torch.relu(a-.9).square(),parameters,retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(torch.relu(b-.9).square(),parameters,retain_graph=True,allow_unused=True)
    for p,x,y in zip(parameters,ga,gb):
        x=torch.zeros_like(p) if x is None else x;y=torch.zeros_like(p) if y is None else y
        torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-8)
    direction=torch.randn_like(R);eps=1e-5
    plus=spectral_radius(jacobian(R+eps*direction,U,Q,M,z,rho),backend)
    minus=spectral_radius(jacobian(R-eps*direction,U,Q,M,z,rho),backend)
    fd=((torch.relu(plus-.9).square()-torch.relu(minus-.9).square())/(2*eps)).sum()
    torch.testing.assert_close(fd,(ga[0]*direction).sum(),atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_spectral_radius_is_not_singular_value_and_complex_gradient_is_finite(backend):
    # A nonnormal matrix has spectral radius 2, while its operator norm is much larger.
    A=torch.eye(32,device='cuda',dtype=torch.float64)*.2
    A[0,0]=A[1,1]=0;A[0,1]=-2;A[1,0]=2;A[2,3]=20;A.requires_grad_()
    radius=spectral_radius(A,backend)
    torch.testing.assert_close(radius,torch.tensor(2.,device='cuda',dtype=torch.float64),rtol=1e-12,atol=1e-12)
    assert torch.linalg.matrix_norm(A,ord=2)>10
    gradient=torch.autograd.grad(torch.relu(radius-.9).square(),A)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_zero_jacobian_has_finite_zero_penalty_gradient(backend):
    A=torch.zeros(2,32,32,device='cuda',requires_grad=True)
    penalty=torch.relu(spectral_radius(A,backend)-.9).square().mean()
    grad=torch.autograd.grad(penalty,A)[0]
    assert penalty==0 and torch.isfinite(grad).all() and not grad.any()


@pytest.mark.parametrize('writer',['oracle_random','joint_fact'])
def test_trace_matches_deployed_reader_every_depth_and_c3_stopping(tmp_path,monkeypatch,writer):
    import numpy as np
    import run_js as driver
    from looplora.sym.model import SymbolicModel,make_task_spec
    from looplora.data.relations import generate
    from looplora.sym.evaluate import stratify_queries
    from looplora.core.stopping import trace_reader,halting_steps,ReaderTrace
    from run_p4s_stopping import reader_batch
    monkeypatch.setattr(driver,'design',lambda:dict(penalty=dict(method=dict(backend='cpu'))))
    c=driver.config(writer,'RS+J',17,True);model=SymbolicModel(c)
    with torch.no_grad(): model.adapter.B.normal_(0,.1)
    checkpoint=tmp_path/'checkpoint.pt';torch.save(dict(config=c,model=model.state_dict()),checkpoint)
    row=dict(config=c,checkpoint=str(checkpoint),name='trace-test',writer=writer,condition='RS+J')
    driver.collect(row,'val',4242,6,tmp_path/'trace',True)
    with np.load(tmp_path/'trace/trace.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    spec=make_task_spec(c,evaluation=True);data=stratify_queries(generate(spec,'val',0,6,4242),spec,list(range(1,9)),4242)
    prepared=model.prepare(data)
    with torch.no_grad():
        for L in range(17):
            output=model.read(prepared,L)
            predictions=output['logits'].masked_fill(~prepared.candidate_mask[:,None,:],-torch.inf).argmax(-1)
            np.testing.assert_array_equal(predictions.cpu().numpy().ravel(),arrays['predictions'][:,L])
            if L in (4,8,16):
                radius=spectral_radius(jacobian(model.adapter.R,model.adapter.U,model.adapter.Q,prepared.memory,output['z'],.25),'cpu')
                np.testing.assert_allclose(radius.cpu().numpy().ravel(),arrays['radii'][:,(4,8,16).index(L)],rtol=1e-5,atol=1e-6)
        trace=trace_reader(model.adapter,reader_batch(prepared),16)
    np.testing.assert_allclose(trace.relative_change[:,1:].cpu().numpy(),arrays['relative_change'][:,1:],rtol=1e-5,atol=1e-6)
    saved=ReaderTrace(torch.as_tensor(arrays['predictions']),torch.zeros(48,17),torch.as_tensor(arrays['relative_change']))
    for threshold in (0,.001,.1,1,4):
        np.testing.assert_array_equal(halting_steps(trace,'relative_change',threshold).cpu().numpy(),halting_steps(saved,'relative_change',threshold).numpy())


def test_paired_initializations_and_registered_new_matrix():
    import run_js as driver
    from looplora.sym.model import SymbolicModel
    seen=set()
    for writer in driver.WRITERS:
        for seed in driver.SEEDS:
            a=driver.config(writer,'RS',seed);b=driver.config(writer,'RS+J',seed)
            assert a['train']==b['train'] and a['train']['steps']==15000
            assert a['train']['batch_size']==64 and a['train']['lr']==.003 and a['train']['depths']==[4,5,6,7,8]
            x=SymbolicModel(a);y=SymbolicModel(b)
            for p,q in zip(x.parameters(),y.parameters()): torch.testing.assert_close(p,q,atol=0,rtol=0)
            seen.update((writer,c,seed) for c in driver.CONDITIONS)
    assert len(seen)==12


def test_confirmation_barrier_precedes_all_reference_and_threshold_access(monkeypatch):
    import run_js as driver
    def stop(): raise ValueError('global training barrier')
    def no_read(*a,**k): pytest.fail('opened reference/threshold before all training')
    monkeypatch.setattr(driver,'all_training',stop);monkeypatch.setattr(Path,'read_text',no_read)
    with pytest.raises(ValueError,match='training barrier'): driver.evaluation_plan()
    with pytest.raises(ValueError,match='training barrier'): driver.collect_stage('confirmation',12)


def test_paired_difference_of_depth_gains_cancels_common_episode_noise(tmp_path):
    import numpy as np
    from analyze_js import Cache,delta_overshoot
    cache=Cache(tmp_path);groups={c:[] for c in ('RS','RS+J')}
    ids=np.repeat(np.arange(8,dtype=np.int64)+2**55,3);query=np.tile(np.arange(3,dtype=np.int16),8)
    for condition in groups:
        for seed in (17,29,43):
            name=f'{condition}-{seed}';groups[condition].append(dict(name=name,seed=seed))
            first=(np.arange(24)+seed)%3==0;last=~first
            arrays=dict(episode_id=np.tile(ids,3),query_id=np.tile(query,3),hop=np.tile(query+1,3),
                answer=np.ones(72,dtype=np.int32),L=np.repeat(np.array([4,16,17]),24),correct=np.r_[first,last,last])
            cache.save(name,arrays)
    stats=delta_overshoot(cache,groups['RS+J'],groups['RS'],'joint_fact')
    assert stats['difference']==stats['ci_low']==stats['ci_high']==0
    assert stats['n_episodes']==8 and stats['n_queries_per_seed']==24
    for condition in groups:
        for row in groups[condition]:
            base=condition=='RS'
            arrays['correct']=np.r_[np.full(24,base),np.full(24,not base),np.full(24,not base)]
            cache.save(row['name'],arrays)
    cache.memory.clear()
    stats=delta_overshoot(cache,groups['RS+J'],groups['RS'],'joint_fact')
    assert stats['difference']==stats['ci_low']==stats['ci_high']==2


def test_overshoot_rule_requires_positive_ci_and_point_floor():
    from analyze_js import decision
    rows=[]
    for w in ('oracle_random','joint_fact'):
        rows.extend([dict(writer=w,endpoint='JS-E1-DeltaS',statistics=dict(difference=.02,ci_low=.001,ci_high=.03)),
            dict(writer=w,endpoint='JS-E1-S',condition='RS+J',statistics=dict(difference=-.01-1e-17,n_queries_per_seed=6000,n_seeds=3))])
    assert all(r['removes_overshoot'] for r in decision(rows))
    rows[0]['statistics']['ci_low']=0;rows[3]['statistics']['difference']=-181/18000
    assert not any(r['removes_overshoot'] for r in decision(rows))


def test_threshold_integer_ties_follow_c3_rule():
    from analyze_js import best_threshold
    cells=[dict(correct=100,steps_total=500,threshold=.001,accuracy=.9),
           dict(correct=100,steps_total=400,threshold=.1,accuracy=.1),
           dict(correct=100,steps_total=400,threshold=.03,accuracy=.2),
           dict(correct=99,steps_total=1,threshold=4,accuracy=1.)]
    assert best_threshold(cells)['threshold']==.03


def test_trace_membership_and_committed_thresholds_before_archive_read(tmp_path,monkeypatch):
    import numpy as np
    import analyze_js as analysis
    import run_js as driver
    from looplora.infra.seeds import stable_hash
    monkeypatch.setattr(driver,'RUNS',tmp_path/'runs');monkeypatch.setattr(driver,'REPORTS',tmp_path/'reports')
    trained=dict(runs=[dict(name=str(i)) for i in range(12)]);monkeypatch.setattr(driver,'all_training',lambda:trained)
    runs=[];paths=[]
    for i in range(24):
        folder=driver.RUNS/'confirmation'/str(i);row=dict(name=str(i),config={'name':str(i)},checkpoint_hash='seal');runs.append(row)
        expected={'name':str(i),'eval':dict(seed=999003,episodes=2000,splits=['test'],depths=list(range(1,17)),hops=list(range(1,9)),counterfactuals=False)}
        driver.write(folder/'evaluation.json',dict(status='complete',aggregation_deferred=True,gate0_passed=True,checkpoint_config=row['config'],
            checkpoint_hash='seal',config=expected,episodes=2000,seed=999003,split='test',radius_depths=[4,8,16]))
        path=folder/'trace.npz';path.touch();paths.append(str(path))
    plan=dict(runs=runs,training_plan_hash=stable_hash(trained))
    driver.write(driver.RUNS/'evaluation_plan.json',plan)
    driver.write(driver.RUNS/'training_barrier.json',dict(status='passed',training_stages=12,evaluation_plan_hash=stable_hash(plan)))
    barrier=dict(status='passed',evaluation_plan_hash=stable_hash(plan),runs=[r['name'] for r in runs],traces=paths[:-1])
    driver.write(driver.RUNS/'confirmation_barrier.json',barrier)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('archive opened before full barrier/threshold lock'))
    with pytest.raises(ValueError,match='trace membership'): analysis.verify_set('confirmation')
    barrier['traces']=paths;driver.write(driver.RUNS/'confirmation_barrier.json',barrier)
    archive=tmp_path/'archive';monkeypatch.setenv('LL_SOURCE_ROOT',str(archive))
    driver.write(driver.REPORTS/'thresholds.json',{'selected':[.03]})
    driver.write(archive/'reports/js/thresholds.json',{'selected':[.1]})
    with pytest.raises(ValueError,match='thresholds differ'): analysis.verify_set('confirmation')


@pytest.mark.parametrize('corruption',['seal','snapshot'])
def test_trace_audit_rejects_modified_archive_and_read_mutation(tmp_path,monkeypatch,corruption):
    import numpy as np
    import analyze_js as analysis
    import run_js as driver
    from analyze_p1s_main import file_hash
    monkeypatch.setattr(driver,'RUNS',tmp_path)
    folder=tmp_path/'confirmation'/'a';folder.mkdir(parents=True)
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'weights')
    ids=np.repeat(np.arange(2000,dtype=np.int64)+2**55,8);query=np.tile(np.arange(8,dtype=np.int16),2000)
    relative=np.ones((16000,17),dtype=np.float32);relative[:,0]=np.nan
    np.savez_compressed(folder/'trace.npz',episode_id=ids,query_id=query,hop=query+1,answer=np.ones(16000,dtype=np.int32),
        candidate_size=np.full(16000,14,dtype=np.int16),predictions=np.ones((16000,17),dtype=np.int32),relative_change=relative,radii=np.full((16000,3),.5,dtype=np.float32))
    hashes=[dict(batch_start=b,L=L,x='1'*32,u='2'*32,memory='3'*32,write_count='4'*32) for b in range(0,2000,64) for L in range(1,17)]
    meta=dict(trace_hash=file_hash(folder/'trace.npz'),memory_hashes=hashes)
    driver.write(folder/'evaluation.json',meta)
    row=dict(name='a',checkpoint=str(checkpoint),checkpoint_hash=file_hash(checkpoint))
    a,labels=analysis.audit_trace(row,'confirmation')
    assert len(a['answer'])==16000 and len(labels)==16000 and labels[:,0].max()>2**55
    if corruption=='seal':
        with (folder/'trace.npz').open('ab') as stream: stream.write(b'tampered')
        with pytest.raises(ValueError,match='trace seal'): analysis.audit_trace(row,'confirmation')
    else:
        hashes[1]['memory']='5'*32;driver.write(folder/'evaluation.json',meta)
        with pytest.raises(ValueError,match='snapshot changed'): analysis.audit_trace(row,'confirmation')
