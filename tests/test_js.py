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
        trace=trace_reader(model.adapter,reader_batch(model,prepared),16)
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
