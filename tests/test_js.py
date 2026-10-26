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
