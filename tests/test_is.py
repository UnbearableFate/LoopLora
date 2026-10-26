"""Registered IS layouts, exact all-valid write execution and load barriers."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_is as driver
import is_runtime as execution
from looplora.data.relations import TaskSpec,generate
from looplora.sym.writers import OracleWriter,FactorizedWriter


def test_layouts_and_update_scope_are_valid():
    for E,chains in driver.LAYOUTS.items():
        spec=TaskSpec(name='t2_chain_end',E=E,chain_lengths=chains,hops=range(1,min(8,max(chains))+1))
        a=generate(spec,'val',0,3,4242)
        assert a.facts.shape==(3,E,3) and a.fact_valid.all()
    with pytest.raises(ValueError,match='sum'): TaskSpec(name='t3_update',base='t2_chain_end',E=12,chain_lengths=[8,4])
    spec=TaskSpec(name='t3_update',base='t2_chain_end',E=14,chain_lengths=[8,4],hops=[1,2,3,4])
    assert spec.n_updates==2 and spec.update_query_rate==.5


@pytest.mark.parametrize('E',[12,24,36])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_factorized_dense_write_forward_and_every_gradient(E,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA needed')
    torch.manual_seed(123)
    ref=FactorizedWriter(192,32,32,17).to(device);fast=copy.deepcopy(ref)
    x=torch.randn(4,E,192,device=device,requires_grad=True);y=x.detach().clone().requires_grad_()
    valid=torch.ones(4,E,dtype=torch.bool,device=device)
    a,av=execution._FACTORIZED(ref,x,valid);b,bv=execution.factorized_forward(fast,y,valid)
    torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(av,bv,atol=0,rtol=0)
    (a.square().sum()+av.square().sum()).backward();(b.square().sum()+bv.square().sum()).backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
    for p,q in zip(ref.parameters(),fast.parameters()): torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('E',[12,24,36])
def test_oracle_dense_exact_and_padding_fallback(E):
    spec=TaskSpec(name='t2_chain_end',E=E,chain_lengths=driver.LAYOUTS[E]);a=generate(spec,'val',0,4,4242)
    writer=OracleWriter(1024,32,31415).cuda();facts=torch.as_tensor(a.facts,device='cuda');valid=torch.as_tensor(a.fact_valid,device='cuda')
    codes=writer.codes(facts,valid,a.episode_seed)
    original=execution._ORACLE(writer,facts,valid,codes)
    dense=execution.oracle_forward(writer,facts,valid,codes)
    torch.testing.assert_close(original,dense,atol=0,rtol=0)
    valid[0,1]=False
    torch.testing.assert_close(execution._ORACLE(writer,facts,valid,codes),execution.oracle_forward(writer,facts,valid,codes),atol=0,rtol=0)


def test_train_and_confirmation_configuration_has_no_selection():
    for E in (24,36):
        for w in driver.WRITERS:
            for v in ('M1','B8','B4'):
                c=driver.new_config(w,v,E,17,.003)
                assert c['train']['steps']==15000 and c['train']['batch_size']==64
                assert c['eval']['splits']==['val'] and c['eval']['seed']==4242
                row=dict(config=c,reader='M1-mixed' if v=='M1' else v)
                confirm=driver.evaluation_config(row,E)
                assert confirm['eval']['seed']==999001 and confirm['eval']['episodes']==2000
                assert c['eval']['splits']==['val']


def test_no_confirmation_or_reference_read_before_new_training(monkeypatch):
    def stop(): raise ValueError('training barrier')
    def no_read(*a,**k): pytest.fail('confirmation/reference opened before training barrier')
    monkeypatch.setattr(driver,'all_training',stop);monkeypatch.setattr(Path,'read_text',no_read)
    with pytest.raises(ValueError,match='training barrier'): driver.evaluation_plan()
