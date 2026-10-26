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
@pytest.mark.parametrize('device,batch',[('cpu',4),('cuda',4),('cuda',64)])
def test_factorized_dense_write_forward_and_every_gradient(E,device,batch):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA needed')
    torch.manual_seed(123)
    ref=FactorizedWriter(192,32,32,17).to(device);fast=copy.deepcopy(ref)
    x=torch.randn(batch,E,192,device=device,requires_grad=True);y=x.detach().clone().requires_grad_()
    valid=torch.ones(batch,E,dtype=torch.bool,device=device)
    a,av=execution._FACTORIZED(ref,x,valid);b,bv=execution.factorized_forward(fast,y,valid)
    torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(av,bv,atol=0,rtol=0)
    (a.square().sum()+av.square().sum()).backward();(b.square().sum()+bv.square().sum()).backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
    for p,q in zip(ref.parameters(),fast.parameters()): torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('E',[12,24,36])
def test_oracle_dense_exact_and_padding_fallback(E):
    spec=TaskSpec(name='t2_chain_end',E=E,chain_lengths=driver.LAYOUTS[E]);a=generate(spec,'val',0,64,4242)
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


def test_update_paths_use_chronological_current_and_obsolete_edges():
    from looplora.sym.evaluate import stratify_queries
    from analyze_is import update_paths
    spec=TaskSpec(name='t3_update',base='t2_chain_end',E=14,Q=4,chain_lengths=[8,4],hops=[1,2,3,4])
    a=stratify_queries(generate(spec,'val',0,20,4242),spec,[1,2,3,4],4242)
    labels=update_paths(a)
    assert len(labels)==80 and {v[0] for v in labels.values()}=={False,True}
    for (episode,q),(changed,old) in labels.items():
        i=np.flatnonzero(a.episode_seed==episode)[0]
        assert (old!=a.answer[i,q]) if changed else old==-1
    # A changed current answer must be rejected before any accuracy is tabulated.
    a.answer[0,0]=(int(a.answer[0,0])+1)%1024
    with pytest.raises(ValueError,match='path labels'): update_paths(a)


@pytest.mark.parametrize('writer',['oracle_random','joint_fact'])
def test_dense_full_model_optimizer_trajectory_matches_reference(writer):
    from looplora.sym.model import SymbolicModel
    c=driver.new_config(writer,'M1',36,17,.003);a=SymbolicModel(c);b=copy.deepcopy(a)
    optim_a=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.003,weight_decay=.01)
    optim_b=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=.003,weight_decay=.01)
    spec=TaskSpec(**c['task'])
    for step in range(3):
        arrays=generate(spec,'val',step*64,64,4242)
        losses=[]
        for model,opt,fast in ((a,optim_a,False),(b,optim_b,True)):
            opt.zero_grad(set_to_none=True)
            original=model.writer.forward
            if fast:
                from types import MethodType
                model.writer.forward=MethodType(execution.oracle_forward if writer=='oracle_random' else execution.factorized_forward,model.writer)
            logits,prepared=model(arrays,4)
            model.writer.forward=original
            restricted=logits.masked_fill(~prepared.candidate_mask[:,None,:],-torch.inf)
            loss=torch.nn.functional.cross_entropy(restricted.flatten(0,1),prepared.answer.flatten());loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.detach())
        torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize('reader',['M1-mixed','B8'])
def test_t3_eval_valid_scope_and_immutable_snapshots(tmp_path,reader):
    from looplora.sym.model import SymbolicModel
    from looplora.sym.evaluate import evaluate
    from audit_p2s import audit_evaluation
    variant='M1' if reader=='M1-mixed' else 'B8'
    original=driver.new_config('joint_fact',variant,24,17,.003)
    row=dict(config=original,reader=reader)
    c=driver.evaluation_config(row,12,True,episodes=6,seed=4242,split='val')
    evaluate(SymbolicModel(original),c,tmp_path,samples_only=True)
    _,audit=audit_evaluation(tmp_path,c)
    assert audit['sample_rows']==6*4*(1 if reader=='B8' else 8)


def test_three_point_capability_threshold_and_strict_ci():
    from analyze_is import capability_passes
    s=dict(n_queries_per_seed=4000,n_seeds=3,difference=.03-1e-17,ci_low=.001)
    assert capability_passes(s)
    s['ci_low']=0;assert not capability_passes(s)
    s['ci_low']=.001;s['difference']=359/12000;assert not capability_passes(s)


def test_independent_load_bootstraps_reproduce_shared_intervals(tmp_path):
    from analyze_is import Cache,paired,episode_bootstrap
    cache=Cache(tmp_path);runs={}
    for E in (12,36):
        runs[E]=[]
        for seed in driver.SEEDS:
            name=f'E{E}-s{seed}';runs[E].append(dict(name=name,seed=seed))
            ids=np.repeat(np.arange(8,dtype=np.int64)+2**55+E*100,2)
            query=np.tile(np.array([0,1],dtype=np.int16),8)
            base=dict(episode_id=np.tile(ids,2),query_id=np.tile(query,2),hop=np.tile(query+2,2),
                L=np.repeat(np.array([1,4],dtype=np.int16),16),answer=np.ones(32,dtype=np.int32),prediction=np.ones(32,dtype=np.int32))
            base['correct']=np.r_[np.zeros(16,dtype=bool),np.arange(16)% (2 if E==12 else 4)==0]
            cache.save(name,'confirmation',base,'test')
    stats={E:paired(cache,runs[E],runs[E],4,1,(2,3),E) for E in runs}
    assert stats[12]['bootstrap_seed']!=stats[36]['bootstrap_seed']
    boot={E:episode_bootstrap(cache,runs[E],runs[E],4,1,(2,3),stats[E]) for E in runs}
    assert np.mean(boot[36]-boot[12])<0
    assert stats[12]['difference']==.5 and stats[36]['difference']==.25


def test_all_360_sets_before_any_archive_load(tmp_path,monkeypatch):
    import analyze_is as analysis
    from looplora.infra.seeds import stable_hash
    monkeypatch.setattr(driver,'RUNS',tmp_path)
    trained=dict(runs=[dict(name=f'train-{i}') for i in range(36)])
    monkeypatch.setattr(driver,'all_training',lambda:trained)
    runs=[];paths=[]
    for i in range(360):
        folder=tmp_path/'evaluation'/str(i);c=driver.new_config('joint_fact','M1',24,17,.003)
        row=dict(name=str(i),run_dir=str(folder),config=c,eval_E=24,update=False,reader='M1-mixed');runs.append(row)
        driver.write(folder/'record.json',dict(status='complete',checkpoint_unchanged=True))
        driver.write(folder/'confirmation/evaluation.json',dict(aggregation_deferred=True,checkpoint_config=c,config=driver.evaluation_config(row,24)))
        path=folder/'confirmation/samples.npz';path.touch();paths.append(str(path))
    plan=dict(runs=runs,training_plan_hash=stable_hash(trained))
    driver.write(tmp_path/'evaluation_plan.json',plan)
    driver.write(tmp_path/'training_barrier.json',dict(status='passed',training_stages=36,evaluation_plan_hash=stable_hash(plan)))
    driver.write(tmp_path/'barrier.json',dict(status='passed',evaluation_plan_hash=stable_hash(plan),runs=[r['name'] for r in runs],samples=paths[:-1]))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('opened archive before full barrier'))
    with pytest.raises(ValueError,match='sample membership'): analysis.verify_release()
