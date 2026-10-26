"""WP-L4 deployed-read algebra, supervision, controls, exports and gates."""
from dataclasses import replace
import json

import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from looplora.core.memory import delta_update
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm.inject import load_tokenizer
from looplora.llm.read_fix import (ReadFixExperiment,configuration,gram_read,raw_read_loss,
    key_decorrelation,question_terms,extra_losses,controlled_score,diagnostics)
from looplora.llm.read_fix_run import save_checkpoint,load_inference,require_prior,choose
from looplora.llm.train import make_task_spec
from looplora.llm.writer_fix import auxiliary_losses


@pytest.fixture(scope='module')
def tokenizer(): return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17,'read-fix-test-model'))
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=151936,hidden_size=16,
            intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield model
    assert not hasattr(model,'looplora_adapters')


def config(arm):
    cfg=configuration(arm,10)
    cfg['model'].update(layer=1,r=4)
    return cfg


def data(cfg):
    return generate(make_task_spec(cfg),'train',0,2,derive_seed(17,'read-fix-fixture'))


def test_triangular_gram_reconstruction_matches_ordered_delta_for_raw_queries():
    g=torch.Generator().manual_seed(derive_seed(17,'read-fix-gram-test'))
    keys=torch.nn.functional.normalize(torch.randn(3,7,5,generator=g,dtype=torch.float64),dim=-1)
    values=torch.randn(3,7,5,generator=g,dtype=torch.float64)
    q=3*torch.randn(3,4,5,generator=g,dtype=torch.float64)
    lam,beta=torch.tensor(.91,dtype=torch.float64),torch.tensor(.7,dtype=torch.float64)
    state=torch.zeros(3,5,5,dtype=torch.float64)
    for j in range(7): state=delta_update(state,keys[:,j],values[:,j],None,lam,beta)
    expected=torch.einsum('bij,bqj->bqi',state,q)
    observed=gram_read(keys,values,q,lam,beta)
    torch.testing.assert_close(observed['reconstructed'],expected,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(gram_read(keys,values,2*q,lam,beta)['d'],2*observed['d'])
    assert not torch.allclose(observed['a'],observed['b'])
    assert (observed['d']<0).any()


def test_read_regression_stops_target_and_norm_gradients():
    read=torch.tensor([[[1.,2.],[3.,1.]]],requires_grad=True)
    target=torch.tensor([[[2.,0.],[1.,2.]]],requires_grad=True)
    loss=raw_read_loss(read,target)
    expected=((read-target.detach()).square().sum(-1)/(target.detach().square().sum(-1)+1e-6)).mean()
    assert torch.equal(loss,expected)
    loss.backward()
    assert read.grad.abs().sum()>0 and target.grad is None
    assert raw_read_loss(torch.zeros_like(read),torch.zeros_like(target))==0


def test_key_penalty_excludes_repeated_addresses_and_signed_cancellation():
    keys=torch.tensor([[[1.,0.],[1.,0.],[-1.,0.]]],requires_grad=True)
    facts=torch.tensor([[[1,0,3],[1,0,4],[2,0,5]]]).numpy()
    assert key_decorrelation(keys,facts)==1
    key_decorrelation(keys,facts).backward()
    assert torch.isfinite(keys.grad).all()
    orthogonal=torch.eye(3)[None]
    distinct=torch.tensor([[[1,0,3],[2,0,4],[3,0,5]]]).numpy()
    assert key_decorrelation(orthogonal,distinct)==0
    assert key_decorrelation(keys,facts*0)==0


@pytest.mark.parametrize('arm',['R','Rrho','RK','OArho'])
def test_extra_losses_have_deployed_gradients_and_inference_needs_no_head(model,tokenizer,tmp_path,arm):
    cfg=config(arm); experiment=ReadFixExperiment(cfg,arm,model=model,tokenizer=tokenizer)
    try:
        with torch.no_grad(): experiment.adapter.B.fill_(.1)
        batch=experiment.prepare(data(cfg))
        logits,x=experiment.score(batch,1,capture_inputs=True)
        terms=question_terms(experiment,batch,x)
        read,decor=extra_losses(experiment,batch,x)
        address,value=auxiliary_losses(experiment,batch,x)
        if arm!='OArho':
            assert read>0 and address>0 and value>0
            gradients=torch.autograd.grad(read,[experiment.adapter.Q,experiment.adapter.V,
                experiment.adapter.K,experiment.adapter.a_beta],retain_graph=True)
            assert all(g.abs().sum()>0 for g in gradients)
            assert torch.equal(terms['m'],torch.einsum('bij,bqj->bqi',batch.snapshot.states[1],terms['q']))
        else:
            assert read==decor==address==value==0
            assert torch.equal(terms['m'],terms['clean'])
        assert (decor>0)==(arm=='RK')
        assert experiment.adapter.cfg.rho==(.5 if arm in {'Rrho','OArho'} else .25)
        optimizer=torch.optim.AdamW(experiment.trainable_parameters())
        save_checkpoint(experiment,optimizer,tmp_path,10)
        exported=torch.load(tmp_path/'writer.pt',weights_only=True)
        assert 'value_head' not in exported and 'optimizer' not in exported
        with torch.no_grad(): expected,_=experiment.score(experiment.prepare(data(cfg)),1)
    finally: experiment.close()
    loaded=load_inference(tmp_path/'writer.pt',model=model,tokenizer=tokenizer)
    try:
        assert loaded.value_head is None and loaded.read_arm==arm
        with torch.no_grad(): actual,_=loaded.score(loaded.prepare(data(cfg)),1)
        assert torch.equal(expected,actual)
    finally: loaded.close()


@pytest.mark.parametrize('arm',['R','OArho'])
def test_controls_change_deployed_memory_keep_question_inputs_and_reproduce_gram(model,tokenizer,arm):
    cfg=config(arm); experiment=ReadFixExperiment(cfg,arm,model=model,tokenizer=tokenizer)
    try:
        with torch.no_grad():
            experiment.adapter.B.fill_(.2)
            batch=experiment.prepare(data(cfg))
            donor=replace(batch,values=batch.values.flip(0),snapshot=replace(batch.snapshot,
                states={1:batch.snapshot.states[1].flip(0)}))
            before=batch.snapshot.states[1].clone()
            observed=[]
            def capture(module,inputs):
                value=(experiment.adapter.oracle_values if arm=='OArho' else module.context.state)
                observed.append(value.clone())
            hook=experiment.bank.modules[1].register_forward_pre_hook(capture)
            try:
                original,x=controlled_score(experiment,batch)
                repeat,rx=controlled_score(experiment,batch)
                zero,zx=controlled_score(experiment,batch,'zero')
                shuffled,sx=controlled_score(experiment,batch,'shuffle',donor)
            finally: hook.remove()
            assert torch.equal(original,repeat)
            assert all(torch.equal(x,y) for y in (rx,zx,sx))
            assert not torch.equal(original,zero)
            assert torch.equal(observed[0],observed[1])
            assert observed[2].count_nonzero()==0
            expected=(donor.snapshot.states[1].repeat_interleave(4,0) if arm=='R' else
                      question_terms(experiment,donor,x)['clean'].flatten(0,1))
            assert torch.equal(observed[3],expected)
            assert not torch.equal(observed[0],observed[3])
            assert torch.equal(before,batch.snapshot.states[1])
            assert experiment.adapter.oracle_values is None and experiment.adapter.phase=='idle'
            diag,exact=diagnostics(experiment,batch,x)
            assert diag['gram_relative_error'].max()<1e-4
            assert exact['d'].shape==(2,4,16)
    finally: experiment.close()


def test_scientific_gate_waits_for_final_L3_report_and_decision(tmp_path):
    decision=tmp_path/'summary.json'; report=tmp_path/'report.md'
    decision.write_text(json.dumps(dict(screen_only=False,selected_arm=None,stop_P1L=True,
                                       export_check=[dict(identical_predictions=True)])))
    with pytest.raises(ValueError): require_prior(decision,report)
    report.write_text('finished report')
    require_prior(decision,report)
    decision.write_text(json.dumps(dict(screen_only=False,selected_arm='W5',stop_P1L=False,export_check=[1])))
    with pytest.raises(ValueError): require_prior(decision,report)


def records(scores):
    return [dict(arm=arm,status='complete',smoke=False,training=dict(actual_steps=10,
        metrics={'10':{'correct':scores.get(arm,0)}})) for arm in ('R','Rrho','RK','OArho')]


def test_selection_oracle_exclusion_and_single_E8_gate():
    assert choose(records({'R':800,'RK':999}))['arm']=='R'
    assert choose(records({'OArho':1000,'RK':499}))['action']=='stop'
    assert choose(records({'R':500,'Rrho':500}))==dict(action='retry_e8',arm='R',E=8,correct={'R':500,'Rrho':500,'RK':0})
    assert choose(records({'Rrho':799,'RK':700}))['arm']=='Rrho'


def test_full_checkpoint_audit_without_optimizer_steps(model,tokenizer,tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_p1l_read_fix import audit_one
    from looplora.llm.writer_fix import WriterFixExperiment
    from looplora.llm.writer_fix_run import configuration as l3_config,save_checkpoint as save_l3
    cfg=l3_config('W5'); cfg['model'].update(layer=1,r=4)
    folder=tmp_path/'source'; folder.mkdir()
    experiment=WriterFixExperiment(cfg,'W5',model=model,tokenizer=tokenizer)
    try:
        with torch.no_grad(): experiment.adapter.B.fill_(.1)
        optimizer=torch.optim.AdamW(experiment.trainable_parameters())
        save_l3(experiment,optimizer,folder,32900)
        (folder/'record.json').write_text(json.dumps(dict(status='complete',extension=True,
            training=dict(actual_steps=32900))))
    finally: experiment.close()
    result=audit_one(folder,tmp_path/'audit',8,model=model,tokenizer=tokenizer)
    assert result['optimizer_steps']==0 and result['gradients']['optimizer_steps']==0
    assert result['identical_repeated_predictions'] and result['max_repeated_logit_drift']==0
    assert result['value_head_available'] and result['max_gram_relative_error']<1e-4
    import numpy as np
    with np.load(tmp_path/'audit/coefficients.npz') as coefficients:
        assert coefficients['d'].shape==(8,1,16)
