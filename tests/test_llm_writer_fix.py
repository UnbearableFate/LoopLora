"""Behavioral controls for WP-L3 value, addressing and paired-position writes."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from looplora.core import memory
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.data.relations import generate
from looplora.data.render import render_episodes,load_name_pool
from looplora.infra.seeds import derive_seed
from looplora.llm.inject import load_tokenizer
from looplora.llm.train import LLMExperiment,make_task_spec
from looplora.llm.writer_fix import (FixAdapter,WriterFixExperiment,fact_positions,auxiliary_losses,alignment)
from looplora.llm.writer_fix_run import configuration,choose,save_checkpoint,load_inference


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17,'writer-fix-test-model'))
        m=Qwen3ForCausalLM(Qwen3Config(vocab_size=151936,hidden_size=16,
            intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield m
    assert not hasattr(m,'looplora_adapters')


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


def cfg(arm):
    config=configuration(arm)
    config['model'].update(layer=1,r=4)
    return config


def arrays(config,split='train'):
    return generate(make_task_spec(config),split,0,2,derive_seed(17,'writer-fix-fixture'))


def test_fact_positions_are_causal_and_align_one_pair_per_fact(tokenizer):
    a=arrays(cfg('W4'))
    rendered=render_episodes(a,tokenizer=tokenizer)
    indices=fact_positions(a,rendered)
    assert indices.shape==(2,16,3)
    pool=load_name_pool()
    for b,row in enumerate(rendered):
        for j,(final,pre,tail) in enumerate(indices[b].tolist()):
            assert pre+1==tail<final
            assert row.fact_final_mask[final]
            assert row.context_ids[tail]==pool.token_ids[int(a.facts[b,j,2])]


@pytest.mark.parametrize('arm',['W0','W4'])
def test_packed_writes_match_explicit_delta_rule_and_code_gradients(arm):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17,'writer-fix-algebra'))
        initial=make_adapter(AdapterConfig(d_in=8,d_out=7,r=4)).double()
        adapter=FixAdapter(initial,arm).double()
        u=torch.randn(2,9,4,dtype=torch.float64,requires_grad=True)
        x=torch.randn(2,9,8,dtype=torch.float64,requires_grad=True)
    adapter.phase='support'
    adapter.positions=torch.tensor([[[2,0,1],[5,3,4],[8,6,7]]]*2)
    adapter.support_x=x
    valid=torch.ones(2,9,dtype=torch.bool)
    write=torch.zeros_like(valid).scatter(1,adapter.positions[:,:,2 if arm=='W4' else 0],True)
    M=torch.zeros(2,4,4,dtype=torch.float64)
    observed=adapter.scan_final(M,u,valid,write)
    expected=M
    for j in range(3):
        if arm=='W4':
            ku=torch.nn.functional.linear(x[:,3*j],adapter.A)
            vu=torch.nn.functional.linear(x[:,3*j+1],adapter.A_v)
        else:
            ku=vu=u[:,3*j+2]
        k=torch.nn.functional.normalize(torch.nn.functional.linear(ku,adapter.K),dim=-1)
        v=torch.nn.functional.linear(vu,adapter.V)
        expected=memory.delta_update(expected,k,v,None,adapter.lam,adapter.beta)
        torch.testing.assert_close(adapter.keys[:,j],k)
        torch.testing.assert_close(adapter.values[:,j],v)
    torch.testing.assert_close(observed,expected,atol=1e-12,rtol=1e-12)
    variables=[adapter.K,adapter.V,adapter.a_lambda,adapter.a_beta]+([adapter.A,adapter.A_v,x] if arm=='W4' else [u])
    left=torch.autograd.grad(observed.square().sum(),variables,retain_graph=True)
    right=torch.autograd.grad(expected.square().sum(),variables)
    for a,b in zip(left,right):
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)


def test_W0_matches_existing_writer_loss_and_memory(model,tokenizer):
    config=cfg('W0'); a=arrays(config)
    base=LLMExperiment(config,model=model,tokenizer=tokenizer)
    try:
        with torch.no_grad():
            base.adapter.B.normal_(0,.03,generator=torch.Generator(device='cuda').manual_seed(derive_seed(17,'W0-B')))
            state={k:v.clone() for k,v in base.adapter.state_dict().items()}
            batch=base.prepare(a); expected,_=base.score(batch,1)
            M=batch.snapshot.states[1].clone()
    finally:
        base.close()
    fix=WriterFixExperiment(config,'W0',model=model,tokenizer=tokenizer)
    try:
        fix.adapter.load_state_dict(state,strict=False)
        with torch.no_grad():
            batch=fix.prepare(a); actual,_=fix.score(batch,1)
        torch.testing.assert_close(batch.snapshot.states[1],M,atol=0,rtol=0)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    finally:
        fix.close()


@pytest.mark.parametrize('arm',['OA','W1','W2','W3','W4','W5'])
def test_intervention_gradients_and_evaluation_does_not_use_aux_head(model,tokenizer,arm):
    config=cfg(arm); fix=WriterFixExperiment(config,arm,model=model,tokenizer=tokenizer)
    try:
        with torch.no_grad():
            fix.adapter.B.normal_(0,.05,generator=torch.Generator(device='cuda').manual_seed(derive_seed(17,'fix-B')))
        batch=fix.prepare(arrays(config))
        assert batch.write_count.eq(16).all() and batch.snapshot.past_key_values is None
        logits,x=fix.score(batch,1,capture_inputs=True)
        a,v=auxiliary_losses(fix,batch,x)
        main=torch.nn.functional.cross_entropy(logits.float().flatten(0,1),batch.answer_ids.flatten())
        (main+a+v).backward()
        assert fix.adapter.A.grad.abs().sum()>0 and fix.adapter.V.grad.abs().sum()>0
        if arm=='OA':
            assert fix.adapter.K.grad is None and fix.adapter.Q.grad is None
            assert fix.adapter.a_lambda.grad is None and fix.adapter.a_beta.grad is None
        if arm in {'W2','W5'}:
            assert a>0 and fix.adapter.Q.grad.abs().sum()>0 and fix.adapter.K.grad.abs().sum()>0
        if arm in {'W3','W5'}:
            assert v>0 and fix.value_head.weight.grad.abs().sum()>0
            head=fix.value_head; fix.value_head=None
            with torch.no_grad():
                independent,_=fix.score(batch,1)
            torch.testing.assert_close(independent,logits,atol=0,rtol=0)
            fix.value_head=head
        if arm=='W4':
            assert fix.adapter.A_v.grad.abs().sum()>0
        metric=alignment(fix,batch,x)
        assert all(t.shape==(2,4) for t in metric.values())
    finally:
        fix.close()


def test_normalization_buffers_and_inference_reload(model,tokenizer,tmp_path):
    config=cfg('W5'); fix=WriterFixExperiment(config,'W5',model=model,tokenizer=tokenizer)
    try:
        stats=fix.standardize()
        assert stats['episodes']==256 and stats['positions']>256*16
        assert fix.adapter.input_mean.dtype==torch.float32
        assert (fix.adapter.input_std>=1e-4).all() and fix.adapter.input_mean.abs().sum()>0
        with torch.no_grad():
            fix.adapter.B.normal_(0,.05,generator=torch.Generator(device='cuda').manual_seed(derive_seed(17,'reload-B')))
            a=arrays(config,'val'); expected,_=fix.score(fix.prepare(a),1)
        optimizer=torch.optim.AdamW(fix.trainable_parameters())
        save_checkpoint(fix,optimizer,tmp_path,1)
        exported=torch.load(tmp_path/'writer.pt',weights_only=True)
        assert 'value_head' not in exported and 'optimizer' not in exported
        mean=fix.adapter.input_mean.clone()
    finally:
        fix.close()
    loaded=load_inference(tmp_path/'writer.pt',model=model,tokenizer=tokenizer)
    try:
        assert loaded.value_head is None
        torch.testing.assert_close(loaded.adapter.input_mean,mean,atol=0,rtol=0)
        with torch.no_grad(): actual,_=loaded.score(loaded.prepare(a),1)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    finally:
        loaded.close()


def records(scores):
    return [dict(arm=arm,status='complete',extension=False,
        training=dict(actual_steps=4800,metrics={'4800':{'correct':scores.get(arm,0)}}))
        for arm in ('OA','W0','W1','W2','W3','W4','W5')]


def test_selection_thresholds_simplicity_and_oracle_exclusion():
    assert choose(records({'OA':1000,'W1':800,'W4':990,'W5':999}))['arm']=='W1'
    assert choose(records({'W4':800,'W3':999}))['arm']=='W4'
    assert choose(records({'OA':1000,'W0':299}))['action']=='stop'
    assert choose(records({'W0':300,'W5':799}))['arm']=='W5'
    assert choose(records({'W0':300,'W1':300}))['arm']=='W0'
    with pytest.raises(ValueError): choose(records({})[:-1])
