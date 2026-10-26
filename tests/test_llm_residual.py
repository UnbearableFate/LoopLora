"""Residual input changes only A's source and preserves the memory protocol."""
from dataclasses import replace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config,Qwen3ForCausalLM

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.core.wrapper import MemoryContext
from looplora.infra.seeds import derive_seed
from looplora.llm.inject import load_tokenizer
from looplora.llm.pilot1_select import writer_config
from looplora.llm.residual import ResidualInputLinear
from looplora.llm.train import LLMExperiment,train,load_checkpoint
from looplora.data.relations import generate
from looplora.llm.train import make_task_spec


@pytest.mark.parametrize('mode',['online','write_only','read_only'])
def test_residual_projection_matches_reference_state_and_branch(mode):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17,'residual-unit'))
        adapter = make_adapter(AdapterConfig(d_in=5,d_out=4,r=3)).double()
        with torch.no_grad():
            adapter.B.normal_(0,.1)
        base = nn.Linear(7,4).double()
        residual = torch.randn(2,4,5,dtype=torch.float64,requires_grad=True)
        attention = torch.randn(2,4,7,dtype=torch.float64,requires_grad=True)
    module = ResidualInputLinear(base,adapter)
    valid = torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
    write = torch.tensor([[1,0,1,0],[0,1,0,1]],dtype=torch.bool)
    reset = torch.tensor([[1,0,0,0],[1,0,0,0]],dtype=torch.bool)
    state = torch.randn(2,3,3,dtype=torch.float64)
    context = MemoryContext(state.clone(),2,valid,write,reset,mode)
    module.context=context
    permitted = torch.zeros_like(write) if mode=='read_only' else write
    branch,expected_state,_ = adapter.process_sequence(residual,state,2,valid,permitted,reset)
    module.capture(residual)
    output = module(attention)
    expected = base(attention) if mode=='write_only' else base(attention)+branch
    torch.testing.assert_close(output,expected)
    torch.testing.assert_close(context.state,expected_state)
    assert module._residual is None and module.base_call_count==1 and module.base_token_count==8
    assert int(context.stats['writes'].sum())==(0 if mode=='read_only' else int((valid&write).sum()))
    (output.square().sum()+context.state.square().sum()).backward()
    assert attention.grad.abs().sum()>0 and residual.grad.abs().sum()>0
    assert adapter.A.grad.abs().sum()>0
    with pytest.raises(RuntimeError,match='pre-hook'):
        module(attention)


def test_residual_qwen_checkpoint_gradients_dimensions_and_teardown(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17,'residual-qwen'))
        # Deliberately different residual/attention widths, as in the real model.
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936,hidden_size=16,
            intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    tokenizer = load_tokenizer()
    cfg = writer_config('residual-test',16,'fact_final',.0003,32,1500)
    cfg['model'].update(layer=1,r=4,input_site='residual')
    cfg['train']['steps']=1
    original = model.model.layers[1].self_attn.o_proj
    original_hooks = len(model.model.layers[1]._forward_pre_hooks)
    experiment,summary = train(cfg,tmp_path,model=model,tokenizer=tokenizer)
    arrays = generate(make_task_spec(cfg),'val',0,2,4242)
    try:
        assert experiment.adapter.A.shape==(4,16)
        assert experiment.bank.modules[1].base.in_features==32
        assert summary['steps']==1 and experiment.adapter.A.grad.abs().sum()>0
        assert experiment.adapter.B.dtype==torch.float32
        with torch.no_grad():
            batch = experiment.prepare(arrays)
            expected,x = experiment.score(batch,1,capture_inputs=True)
        assert x.shape==(*batch.question_ids.shape,16)
        assert batch.snapshot.past_key_values is None
        assert batch.write_count.eq(16).all()
        signature = experiment.writer_signature()
        assert signature['input_site']=='residual' and signature['d_in']==16
    finally:
        experiment.close()
    assert model.model.layers[1].self_attn.o_proj is original
    assert len(model.model.layers[1]._forward_pre_hooks)==original_hooks
    restored = load_checkpoint(tmp_path/'checkpoint.pt',model=model,tokenizer=tokenizer)
    try:
        with torch.no_grad():
            observed,_ = restored.score(restored.prepare(arrays),1)
        torch.testing.assert_close(observed,expected,atol=0,rtol=0)
    finally:
        restored.close()
    record = torch.load(tmp_path/'writer.pt',weights_only=True)
    current = {**cfg,'stage':'reader','writer_checkpoint':str(tmp_path/'writer.pt'),
               'model':{**cfg['model'],'input_site':'o_proj'}}
    with pytest.raises(ValueError,match='signature'):
        LLMExperiment(current,model=model,tokenizer=tokenizer,writer_record=record)
    assert model.model.layers[1].self_attn.o_proj is original
    assert len(model.model.layers[1]._forward_pre_hooks)==original_hooks
