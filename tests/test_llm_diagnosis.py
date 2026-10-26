"""Output oracle bypasses identity inference and writes no support or answers."""
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm.diagnosis import inject_last, OutputOracle, fit
from looplora.llm.inject import load_tokenizer
from looplora.llm.pilot1_select import writer_config
from looplora.llm.train import make_task_spec
from looplora.llm.train import LLMExperiment
from looplora.sym.stats import load_samples


def test_oracle_injects_only_actual_last_token_and_preserves_gradients():
    x = torch.randn(2,4,6)
    codes = torch.randn(2,3,requires_grad=True)
    B = torch.randn(6,3,requires_grad=True)
    mask = torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    output = inject_last(x,codes,B,mask)
    delta = output-x
    torch.testing.assert_close(delta[0,1],B@codes[0])
    torch.testing.assert_close(delta[1,3],B@codes[1])
    assert delta[0,[0,2,3]].eq(0).all() and delta[1,:3].eq(0).all()
    output.square().sum().backward()
    assert codes.grad.abs().sum()>0 and B.grad.abs().sum()>0


def test_oracle_only_table_and_B_train_and_disabled_equals_frozen_backbone():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17,'D2-tiny'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936,hidden_size=16,
            intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=4,attention_dropout=0.)).cuda().eval()
    config = writer_config('test',16,'fact_final',.0003,32,1500)
    config['model'].update(layer=1,r=4)
    oracle = OutputOracle(config,model=model,tokenizer=load_tokenizer())
    try:
        assert not any(parameter.requires_grad for parameter in model.parameters())
        assert {id(p) for p in oracle.trainable_parameters()}=={id(oracle.B),id(oracle.table.weight)}
        arrays = generate(make_task_spec(config),'train',0,2,derive_seed(17,'D2-fixture'))
        batch = oracle.prepare(arrays)
        assert not batch.snapshot.states and batch.snapshot.past_key_values is None
        assert batch.write_count.eq(0).all()
        oracle.enabled=False
        with torch.no_grad():
            frozen,_ = oracle.score(batch)
        oracle.enabled=True
        logits,_ = oracle.score(batch)
        torch.testing.assert_close(logits,frozen,atol=0,rtol=0)
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0,1),batch.answer_ids.flatten())
        loss.backward()
        assert oracle.B.grad.abs().sum()>0
        assert oracle.table.weight.grad.eq(0).all()  # B is exactly zero at initialization
        with torch.no_grad():
            oracle.B.normal_(0,.1,generator=torch.Generator(device='cuda').manual_seed(derive_seed(17,'D2-B')))
        oracle.table.weight.grad=None
        logits,_ = oracle.score(batch)
        logits.float().square().mean().backward()
        assert oracle.table.weight.grad.abs().sum()>0
        assert oracle.codes is None and oracle.mask is None
    finally:
        oracle.close()
    assert not hasattr(model,'looplora_adapters')


def test_budget_writer_preserves_val_samples_and_records_source_key_matching(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17,'D3-tiny'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936,hidden_size=16,
            intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=4,attention_dropout=0.)).cuda().eval()
    config = writer_config('test',16,'fact_final',.0003,32,1500)
    config['model'].update(layer=1,r=4)
    experiment = LLMExperiment(config,model=model,tokenizer=load_tokenizer())
    try:
        result = fit(experiment,tmp_path,steps=1,checkpoints=[1])
        assert result['actual_steps']==1 and result['train_episodes']==32
        metric = result['evaluations']['1']
        assert metric['n']==1000 and metric['loss']>0
        assert 0<=metric['matching']['key_top1']<=1
        assert metric['matching']['chance']==1/16
        sample = load_samples(tmp_path/'val-1'/'samples.npz')
        assert len(sample['correct'])==1000
        assert metric['correct']==int(sample['correct'].sum())
        assert metric['em']==float(sample['correct'].mean())
        assert (sample['split']=='val').all() and (sample['hop']==1).all()
        assert (tmp_path/'checkpoint.pt').exists()
    finally:
        experiment.close()
