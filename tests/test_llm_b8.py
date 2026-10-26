"""B8 LLM depth, protocol, writer compatibility and WP-L3 controls."""
import json
from types import MethodType

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.core.adapters import UnsharedAdapter, make_adapter
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm.config import resolve_config
from looplora.llm.evaluate import _settings, evaluate
from looplora.llm.inject import load_tokenizer
from looplora.llm.train import LLMExperiment, load_checkpoint, make_task_spec, train
from looplora.llm.writer_fix import WriterFixExperiment, auxiliary_losses, alignment, query_codes
from looplora.llm.writer_fix_run import configuration, save_checkpoint, load_inference


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'llm-b8-model'))
        value = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield value
    assert not hasattr(value, 'looplora_adapters')


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


def config(site='o_proj', variant='B8', stage='joint', **updates):
    return resolve_config(dict(name='b8-unit', variant=variant, stage=stage, seed=17,
        model=dict(r=8, layer=1, input_site=site),
        task=dict(name='t2_chain_end', E=4, Q=2, chain_lengths=[4]),
        train=dict(steps=2, batch_size=2, depths=[4], log_every=1, warmup_steps=1),
        eval=dict(episodes=2, batch_size=2, hops=[1,2,3,4], counterfactuals=False), **updates))


def data(cfg, split='train'):
    return generate(make_task_spec(cfg), split, 0, 2, derive_seed(17, 'llm-b8-data'))


@pytest.mark.parametrize('stage', ['writer', 'reader', 'joint'])
def test_b8_depth_and_protocol_resolution(stage):
    cfg = resolve_config(dict(variant='B8',stage=stage,
        **({'writer_checkpoint':'/writer.pt'} if stage=='reader' else {}),
        train=dict(depths=[1,2]),eval=dict(depths=[0,1,8],protocols=['memory_only','context_retained'])))
    assert cfg['train']['depths'] == cfg['eval']['depths'] == [4]
    assert _settings(cfg)[0] == [4]
    assert cfg['variant'] == 'B8'


@pytest.mark.parametrize('site', ['o_proj', 'residual'])
@pytest.mark.parametrize('protocol', ['memory_only', 'context_retained'])
def test_b8_all_step_gradients_and_direct_core_loss(model, tokenizer, site, protocol):
    cfg = config(site, activation='identity')
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        adapter = experiment.adapter
        assert isinstance(adapter, UnsharedAdapter) and adapter.cfg.unshared_steps == 4
        with torch.no_grad():
            adapter.B.normal_(0,.03,generator=torch.Generator(device='cuda').manual_seed(derive_seed(17,'b8-B')))
        prepared = experiment.prepare(data(cfg), protocol=protocol)
        logits, _ = experiment.score(prepared, 4)
        loss = F.cross_entropy(logits.float().flatten(0,1), prepared.answer_ids.flatten())
        loss.backward()
        for maps in (adapter.R,adapter.U,adapter.Q):
            assert len(maps) == 4
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in maps)
        assert (prepared.snapshot.past_key_values is None) == (protocol=='memory_only')
        assert prepared.write_count.eq(4).all()
        direct = make_adapter(adapter.cfg).to(device='cuda',dtype=torch.float32)
        direct.load_state_dict(adapter.state_dict())
        module = experiment.bank.modules[1]
        original = module.forward

        def forward(wrapper,x):
            context = wrapper.context
            assert context.mode == 'read_only'
            source = wrapper._residual if site=='residual' else x
            if site=='residual':
                wrapper._residual = None
            source = source.to(direct.A.dtype)
            valid = context.valid
            u = direct.u(torch.where(valid[...,None],source,torch.zeros_like(source)))
            # Direct core computation over all question tokens from one immutable snapshot.
            state = context.state[:,None].expand(-1,u.shape[1],-1,-1)
            branch = direct.branch(direct.read(u,state,4))
            branch = torch.where(valid[...,None],branch,torch.zeros_like(branch))
            base = wrapper.base(x)
            return base+branch.to(base.dtype)

        module.forward = MethodType(forward,module)
        try:
            with torch.no_grad():
                expected, _ = experiment.score(prepared, 4)
                expected_loss = F.cross_entropy(expected.float().flatten(0,1),prepared.answer_ids.flatten())
            torch.testing.assert_close(loss,expected_loss,rtol=1e-5,atol=1e-6)
        finally:
            module.forward = original
        with pytest.raises(ValueError, match='unshared_steps'):
            experiment.score(prepared, 1)
    finally:
        experiment.close()


@pytest.mark.parametrize('site', ['o_proj', 'residual'])
def test_b8_writer_reader_joint_checkpoint_compatibility(model, tokenizer, tmp_path, site):
    cfg = config(site, stage='writer')
    writer, summary = train(cfg, tmp_path/'writer',model=model,tokenizer=tokenizer)
    try:
        assert summary['mean_L'] == 4 and writer.config['variant']=='B8'
        record = torch.load(tmp_path/'writer/checkpoint.pt',weights_only=True)
        assert all(f'{name}.{s}' in record['adapter'] for name in ('R','U','Q') for s in range(4))
        state = writer.writer_state()
        signature = writer.writer_signature()
    finally:
        writer.close()
    # The writer signature is independent of shared/unshared reader parameters.
    for variant in ('M1','B8'):
        cfg = config(site,variant=variant,stage='reader',writer_checkpoint=str(tmp_path/'writer/writer.pt'))
        experiment, summary = train(cfg,tmp_path/variant,model=model,tokenizer=tokenizer)
        try:
            assert experiment.writer_signature() == signature
            assert summary['writer_frozen'] and summary['mean_L']==4
            for name,value in state.items():
                assert torch.equal(value,experiment.writer_state()[name])
                assert not getattr(experiment.writer,name).requires_grad
            a = data(cfg,'val')
            with torch.no_grad(): expected,_ = experiment.score(experiment.prepare(a),4)
        finally:
            experiment.close()
        restored = load_checkpoint(tmp_path/variant/'checkpoint.pt',model=model,tokenizer=tokenizer)
        try:
            with torch.no_grad(): actual,_ = restored.score(restored.prepare(a),4)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        finally:
            restored.close()


@pytest.mark.parametrize('variant', ['M1','B5','B3'])
def test_explicit_fixed_depth_training_is_not_resampled(model,tokenizer,tmp_path,variant):
    cfg = config(variant=variant)
    assert cfg['train']['depths']==[4]
    experiment, summary = train(cfg,tmp_path,model=model,tokenizer=tokenizer)
    try:
        rows = [json.loads(r) for r in (tmp_path/'train_log.jsonl').read_text().splitlines()]
        assert summary['mean_L']==4 and {r['L'] for r in rows}=={4}
    finally:
        experiment.close()


@pytest.mark.parametrize('site', ['o_proj','residual'])
@pytest.mark.parametrize('arm', ['OA','W0','W1','W2','W3','W4','W5'])
def test_all_wp_l3_arms_support_b8(model,tokenizer,site,arm):
    cfg = configuration(arm,variant='B8')
    cfg.update(activation='identity')
    cfg['model'].update(r=8,layer=1,input_site=site)
    assert cfg['train']['depths']==cfg['eval']['depths']==[4]
    experiment = WriterFixExperiment(cfg,arm,model=model,tokenizer=tokenizer)
    try:
        with torch.no_grad():
            experiment.adapter.B.fill_(.03)
        batch = experiment.prepare(data(cfg))
        logits,x = experiment.score(batch,4,capture_inputs=True)
        address,value = auxiliary_losses(experiment,batch,x)
        main = F.cross_entropy(logits.float().flatten(0,1),batch.answer_ids.flatten())
        (main+address+value).backward()
        assert experiment.adapter.V.grad.abs().sum()>0
        for maps in (experiment.adapter.R,experiment.adapter.U):
            assert all(p.grad is not None and p.grad.abs().sum()>0 for p in maps)
        if arm=='OA':
            assert all(p.grad is None for p in experiment.adapter.Q)
        else:
            assert all(p.grad is not None and p.grad.abs().sum()>0 for p in experiment.adapter.Q)
        if arm in {'W2','W5'}:
            assert address>0 and experiment.adapter.K.grad.abs().sum()>0
            last = batch.question_mask.sum(-1)-1
            expected = experiment.adapter.u(x[torch.arange(2,device='cuda')[:,None],
                torch.arange(4,device='cuda')[None,:],last].float())
            expected = F.normalize(F.linear(expected,experiment.adapter.Q[0]),dim=-1,eps=experiment.adapter.cfg.eps)
            torch.testing.assert_close(query_codes(experiment,batch,x),expected,rtol=0,atol=0)
        if arm in {'W3','W5'}:
            assert value>0 and experiment.value_head.weight.grad.abs().sum()>0
        if arm=='W4':
            assert experiment.adapter.A_v.grad.abs().sum()>0
        assert all(v.shape==(2,4) for v in alignment(experiment,batch,x).values())
    finally:
        experiment.close()


def test_b8_wp_l3_normalization_and_inference_reload(model,tokenizer,tmp_path):
    cfg=configuration('W5',variant='B8')
    cfg['model'].update(r=8,layer=1,input_site='residual')
    experiment=WriterFixExperiment(cfg,'W5',model=model,tokenizer=tokenizer)
    try:
        assert experiment.standardize()['episodes']==256
        assert experiment.adapter.input_mean.abs().sum()>0
        with torch.no_grad():
            experiment.adapter.B.fill_(.01)
            a=data(cfg,'val')
            expected,_=experiment.score(experiment.prepare(a),4)
        save_checkpoint(experiment,torch.optim.AdamW(experiment.trainable_parameters()),tmp_path,1)
    finally:
        experiment.close()
    restored=load_inference(tmp_path/'writer.pt',model=model,tokenizer=tokenizer)
    try:
        with torch.no_grad(): actual,_=restored.score(restored.prepare(a),4)
        assert restored.value_head is None
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    finally:
        restored.close()


def test_b8_evaluation_only_reads_four_steps_in_both_protocols(model,tokenizer,tmp_path):
    cfg=config()
    experiment=LLMExperiment(cfg,model=model,tokenizer=tokenizer)
    try:
        metrics,_=evaluate(experiment,cfg,tmp_path)
        assert {r['L'] for r in metrics['table']}=={4}
        assert {r['protocol'] for r in metrics['table']}=={'memory_only','context_retained'}
    finally:
        experiment.close()
