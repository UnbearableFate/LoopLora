"""Fact-only oracle equivalence and unchanged LLM reader contracts on compute."""
import copy
import json
from dataclasses import fields
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.data.relations import generate
from looplora.data.render import render_episodes
from looplora.infra.seeds import derive_seed
from looplora.llm.batching import TokenBatch, tokenize_batch
from looplora.llm.config import resolve_config
from looplora.llm.inject import load_tokenizer
from looplora.llm.oracle import OracleRandomWriter
from looplora.llm.prefetch import BatchStream
from looplora.llm.run import compare_suite, run_one
from looplora.llm.train import LLMExperiment, load_checkpoint, make_task_spec, train
from looplora.sym.writers import OracleWriter


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "oracle-tiny-qwen"))
        value = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield value
    assert not hasattr(value, "looplora_adapters")


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def config(variant="M1", site="o_proj", **updates):
    return resolve_config(dict(name="oracle-unit", writer="oracle_random", stage="reader",
        variant=variant, model=dict(r=32, layer=1, input_site=site),
        task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
        train=dict(steps=2, batch_size=2, depths=[4], log_every=1, warmup_steps=1),
        eval=dict(episodes=2, batch_size=2, hops=[1,2,3,4], counterfactuals=False), **updates))


def data(cfg):
    return generate(make_task_spec(cfg), "train", 0, 2, derive_seed(17, "oracle-unit-data"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rank", [32, 128])
def test_exact_symbolic_oracle_rule_with_pair_keys_and_padding(device, rank):
    writer = OracleRandomWriter(V=32, n_rel=4, seed=17, r_mem=rank).to(device)
    reference = OracleWriter(V=32, r_mem=rank, seed=17).to(device)
    facts = torch.tensor([[[1,0,2],[1,1,3],[1,0,4],[-1,-1,-1]],
                          [[2,3,5],[5,3,8],[-1,-1,-1],[-1,-1,-1]]],device=device)
    valid = torch.tensor([[1,1,1,0],[1,1,0,0]],dtype=torch.bool,device=device)
    clean = facts.masked_fill(~valid[...,None],0)
    # Symbolic OracleWriter on the identical ordered facts with head IDs lifted
    # into the (entity, relation) namespace and tail IDs into entity values.
    lifted = torch.stack((clean[...,0]*4+clean[...,1],clean[...,1],128+clean[...,2]),-1)
    table = torch.cat((writer.pair_codes.flatten(0,1),reference.global_codes),0)
    expected = reference(lifted,valid,table.unsqueeze(0).expand(2,-1,-1))
    assert torch.equal(writer(facts,valid),expected)
    assert torch.equal(writer.global_codes,reference.global_codes)
    assert not torch.equal(writer.pair_codes[1,0],writer.pair_codes[1,1])
    assert not list(writer.parameters())
    assert all(not b.requires_grad and b.grad is None for b in writer.buffers())
    assert torch.equal(writer(facts,torch.zeros_like(valid)),torch.zeros_like(expected))
    assert torch.equal(writer(facts[:,:0],valid[:,:0]),torch.zeros_like(expected))


@pytest.mark.parametrize("updates", [
    {"writer":"unknown"}, {"oracle_seed":True}, {"model":{"r":8}},
    {"stage":"writer"}, {"writer_checkpoint":"/writer.pt"},
    {"eval":{"protocols":["context_retained"]}}, {"protocol":"context_retained"},
])
def test_oracle_config_rejects_incompatible_contracts(updates):
    raw = dict(writer="oracle_random",stage="reader")
    raw.update(updates)
    with pytest.raises(ValueError):
        resolve_config(raw)


@pytest.mark.parametrize("variant", ["M1", "B8", "B4", "B5", "B3"])
def test_rank128_training_roundtrip_and_writer_dimension_validation(model, tokenizer, tmp_path, variant):
    cfg = config(variant)
    cfg["model"]["r"] = 128
    cfg["train"]["steps"] = 1
    cfg = resolve_config(cfg)
    depth = 1 if variant == "B4" else 4
    experiment, summary = train(cfg, tmp_path / variant, model=model, tokenizer=tokenizer)
    try:
        assert experiment.writer_signature()["r_mem"] == 128
        assert experiment.writer.pair_codes.shape == (1024, 4, 128)
        assert summary["writer_frozen"]
        arrays = data(cfg)
        with torch.no_grad():
            expected, _ = experiment.score(experiment.prepare(arrays), depth)
        writer = dict(writer=experiment.writer_state(), signature=experiment.writer_signature())
        writer["signature"]["r_mem"] = 32
        with pytest.raises(ValueError, match="signature"):
            experiment._load_writer(writer)
    finally:
        experiment.close()
    restored = load_checkpoint(tmp_path / variant / "checkpoint.pt", model=model, tokenizer=tokenizer)
    try:
        with torch.no_grad():
            observed, _ = restored.score(restored.prepare(arrays), depth)
        assert torch.equal(observed, expected)
        assert restored.writer_signature()["r_mem"] == 128
    finally:
        restored.close()


def test_question_only_rendering_preserves_exact_queries_and_tokens(tokenizer):
    arrays = data(config())
    full = render_episodes(arrays,tokenizer=tokenizer)
    # __call__ is used for support tokenization; query rendering uses encode.
    class QuestionsOnly:
        def __getattr__(self,name): return getattr(tokenizer,name)
        def __call__(self,*args,**kwargs): raise AssertionError("support text was tokenized")
    short = render_episodes(arrays,tokenizer=QuestionsOnly(),include_support=False)
    for a,b in zip(full,short,strict=True):
        assert a.queries==b.queries and b.context_text=="" and b.context_ids==() and b.fact_spans==()
    expected = tokenize_batch(arrays,tokenizer,device="cpu")
    actual = tokenize_batch(arrays,QuestionsOnly(),device="cpu",include_support=False)
    for field in fields(TokenBatch):
        if field.name.startswith("support") or field.name=="fact_final":
            assert getattr(actual,field.name).shape==(2,0)
        else:
            assert torch.equal(getattr(actual,field.name),getattr(expected,field.name))


@pytest.mark.parametrize("variant", ["M1","M1-lin","B8","B4","B5","B3"])
@pytest.mark.parametrize("site", ["o_proj","residual"])
def test_all_readers_train_and_reload_without_support_forward(model,tokenizer,tmp_path,variant,site):
    cfg=config(variant,site)
    experiment,summary=train(cfg,tmp_path,model=model,tokenizer=tokenizer)
    arrays=data(cfg)
    try:
        assert summary["writer_frozen"] and summary["mean_L"]==(1 if variant=="B4" else 4)
        assert experiment.adapter.A.requires_grad and experiment.adapter.A.grad.abs().sum()>0
        assert all(b.grad is None and not b.requires_grad for b in experiment.writer.buffers())
        assert all(p.grad is None for p in experiment.model.parameters() if not p.requires_grad)
        before=experiment.writer_state()
        calls=[]
        hook=model.model.register_forward_pre_hook(lambda *args:calls.append(1))
        try:
            with patch("looplora.core.memory.scan_final",side_effect=AssertionError("LM writer used")), \
                 patch("looplora.llm.train.encode_support",side_effect=AssertionError("support LM used")):
                prepared=experiment.prepare(arrays)
        finally:
            hook.remove()
        assert not calls and prepared.support_tokens==0 and prepared.support_ids.shape==(2,0)
        assert prepared.write_count.tolist()==[4,4] and prepared.snapshot.past_key_values is None
        memory=prepared.snapshot.states[1].clone()
        assert not memory.requires_grad
        with torch.no_grad():
            expected,_=experiment.score(prepared,1 if variant=="B4" else 4)
        assert torch.equal(memory,prepared.snapshot.states[1])
        with pytest.raises(ValueError,match="memory_only"):
            experiment.prepare(arrays,protocol="context_retained")
        record=torch.load(tmp_path/"checkpoint.pt",weights_only=True)
        assert set(record["writer"])=={"global_codes","pair_codes"}
        signature=experiment.writer_signature()
    finally:
        experiment.close()
    (tmp_path/"writer.pt").unlink()  # The full checkpoint must be self-contained.
    restored=load_checkpoint(tmp_path/"checkpoint.pt",model=model,tokenizer=tokenizer)
    try:
        assert restored.writer_signature()==signature
        assert all(torch.equal(v,restored.writer_state()[k]) for k,v in before.items())
        with torch.no_grad():
            actual_batch=restored.prepare(arrays)
            actual,_=restored.score(actual_batch,1 if variant=="B4" else 4)
        assert torch.equal(memory,actual_batch.snapshot.states[1])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    finally:
        restored.close()


def test_prefetch_two_matches_synchronous_training_exactly(model,tokenizer,tmp_path):
    cfg=config("B8")
    records,states=[],[]
    for workers in (0,2):
        cfg["train"]["prefetch_workers"]=workers
        out=tmp_path/f"workers{workers}"
        experiment,_=train(cfg,out,model=model,tokenizer=tokenizer)
        try:
            states.append(copy.deepcopy(experiment.adapter.state_dict()))
            records.append([{k:v for k,v in json.loads(line).items() if k!="step_seconds"}
                            for line in (out/"train_log.jsonl").read_text().splitlines()])
        finally:
            experiment.close()
    assert records[0]==records[1]
    assert all(torch.equal(value,states[1][key]) for key,value in states[0].items())
    with BatchStream(make_task_spec(cfg),"train",2,17,1,tokenizer=tokenizer,workers=2,
                     include_support=False) as stream:
        arrays,tokens=next(stream)
        assert tokens.support_ids.shape==(2,0) and tokens.support_ids.device.type=="cpu"
        expected=tokenize_batch(arrays,tokenizer,device="cpu",include_support=False)
        assert all(torch.equal(getattr(tokens,f.name),getattr(expected,f.name)) for f in fields(TokenBatch))


def test_seed_is_global_and_reader_depth_does_not_change_memory(model,tokenizer):
    cfg=config()
    experiment=LLMExperiment(cfg,model=model,tokenizer=tokenizer)
    try:
        batch=experiment.prepare(data(cfg))
        initial=batch.snapshot.states[1].clone()
        for L in (0,1,2,4,8):
            with torch.no_grad(): experiment.score(batch,L)
            assert torch.equal(initial,batch.snapshot.states[1])
        expected=experiment.writer_state()
    finally:
        experiment.close()
    cfg.update(seed=43,variant="B8")
    experiment=LLMExperiment(cfg,model=model,tokenizer=tokenizer)
    try:
        assert all(torch.equal(v,experiment.writer_state()[k]) for k,v in expected.items())
    finally:
        experiment.close()


def test_checkpoint_restores_buffers_and_rejects_incompatible_writer(model,tokenizer,tmp_path):
    cfg=config("B8")
    experiment,_=train(cfg,tmp_path/"original",model=model,tokenizer=tokenizer)
    experiment.close()
    record=torch.load(tmp_path/"original/checkpoint.pt",weights_only=True)
    # Saved tables are authoritative, rather than silently regenerated at load.
    record["writer"]["pair_codes"]=-record["writer"]["pair_codes"]
    torch.save(record,tmp_path/"changed.pt")
    restored=load_checkpoint(tmp_path/"changed.pt",model=model,tokenizer=tokenizer)
    try:
        assert torch.equal(restored.writer.pair_codes.cpu(),record["writer"]["pair_codes"])
    finally:
        restored.close()
    for change in ("signature","dtype"):
        bad=copy.deepcopy(record)
        if change=="signature": bad["signature"]["n_rel"]=3
        else: bad["writer"]["pair_codes"]=bad["writer"]["pair_codes"].bfloat16()
        torch.save(bad,tmp_path/f"{change}.pt")
        with pytest.raises(ValueError,match="signature|fp32"):
            load_checkpoint(tmp_path/f"{change}.pt",model=model,tokenizer=tokenizer)
        assert not hasattr(model,"looplora_adapters")


def test_run_record_evaluation_and_oracle_cross_reader_gate(model,tokenizer,tmp_path):
    jobs=[]
    for variant in ("M1","B4"):
        cfg=config(variant)
        cfg["name"]=variant
        record=run_one(cfg,tmp_path/variant,model=model,tokenizer=tokenizer)
        assert record["writer"]=="oracle_random" and record["consumed_writer_checkpoint"] is None
        assert record["writer_hash"] and record["writer_signature"]["seed"]==17
        assert record["status"]=="complete"
        jobs.append(dict(name=variant,config=cfg))
    # Reader A is trained independently; x/u may differ across variants. Memory
    # and write counts must still match. Per-depth x/u checks happen in evaluate.
    result=compare_suite(jobs,tmp_path)
    assert result["frozen_writer_hashes_verified"]==[["M1","B4"]]
    path=tmp_path/"B4/metrics.json"
    metrics=json.loads(path.read_text())
    metrics["invariant_hashes"][0]["M"]="different"
    path.write_text(json.dumps(metrics))
    with pytest.raises(AssertionError,match="hashes differ"):
        compare_suite(jobs,tmp_path)
