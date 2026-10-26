"""Meaningful controls for frozen feature positions and probe rank."""
import numpy as np
import torch
from looplora.data.relations import generate
from looplora.data.render import render_episodes, load_name_pool
from looplora.llm.inject import load_tokenizer
from looplora.llm.pilot1_select import writer_config
from looplora.llm.train import make_task_spec
from looplora.llm.diagnosis_probes import positions, fit_probe, LinearProbe
from looplora.infra.seeds import derive_seed


def test_probe_positions_use_actual_tail_and_question_name_tokens():
    tokenizer = load_tokenizer()
    arrays = generate(make_task_spec(writer_config('test',16,'fact_final',.0003,32,1500)),
                      'val',0,3,4242)
    rendered = render_episodes(arrays,tokenizer=tokenizer)
    w,r,wy,ry = positions(arrays,rendered,tokenizer)
    assert w.shape==(48,3) and r.shape==(12,4)
    pool = load_name_pool()
    for index,(b,final,tail) in enumerate(w.tolist()):
        assert final > tail
        assert rendered[b].context_ids[tail]==pool.token_ids[int(wy[index,0])]
        assert rendered[b].fact_final_mask[final]
    for index,(bq,final,head,rel) in enumerate(r.tolist()):
        query = rendered[bq//4].queries[bq%4]
        assert final==len(query.question_ids)-1 and head<rel<final
        assert query.question_ids[head]==pool.token_ids[int(ry[index,0])]


def test_low_rank_probe_is_linear_rank_bounded_and_recovers_known_signal():
    generator = torch.Generator().manual_seed(derive_seed(17,'D1-test'))
    x = torch.randn(500,10,generator=generator)
    y = x[:,:3].argmax(-1)
    for rank in (None,3):
        row,_ = fit_probe(x[:400],y[:400],x[400:],y[400:],classes=3,rank=rank,
                          purpose='positive-control',steps=250,batch_size=64,device='cpu')
        assert row['accuracy'] >= .92
    probe = LinearProbe(10,8,3)
    weight = probe.map[1].weight @ probe.map[0].weight
    assert int(torch.linalg.matrix_rank(weight))<=3
    torch.testing.assert_close(probe(x),x @ weight.T+probe.map[1].bias)
