"""Rendering regressions, using the pinned offline Qwen tokenizer on compute."""
from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
from transformers import AutoTokenizer

from looplora.data.render import (
    ASSET_PATH, answer_em, load_name_pool, relation_phrase, render_episode,
    render_episodes, validate_name_pool,
)
from looplora.infra.seeds import derive_seed


@pytest.fixture(scope="module")
def tokenizer():
    pool = load_name_pool()
    return AutoTokenizer.from_pretrained(pool.model, revision=pool.revision, local_files_only=True)


def episode(task="t1", *, hops=1, answer=2, base=None):
    if base is None:
        base = {"t3_update": "t1", "t3_missing": "t2_walk"}.get(task)
    return SimpleNamespace(
        facts=np.asarray([[[0, 0, 1], [-1, -1, -1], [1, 0, 2], [0, 1, 3]]], dtype=np.int32),
        fact_valid=np.asarray([[True, False, True, True]]),
        q_start=np.asarray([[0, 1]], dtype=np.int32),
        q_rel=np.asarray([[0, 0]], dtype=np.int32),
        q_hop_given=np.asarray([[hops, hops]], dtype=np.int32),
        q_hop_true=np.asarray([[2, 1]], dtype=np.int32),
        answer=np.asarray([[answer, 2]], dtype=np.int32),
        episode_seed=np.asarray([17], dtype=np.int64),
        meta={"spec": {"name": task, "V": 8, "base": base}},
    )


def test_name_pool_and_reproducible_selection(tokenizer):
    pool = load_name_pool()
    assert len(pool.names) == 1024
    assert len(set(pool.names)) == len(set(pool.token_ids)) == 1024
    assert pool.revision == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    assert pool.insufficient_name not in pool.names
    validate_name_pool(tokenizer, pool)
    builder_path = ASSET_PATH.parent / "build_name_pool.py"
    spec = importlib.util.spec_from_file_location("name_pool_builder", builder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = json.loads((ASSET_PATH.parent / "name_candidates.v1.json").read_text())
    assert module.build(tokenizer, candidates) == json.loads(ASSET_PATH.read_text())
    assert list(pool.names) == sorted(pool.names, key=lambda name: (derive_seed(0, "name-pool-order", name), name))
    # Both split pools include both categories, rather than a category boundary.
    kinds = json.loads(ASSET_PATH.read_text())["kinds"]
    assert set(kinds[:768]) == set(kinds[768:]) == {"given", "surname"}


@pytest.mark.parametrize("task,hops", [
    ("t1", 1), ("t2_chain_end", -1), ("t2_walk", 2),
    ("t2_typed", -1), ("t3_update", -1), ("t3_missing", 2),
])
def test_fact_spans_answer_boundary_and_write_mask(tokenizer, task, hops):
    arrays = episode(task, hops=hops)
    result = render_episode(arrays, tokenizer=tokenizer)
    pool = load_name_pool()
    expected_lines = [f"{pool.names[0]} knows {pool.names[1]}.\n",
                      f"{pool.names[1]} knows {pool.names[2]}.\n",
                      f"{pool.names[0]} trusts {pool.names[3]}.\n"]
    assert result.context_text == "".join(expected_lines)
    assert [span.fact_index for span in result.fact_spans] == [0, 2, 3]
    assert sum(result.fact_final_mask) == 3
    for span, line in zip(result.fact_spans, expected_lines, strict=True):
        assert result.context_text[span.char_start:span.char_end] == line
        assert tokenizer.decode(result.context_ids[span.token_start:span.token_end]) == line
        assert result.fact_final_mask[span.token_end - 1]
    assert tuple(tokenizer.encode(result.context_text, add_special_tokens=False)) == result.context_ids
    for query in result.queries:
        assert query.answer_text == " " + pool.names[2]
        assert tokenizer.encode(query.answer_text, add_special_tokens=False) == list(query.answer_ids)
        assert tuple(tokenizer.encode(query.question_text + query.answer_text, add_special_tokens=False)) == query.question_ids + query.answer_ids
        assert answer_em(query.answer_token_id, query)
        assert not answer_em(pool.token_ids[0], query)


def test_query_semantics_and_oracle_hop_is_not_leaked(tokenizer):
    chain = episode("t2_chain_end", hops=-1)
    rendered = render_episode(chain, tokenizer=tokenizer)
    assert "until there is no further fact" in rendered.queries[0].question_text
    assert "exactly" not in rendered.queries[0].question_text
    chain.q_hop_true[:] = 917
    assert render_episode(chain, tokenizer=tokenizer) == rendered
    walk = render_episode(episode("t2_walk", hops=2), tokenizer=tokenizer)
    assert "exactly 2 steps" in walk.queries[0].question_text
    typed = render_episode(episode("t2_typed", hops=-1), tokenizer=tokenizer)
    assert "'knows'" in typed.queries[0].question_text
    assert "trusts" in typed.context_text


def test_update_keeps_fact_order_and_latest_fact_instruction(tokenizer):
    arrays = episode("t3_update", hops=1)
    arrays.facts[0, 2] = [0, 0, 2]
    result = render_episode(arrays, tokenizer=tokenizer)
    pool = load_name_pool()
    assert result.context_text.index(f"knows {pool.names[1]}") < result.context_text.index(f"knows {pool.names[2]}")
    assert "Use the latest fact" in result.queries[0].question_text
    assert result.queries[0].answer_token_id == pool.token_ids[2]


def test_missing_label_is_one_reserved_token(tokenizer):
    result = render_episode(episode("t3_missing", hops=2, answer=8), tokenizer=tokenizer)
    query = result.queries[0]
    pool = load_name_pool()
    assert query.answer_text == " Unknown"
    assert query.answer_ids == (pool.insufficient_token_id,)
    assert "If a required fact is missing, answer Unknown." in query.question_text
    assert tokenizer.decode(query.answer_ids) == query.answer_text
    assert query.answer_token_id not in pool.token_ids


@pytest.mark.parametrize("task,base,hops,fragment", [
    ("t3_update", "t1", -1, "Which person is linked"),
    ("t3_update", "t2_chain_end", -1, "until there is no further fact"),
    ("t3_missing", "t1", -1, "Which person is linked"),
    ("t3_missing", "t2_walk", 2, "exactly 2 steps"),
])
def test_t3_questions_use_base_semantics(tokenizer, task, base, hops, fragment):
    result = render_episode(episode(task, base=base, hops=hops), tokenizer=tokenizer)
    assert fragment in result.queries[0].question_text


def test_empty_context_and_batch_rendering(tokenizer):
    arrays = episode("t3_missing", hops=1, answer=8)
    arrays.fact_valid[:] = False
    result = render_episode(arrays, tokenizer=tokenizer)
    assert result.context_text == ""
    assert result.context_ids == result.fact_spans == result.fact_final_mask == ()
    assert render_episodes(arrays, tokenizer=tokenizer) == (result,)


def test_explicit_fixture_metadata_and_invalid_ids(tokenizer):
    arrays = episode()
    arrays.meta = {}
    result = render_episode(arrays, tokenizer=tokenizer, task="t1", entity_count=8)
    assert len(result.queries) == 2
    with pytest.raises(ValueError, match="Unsupported task"):
        render_episode(arrays, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="does not fit"):
        render_episode(arrays, tokenizer=tokenizer, task="t1", entity_count=1025)
    arrays.facts[0, 0, 0] = 8
    with pytest.raises(ValueError, match="outside"):
        render_episode(arrays, tokenizer=tokenizer, task="t1", entity_count=8)
    assert relation_phrase(17) == "has relation 17 to"
    with pytest.raises(ValueError, match="nonnegative"):
        relation_phrase(-1)


@pytest.mark.parametrize("task,base", [
    ("t1", None), ("t2_chain_end", None), ("t2_walk", None), ("t2_typed", None),
    ("t3_update", "t1"), ("t3_update", "t2_chain_end"),
    ("t3_missing", "t1"), ("t3_missing", "t2_walk"),
])
@pytest.mark.parametrize("split", ["train", "test_unseen"])
def test_generated_episode_integration(tokenizer, task, base, split):
    from looplora.data import relations
    kwargs = {"E": 8, "Q": 3}
    if base is not None:
        kwargs["base"] = base
    if task == "t2_typed":
        kwargs.update(E=10, n_distract=2)
    if task == "t3_update":
        kwargs["E"] = 10
    if task in {"t2_chain_end", "t2_walk", "t2_typed"} or base in {"t2_chain_end", "t2_walk"}:
        kwargs["chain_lengths"] = (4, 4)
    if task == "t3_missing":
        kwargs["missing_rate"] = 1.0
    spec = relations.TaskSpec(task, **kwargs)
    arrays = relations.generate(spec, split, 0, 2, 19)
    pool = load_name_pool()
    result = render_episodes(arrays, tokenizer=tokenizer)
    for index, rendered in enumerate(result):
        expected = "".join(f"{pool.names[int(head)]} {relation_phrase(int(rel))} {pool.names[int(tail)]}.\n"
                           for head, rel, tail in arrays.facts[index])
        assert rendered.context_text == expected
        assert len(rendered.fact_spans) == sum(rendered.fact_final_mask) == spec.E
        for query, answer_id in zip(rendered.queries, arrays.answer[index], strict=True):
            expected_id = pool.insufficient_token_id if answer_id == spec.V else pool.token_ids[int(answer_id)]
            assert query.answer_token_id == expected_id
        if task == "t3_missing":
            assert rendered.queries[0].answer_token_id == pool.insufficient_token_id
