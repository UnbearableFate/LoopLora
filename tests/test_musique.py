"""MuSiQue contracts; execute on compute with the pinned offline tokenizer."""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from pathlib import Path
import re

import pytest

from looplora.data.musique import (
    MuSiQueExample, Paragraph, answer_scores, answer_substitution, build_donor_pool,
    load_split, make_splits, normalize_answer, read_examples, remove_final_support,
    render_example, write_episodes,
)


HOP_TYPES = ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3")


@pytest.fixture(scope="module")
def tokenizer():
    # Read literal pins without importing inject.py (which also imports torch).
    from transformers import AutoTokenizer

    source = Path(__file__).parents[1] / "src/looplora/llm/inject.py"
    pins = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "MODEL_ID", "MODEL_REVISION", "CACHE_DIR",
                }:
                    pins[target.id] = ast.literal_eval(node.value)
    result = AutoTokenizer.from_pretrained(
        pins["MODEL_ID"], revision=pins["MODEL_REVISION"],
        cache_dir=pins["CACHE_DIR"], local_files_only=True,
    )
    assert result.is_fast
    return result


def raw_example(hop="2hop", serial=1):
    hops = int(hop[0])
    decomposition_ids = [serial * 10 + index for index in range(hops)]
    supports = tuple(reversed(range(hops)))
    return {
        "id": hop + "__" + "_".join(map(str, decomposition_ids)),
        "paragraphs": [
            {"idx": index, "title": f"Title {index}",
             "paragraph_text": "York is the answer." if index == 0 else f"Fact {index}.",
             "is_supporting": index < hops}
            for index in range(hops + 2)
        ],
        "question": "Which place completes the chain?",
        "question_decomposition": [
            {"id": qid, "question": f"Step {index + 1}?", "answer": "York",
             "paragraph_support_idx": support}
            for index, (qid, support) in enumerate(zip(decomposition_ids, supports, strict=True))
        ],
        "answer": "York", "answer_aliases": ["New York", "NYC"],
        "answerable": True,
    }


def example(hop="2hop", serial=1, *, answer="York", aliases=("New York", "NYC")):
    raw = raw_example(hop, serial)
    return MuSiQueExample(
        id=raw["id"], hop_type=hop, question=raw["question"], answer=answer,
        answer_aliases=aliases,
        paragraphs=tuple(Paragraph(p["idx"], p["title"], p["paragraph_text"],
                                   p["is_supporting"]) for p in raw["paragraphs"]),
        decomposition_order=tuple(p["paragraph_support_idx"] for p in raw["question_decomposition"]),
        decomposition_ids=tuple(str(p["id"]) for p in raw["question_decomposition"]),
    )


def write_raw(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def rank(identifier):
    return hashlib.blake2b(identifier.encode("utf-8"), digest_size=16).digest()


def check_encoding(episode, tokenizer):
    assert episode.context_ids == tuple(tokenizer.encode(episode.context_text, add_special_tokens=False))
    assert episode.question_ids == tuple(tokenizer.encode(episode.question_text, add_special_tokens=False))
    assert episode.answer_ids == tuple(tokenizer.encode(episode.answer_text, add_special_tokens=False))
    assert tuple(tokenizer.encode(episode.question_text + episode.answer_text,
                                  add_special_tokens=False)) == episode.question_ids + episode.answer_ids
    assert tokenizer.decode(episode.answer_ids, clean_up_tokenization_spaces=False) == episode.answer_text
    assert len(episode.paragraph_spans) == len(episode.paragraphs)
    assert len(episode.paragraph_final_mask) == len(episode.context_ids)
    assert sum(episode.paragraph_final_mask) == len(episode.paragraphs)
    encoded = tokenizer(episode.context_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    char_cursor = token_cursor = 0
    for paragraph, span in zip(episode.paragraphs, episode.paragraph_spans, strict=True):
        block = f"{paragraph.title}\n{paragraph.text}\n\n"
        assert span.index == paragraph.index
        assert span.char_start == char_cursor
        assert span.token_start == token_cursor
        assert episode.context_text[span.char_start:span.char_end] == block
        assert tokenizer.decode(episode.context_ids[span.token_start:span.token_end],
                                clean_up_tokenization_spaces=False) == block
        assert offsets[span.token_start][0] == span.char_start
        assert offsets[span.token_end - 1][1] == span.char_end
        assert episode.paragraph_final_mask[span.token_end - 1]
        char_cursor, token_cursor = span.char_end, span.token_end
    assert char_cursor == len(episode.context_text)
    assert token_cursor == len(episode.context_ids)


def test_read_raw_schema_and_frozen_records(tmp_path):
    raw = raw_example()
    loaded, = read_examples(write_raw(tmp_path / "raw.jsonl", [raw]), expected_count=1)
    assert loaded == example()
    assert loaded.final_support_idx == 0
    assert loaded.decomposition_order == (1, 0)
    assert loaded.answer_aliases == ("New York", "NYC")
    with pytest.raises(FrozenInstanceError):
        loaded.answer = "Paris"
    with pytest.raises(FrozenInstanceError):
        loaded.paragraphs[0].title = "Changed"


@pytest.mark.parametrize("defect", [
    "missing_question", "unanswerable", "bad_support", "wrong_hop_count",
    "duplicate_paragraph", "nonboolean_support", "bad_aliases",
])
def test_read_rejects_invalid_schema(tmp_path, defect):
    raw = deepcopy(raw_example())
    if defect == "missing_question":
        del raw["question"]
    elif defect == "unanswerable":
        raw["answerable"] = False
    elif defect == "bad_support":
        raw["question_decomposition"][-1]["paragraph_support_idx"] = 999
    elif defect == "wrong_hop_count":
        raw["question_decomposition"].pop()
    elif defect == "duplicate_paragraph":
        raw["paragraphs"][1]["idx"] = raw["paragraphs"][0]["idx"]
    elif defect == "nonboolean_support":
        raw["paragraphs"][0]["is_supporting"] = "true"
    elif defect == "bad_aliases":
        raw["answer_aliases"] = "New York"
    with pytest.raises(ValueError):
        read_examples(write_raw(tmp_path / "invalid.jsonl", [raw]))


def test_read_rejects_duplicate_ids_and_wrong_count(tmp_path):
    row = raw_example()
    with pytest.raises(ValueError):
        read_examples(write_raw(tmp_path / "duplicate.jsonl", [row, row]))
    with pytest.raises(ValueError):
        read_examples(write_raw(tmp_path / "count.jsonl", [row]), expected_count=2)


def test_splits_are_disjoint_hash_ranked_stratified_and_order_independent():
    counts = (17, 11, 9, 7, 5, 3)
    train = tuple(example(hop, serial=1000 * (index + 1) + serial)
                  for index, (hop, count) in enumerate(zip(HOP_TYPES, counts, strict=True))
                  for serial in range(count))
    dev = tuple(example(hop, serial=100_000 + 1000 * index + serial)
                for index, hop in enumerate(HOP_TYPES) for serial in range(10))
    splits = make_splits(train, dev, val_count=13)
    assert set(splits) == {"train", "val", "test", "confirm"}
    ids = {name: {row.id for row in rows} for name, rows in splits.items()}
    names = tuple(splits)
    for index, name in enumerate(names):
        assert len(ids[name]) == len(splits[name])
        for other in names[index + 1:]:
            assert ids[name].isdisjoint(ids[other])
    assert ids["train"] | ids["val"] == {row.id for row in train}
    assert ids["test"] | ids["confirm"] == {row.id for row in dev}
    # 13 * (17,11,9,7,5,3)/52; three .75 remainders receive the extra slots.
    quotas = dict(zip(HOP_TYPES, (4, 3, 2, 2, 1, 1), strict=True))
    assert Counter(row.hop_type for row in splits["val"]) == quotas
    for hop, quota in quotas.items():
        ranked = sorted((row for row in train if row.hop_type == hop), key=lambda row: (rank(row.id), row.id))
        assert ids["val"] & {row.id for row in ranked} == {row.id for row in ranked[:quota]}
    assert ids["test"] == {row.id for row in dev if int.from_bytes(rank(row.id), "big") % 3 == 0}
    assert ids["confirm"] == {row.id for row in dev if int.from_bytes(rank(row.id), "big") % 3 != 0}
    assert make_splits(tuple(reversed(train)), tuple(reversed(dev)), val_count=13) == splits


@pytest.mark.parametrize("val_count", [-1, 3])
def test_split_rejects_impossible_validation_size(val_count):
    with pytest.raises(ValueError):
        make_splits((example(serial=1), example(serial=2)), (), val_count=val_count)


def test_split_rejects_duplicate_or_overlapping_official_ids():
    source = example()
    with pytest.raises(ValueError):
        make_splits((source, source), (), val_count=0)
    with pytest.raises(ValueError):
        make_splits((source,), (source,), val_count=0)


@pytest.mark.parametrize("answer", ["York", "São Tomé", "1998", "3.14", "Jean-Luc Picard"])
def test_real_tokenizer_paragraph_spans_and_answer_boundary(tokenizer, answer):
    source = example(answer=answer)
    source = replace(source, paragraphs=(
        Paragraph(0, "Café — 東京", f"Answer: {answer}.\nA second line; naïve words.", True),
        *source.paragraphs[1:],
    ))
    rendered = render_example(source, tokenizer=tokenizer, base_seed=917)
    assert rendered.question_text == f"Question: {source.question}\nAnswer:"
    assert rendered.answer_text == " " + answer
    assert rendered.answer_aliases == source.answer_aliases
    assert rendered.final_support_idx == source.final_support_idx
    assert rendered.condition == "original"
    assert not rendered.sensitivity_only
    assert {p.index for p in rendered.paragraphs} == set(source.decomposition_order)
    assert render_example(source, tokenizer=tokenizer, base_seed=917) == rendered
    check_encoding(rendered, tokenizer)


@pytest.mark.parametrize("hop", ["2hop", "3hop1", "4hop1"])
def test_shuffle_matches_decomposition_only_at_chance(tokenizer, hop):
    trials = 384
    matches = 0
    orders = set()
    for serial in range(trials):
        source = example(hop, serial=20_000 + serial)
        rendered = render_example(source, tokenizer=tokenizer, base_seed=73)
        order = tuple(p.index for p in rendered.paragraphs)
        orders.add(order)
        assert set(order) == set(source.decomposition_order)
        matches += order == source.decomposition_order
    probability = 1 / math.factorial(int(hop[0]))
    # A fixed sample, with a four-standard-deviation binomial upper allowance.
    upper = math.ceil(trials * probability + 4 * math.sqrt(trials * probability * (1 - probability)))
    assert 0 < matches <= upper
    assert len(orders) > 1


def test_distractors_come_only_from_the_examples_own_paragraphs(tokenizer):
    source = example()
    plain = render_example(source, tokenizer=tokenizer, base_seed=19)
    one = render_example(source, tokenizer=tokenizer, base_seed=19, n_distractors=1)
    all_ = render_example(source, tokenizer=tokenizer, base_seed=19, n_distractors=2)
    assert len(plain.paragraphs) == 2
    assert len(one.paragraphs) == 3
    assert sum(not p.is_supporting for p in one.paragraphs) == 1
    assert set(one.paragraphs).issubset(set(source.paragraphs))
    assert set(all_.paragraphs) == set(source.paragraphs)
    assert render_example(source, tokenizer=tokenizer, base_seed=19, n_distractors=1) == one
    assert render_example(replace(source, paragraphs=tuple(reversed(source.paragraphs))),
                          tokenizer=tokenizer, base_seed=19, n_distractors=1) == one
    check_encoding(one, tokenizer)


@pytest.mark.parametrize("count", [-1, 3])
def test_distractors_reject_invalid_counts(tokenizer, count):
    with pytest.raises(ValueError):
        render_example(example(), tokenizer=tokenizer, n_distractors=count)


@pytest.mark.parametrize("text,expected", [
    ("  The, QUICK! brown fox.  ", "quick brown fox"),
    ("An apple and a pear", "apple and pear"),
    ("U.S.A.", "usa"), ("Theatre", "theatre"),
    ("a-b", "ab"), ("l’été", "l’été"), ("\t A\nTHE an ", ""),
])
def test_official_ascii_answer_normalization(text, expected):
    assert normalize_answer(text) == expected


@pytest.mark.parametrize("prediction,answer,aliases,em,f1", [
    ("The U.S.A.!", "USA", (), 1, 1),
    ("NYC", "New York City", ("NYC", "New York"), 1, 1),
    ("new york", "York", ("New York City",), 0, 0.8),
    ("red red blue", "red blue blue", (), 0, 2 / 3),
    ("London", "York", (), 0, 0),
    ("", "", (), 1, 1), ("the", "", (), 1, 1),
    ("", "York", (), 0, 0), ("York", "", (), 0, 0),
    ("", "York", ("the",), 1, 1),
])
def test_official_metric_handcases(prediction, answer, aliases, em, f1):
    scores = answer_scores(prediction, answer, aliases)
    assert scores["em"] == em
    assert scores["f1"] == pytest.approx(f1)


def substitution_source():
    source = example()
    return replace(source, paragraphs=(
        Paragraph(0, "NEW YORK / York", "York is tied to NEW YORK and nyc; Yorkish is a word.", True),
        Paragraph(1, "Earlier support", "York met NYC.", True),
        Paragraph(2, "Safe distractor", "An unrelated passage.", False),
        Paragraph(3, "Other distractor", "No answer here.", False),
    ))


def test_substitution_replaces_gold_aliases_titles_and_case_preserving_boundaries(tokenizer):
    source = substitution_source()
    original = render_example(source, tokenizer=tokenizer, base_seed=8)
    donor = example(serial=901, answer="Rome", aliases=("Roma",))
    changed = answer_substitution(original, [donor], tokenizer=tokenizer, base_seed=8)
    assert changed is not None
    assert changed.condition == "answer_substitution"
    assert not changed.sensitivity_only
    assert changed.answer_text == " Rome"
    assert changed.answer_aliases == ("Roma",)
    assert changed.question_text == original.question_text
    assert changed.question_ids == original.question_ids
    assert [p.index for p in changed.paragraphs] == [p.index for p in original.paragraphs]
    paragraphs = {p.index: p for p in changed.paragraphs}
    assert paragraphs[0].title == "Rome / Rome"
    assert paragraphs[0].text == "Rome is tied to Rome and Rome; Yorkish is a word."
    assert paragraphs[1].text == "Rome met Rome."
    for variant in (source.answer, *source.answer_aliases):
        assert re.search(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", changed.context_text, re.I) is None
    assert "Yorkish" in changed.context_text
    assert original.context_text == render_example(source, tokenizer=tokenizer, base_seed=8).context_text
    check_encoding(changed, tokenizer)


def test_substitution_final_title_is_part_of_support_and_overlapping_aliases_are_filtered(tokenizer):
    source = example()
    source = replace(source, paragraphs=(Paragraph(0, "York", "The final title gives the answer.", True),
                                         *source.paragraphs[1:]))
    rendered = render_example(source, tokenizer=tokenizer)
    donor = example(serial=903, answer="Rome", aliases=("Roma", "NYC"))
    changed = answer_substitution(rendered, [donor], tokenizer=tokenizer)
    assert changed is not None
    assert changed.answer_text == " Rome"
    assert changed.answer_aliases == ("Roma",)
    assert next(p for p in changed.paragraphs if p.index == 0).title == "Rome"
    check_encoding(changed, tokenizer)


def test_substitution_rejects_metric_equivalent_donor_gold_and_filters_equivalent_aliases(tokenizer):
    source = example(answer="United States", aliases=("U.S.A.",))
    source = replace(source, paragraphs=(
        Paragraph(0, "Final support", "United States is the answer.", True),
        *source.paragraphs[1:],
    ))
    rendered = render_example(source, tokenizer=tokenizer)
    equivalent = example(serial=904, answer="USA", aliases=())
    assert answer_scores(equivalent.answer, source.answer, source.answer_aliases)["em"] == 1
    assert answer_substitution(rendered, [equivalent], tokenizer=tokenizer) is None

    donor = example(serial=905, answer="Canada", aliases=("Dominion of Canada", "USA", "the USA"))
    changed = answer_substitution(rendered, [donor], tokenizer=tokenizer)
    assert changed is not None
    assert changed.answer_text == " Canada"
    assert changed.answer_aliases == ("Dominion of Canada",)
    assert answer_scores(changed.answer_text, source.answer, source.answer_aliases)["em"] == 0
    for alias in changed.answer_aliases:
        assert answer_scores(alias, source.answer, source.answer_aliases)["em"] == 0
    check_encoding(changed, tokenizer)


@pytest.mark.parametrize("final_text", ["york is lower-case.", "Yorkshire is a county.", "NYC is only an alias.", "No answer."])
def test_substitution_requires_exact_gold_word_in_final_support(tokenizer, final_text):
    source = example()
    source = replace(source, paragraphs=(Paragraph(0, "Last support", final_text, True),
                                         Paragraph(1, "Earlier", "York is here.", True),
                                         *source.paragraphs[2:]))
    rendered = render_example(source, tokenizer=tokenizer)
    assert answer_substitution(rendered, [example(serial=902, answer="Rome")], tokenizer=tokenizer) is None


@pytest.mark.parametrize("answer,compatible,incompatible", [
    ("1998", "2005", ("12", "Rome")),
    ("12", "37", ("2005", "Rome")),
    ("York", "Rome", ("2005", "12")),
])
def test_substitution_uses_same_coarse_answer_kind(tokenizer, answer, compatible, incompatible):
    source = example(answer=answer, aliases=())
    source = replace(source, paragraphs=(Paragraph(0, "Last", f"The value is {answer}.", True),
                                         *source.paragraphs[1:]))
    rendered = render_example(source, tokenizer=tokenizer)
    wrong = [example(serial=910 + index, answer=value, aliases=()) for index, value in enumerate(incompatible)]
    assert answer_substitution(rendered, wrong, tokenizer=tokenizer) is None
    donor = example(serial=920, answer=compatible, aliases=())
    changed = answer_substitution(rendered, [*wrong, donor], tokenizer=tokenizer)
    assert changed is not None
    assert changed.answer_text == " " + compatible
    assert compatible in changed.context_text
    check_encoding(changed, tokenizer)


def test_substitution_donor_order_independent_and_rejects_original_variants(tokenizer):
    source = substitution_source()
    rendered = render_example(source, tokenizer=tokenizer)
    rejected = [source, replace(source, answer="Rome"), example(serial=930, answer="York"),
                example(serial=931, answer="New York City")]
    assert answer_substitution(rendered, rejected, tokenizer=tokenizer) is None
    donors = [example(serial=940 + index, answer=value, aliases=())
              for index, value in enumerate(("Rome", "Paris", "Oslo"))]
    observed = set()
    for seed in (0, 1, 3, 7, 11, 19, 37, 101):
        first = answer_substitution(rendered, [*rejected, *donors], tokenizer=tokenizer, base_seed=seed)
        second = answer_substitution(rendered, list(reversed([*rejected, *donors])),
                                     tokenizer=tokenizer, base_seed=seed)
        assert first is not None and first == second
        assert answer_substitution(rendered, build_donor_pool([*rejected, *donors]),
                                   tokenizer=tokenizer, base_seed=seed) == first
        observed.add(first.answer_text)
    assert len(observed) > 1


def test_substitution_rejects_selected_distractor_with_old_alias(tokenizer):
    source = substitution_source()
    source = replace(source, paragraphs=(*source.paragraphs[:2],
                                         Paragraph(2, "NYC", "Distractor leaks the old answer.", False),
                                         source.paragraphs[3]))
    donor = example(serial=950, answer="Rome")
    plain = render_example(source, tokenizer=tokenizer)
    assert answer_substitution(plain, [donor], tokenizer=tokenizer) is not None
    distracted = render_example(source, tokenizer=tokenizer, n_distractors=2)
    assert answer_substitution(distracted, [donor], tokenizer=tokenizer) is None


def test_remove_final_support_is_sensitivity_only_and_keeps_target_order(tokenizer):
    rendered = render_example(substitution_source(), tokenizer=tokenizer, base_seed=17, n_distractors=2)
    changed = remove_final_support(rendered, tokenizer=tokenizer)
    assert changed.condition == "remove_final_support"
    assert changed.sensitivity_only
    assert changed.answer_text == rendered.answer_text
    assert changed.answer_ids == rendered.answer_ids
    assert changed.answer_aliases == rendered.answer_aliases
    assert changed.question_text == rendered.question_text
    assert changed.question_ids == rendered.question_ids
    assert changed.paragraphs == tuple(p for p in rendered.paragraphs if p.index != rendered.final_support_idx)
    assert len(rendered.paragraphs) == len(changed.paragraphs) + 1
    check_encoding(changed, tokenizer)


@pytest.mark.parametrize("condition", ["original", "answer_substitution", "remove_final_support"])
def test_jsonl_roundtrip_restores_tuple_dataclasses_and_confirmation_guard(tmp_path, tokenizer, condition):
    original = render_example(substitution_source(), tokenizer=tokenizer)
    if condition == "answer_substitution":
        episode = answer_substitution(original, [example(serial=960, answer="Rome")], tokenizer=tokenizer)
        assert episode is not None
    elif condition == "remove_final_support":
        episode = remove_final_support(original, tokenizer=tokenizer)
    else:
        episode = original
    suffix = "" if condition == "original" else "." + condition
    write_episodes(tmp_path / f"train{suffix}.jsonl", [episode])
    loaded = load_split("train", tmp_path, condition=condition)
    assert loaded == (episode,)
    assert isinstance(loaded, tuple)
    assert isinstance(loaded[0].paragraphs, tuple)
    assert isinstance(loaded[0].paragraphs[0], Paragraph)
    assert isinstance(loaded[0].paragraph_spans, tuple)
    check_encoding(loaded[0], tokenizer)
    # Guard must win even when no confirmation file exists (no accidental read).
    with pytest.raises((ValueError, PermissionError), match="[Cc]onfirm|allow_confirmation"):
        load_split("confirm", tmp_path, condition=condition)
    write_episodes(tmp_path / f"confirm{suffix}.jsonl", [episode])
    with pytest.raises((ValueError, PermissionError), match="[Cc]onfirm|allow_confirmation"):
        load_split("confirm", tmp_path, condition=condition)
    assert load_split("confirm", tmp_path, condition=condition, allow_confirmation=True) == (episode,)
