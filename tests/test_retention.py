import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from looplora.data.retention import (
    MCItem,
    _MMLU_SUBJECTS,
    _convert,
    _SOURCES,
    load_retention,
    prepare_retention,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _arc_rows() -> list[dict]:
    return [
        {"id": f"arc-{index}", "question": f"Which item {index}?",
         "choices": {"text": ["first", "second"], "label": ["3", "7"]},
         "answerKey": "7"}
        for index in range(3)
    ]


@pytest.fixture
def cached_sources(tmp_path):
    hf_home = tmp_path / "hf"
    snapshots = {}
    for dataset, _ in _SOURCES.values():
        cache = hf_home / "hub" / ("datasets--" + dataset.replace("/", "--"))
        snapshot = cache / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True, exist_ok=True)
        (cache / "refs").mkdir(exist_ok=True)
        (cache / "refs" / "main").write_text("a" * 40)
        snapshots[dataset] = snapshot
    arc = snapshots["allenai/ai2_arc"]
    for config in ("ARC-Easy", "ARC-Challenge"):
        _write(arc / config / "validation-00000-of-00001.parquet", _arc_rows())
        _write(arc / config / "test-00000-of-00001.parquet", _arc_rows() + [
            {**_arc_rows()[0], "id": "test-only"}
        ])
    _write(snapshots["baber/piqa"] / "piqa_validation.parquet", [
        {"goal": f"Open box {index}", "sol1": "Use scissors", "sol2": "Use water", "label": 0}
        for index in range(3)
    ])
    _write(snapshots["Rowan/hellaswag"] / "data" / "validation-00000-of-00001.parquet", [
        {"ind": index, "ctx": f"A person {index} prepares", "endings": ["food", "nothing"],
         "label": "1", "source_id": "same-video"}
        for index in range(3)
    ])
    _write(snapshots["allenai/winogrande"] / "winogrande_xl" / "validation-00000-of-00001.parquet", [
        {"sentence": f"In scene {index}, _ carried the box.", "option1": "Alice", "option2": "Bob", "answer": "2"}
        for index in range(3)
    ])
    _write(snapshots["cais/mmlu"] / "all" / "validation-00000-of-00001.parquet", [
        {"question": "Shared question?", "subject": subject, "choices": ["Yes", "No"], "answer": 0}
        for subject in sorted(_MMLU_SUBJECTS)
    ])
    return hf_home, snapshots


def test_all_suites_rank_ids_record_sources_and_roundtrip(cached_sources, tmp_path):
    hf_home, snapshots = cached_sources
    output = tmp_path / "prepared"
    manifest = prepare_retention(hf_home, output, count=2)
    assert set(manifest["datasets"]) == set(_SOURCES)
    assert manifest == json.loads((output / "manifest.json").read_text())
    for name, entry in manifest["datasets"].items():
        items = load_retention(name, output)
        assert len(items) == entry["count"] == 2
        assert entry["split"] == "validation"
        assert entry["revision"] == "a" * 40
        assert entry["selected_ids"] == [item.id for item in items]
        assert all(isinstance(item.choices, tuple) for item in items)
        for source in entry["source_files"]:
            path = snapshots[entry["source_dataset"]] / source["path"]
            assert source["blake2b_128"] == hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
    expected_arc_ids = sorted(
        [row["id"] for row in _arc_rows()],
        key=lambda item_id: (hashlib.blake2b(item_id.encode(), digest_size=16).hexdigest(), item_id),
    )[:2]
    assert manifest["datasets"]["arc_easy"]["selected_ids"] == expected_arc_ids


def test_subset_is_independent_of_parquet_order(cached_sources, tmp_path):
    hf_home, snapshots = cached_sources
    first = prepare_retention(hf_home, tmp_path / "first", count=2)
    for snapshot in set(snapshots.values()):
        for path in snapshot.rglob("*.parquet"):
            _write(path, list(reversed(pq.ParquetFile(path).read().to_pylist())))
    second = prepare_retention(hf_home, tmp_path / "second", count=2)
    for name in _SOURCES:
        assert first["datasets"][name]["selected_ids"] == second["datasets"][name]["selected_ids"]
        assert load_retention(name, tmp_path / "first") == load_retention(name, tmp_path / "second")


def test_label_mapping_and_exact_continuations():
    arc = _convert("arc_easy", _arc_rows()[0], "ARC-Easy", "validation")
    assert arc.label == 1  # Numeric ARC labels are symbols, not zero-based indices.
    assert arc.context + arc.choices[arc.label] == "Question: Which item 0?\nAnswer: second"
    alphabetic = {**_arc_rows()[0], "choices": {"text": ["first", "second"], "label": ["B", "A"]}, "answerKey": "A"}
    assert _convert("arc_easy", alphabetic, "ARC-Easy", "validation").label == 1
    wino = _convert("winogrande", {
        "sentence": "The trophy does not fit because _ is too large.",
        "option1": "the trophy", "option2": "the suitcase", "answer": "1"
    }, "winogrande_xl", "validation")
    assert wino.label == 0
    assert wino.context + wino.choices[0] == "The trophy does not fit because the trophy is too large."
    assert wino.context + wino.choices[1] == "The trophy does not fit because the suitcase is too large."
    for subject in ("anatomy", "astronomy"):
        item = _convert("mmlu", {"subject": subject, "question": "Q?", "choices": ["a", "b"], "answer": 1}, "all", "validation")
        assert item.label == 1
        assert item.id.startswith(subject + ":")


def test_semantic_ids_ignore_label_and_include_mmlu_subject():
    row = {"goal": "Do something", "sol1": "First", "sol2": "Second", "label": 0}
    original = _convert("piqa", row, "default", "validation")
    assert original.id == _convert("piqa", {**row, "label": 1}, "default", "validation").id
    mmlu_row = {"question": "Same?", "choices": ["Yes", "No"], "answer": 0}
    assert _convert("mmlu", mmlu_row, "anatomy", "validation").id != _convert("mmlu", mmlu_row, "astronomy", "validation").id


def test_too_small_never_silently_switches_split(cached_sources, tmp_path):
    hf_home, _ = cached_sources
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="arc_easy/validation.*fewer than 4"):
        prepare_retention(hf_home, output, count=4)
    assert not output.exists()
    manifest = prepare_retention(hf_home, tmp_path / "smaller", count=4, allow_smaller=set(_SOURCES) - {"mmlu"})
    assert manifest["datasets"]["arc_challenge"]["split"] == "validation"
    assert manifest["datasets"]["arc_challenge"]["count"] == 3
    assert "arc_challenge" in manifest["allow_smaller"]
    manifest = prepare_retention(hf_home, tmp_path / "override", count=4,
                                 split_overrides={"arc_easy": "test", "arc_challenge": "test"},
                                 allow_smaller=("piqa", "hellaswag", "winogrande"))
    assert manifest["datasets"]["arc_challenge"]["count"] == 4
    assert manifest["split_overrides"]["arc_challenge"] == "test"


def test_dev_then_test_fallback_when_earlier_splits_absent(cached_sources, tmp_path):
    hf_home, snapshots = cached_sources
    piqa = snapshots["baber/piqa"]
    validation = piqa / "piqa_validation.parquet"
    validation.rename(piqa / "piqa_dev.parquet")
    assert prepare_retention(hf_home, tmp_path / "dev", count=1)["datasets"]["piqa"]["split"] == "dev"
    (piqa / "piqa_dev.parquet").rename(piqa / "piqa_test.parquet")
    assert prepare_retention(hf_home, tmp_path / "test", count=1)["datasets"]["piqa"]["split"] == "test"


def test_mmlu_aggregate_preferred_and_partial_cache_rejected(cached_sources, tmp_path):
    hf_home, snapshots = cached_sources
    mmlu = snapshots["cais/mmlu"]
    # A cached subject must not be double-counted alongside the aggregate.
    row = {"subject": "anatomy", "question": "Q?", "choices": ["a", "b"], "answer": 0}
    _write(mmlu / "anatomy" / "validation-00000-of-00001.parquet", [row])
    assert prepare_retention(hf_home, tmp_path / "full", count=1)["datasets"]["mmlu"]["source_rows"] == 57
    aggregate = mmlu / "all" / "validation-00000-of-00001.parquet"
    aggregate.rename(mmlu / "all" / "unselected.parquet")
    with pytest.raises(ValueError, match="Incomplete MMLU validation cache"):
        prepare_retention(hf_home, tmp_path / "partial", count=1, allow_smaller=("mmlu",))
    _write(aggregate, [row])
    with pytest.raises(ValueError, match="MMLU must cover all 57 subjects"):
        prepare_retention(hf_home, tmp_path / "partial-aggregate", count=1)


def test_all_mmlu_subject_files_and_missing_main_ref(cached_sources, tmp_path):
    hf_home, snapshots = cached_sources
    mmlu = snapshots["cais/mmlu"]
    aggregate = mmlu / "all" / "validation-00000-of-00001.parquet"
    rows = pq.ParquetFile(aggregate).read().to_pylist()
    aggregate.rename(mmlu / "all" / "unselected.parquet")
    for row in rows:
        _write(mmlu / row["subject"] / "validation-00000-of-00001.parquet", [row])
    main_ref = mmlu.parent.parent / "refs" / "main"
    main_ref.rename(main_ref.with_name("other"))
    manifest = prepare_retention(hf_home, tmp_path / "subjects", count=2)
    assert len(manifest["datasets"]["mmlu"]["source_files"]) == 57
    assert manifest["datasets"]["mmlu"]["revision"] == "a" * 40


def test_missing_shard_rejected(cached_sources, tmp_path):
    hf_home, snapshots = cached_sources
    source = snapshots["allenai/ai2_arc"] / "ARC-Easy" / "validation-00000-of-00001.parquet"
    source.rename(source.with_name("validation-00000-of-00002.parquet"))
    with pytest.raises(ValueError, match="Incomplete cached Parquet shards"):
        prepare_retention(hf_home, tmp_path / "rejected", count=1)


def test_conflicting_duplicates_rejected_and_identical_rows_counted(cached_sources, tmp_path):
    hf_home, snapshots = cached_sources
    source = snapshots["allenai/ai2_arc"] / "ARC-Easy" / "validation-00000-of-00001.parquet"
    rows = _arc_rows()
    _write(source, rows + [rows[0]])
    entry = prepare_retention(hf_home, tmp_path / "dedup", count=2)["datasets"]["arc_easy"]
    assert entry["source_rows"] == 4
    assert entry["unique_items"] == 3
    assert entry["duplicate_rows"] == 1
    _write(source, rows + [{**rows[0], "answerKey": "3"}])
    with pytest.raises(ValueError, match="Conflicting rows"):
        prepare_retention(hf_home, tmp_path / "rejected", count=1)


@pytest.mark.parametrize("bad_label", ["", "-1", "A", -1, 2, None, True, 0.5])
def test_unlabeled_or_invalid_examples_fail(bad_label):
    with pytest.raises(ValueError):
        _convert("piqa", {"goal": "Goal", "sol1": "One", "sol2": "Two", "label": bad_label}, "default", "test")


@pytest.mark.parametrize("kwargs", [
    {"count": 0}, {"count": True}, {"allow_smaller": ("typo",)},
    {"split_overrides": {"typo": "test"}}, {"split_overrides": {"piqa": "train"}},
])
def test_invalid_configuration_rejected_before_io(tmp_path, kwargs):
    with pytest.raises(ValueError):
        prepare_retention(tmp_path / "missing", tmp_path / "output", **kwargs)


def test_loader_detects_tampering_and_bad_names(cached_sources, tmp_path):
    hf_home, _ = cached_sources
    output = tmp_path / "prepared"
    prepare_retention(hf_home, output, count=1)
    with pytest.raises(ValueError, match="Unknown retention"):
        load_retention("../piqa", output)
    path = output / "piqa.jsonl"
    path.write_text(path.read_text().replace("Use scissors", "Use fire"))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_retention("piqa", output)


def test_mcitem_rejects_out_of_range_label():
    with pytest.raises(ValueError, match="zero-based"):
        MCItem("id", "Context", (" first", " second"), 2, "dataset", "config", "validation")
