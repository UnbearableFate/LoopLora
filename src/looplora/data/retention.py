"""Offline, hash-ranked multiple-choice retention subsets from cached HF Parquet.

Choices are continuations: score ``item.context + choice`` while masking context.
No dataset or model download is attempted. PyArrow is imported only when preparing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MCItem:
    id: str
    context: str
    choices: tuple[str, ...]
    label: int
    source_dataset: str
    source_config: str
    source_split: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("MCItem requires a nonempty string id")
        if not isinstance(self.context, str):
            raise ValueError("MCItem context must be text")
        if not isinstance(self.choices, tuple) or len(self.choices) < 2:
            raise ValueError("MCItem requires at least two choices in a tuple")
        if any(not isinstance(choice, str) or not choice.strip() for choice in self.choices):
            raise ValueError("MCItem choices must be nonempty text")
        if type(self.label) is not int or not 0 <= self.label < len(self.choices):
            raise ValueError("MCItem label must be a valid zero-based choice index")
        if any(not isinstance(value, str) or not value for value in (
            self.source_dataset, self.source_config, self.source_split
        )):
            raise ValueError("MCItem source fields must be nonempty strings")


_SOURCES = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
    "piqa": ("baber/piqa", "default"),
    "hellaswag": ("Rowan/hellaswag", "default"),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "mmlu": ("cais/mmlu", "all"),
}
_SPLITS = ("validation", "dev", "test")
_MMLU_SUBJECTS = frozenset("""
abstract_algebra anatomy astronomy business_ethics clinical_knowledge college_biology
college_chemistry college_computer_science college_mathematics college_medicine
college_physics computer_security conceptual_physics econometrics electrical_engineering
elementary_mathematics formal_logic global_facts high_school_biology high_school_chemistry
high_school_computer_science high_school_european_history high_school_geography
high_school_government_and_politics high_school_macroeconomics high_school_mathematics
high_school_microeconomics high_school_physics high_school_psychology high_school_statistics
high_school_us_history high_school_world_history human_aging human_sexuality international_law
jurisprudence logical_fallacies machine_learning management marketing medical_genetics
miscellaneous moral_disputes moral_scenarios nutrition philosophy prehistory
professional_accounting professional_law professional_medicine professional_psychology
public_relations security_studies sociology us_foreign_policy virology world_religions
""".split())


def _digest(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(hf_home: Path, dataset: str) -> tuple[Path, str]:
    cache = hf_home / "hub" / ("datasets--" + dataset.replace("/", "--"))
    main_ref = cache / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Invalid cached revision in {main_ref}")
        snapshot = cache / "snapshots" / revision
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Cached revision {revision} is missing for {dataset}")
        return snapshot, revision
    snapshots = sorted(
        path for path in (cache / "snapshots").glob("*")
        if path.is_dir() and re.fullmatch(r"[0-9a-f]{40}", path.name)
    )
    if len(snapshots) != 1:
        raise ValueError(f"Need one unambiguous cached revision for {dataset}; found {len(snapshots)}")
    return snapshots[0], snapshots[0].name


def _split_files(directory: Path, split: str, prefix: str = "") -> tuple[Path, ...]:
    files = sorted(directory.glob(f"{prefix}{split}*.parquet"))
    # Do not accept unrelated names such as validation_extra.parquet.
    pattern = re.compile(rf"{re.escape(prefix + split)}(?:-\d+-of-\d+)?\.parquet")
    files = [path for path in files if path.is_file() and pattern.fullmatch(path.name)]
    if not files:
        return ()
    shard_pattern = re.compile(r"-(\d+)-of-(\d+)\.parquet$")
    shards = [shard_pattern.search(path.name) for path in files]
    if any(shards):
        if not all(shards):
            raise ValueError(f"Mixed sharded and unsharded files in {directory} for {split}")
        parts = [(int(match[1]), int(match[2])) for match in shards if match]
        totals = {total for _, total in parts}
        if len(totals) != 1 or {index for index, _ in parts} != set(range(parts[0][1])):
            raise ValueError(f"Incomplete cached Parquet shards in {directory} for {split}")
    return tuple(files)


def _sources_for_split(snapshot: Path, name: str, split: str) -> tuple[tuple[str, Path], ...]:
    if name == "mmlu":
        aggregate = _split_files(snapshot / "all", split)
        if aggregate:
            return tuple(("all", path) for path in aggregate)
        present = {subject: _split_files(snapshot / subject, split) for subject in _MMLU_SUBJECTS}
        if not any(present.values()):
            return ()
        missing = sorted(subject for subject, files in present.items() if not files)
        if missing:
            raise ValueError(f"Incomplete MMLU {split} cache; missing {len(missing)} subjects: {', '.join(missing)}")
        return tuple((subject, path) for subject in sorted(present) for path in present[subject])
    _, config = _SOURCES[name]
    directory = snapshot
    prefix = ""
    if name.startswith("arc_") or name == "winogrande":
        directory = snapshot / config
    elif name == "hellaswag":
        directory = snapshot / "data"
    elif name == "piqa":
        prefix = "piqa_"
    return tuple((config, path) for path in _split_files(directory, split, prefix))


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty text in field {key}")
    return value


def _label(value: Any, size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Invalid multiple-choice label: {value!r}")
    if isinstance(value, str) and not re.fullmatch(r"\d+", value):
        raise ValueError(f"Missing or invalid multiple-choice label: {value!r}")
    label = int(value)
    if not 0 <= label < size:
        raise ValueError(f"Multiple-choice label {label} outside [0, {size})")
    return label


def _continuations(values: Iterable[Any]) -> tuple[str, ...]:
    values = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Choices must be nonempty text")
    return tuple(" " + value.strip() for value in values)


def _convert(name: str, row: Mapping[str, Any], config: str, split: str) -> MCItem:
    source_dataset = _SOURCES[name][0]
    explicit_id = row.get("id")
    if name.startswith("arc_"):
        context = f"Question: {_text(row, 'question')}\nAnswer:"
        raw_choices = row["choices"]
        if not isinstance(raw_choices, Mapping):
            raise ValueError("ARC choices must contain text and label arrays")
        choice_labels = tuple(raw_choices["label"])
        choices = _continuations(raw_choices["text"])
        if len(choice_labels) != len(choices) or len(set(choice_labels)) != len(choice_labels):
            raise ValueError("ARC choice labels must be unique and align with choices")
        try:
            label = choice_labels.index(str(row["answerKey"]))
        except ValueError as error:
            raise ValueError("ARC answerKey is absent from choice labels") from error
    elif name == "piqa":
        context = f"Question: {_text(row, 'goal')}\nAnswer:"
        choices = _continuations((_text(row, "sol1"), _text(row, "sol2")))
        label = _label(row["label"], len(choices))
    elif name == "hellaswag":
        context = _text(row, "ctx").rstrip()
        choices = _continuations(row["endings"])
        label = _label(row["label"], len(choices))
        explicit_id = row.get("ind", explicit_id)
    elif name == "winogrande":
        sentence = _text(row, "sentence")
        if sentence.count("_") != 1:
            raise ValueError("WinoGrande sentence requires exactly one blank")
        context, suffix = sentence.split("_")
        choices = tuple(_text(row, field) + suffix for field in ("option1", "option2"))
        answer = _text(row, "answer")
        if answer not in ("1", "2"):
            raise ValueError(f"Invalid WinoGrande answer: {answer!r}")
        label = int(answer) - 1
        explicit_id = row.get("qID", explicit_id)
    elif name == "mmlu":
        subject = _text(row, "subject") if config == "all" else config
        if subject not in _MMLU_SUBJECTS:
            raise ValueError(f"Unknown MMLU subject {subject!r}")
        if "subject" in row and row["subject"] != subject:
            raise ValueError("MMLU row subject disagrees with source configuration")
        config = subject
        context = f"Question: {_text(row, 'question')}\nAnswer:"
        choices = _continuations(row["choices"])
        label = _label(row["answer"], len(choices))
    else:
        raise ValueError(f"Unknown retention dataset {name!r}")
    if explicit_id is None:
        identity = {"context": context, "choices": choices, "config": config}
        item_id = _digest(_json(identity).encode("utf-8"))
    else:
        item_id = str(explicit_id)
    if name == "mmlu":
        item_id = f"{config}:{item_id}"
    return MCItem(item_id, context, choices, label, source_dataset, config, split)


def _read_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError("Retention preparation requires PyArrow; install an approved compatible wheel") from error
    with parquet.ParquetFile(path) as source:
        for batch in source.iter_batches(batch_size=1024, use_threads=False):
            yield from batch.to_pylist()


def prepare_retention(
    hf_home: str | Path,
    output_dir: str | Path,
    *,
    count: int = 500,
    split_overrides: Mapping[str, str] | None = None,
    allow_smaller: Iterable[str] = (),
) -> dict[str, Any]:
    """Prepare all six suites; reject incomplete sources before writing any files.

    ``split_overrides`` maps output names to validation/dev/test. ``allow_smaller``
    permits a count shortfall only, never missing MMLU subjects or missing shards.
    The default split is the first cached validation, dev, test, in that order.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    overrides = dict(split_overrides or {})
    smaller = set(allow_smaller)
    unknown = (set(overrides) | smaller) - set(_SOURCES)
    if unknown:
        raise ValueError(f"Unknown retention dataset names: {sorted(unknown)}")
    if any(split not in _SPLITS for split in overrides.values()):
        raise ValueError("Split overrides must be validation, dev, or test")
    manifest: dict[str, Any] = {
        "schema_version": 1, "hash_algorithm": "blake2b-128",
        "selection": "ascending blake2b-128 of UTF-8 id, then id",
        "requested_count": count, "split_overrides": overrides,
        "allow_smaller": sorted(smaller), "datasets": {},
    }
    prepared: dict[str, tuple[MCItem, ...]] = {}
    for name, (dataset, _) in _SOURCES.items():
        snapshot, revision = _snapshot(Path(hf_home), dataset)
        candidates = (overrides[name],) if name in overrides else _SPLITS
        files: tuple[tuple[str, Path], ...] = ()
        for split in candidates:
            files = _sources_for_split(snapshot, name, split)
            if files:
                break
        if not files:
            raise FileNotFoundError(f"No cached labeled split {candidates} for {name}")
        items: dict[str, MCItem] = {}
        source_rows = 0
        for config, path in files:
            for row in _read_rows(path):
                item = _convert(name, row, config, split)
                source_rows += 1
                if item.id in items and items[item.id] != item:
                    raise ValueError(f"Conflicting rows for {name} id {item.id}")
                items[item.id] = item
        if name == "mmlu":
            subjects = {item.source_config for item in items.values()}
            if subjects != _MMLU_SUBJECTS:
                raise ValueError(f"MMLU must cover all 57 subjects; found {len(subjects)}")
        if not items:
            raise ValueError(f"Empty retention source for {name}")
        if len(items) < count and name not in smaller:
            raise ValueError(f"{name}/{split} has {len(items)} unique items, fewer than {count}; use an explicit split override or allow_smaller")
        ranked = sorted(items.values(), key=lambda item: (_digest(item.id.encode("utf-8")), item.id))
        selected = tuple(ranked[:count])
        prepared[name] = selected
        manifest["datasets"][name] = {
            "source_dataset": dataset, "revision": revision, "split": split,
            "configs": sorted({item.source_config for item in items.values()}),
            "source_rows": source_rows, "unique_items": len(items),
            "duplicate_rows": source_rows - len(items), "count": len(selected),
            "selected_ids": [item.id for item in selected],
            "source_files": [
                {"path": path.relative_to(snapshot).as_posix(), "blake2b_128": _file_hash(path)}
                for _, path in files
            ],
        }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in prepared.items():
        target = output_dir / f"{name}.jsonl"
        target.write_text("".join(_json(asdict(item)) + "\n" for item in selected), encoding="utf-8")
        manifest["datasets"][name]["output_blake2b_128"] = _file_hash(target)
    (output_dir / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


def load_retention(name: str, output_dir: str | Path) -> tuple[MCItem, ...]:
    """Load a prepared suite, checking manifest identity and artifact integrity."""
    if name not in _SOURCES:
        raise ValueError(f"Unknown retention dataset {name!r}")
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("hash_algorithm") != "blake2b-128":
        raise ValueError("Unsupported retention manifest schema")
    entry = manifest["datasets"][name]
    target = output_dir / f"{name}.jsonl"
    if _file_hash(target) != entry["output_blake2b_128"]:
        raise ValueError(f"Retention artifact hash mismatch for {name}")
    items = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["choices"] = tuple(row["choices"])
            item = MCItem(**row)
            if item.source_dataset != _SOURCES[name][0] or item.source_split != entry["split"]:
                raise ValueError(f"Retention provenance mismatch for {name}")
            items.append(item)
    if len(items) != entry["count"] or [item.id for item in items] != entry["selected_ids"]:
        raise ValueError(f"Retention manifest selection mismatch for {name}")
    return tuple(items)
