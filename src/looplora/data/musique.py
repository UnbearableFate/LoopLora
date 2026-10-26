"""Offline MuSiQue given-support episodes, answer metrics and controlled edits.

The answer metric follows StonyBrookNLP/musique metrics/answer.py at revision
922ac98f19a201998dbdae6d7f2887a5258dbdeb. No tokenizer or model is loaded here.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import re
import string
from typing import Any, Iterable, Mapping, Sequence

from looplora.infra.seeds import derive_seed

VERSION = "musique-given-support-v1"
DATASET_ID = "dgslibisey/MuSiQue"
DATASET_REVISION = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
SOURCE_COUNTS = {"train": 19_938, "dev": 2_417}
HOP_TYPES = ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3")
SPLITS = ("train", "val", "test", "confirm")
CONDITIONS = ("original", "answer_substitution", "remove_final_support")
DEFAULT_ROOT = Path("/work/xg24i002/x10041/LoopLora/cache/musique")


@dataclass(frozen=True)
class Paragraph:
    index: int
    title: str
    text: str
    is_supporting: bool


@dataclass(frozen=True)
class MuSiQueExample:
    id: str
    hop_type: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    paragraphs: tuple[Paragraph, ...]
    decomposition_order: tuple[int, ...]
    decomposition_ids: tuple[str, ...]

    @property
    def final_support_idx(self) -> int:
        return self.decomposition_order[-1]


@dataclass(frozen=True)
class ParagraphSpan:
    index: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class MuSiQueEpisode:
    id: str
    hop_type: str
    paragraphs: tuple[Paragraph, ...]
    final_support_idx: int
    context_text: str
    context_ids: tuple[int, ...]
    question_text: str
    question_ids: tuple[int, ...]
    answer_text: str
    answer_ids: tuple[int, ...]
    answer_aliases: tuple[str, ...]
    paragraph_spans: tuple[ParagraphSpan, ...]
    condition: str
    sensitivity_only: bool
    meta: dict[str, Any]

    @property
    def paragraph_final_mask(self) -> tuple[bool, ...]:
        final = {span.token_end - 1 for span in self.paragraph_spans}
        return tuple(index in final for index in range(len(self.context_ids)))


def id_digest(identifier: str) -> bytes:
    return hashlib.blake2b(identifier.encode("utf-8"), digest_size=16).digest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_example(record: dict[str, Any]) -> MuSiQueExample:
    identifier = record["id"]
    if not isinstance(identifier, str):
        raise ValueError("id must be a string")
    hop_type = identifier.split("__", 1)[0]
    if hop_type not in HOP_TYPES:
        raise ValueError(f"unknown hop type {hop_type!r}")
    if record["answerable"] is not True:
        raise ValueError("MuSiQue-Ans examples must be answerable")
    for name in ("question", "answer"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{name} must be nonempty text")
    aliases = record["answer_aliases"]
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        raise ValueError("answer_aliases must be a list of strings")
    paragraphs = []
    for raw in record["paragraphs"]:
        if type(raw["idx"]) is not int or type(raw["is_supporting"]) is not bool:
            raise ValueError("invalid paragraph index/support flag")
        if not isinstance(raw["title"], str) or not isinstance(raw["paragraph_text"], str):
            raise ValueError("paragraph title/text must be strings")
        paragraphs.append(Paragraph(raw["idx"], raw["title"], raw["paragraph_text"], raw["is_supporting"]))
    indices = {paragraph.index for paragraph in paragraphs}
    if len(indices) != len(paragraphs):
        raise ValueError("duplicate paragraph index")
    steps = record["question_decomposition"]
    if len(steps) != int(hop_type[0]):
        raise ValueError("decomposition length disagrees with hop type")
    supporting = {paragraph.index for paragraph in paragraphs if paragraph.is_supporting}
    order, step_ids = [], []
    for step in steps:
        if not isinstance(step["question"], str) or not isinstance(step["answer"], str):
            raise ValueError("invalid decomposition question/answer")
        index = step["paragraph_support_idx"]
        if type(index) is not int or index not in supporting:
            raise ValueError("decomposition references a missing/non-supporting paragraph")
        order.append(index)
        step_ids.append(str(step["id"]))
    if set(order) != supporting:
        raise ValueError("supporting paragraphs and decomposition references disagree")
    return MuSiQueExample(identifier, hop_type, record["question"], record["answer"],
                          tuple(aliases), tuple(paragraphs), tuple(order), tuple(step_ids))


def read_examples(path: str | Path, *, expected_count: int | None = None) -> tuple[MuSiQueExample, ...]:
    examples, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example = _parse_example(json.loads(line))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if example.id in seen:
                raise ValueError(f"duplicate example id {example.id!r}")
            seen.add(example.id)
            examples.append(example)
    if expected_count is not None and len(examples) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} examples, got {len(examples)}")
    return tuple(examples)


def _ranked(examples: Iterable[MuSiQueExample]) -> tuple[MuSiQueExample, ...]:
    return tuple(sorted(examples, key=lambda example: (id_digest(example.id), example.id)))


def make_splits(
    train: Sequence[MuSiQueExample], dev: Sequence[MuSiQueExample], *, val_count: int = 1000,
) -> dict[str, tuple[MuSiQueExample, ...]]:
    if type(val_count) is not int or not 0 <= val_count <= len(train):
        raise ValueError("val_count must be an integer within the training pool")
    all_ids = [example.id for example in (*train, *dev)]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("official splits contain duplicate or overlapping ids")
    groups: dict[str, list[MuSiQueExample]] = defaultdict(list)
    for example in train:
        groups[example.hop_type].append(example)
    quotas = {hop: val_count * len(group) // len(train) for hop, group in groups.items()} if train else {}
    remainders = sorted(groups, key=lambda hop: (-(val_count * len(groups[hop]) % len(train)), hop))
    for hop in remainders[:val_count - sum(quotas.values())]:
        quotas[hop] += 1
    validation = {example.id for hop, group in groups.items() for example in _ranked(group)[:quotas[hop]]}
    screening = {example.id for example in dev if int.from_bytes(id_digest(example.id), "big") % 3 == 0}
    return {
        "train": _ranked(example for example in train if example.id not in validation),
        "val": _ranked(example for example in train if example.id in validation),
        "test": _ranked(example for example in dev if example.id in screening),
        "confirm": _ranked(example for example in dev if example.id not in screening),
    }


def _render(
    identifier: str, hop_type: str, paragraphs: tuple[Paragraph, ...], final_support_idx: int,
    question_text: str, answer: str, aliases: tuple[str, ...], tokenizer: Any,
    condition: str, sensitivity_only: bool, meta: dict[str, Any],
) -> MuSiQueEpisode:
    blocks = [f"{paragraph.title}\n{paragraph.text}\n\n" for paragraph in paragraphs]
    context = "".join(blocks)
    encoded = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    ids = tuple(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    starts, ends = [pair[0] for pair in offsets], [pair[1] for pair in offsets]
    spans, cursor = [], 0
    for paragraph, block in zip(paragraphs, blocks, strict=True):
        end = cursor + len(block)
        first, last = bisect_right(ends, cursor), bisect_left(starts, end)
        if first >= last or starts[first] != cursor or ends[last - 1] != end:
            raise ValueError(f"{identifier}: tokenizer crosses paragraph boundary {paragraph.index}")
        if tokenizer.decode(ids[first:last], clean_up_tokenization_spaces=False) != block:
            raise ValueError(f"{identifier}: paragraph span does not round-trip")
        spans.append(ParagraphSpan(paragraph.index, cursor, end, first, last))
        cursor = end
    question_ids = tuple(tokenizer.encode(question_text, add_special_tokens=False))
    answer_text = " " + answer
    answer_ids = tuple(tokenizer.encode(answer_text, add_special_tokens=False))
    if tuple(tokenizer.encode(question_text + answer_text, add_special_tokens=False)) != question_ids + answer_ids:
        raise ValueError(f"{identifier}: tokenization crosses the question/answer boundary")
    return MuSiQueEpisode(identifier, hop_type, paragraphs, final_support_idx, context, ids,
                           question_text, question_ids, answer_text, answer_ids, aliases,
                           tuple(spans), condition, sensitivity_only, meta)


def render_example(
    example: MuSiQueExample, *, tokenizer: Any, base_seed: int = 0, n_distractors: int = 0,
) -> MuSiQueEpisode:
    if type(n_distractors) is not int or n_distractors < 0:
        raise ValueError("n_distractors must be a nonnegative integer")
    support = sorted((paragraph for paragraph in example.paragraphs if paragraph.is_supporting),
                     key=lambda paragraph: paragraph.index)
    distractors = sorted((paragraph for paragraph in example.paragraphs if not paragraph.is_supporting),
                         key=lambda paragraph: paragraph.index)
    if n_distractors > len(distractors):
        raise ValueError(f"{example.id}: not enough own non-supporting paragraphs")
    seed = derive_seed(base_seed, VERSION, "paragraph-order", example.id, n_distractors)
    rng = random.Random(seed)
    selected = support + rng.sample(distractors, n_distractors)
    rng.shuffle(selected)
    return _render(example.id, example.hop_type, tuple(selected), example.final_support_idx,
                   f"Question: {example.question}\nAnswer:", example.answer, example.answer_aliases,
                   tokenizer, "original", False, {"version": VERSION, "episode_seed": seed,
                   "base_seed": base_seed, "n_distractors": n_distractors,
                   "decomposition_order": list(example.decomposition_order),
                   "decomposition_ids": list(example.decomposition_ids)})


def normalize_answer(text: str) -> str:
    lowered = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", lowered).split())


def answer_scores(prediction: str, answer: str, aliases: Sequence[str] = ()) -> dict[str, float]:
    predicted = normalize_answer(prediction)
    pred_tokens = predicted.split()
    exact, best_f1 = 0.0, 0.0
    for gold in (answer, *aliases):
        normalized = normalize_answer(gold)
        tokens = normalized.split()
        exact = max(exact, float(predicted == normalized))
        overlap = sum((Counter(pred_tokens) & Counter(tokens)).values())
        f1 = (2 * overlap / (len(pred_tokens) + len(tokens)) if pred_tokens and tokens
              else float(pred_tokens == tokens))
        best_f1 = max(best_f1, f1)
    return {"em": exact, "f1": best_f1}


def answer_kind(answer: str) -> str:
    if re.fullmatch(r"[12]\d{3}", answer.strip()):
        return "year"
    if re.fullmatch(r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?", answer.strip()):
        return "other_number"
    return "other"


def build_donor_pool(examples: Sequence[MuSiQueExample]) -> dict[str, tuple[MuSiQueExample, ...]]:
    grouped: dict[str, list[MuSiQueExample]] = defaultdict(list)
    for example in sorted(examples, key=lambda example: example.id):
        grouped[answer_kind(example.answer)].append(example)
    return {kind: tuple(group) for kind, group in grouped.items()}


def _answer_pattern(answers: Sequence[str], *, ignore_case: bool = True) -> re.Pattern[str]:
    variants = sorted({text for text in answers if text.strip()}, key=lambda text: (-len(text), text))
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(text) for text in variants) + r")(?!\w)",
                      re.IGNORECASE if ignore_case else 0)


def answer_substitution(
    episode: MuSiQueEpisode,
    donor_pool: Sequence[MuSiQueExample] | Mapping[str, Sequence[MuSiQueExample]],
    *, tokenizer: Any, base_seed: int = 0,
) -> MuSiQueEpisode | None:
    """Replace support mentions with a same-kind donor; None denotes uncovered.

    Eligibility is exact-case verbatim gold presence in the final support.
    Replacements cover case variants and aliases. Donors/selected distractors
    that would leave a source answer mention are excluded, without editing the
    question or inventing aliases. Preparation supplies donors from the same split.
    """
    gold = episode.answer_text[1:]
    final = next((paragraph for paragraph in episode.paragraphs
                  if paragraph.index == episode.final_support_idx and paragraph.is_supporting), None)
    if final is None or not _answer_pattern((gold,), ignore_case=False).search(f"{final.title}\n{final.text}"):
        return None
    pattern = _answer_pattern((gold, *episode.answer_aliases))
    original_labels = {normalize_answer(text) for text in (gold, *episode.answer_aliases)}
    if any(pattern.search(f"{paragraph.title}\n{paragraph.text}")
           for paragraph in episode.paragraphs if not paragraph.is_supporting):
        return None
    kind = answer_kind(gold)
    candidates = (donor_pool.get(kind, ()) if isinstance(donor_pool, Mapping)
                  else sorted((donor for donor in donor_pool if answer_kind(donor.answer) == kind),
                              key=lambda donor: donor.id))
    if not candidates:
        return None
    seed = derive_seed(base_seed, VERSION, "answer-substitution", episode.id)
    begin = random.Random(seed).randrange(len(candidates))
    for offset in range(len(candidates)):
        donor = candidates[(begin + offset) % len(candidates)]
        if donor.id == episode.id or normalize_answer(donor.answer) in original_labels:
            continue
        if pattern.search(donor.answer):
            continue
        paragraphs = tuple(replace(paragraph,
                                   title=pattern.sub(lambda _: donor.answer, paragraph.title),
                                   text=pattern.sub(lambda _: donor.answer, paragraph.text))
                           if paragraph.is_supporting else paragraph for paragraph in episode.paragraphs)
        if any(pattern.search(f"{paragraph.title}\n{paragraph.text}") for paragraph in paragraphs):
            continue
        aliases = tuple(alias for alias in donor.answer_aliases if alias.strip() and not pattern.search(alias)
                        and normalize_answer(alias) not in original_labels)
        return _render(episode.id, episode.hop_type, paragraphs, episode.final_support_idx,
                       episode.question_text, donor.answer, aliases, tokenizer, "answer_substitution", False,
                       {**episode.meta, "counterfactual_seed": seed, "donor_id": donor.id,
                        "original_answer": gold, "original_aliases": list(episode.answer_aliases),
                        "answer_kind": kind})
    return None


def remove_final_support(episode: MuSiQueEpisode, *, tokenizer: Any) -> MuSiQueEpisode:
    paragraphs = tuple(paragraph for paragraph in episode.paragraphs if paragraph.index != episode.final_support_idx)
    if len(paragraphs) == len(episode.paragraphs):
        raise ValueError("final supporting paragraph is absent")
    return _render(episode.id, episode.hop_type, paragraphs, episode.final_support_idx,
                   episode.question_text, episode.answer_text[1:], episode.answer_aliases, tokenizer,
                   "remove_final_support", True, {**episode.meta, "removed_paragraph_idx": episode.final_support_idx})


def episode_path(root: str | Path, split: str, condition: str = "original") -> Path:
    if split not in SPLITS or condition not in CONDITIONS:
        raise ValueError("unknown MuSiQue split or condition")
    suffix = "" if condition == "original" else "." + condition
    return Path(root) / f"{split}{suffix}.jsonl"


def write_episodes(path: str | Path, episodes: Iterable[MuSiQueEpisode]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(asdict(episode), ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_split(
    split: str, root: str | Path = DEFAULT_ROOT, *, condition: str = "original",
    allow_confirmation: bool = False,
) -> tuple[MuSiQueEpisode, ...]:
    if split == "confirm" and not allow_confirmation:
        raise PermissionError("confirm requires explicit allow_confirmation=True")
    episodes = []
    with episode_path(root, split, condition).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record["paragraphs"] = tuple(Paragraph(**paragraph) for paragraph in record["paragraphs"])
            record["paragraph_spans"] = tuple(ParagraphSpan(**span) for span in record["paragraph_spans"])
            for name in ("context_ids", "question_ids", "answer_ids", "answer_aliases"):
                record[name] = tuple(record[name])
            episodes.append(MuSiQueEpisode(**record))
    return tuple(episodes)
