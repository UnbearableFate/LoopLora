"""Deterministic text rendering of symbolic relation episodes for Qwen3.

Entity IDs index the pinned name asset. Answer text includes its leading space,
so every target (including the missing-information label) is exactly one token.
No tokenizer/model library is imported here; callers supply a fast tokenizer.
"""
from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .relations import EpisodeArrays

RENDER_VERSION = "relations-text-v1"
ASSET_PATH = Path(__file__).with_name("assets") / "qwen3_names.v1.json"
RELATION_PHRASES = (
    "knows", "trusts", "visits", "helps", "follows", "mentors", "admires", "supports",
    "greets", "teaches", "calls", "thanks", "hires", "invites", "guides", "remembers",
)
TASKS = frozenset({"t1", "t2_chain_end", "t2_walk", "t2_typed", "t3_update", "t3_missing"})


@dataclass(frozen=True)
class NamePool:
    version: str
    names: tuple[str, ...]
    token_ids: tuple[int, ...]
    model: str
    revision: str
    insufficient_name: str
    insufficient_token_id: int


@dataclass(frozen=True)
class FactSpan:
    """Half-open character/token spans, including the fact's trailing newline."""

    fact_index: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class RenderedQuery:
    question_text: str
    question_ids: tuple[int, ...]
    answer_text: str
    answer_token_id: int
    answer_id: int

    @property
    def answer_ids(self) -> tuple[int, ...]:
        return (self.answer_token_id,)


@dataclass(frozen=True)
class RenderedEpisode:
    context_text: str
    context_ids: tuple[int, ...]
    fact_spans: tuple[FactSpan, ...]
    queries: tuple[RenderedQuery, ...]
    version: str = RENDER_VERSION

    @property
    def fact_final_mask(self) -> tuple[bool, ...]:
        """Write exactly once per valid fact, at the final token in its span."""
        final = {span.token_end - 1 for span in self.fact_spans}
        return tuple(index in final for index in range(len(self.context_ids)))


def load_name_pool(path: str | Path = ASSET_PATH) -> NamePool:
    data = json.loads(Path(path).read_text())
    pool = NamePool(data["version"], tuple(data["names"]), tuple(data["token_ids"]),
                    data["model"], data["revision"], data["insufficient_name"],
                    data["insufficient_token_id"])
    if len(pool.names) != len(pool.token_ids) or len(set(pool.names)) != len(pool.names):
        raise ValueError("Name asset has inconsistent lengths or duplicate names")
    if len(set(pool.token_ids)) != len(pool.token_ids):
        raise ValueError("Name asset has duplicate token IDs")
    if pool.insufficient_name in pool.names or pool.insufficient_token_id in pool.token_ids:
        raise ValueError("Missing-information label overlaps an entity")
    return pool


def validate_name_pool(tokenizer: Any, pool: NamePool | None = None) -> None:
    """Check the complete pinned pool against the supplied tokenizer."""
    pool = load_name_pool() if pool is None else pool
    if len(pool.names) < 1024:
        raise ValueError("Name pool must contain at least 1024 entities")
    for name, token_id in zip((*pool.names, pool.insufficient_name),
                              (*pool.token_ids, pool.insufficient_token_id), strict=True):
        text = " " + name
        if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Name {name!r} is not its recorded single leading-space token")
        if tokenizer.decode([token_id], clean_up_tokenization_spaces=False) != text:
            raise ValueError(f"Name {name!r} does not decode exactly")


def relation_phrase(relation_id: int) -> str:
    if relation_id < 0:
        raise ValueError("Relation IDs must be nonnegative")
    if relation_id < len(RELATION_PHRASES):
        return RELATION_PHRASES[relation_id]
    return f"has relation {relation_id} to"


def _question(task: str, kind: str, start: str, relation: str, hops: int) -> str:
    if kind == "t1":
        text = f"Which person is linked from {start} by the relation '{relation}'?"
    elif kind == "t2_chain_end":
        text = (f"Starting from {start}, keep following the relation '{relation}' until "
                "there is no further fact for that relation. Which person is reached?")
    else:
        if hops < 1:
            raise ValueError("A walk question must state a positive step count")
        text = (f"Starting from {start}, follow the relation '{relation}' exactly {hops} "
                f"{'step' if hops == 1 else 'steps'}. Which person is reached?")
    if task == "t3_update":
        text = "Use the latest fact for each person and relation. " + text
    if task == "t3_missing":
        text += " If a required fact is missing, answer Unknown."
    return text + "\nAnswer:"


def render_episode(
    arrays: EpisodeArrays,
    episode_index: int = 0,
    *,
    tokenizer: Any,
    name_pool: NamePool | None = None,
    task: str | None = None,
    entity_count: int | None = None,
    include_support: bool = True,
) -> RenderedEpisode:
    """Render one episode without consulting oracle hop lengths or changing IDs.

    ``task`` and ``entity_count`` can explicitly describe an EpisodeArrays-like
    fixture. Otherwise they come from the generator's ``meta['spec']``.
    Invalid fact slots are omitted; spans retain the original fact slot index.
    """
    pool = load_name_pool() if name_pool is None else name_pool
    spec = arrays.meta.get("spec", {})
    task = spec.get("name") if task is None else task
    entity_count = spec.get("V") if entity_count is None else entity_count
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}")
    if entity_count is None or not 1 <= entity_count <= len(pool.names):
        raise ValueError("Entity vocabulary does not fit the name pool")
    kind = spec.get("base") if task in {"t3_update", "t3_missing"} else task
    if kind == "t2_typed":
        kind = "t2_chain_end"
    if kind not in {"t1", "t2_chain_end", "t2_walk"}:
        raise ValueError("The t3 task requires its generator base in meta['spec']")
    if not 0 <= episode_index < len(arrays.facts):
        raise IndexError("Episode index is outside the arrays")

    def entity(entity_id: int) -> str:
        if not 0 <= entity_id < entity_count:
            raise ValueError(f"Entity ID {entity_id} is outside 0..{entity_count - 1}")
        return pool.names[entity_id]

    lines, chars = [], []
    char_cursor = 0
    fact_rows = (zip(arrays.facts[episode_index], arrays.fact_valid[episode_index], strict=True)
                 if include_support else ())
    for fact_index, (fact, valid) in enumerate(fact_rows):
        if not valid:
            continue
        head, relation, tail = map(int, fact)
        line = f"{entity(head)} {relation_phrase(relation)} {entity(tail)}.\n"
        lines.append(line)
        chars.append((fact_index, char_cursor, char_cursor + len(line)))
        char_cursor += len(line)
    context = "".join(lines)
    encoded = (tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
               if include_support else {"input_ids": [], "offset_mapping": []})
    ids = tuple(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    starts, ends = [p[0] for p in offsets], [p[1] for p in offsets]
    spans = []
    for fact_index, char_start, char_end in chars:
        token_start = bisect_right(ends, char_start)
        token_end = bisect_left(starts, char_end)
        if token_start >= token_end or starts[token_start] != char_start or ends[token_end - 1] != char_end:
            raise ValueError("Tokenizer crosses a fact boundary; exact fact spans are unavailable")
        if tokenizer.decode(ids[token_start:token_end], clean_up_tokenization_spaces=False) != context[char_start:char_end]:
            raise ValueError("Fact token span does not round-trip to its text")
        spans.append(FactSpan(fact_index, char_start, char_end, token_start, token_end))

    queries = []
    for start, relation, hops, answer in zip(arrays.q_start[episode_index], arrays.q_rel[episode_index],
                                            arrays.q_hop_given[episode_index], arrays.answer[episode_index], strict=True):
        start, relation, hops, answer = map(int, (start, relation, hops, answer))
        question = _question(task, kind, entity(start), relation_phrase(relation), hops)
        question_ids = tuple(tokenizer.encode(question, add_special_tokens=False))
        if answer == entity_count:
            answer_text = " " + pool.insufficient_name
            answer_token_id = pool.insufficient_token_id
        else:
            answer_text = " " + entity(answer)
            answer_token_id = pool.token_ids[answer]
        if tuple(tokenizer.encode(question + answer_text, add_special_tokens=False)) != question_ids + (answer_token_id,):
            raise ValueError("Question/answer boundary changes the recorded answer token")
        queries.append(RenderedQuery(question, question_ids, answer_text, answer_token_id, answer))
    return RenderedEpisode(context, ids, tuple(spans), tuple(queries))


def render_episodes(arrays: EpisodeArrays, *, tokenizer: Any, name_pool: NamePool | None = None,
                    task: str | None = None, entity_count: int | None = None,
                    include_support: bool = True) -> tuple[RenderedEpisode, ...]:
    pool = load_name_pool() if name_pool is None else name_pool
    return tuple(render_episode(arrays, index, tokenizer=tokenizer, name_pool=pool,
                                task=task, entity_count=entity_count, include_support=include_support)
                 for index in range(len(arrays.facts)))


def answer_em(first_token_id: int, query: RenderedQuery) -> bool:
    """Exact match is equality of the greedy first answer token and its label."""
    return int(first_token_id) == query.answer_token_id
