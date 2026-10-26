"""Render episode batches once and fork their support snapshots by question."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from looplora.data.relations import EpisodeArrays
from looplora.data.render import NamePool, load_name_pool, render_episodes
from .protocols import SupportSnapshot


@dataclass(frozen=True)
class TokenBatch:
    support_ids: Tensor
    support_mask: Tensor
    fact_final: Tensor
    question_ids: Tensor
    question_mask: Tensor
    answer_ids: Tensor
    candidate_ids: Tensor
    candidate_mask: Tensor

    def to(self, device) -> TokenBatch:
        return TokenBatch(**{field.name: getattr(self, field.name).to(device)
                             for field in fields(self)})


@dataclass(frozen=True)
class PreparedBatch:
    snapshot: SupportSnapshot
    question_ids: Tensor
    question_mask: Tensor
    answer_ids: Tensor
    candidate_ids: Tensor
    candidate_mask: Tensor
    arrays: EpisodeArrays
    support_tokens: int
    write_count: Tensor
    support_ids: Tensor
    support_mask: Tensor
    fact_final: Tensor


def tokenize_batch(arrays: EpisodeArrays, tokenizer, *, device,
                   name_pool: NamePool | None = None, rendered=None) -> TokenBatch:
    pool = load_name_pool() if name_pool is None else name_pool
    rendered = (render_episodes(arrays, tokenizer=tokenizer, name_pool=pool)
                if rendered is None else rendered)
    if len(rendered) != len(arrays.facts):
        raise ValueError("rendered rows must align with episode arrays")
    if not rendered or not rendered[0].queries:
        raise ValueError("a batch requires episodes and queries")
    B, Q = len(rendered), len(rendered[0].queries)
    if any(len(row.queries) != Q or not row.context_ids for row in rendered):
        raise ValueError("episodes need nonempty supports and equal query counts")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("tokenizer needs a pad or EOS token")
    S = max(len(row.context_ids) for row in rendered)
    T = max(len(query.question_ids) for row in rendered for query in row.queries)
    support = torch.full((B, S), pad, dtype=torch.long, device=device)
    valid = torch.zeros((B, S), dtype=torch.bool, device=device)
    final = torch.zeros_like(valid)
    questions = torch.full((B, Q, T), pad, dtype=torch.long, device=device)
    qvalid = torch.zeros((B, Q, T), dtype=torch.bool, device=device)
    answers = torch.empty((B, Q), dtype=torch.long, device=device)
    candidates = []
    for b, row in enumerate(rendered):
        n = len(row.context_ids)
        support[b, :n] = torch.tensor(row.context_ids, device=device)
        valid[b, :n] = True
        final[b, :n] = torch.tensor(row.fact_final_mask, dtype=torch.bool, device=device)
        for q, query in enumerate(row.queries):
            if not query.question_ids:
                raise ValueError("questions must contain at least one token")
            n = len(query.question_ids)
            questions[b, q, :n] = torch.tensor(query.question_ids, device=device)
            qvalid[b, q, :n] = True
            answers[b, q] = query.answer_token_id
        facts = arrays.facts[b, arrays.fact_valid[b]]
        entities = sorted(set(facts[:, (0, 2)].ravel().tolist()))
        ids = [pool.token_ids[entity] for entity in entities]
        if arrays.meta["spec"]["name"] == "t3_missing":
            ids.append(pool.insufficient_token_id)
        candidates.append(ids)
    C = max(map(len, candidates))
    candidate_ids = torch.full((B, C), pad, dtype=torch.long, device=device)
    candidate_mask = torch.zeros((B, C), dtype=torch.bool, device=device)
    for b, ids in enumerate(candidates):
        candidate_ids[b, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        candidate_mask[b, :len(ids)] = True
    return TokenBatch(support, valid, final, questions, qvalid, answers,
                      candidate_ids, candidate_mask)


def repeat_snapshot(snapshot: SupportSnapshot, queries: int) -> SupportSnapshot:
    """Episode-major B,Q flattening; retain the support autograd graph."""
    if queries < 1:
        raise ValueError("queries must be positive")
    states = {layer: state.repeat_interleave(queries, dim=0)
              for layer, state in snapshot.states.items()}
    cache = None if snapshot.past_key_values is None else tuple(
        (key.repeat_interleave(queries, dim=0), value.repeat_interleave(queries, dim=0))
        for key, value in snapshot.past_key_values
    )
    return SupportSnapshot(states, cache,
                           snapshot.attention_mask.repeat_interleave(queries, dim=0),
                           snapshot.protocol)
