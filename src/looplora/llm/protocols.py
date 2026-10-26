"""Padding-aware support writing and read-only QA with independent KV forks.

Questions are [batch, tokens], aligned with the support rows. Call once for each
question (and each L) using the same SupportSnapshot. Answer logits predict each
answer token from the preceding real token, including across padding holes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import DynamicCache


@dataclass(frozen=True)
class SupportSnapshot:
    states: dict[int, Tensor]
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None
    attention_mask: Tensor
    protocol: str


@dataclass(frozen=True)
class AnswerScores:
    logits: Tensor
    loss: Tensor
    token_loss: Tensor


class _SupportComplete(Exception):
    """Internal control flow: the single memory writer has finished its work."""


def _mask(ids: Tensor, mask: Tensor | None, name: str) -> Tensor:
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError(f"{name} ids must be nonempty [batch, tokens]")
    if mask is None:
        return torch.ones_like(ids, dtype=torch.bool)
    if mask.shape != ids.shape:
        raise ValueError(f"{name} mask shape differs from ids")
    return mask.to(device=ids.device, dtype=torch.bool)


def _positions(mask: Tensor) -> Tensor:
    return (mask.long().cumsum(-1) - 1).clamp_min(0)


def _support_attention(valid: Tensor, reset: Tensor | None, dtype: torch.dtype):
    if reset is None:
        return valid, _positions(valid), valid
    if reset.shape != valid.shape:
        raise ValueError("reset mask shape differs from support")
    reset = reset.to(device=valid.device, dtype=torch.bool)
    segments = reset.long().cumsum(-1)
    count = valid.long().cumsum(-1)
    offsets = torch.where(reset, count - valid.long(), 0).cummax(-1).values
    positions = (count - offsets - 1).clamp_min(0)
    length = valid.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
    allowed = (segments[:, :, None] == segments[:, None, :]) & causal & valid[:, None, :]
    attention = torch.zeros(allowed.shape, device=valid.device, dtype=dtype)
    attention = attention.masked_fill(~allowed, torch.finfo(dtype).min)[:, None, :, :]
    final_valid = valid & (segments == segments[:, -1:])
    return attention, positions, final_valid


def encode_support(
    model, context_ids: Tensor, attn_mask: Tensor, write_policy: str = "all_tokens", *,
    fact_final: Tensor | None = None, protocol: str = "memory_only", loop_steps: int = 1,
    reset: Tensor | None = None, truncate: bool = False,
) -> SupportSnapshot:
    """Write support and retain only M, or M plus a forkable backbone cache.

    Optional reset masks also isolate backbone attention between packed episodes;
    the returned cache exposes only the final episode in each row.
    Gradients are preserved; inference callers can use torch.no_grad().
    With one adapter, truncate=True stops after its support write. This is only
    valid for memory-only: no upper-layer output or backbone KV is requested.
    """
    if protocol not in {"memory_only", "context_retained"}:
        raise ValueError(f"unknown protocol: {protocol}")
    valid = _mask(context_ids, attn_mask, "support")
    if write_policy == "all_tokens":
        write = valid
    elif write_policy == "fact_final":
        if fact_final is None or fact_final.shape != valid.shape:
            raise ValueError("fact_final write policy requires a [batch, tokens] mask")
        write = valid & fact_final.to(device=valid.device, dtype=torch.bool)
    else:
        raise ValueError(f"unknown write policy: {write_policy}")
    if reset is not None:
        reset = reset.to(device=valid.device, dtype=torch.bool)
    attention, positions, retained_valid = _support_attention(valid, reset, model.dtype)
    bank = model.looplora_adapters
    keep_cache = protocol == "context_retained"
    if truncate and (keep_cache or len(bank.modules) != 1):
        raise ValueError("truncated support requires memory_only and exactly one adapter")
    hook = None
    if truncate:
        def stop_after_write(module, inputs, output):
            raise _SupportComplete

        hook = next(iter(bank.modules.values())).register_forward_hook(stop_after_write)
    try:
        mode = "write_only" if truncate else "online"
        with bank.use(loop_steps=loop_steps, valid=valid, write=write, reset=reset, mode=mode):
            try:
                output = model.model(
                    input_ids=context_ids, attention_mask=attention, position_ids=positions,
                    use_cache=keep_cache,
                )
            except _SupportComplete:
                if not truncate:
                    raise
                output = None
            else:
                if truncate:
                    raise RuntimeError("support forward did not reach the installed adapter")
            states = bank.states()
    finally:
        if hook is not None:
            hook.remove()
    cache = output.past_key_values.to_legacy_cache() if keep_cache else None
    return SupportSnapshot(states, cache, retained_valid.clone(), protocol)


def _fork_cache(snapshot: SupportSnapshot, model):
    if snapshot.protocol == "memory_only":
        if snapshot.past_key_values is not None:
            raise ValueError("memory-only snapshot cannot contain backbone KV")
        return None, snapshot.attention_mask[:, :0]
    if snapshot.protocol != "context_retained" or snapshot.past_key_values is None:
        raise ValueError("context-retained snapshot requires backbone KV")
    cache = DynamicCache(
        ((key.clone(), value.clone()) for key, value in snapshot.past_key_values),
        config=model.config,
    )
    return cache, snapshot.attention_mask


def _check_questions(snapshot: SupportSnapshot, ids: Tensor, mask: Tensor):
    if ids.shape[0] != snapshot.attention_mask.shape[0]:
        raise ValueError("question batch must align with support snapshot rows")
    if not bool(mask.any(-1).all()):
        raise ValueError("each question needs at least one real token")


def _scores(logits: Tensor, answer_ids: Tensor, answer_mask: Tensor) -> AnswerScores:
    labels = answer_ids.masked_fill(~answer_mask, -100)
    losses = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none")
    return AnswerScores(logits, losses.sum() / answer_mask.sum().clamp_min(1), losses)


def answer_scores(
    model, snapshot: SupportSnapshot, question_ids: Tensor, question_mask: Tensor,
    answer_ids: Tensor, answer_mask: Tensor | None = None, *, loop_steps: int = 1,
) -> AnswerScores:
    """Teacher-forced answer-token scores without writing question/answer tokens."""
    qvalid = _mask(question_ids, question_mask, "question")
    avalid = _mask(answer_ids, answer_mask, "answer")
    _check_questions(snapshot, question_ids, qvalid)
    if answer_ids.shape[0] != question_ids.shape[0]:
        raise ValueError("answer batch must align with question batch")
    cache, prefix = _fork_cache(snapshot, model)
    all_valid = torch.cat((qvalid, avalid), -1)
    ids = torch.cat((question_ids, answer_ids[:, :-1]), -1)
    valid = all_valid[:, :-1]
    attention = torch.cat((prefix, valid), -1)
    positions = _positions(attention)[:, prefix.shape[1]:]
    # Locate the previous real token for every label, even with left/right padding.
    indices = torch.arange(all_valid.shape[1], device=ids.device).expand_as(all_valid)
    previous = torch.where(all_valid, indices, -1).cummax(-1).values
    pred = previous[:, question_ids.shape[1] - 1: -1]
    with model.looplora_adapters.use(
        snapshot.states, loop_steps=loop_steps, valid=valid, mode="read_only",
    ):
        output = model.model(
            input_ids=ids, attention_mask=attention, position_ids=positions,
            past_key_values=cache, use_cache=False,
        )
    hidden = output.last_hidden_state.gather(
        1, pred[:, :, None].expand(-1, -1, output.last_hidden_state.shape[-1]),
    )
    return _scores(model.lm_head(hidden), answer_ids, avalid)


def _prefill(model, snapshot, question_ids, question_mask, loop_steps):
    valid = _mask(question_ids, question_mask, "question")
    _check_questions(snapshot, question_ids, valid)
    cache, prefix = _fork_cache(snapshot, model)
    attention = torch.cat((prefix, valid), -1)
    with model.looplora_adapters.use(
        snapshot.states, loop_steps=loop_steps, valid=valid, mode="read_only",
    ):
        output = model.model(
            input_ids=question_ids, attention_mask=attention,
            position_ids=_positions(attention)[:, prefix.shape[1]:],
            past_key_values=cache, use_cache=True,
        )
    last = torch.where(
        valid, torch.arange(valid.shape[1], device=valid.device), -1,
    ).amax(-1)
    hidden = output.last_hidden_state[torch.arange(valid.shape[0], device=valid.device), last]
    return model.lm_head(hidden), output.past_key_values, attention


def _decode_step(model, snapshot, token, valid, cache, attention, loop_steps):
    attention = torch.cat((attention, valid[:, None]), -1)
    with model.looplora_adapters.use(
        snapshot.states, loop_steps=loop_steps, valid=valid[:, None], mode="read_only",
    ):
        output = model.model(
            input_ids=token[:, None], attention_mask=attention,
            position_ids=_positions(attention)[:, -1:], past_key_values=cache, use_cache=True,
        )
    return model.lm_head(output.last_hidden_state[:, 0]), output.past_key_values, attention


def incremental_answer_scores(
    model, snapshot: SupportSnapshot, question_ids: Tensor, question_mask: Tensor,
    answer_ids: Tensor, answer_mask: Tensor | None = None, *, loop_steps: int = 1,
) -> AnswerScores:
    """Teacher forcing one token at a time, using the same cache path as decode."""
    valid = _mask(answer_ids, answer_mask, "answer")
    logits, cache, attention = _prefill(model, snapshot, question_ids, question_mask, loop_steps)
    result = []
    for step in range(answer_ids.shape[1]):
        result.append(logits)
        if step + 1 < answer_ids.shape[1]:
            next_logits, cache, attention = _decode_step(
                model, snapshot, answer_ids[:, step], valid[:, step], cache, attention, loop_steps,
            )
            logits = torch.where(valid[:, step, None], next_logits, logits)
    return _scores(torch.stack(result, 1), answer_ids, valid)


@torch.no_grad()
def greedy_answer(
    model, snapshot: SupportSnapshot, question_ids: Tensor, question_mask: Tensor, *,
    loop_steps: int = 1, answer_budget: int = 1,
) -> Tensor:
    """Decode exactly answer_budget tokens; EM compares only the first token."""
    if answer_budget < 1:
        raise ValueError("answer_budget must be positive")
    logits, cache, attention = _prefill(model, snapshot, question_ids, question_mask, loop_steps)
    result = []
    for step in range(answer_budget):
        token = logits.argmax(-1)
        result.append(token)
        if step + 1 < answer_budget:
            logits, cache, attention = _decode_step(
                model, snapshot, token, torch.ones_like(token, dtype=torch.bool),
                cache, attention, loop_steps,
            )
    return torch.stack(result, 1)
