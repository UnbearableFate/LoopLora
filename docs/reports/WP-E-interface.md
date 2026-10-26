# WP-E implementation interface (working coordination)

WP-E owns `llm/`, `data/render.py`, name assets, `test_llm_*.py`, pilot config.
Orchestrator confirmed WP-A/WP-B merged at master `dbead6f`; merged into `wp/llm`.
Codegraph CLI was checked: this worktree is not indexed and the master index contains only one
file, so implementation inspection uses the small source tree directly (no login-side indexing).

Resolved integration blocker observed in WP-A worktree `core/wrapper.py` (2026-10-26):
`AdaptedLinear.forward` passes the original bf16 `sequence` to the fp32 adapter's `F.linear`
and returns `base + branch` without a dtype conversion. Requested core fix: pass
`sequence.to(adapter.A.dtype)` only to the adapter; preserve original x for base; cast branch
to base.dtype before addition. These casts preserve autograd and let both reference fp64 and
Qwen bf16+fp32 configurations work. WP-A implemented this as addendum 3 before merge.
WP-E does not edit core. Its full-model bf16 test checks this.

LLM API being implemented:
- `load_model(dtype=torch.bfloat16, device="cuda")`, `load_tokenizer()` in `llm/inject.py`; pinned offline Qwen.
- `attach_adapters(model, cfg, layers=(14,), seed=0) -> AdapterBank`; `cfg` is core AdapterConfig
  (dimensions replaced with real projection sizes); `detach_adapters(model)`.
- `AdapterBank` accessible as `model.looplora_adapters`; `.modules` dict indexed by layer int;
  `.states()` -> dict[int,Tensor]; `.use(states=None, loop_steps=1, valid=None, write=None,
  reset=None, mode="read_only")` context manager; `.zero_states(batch)`.
- `encode_support(model, context_ids, attn_mask, write_policy="all_tokens", *, fact_final=None,
  protocol="memory_only", loop_steps=1) -> SupportSnapshot`;
  snapshot `.states`, `.past_key_values` (immutable tuple of K/V pairs or None), `.attention_mask`.
  Optional `reset` isolates both backbone attention and memory for packed episodes.
- `answer_scores(model, snapshot, question_ids, question_mask, answer_ids, answer_mask=None,
  *, loop_steps=1) -> AnswerScores` with `.logits [B,A,V]`, `.loss`, `.token_loss [B,A]`.
- `greedy_answer(model, snapshot, question_ids, question_mask, *, loop_steps=1,
  answer_budget=1) -> Tensor[B,budget]`; `incremental_answer_scores` same inputs/return
  as answer_scores for the teacher forcing equivalence test.
- Question batch rows align with snapshot rows; one call per question forks the snapshot.
- Attention position ids use cumulative valid masks; memory-only starts fresh positions.
- Rendered outputs expected from data worker agent, independent of protocol input tensors.

Pilot uses synthetic token sequences with declared lengths (support 64/question 16/answer 1 by
initial default), avoiding linguistic correctness claims. Full 20-cell requested rank/L/batch grid.
