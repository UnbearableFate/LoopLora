# WP-B3 — T-08 real-task data: MuSiQue (given-support) and a retention suite

Read first: plan §7.4–7.5 (shortcuts, protocols), §9.4; `docs/reports/WP-E-interface.md` (how the LLM pipeline
consumes context/question/answer ids), `src/looplora/data/render.py`. Branch/worktree: `wp/data` (merge `master`
first). Report: `docs/reports/WP-B3.md`. Purpose: P3 (Gate 4) needs a reproducible real multi-hop task with the
same information-access protocols as P1-L; this task only prepares data, metrics and counterfactuals.

## 1. Source (login node for the download, `timeout`, HF_HOME as in docs/README.md)
- HF dataset `dgslibisey/MuSiQue`, revision `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321`, files
  `musique_ans_v1.0_train.jsonl`, `musique_ans_v1.0_dev.jsonl` (third-party mirror of MuSiQue-Ans v1.0).
  Verify counts against the paper (train 19,938, dev 2,417) and the field schema (`id`, `paragraphs` with
  `is_supporting`, `question`, `question_decomposition` with `paragraph_support_idx`, `answer`,
  `answer_aliases`, `answerable`); record blake2b-128 of each file. Jobs read it offline.

## 2. Splits (fixed; record ids and counts)
- `train` = MuSiQue train minus `val`; `val` = 1,000 MuSiQue-train examples chosen by blake2b(id) rank,
  stratified by hop type (`2hop`, `3hop1`, `3hop2`, `4hop1`, `4hop2`, `4hop3`, from the id prefix).
- MuSiQue dev → `test` (screening: blake2b(id) mod 3 = 0) and `confirm` (the rest); report the per-hop-type
  proportions. `confirm` loads only with an explicit `allow_confirmation=True`.
- State in the report that `val` may share single-hop sub-questions with `train` (MuSiQue limits leakage only
  between its official splits).

## 3. Episodes (given-support setting)
- Context = the supporting paragraphs only, each rendered `"{title}\n{paragraph_text}"`, in a random order per
  example (seeded via `derive_seed`; never the decomposition order). Option `n_distractors` (default 0) adds that
  many of the example's own non-supporting paragraphs, shuffled in.
- Question rendered with a fixed prompt (`"Question: {question}\nAnswer:"`), target `" {answer}"`; keep
  `answer_aliases`. Provide per-paragraph spans in token space (Qwen3-0.6B-Base tokenizer, pinned in
  `llm/inject.py`) so a writer can use paragraph or all-token write policies.
- Length: report the token-count distribution of context, question and answer per hop type. Fix one context
  cap (choose from the distribution, e.g. the 95th percentile rounded up) and drop longer examples; report the
  number dropped per split and hop type (selection must not depend on any model output).

## 4. Metrics
- Answer EM and F1 as in the official MuSiQue answer metric (SQuAD-style normalisation: lower-case, remove
  punctuation and articles, collapse whitespace; maximum over answer and aliases). Test against hand-checked
  cases, including aliases and punctuation. Record any difference from the official script.
- Generation contract for callers: greedy, stop at newline or EOS, `max_answer_tokens` fixed per split from the
  answer-length distribution (report the value).

## 5. Counterfactuals (memory reading vs. pretraining knowledge)
- `answer_substitution`: in examples whose gold answer occurs verbatim (word boundaries) in the last-hop
  supporting paragraph, replace every occurrence of the answer and its aliases in the supporting paragraphs
  with a substitute gold answer of another example of the same coarse kind (year, other number, other), seeded;
  the new target is the substitute. Report coverage per hop type.
- `remove_final_support`: drop the last-hop supporting paragraph; sensitivity only (no target change; callers
  report the accuracy drop and do not score a refusal as an error).
- A closed-book control needs no data change (question only); document it as the caller's protocol.

## 6. Retention suite (data side only; scoring belongs to the LLM pipeline)
Fixed subsets from the cached HF datasets under HF_HOME (record each dataset's cached revision): ARC-Easy,
ARC-Challenge, PIQA, HellaSwag, WinoGrande, MMLU — 500 examples each from a validation/dev split where one
exists (else test, stated), chosen by blake2b(id) rank. Format as multiple-choice items
(`context`, `choices`, `label`) for log-likelihood scoring; no model calls here.

## Tests and deliverable boundary
Unit tests (compute node via `scripts/submit.py`): split disjointness and determinism, hop stratification,
the shuffled order matches the decomposition order no more often than chance, span alignment with
the tokenizer, metric cases, counterfactual targets (substitute present, original answer and aliases absent from
the context), retention subset determinism. Processed data under `/work/xg24i002/x10041/LoopLora/cache/musique/`
and `cache/retention/` (git-ignored); code in `src/looplora/data/musique.py`, `src/looplora/data/retention.py`,
`scripts/prepare_musique.py`. Do not change existing generators or `render.py` behaviour.
