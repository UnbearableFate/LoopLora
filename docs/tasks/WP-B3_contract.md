# WP-B3 implementation coordination

Scope: new `musique.py`, `retention.py`, `prepare_musique.py`, tests and report;
existing generators/rendering remain unchanged. Runtime work goes through
`scripts/submit.py`; downloads are login-side with `timeout`.

- MuSiQue raw files: `/work/xg24i002/x10041/LoopLora/cache/musique/raw/`.
  Prepared artifacts: the same parent plus `cache/retention/`, shared and ignored.
  Preparation and loaders are offline, with no implicit downloads/model calls.
- Module `retention.py` provides frozen `MCItem` records (`id`, `context`,
  `choices: tuple[str,...]`, `label: int`, source dataset/config/split fields),
  `prepare_retention(hf_home, output_dir, *, count=500, split_overrides=None,
  allow_smaller=()) -> dict` and `load_retention(name, output_dir) -> tuple[MCItem,...]`.
  It reads cached Parquet with a lazy PyArrow import, writes six JSONL files and a
  manifest with pinned revisions, selected IDs, counts and source file hashes.
  Output names: arc_easy, arc_challenge, piqa, hellaswag, winogrande, mmlu.
- Choose validation first, then dev, then test only when no earlier split exists;
  explicit overrides and allowed smaller subsets must be recorded in the manifest.
  Hash-rank IDs with blake2b-128 (plain UTF-8 ID). Use stable semantic content IDs
  for datasets without IDs; include MMLU subject to prevent cross-subject collision.
  WinoGrande uses the text preceding `_` as context and each option plus the
  sentence suffix as its choice, so likelihood scoring compares full continuations.
- Main script owns CLI and invokes retention preparation; retention worker owns
  only its module, tests and `docs/reports/WP-B3_retention_notes.md`.
- Orchestrator decisions: PyArrow 21.0.0 approved and installed in the shared
  venv; dependency files come from master, with no worker-side `uv sync`.
  ARC-Challenge uses all 299 validation items, with no test items. MMLU uses the
  complete validation shard downloaded at cached revision
  `c30699e8356da336a370243923dbaf21066bb9fe`, then 500 items by ID hash rank.
  Preparation therefore passes `--allow-smaller-retention arc_challenge`;
  no split override or restricted MMLU subject selection is used.

MuSiQue API for independent tests:
- Frozen `Paragraph(index, title, text, is_supporting)`; frozen
  `MuSiQueExample(id, hop_type, question, answer, answer_aliases, paragraphs,
  decomposition_order, decomposition_ids)` with `.final_support_idx` property.
- `read_examples(path, *, expected_count=None)` validates raw schema, unique IDs,
  hop/decomposition agreement and support references. `make_splits(train, dev,
  *, val_count=1000)` uses proportional largest-remainder hop quotas, hash-rank
  selections and big-endian blake2b-128(id) modulo three for test/confirm.
- `render_example(example, *, tokenizer, base_seed=0, n_distractors=0)` returns
  `MuSiQueEpisode` with context/question/answer text and tuple IDs, aliases,
  ordered paragraphs, final_support_idx, half-open `.paragraph_spans`
  (`index,char_start,char_end,token_start,token_end`), `.paragraph_final_mask`,
  condition, sensitivity_only and meta. Paragraph blocks include trailing `\n\n`.
- `answer_scores(prediction, answer, aliases=()) -> dict` with `em`, `f1`;
  `normalize_answer(text)` follows the pinned official ASCII punctuation/article
  normalization, including empty-token F1 and independent alias maxima.
- `answer_substitution(episode, donor_pool, *, tokenizer, base_seed=0)` returns a
  new episode or None for uncovered examples. Exact-case gold presence in the
  final support is required; longest-first case-insensitive word-boundary
  replacement covers gold+aliases in support titles/text. It rejects donors or
  selected distractors that would leave an original variant in the context.
  Donors are sorted by id, same coarse kind, different example/gold, from the
  caller-supplied pool (preparation supplies the same retained split only).
- `remove_final_support(episode, *, tokenizer)` returns a sensitivity-only copy
  with unchanged target. Original episode/query ordering is preserved by both.
- `write_episodes(path, episodes)` serializes JSONL; `load_split(split, root,
  *, condition='original', allow_confirmation=False)` restores tuples/dataclasses
  and rejects confirm access before opening files unless explicitly allowed.
  Conditions: original, answer_substitution, remove_final_support.
