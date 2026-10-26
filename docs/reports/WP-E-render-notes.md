# WP-E rendering notes

Primary integration is complete; see [WP-E.md](WP-E.md) for the final 54-test
run and implementation commits. The following records the rendering handoff.

Implemented `src/looplora/data/render.py`, versioned name assets, and
`tests/test_llm_render.py`. No core, dependencies, data generator, or data
`__init__.py` changes. No commits (WP-E primary owns integration/commits).

## API and choices

- `load_name_pool()` returns immutable names, exact token IDs, model/revision,
  and the reserved missing-information label. `validate_name_pool(tokenizer)`
  validates the entire list against a supplied tokenizer; no model/tokenizer
  imports or file writes occur in renderer imports.
- `render_episode(arrays, episode_index=0, *, tokenizer, name_pool=None,
  task=None, entity_count=None)` and `render_episodes(...)` use WP-B's concrete
  `meta['spec']['name']`, `['V']`, `['base']`. Optional explicit task/vocabulary
  describe array-compatible fixtures; T3 still needs its declared base.
- Result: `context_text`, `context_ids`, valid `fact_spans` with original slot
  indices and half-open character/token bounds, `fact_final_mask`, `queries`.
  Each query has `question_text`, `question_ids`, leading-space `answer_text`,
  `answer_token_id`, `answer_ids` (one token), and original `answer_id`.
- Every fact is one line; its span includes the trailing newline. Write-at-final
  means the last token of that complete line. Padding facts are omitted. Facts
  stay in presentation order, including obsolete/update versions.
- T1/direct, terminal-chain, explicit-hop walk and typed-chain questions follow
  generator semantics; T3 uses the declared base and adds latest/missing
  instructions. `q_hop_true` is never read. `answer_em(first_token, query)` is
  greedy first-token equality.
- `INSUFFICIENT=V` maps to reserved ` Unknown`, one tokenizer token and excluded
  from entity labels. Entity IDs directly index the pool, retaining seen/unseen
  ID partitioning. Question+answer tokenization is checked for a clean boundary.

## Asset provenance and validation

`assets/qwen3_names.v1.json`: 1024 labels (776 given names, 248 surnames), exact
leading-space token IDs, Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Candidates are explicit names extracted
from six Faker v37.12.0 providers; pinned source URLs, blake2b digests and MIT
license included. Selected records are deterministically ordered by
`(derive_seed(0, 'name-pool-order', name), name)` before IDs are assigned, avoiding
given-name/surname ordering aligned with seen/unseen pools. No Faker dependency. Selection and rebuild procedure are
versioned in `assets/build_name_pool.py` and `assets/README.md`. Jobs are offline.

- Plane: login/control `miyabi-g2`; all tokenizer/tests run through
  `scripts/submit.py` on compute. No torch/transformers/pytest login imports.
- Asset build: `scripts/submit.py --name names-llm --walltime 00:05:00
  --worktree . -- src/looplora/data/assets/build_name_pool.py --extract-faker
  /work/xg24i002/x10041/LoopLora/logs/name-candidates-v1`;
  **3711307.opbs**, `mg0044`, exit **0**, 2026-10-26 18:55 JST.
  Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/names-llm-20261026-185441-1154883.log`.
  Initial 3711300 was canceled before execution to move inputs from node-local
  `/tmp` to shared storage; no downloads occur in jobs.
- Initial renderer test command: `scripts/submit.py --name t-llm-render --walltime
  00:05:00 --worktree . -- -m pytest -q tests/test_llm_render.py`;
  **3711336.opbs**, `mg0025`, exit **0**, **16 tests passed**, 19:00:44 JST.
  Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/t-llm-render-20261026-185727-1170483.log`.
- Final asset-permutation build + rendering/generator integration:
  `scripts/submit.py --name t-llm-render --walltime 00:05:00 --worktree . --
  /work/xg24i002/x10041/LoopLora/logs/name-candidates-v1/check_render.py`;
  **3711392.opbs**, `mg0007`, exit **0**, **32 tests passed, none skipped**,
  19:02:37 JST. The runner rebuilds the local asset,
  temporarily adds WP-B's sibling worktree package path, then runs the renderer
  test file. The added 16 generator cases span all six tasks/all T3 bases and
  train/test_unseen, with 32 generated episodes. Final persistent generator tests
  require the normal WP-B import after merge; they no longer skip missing imports.
  Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/t-llm-render-20261026-190140-1211103.log`.

No remaining rendering blockers. Post-merge full WP-E testing belongs to the
primary; this validation imported WP-B's actual worktree source without editing
or copying it. No model training was run.
