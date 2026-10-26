# WP-E — Qwen3 injection, memory protocols, text rendering, throughput pilot (plan T-07, §7.5–§7.6, §10.6)

Read first: `docs/README.md`, `docs/notation.md` §4–§5, `docs/tasks/WP-A_core.md` (interface you build on),
`docs/tasks/WP-B_data.md` (episode arrays you render). Branch/worktree: `wp/llm` at
`/work/xg24i002/x10041/LoopLora-wt/llm`. Report: `docs/reports/WP-E.md`.

WP-A (adapters, `AdaptedLinear`) and WP-B (episodes) are being built in parallel. Start with the parts that do
not need them (model loading, hook points, name pool, templates, protocol drivers with a zero branch), then
integrate when the orchestrator tells you `wp/core` / `wp/data` are merged into `master` (merge `master` into
your branch then).

## Fixed choices
- Backbone: `Qwen/Qwen3-0.6B-Base`, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, from
  `HF_HOME=/work/xg24i002/x10041/hf_home`, offline. bf16 weights, `attn_implementation="sdpa"`; the adapter and
  its memory state run in fp32. 28 layers, `o_proj: 2048 → 1024`.
- Insertion: `model.model.layers[ℓ].self_attn.o_proj` replaced by `AdaptedLinear`; default ℓ = 14; support a
  list of layers (later multi-layer stage). Backbone parameters frozen (`requires_grad=False`), but no
  `no_grad`/`detach` on activations: gradients must reach adapters through frozen upper layers.

## Deliverables (`src/looplora/llm/`, `src/looplora/data/render.py`)
1. `render.py`: entity name pool of ≥ 1024 names that are single tokens with a leading space in the Qwen
   tokenizer (prefer given names; record the selection rule and the list as a versioned asset under
   `src/looplora/data/assets/`); relation phrases; templates for `t1`, `t2_chain_end`, `t2_walk`, `t2_typed`,
   `t3_*` rendering `EpisodeArrays` into (context text, per-query question text, answer text = one name
   token). Every fact is one line; record per-fact token spans (needed for "write at fact-final token" masks).
   Answer EM = greedy first answer token equals the name token.
2. `inject.py`: attach/detach adapters, collect per-module states, set a per-call `MemoryContext` for all
   modules (states, loop_steps, valid/write/reset masks, mode).
3. `protocols.py` (batched, padding-aware):
   - `encode_support(model, context_ids, attn_mask, write_policy)` → per-module memory snapshots;
     `write_policy ∈ {all_tokens, fact_final}`; backbone KV discarded (memory-only) or returned
     (context-retained).
   - `answer_scores(...)`: teacher-forced loss/logits on answer tokens for memory-only (question run as a fresh
     sequence reading the snapshot, writes off) and context-retained (context KV kept, question/answer
     tokens read-only) protocols, for a given L; each question forks the same snapshot.
   - `greedy_answer(...)`: incremental decode with KV cache under the same protocols (fixed answer budget).
4. `bench.py` + `configs/pilot_throughput.yaml`: throughput pilot on one GH200 (plan §10.6): for r ∈ {16, 32},
   L ∈ {0, 1, 2, 4, 8}, batch ∈ {16, 64}: support encoding (with writer) and question reading time, tokens/s,
   peak memory, and one reader-training step time (backward through layers ℓ..27) — warmup, CUDA sync, ≥ 5
   repeats; write `runs/pilot/throughput-<stamp>.json`. Submit it as one job (≤ 20 min, debug-g).

## Tests (`tests/test_llm_*.py`, compute node; load the model in fp32 for tight tolerances)
- B = 0 adapter: logits equal the unmodified model (both protocols).
- Teacher-forced logits equal incremental-decode logits at every answer position (both protocols, L ∈ {0,1,4}).
- Base `o_proj` called once per token for any L; writes happen only on tokens allowed by the policy.
- Episode reset: two episodes processed in one batch row with reset equal separate runs; left/right padding
  does not change states or logits of real tokens.
- Two questions forked from one snapshot do not affect each other; question tokens never write.
- Gradients reach adapter params through frozen layers; backbone params receive no grad.
- Name pool: every name is one token with a leading space; no duplicates; rendering round-trips spans.
Run: `scripts/submit.py --name t-llm --walltime 00:20:00 --worktree . --wait -- -m pytest -q tests/test_llm_*.py`.

## Constraints
No training runs in this WP beyond the pilot's timing step. No downloads in jobs. Do not edit `src/looplora/core`
(ask the orchestrator if its interface blocks you).
