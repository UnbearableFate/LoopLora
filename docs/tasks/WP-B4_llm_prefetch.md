# WP-B4 — background batch generation and rendering for LLM training (exact)

Read first: `docs/reports/WP-C5.md` (profile: at batch 128, one process, a P1-L T1 step costs ≈ 343 ms, of which
≈ 154 ms is CPU episode generation and rendering; two processes per GPU reach ≈ 474 ms of preparation),
`src/looplora/llm/train.py` (`_fit`: `generate(spec, "train", step * batch, batch, data_seed)` then
`experiment.prepare(arrays)`), `src/looplora/llm/batching.py`, `src/looplora/data/render.py`. Branch/worktree:
`wp/data` (merge `master` first). Report: `docs/reports/WP-B4.md`.

## Task
- `generate` is a pure function of (split, start, count, seed), so the batch for step s can be produced ahead of
  time. Add a prefetcher that produces, in step order, each step's episode arrays and its rendered CPU tensors
  (everything in `prepare` before the first GPU operation) in background worker processes, with a bounded queue.
  The main loop moves tensors to the GPU and runs the unchanged GPU part.
- Keep the change isolated: a new module plus a minimal hook in `_fit` behind `train.prefetch_workers`
  (default 0 = current behaviour). wk-llm is editing `_fit` for WP-L3; touch as few lines there as possible.
- Exactness tests (on compute): for 50 steps, prefetched arrays and token tensors equal the sequential ones
  exactly, for T1 and T2 tasks, both write policies, and evaluation batches if you route them too; worker
  start-up, shutdown, exceptions and a resumed start step behave correctly.
- Benchmark (on compute, WP-C5 protocol): ms/step and episodes/s for batch 32 / 64 / 128, one and two training
  processes per GPU, prefetch workers ∈ {0, 2, 4, 8}; recommend a setting.

## Limits
No change to the generated data, tokenization or numerics; no new dependencies. One node per job, ≤ 1 h; total
≤ 0.5 node·h. Final line: `WP-B4 done: docs/reports/WP-B4.md`.
