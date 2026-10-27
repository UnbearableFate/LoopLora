# Study WP-L6 — LLM oracle code designs at r = 32 (pilot), pre-registered (draft)

Drafted 2026-10-27 by sn-oracle (Claude Sonnet-5.5) for the orchestrator's review; the registration time is the
orchestrator's commit time. Source: `plan/followup_2026-10-27.md` §3 F5 and §6. Written before any WP-L6 run
(none is possible before the Miyabi maintenance ends, 2026-10-28 22:00 JST). Validation episodes only: no test or
confirmation episode is read. A pass licenses `docs/prereg/P1L-S.md`; it is not a claim about multiple reads.

Information seen before this draft: P1-L Pilot-2 at r = 32 (seed 17, 5,666 steps, validation): every memory reader
at chance, M1-L4 at lr 3e-3 hop-1 full EM 0.1% (the reference, not rerun); the 17,000-step training logs
(Amendment B7); `scripts/analyze_oracle_capacity.py` (one-step hop-2 addressing among the 12 keys: 48% at r = 32,
99.8% at r = 128 with P1-L's random keys; 99% / 100% with rotation keys); WP-L2 (`docs/reports/P1L-diagnosis.md`):
supplying the answer code directly, the output path reaches val EM 98.7% in 1,500 steps at lr 3e-3. No result of any
rotated-key or shared-code run exists.

## Question
At r = r_mem = 32, can an M1-L4 reader learn to use the oracle memory inside the LM when the oracle's keys are
related to its values (O-rot), and, if that is not enough, when the entity codes are also trained with the reader
(O-shared)?

## Fixed design (everything not named here is P1-L's r = 32 `oracle_random` M1-L4 run)
- Backbone `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, bf16; module at layer 14 `o_proj`;
  fp32 adapter and state; ρ = 0.25, α = 1; r = r_mem = 32; fast path; `train.prefetch_workers: 2`.
- Task `t2_chain_end`, E = 12, chains (8, 4), n_rel 4, head sharing 0.3, Q = 4, V = 1024; train hops {1, 2, 3};
  memory-only, facts written from the fact list only (no support text, no support LM call), as P1-L.
- Reader M1 trained at fixed L = 4 (M1-L4), seed 17, oracle seed 17. AdamW, batch 64, 20 warmup steps, weight decay
  0.01, clip 1.0, constant lr 3e-3 (P1-L's selected lr for M1-L4 at r = 32; not retuned), **5,666 steps** (⌊17,000/3⌋,
  P1-L's Pilot-2 length). Both arms use the same reader initialization and the same training episodes.
- Evaluation: split `val`, eval seed 4242, 1,000 episodes, L ∈ {1, 2, 3, 4}, hops 1–8, memory-only, no
  counterfactuals. `configs/wp_l6/{oracle_rot,oracle_shared}.json` differ from `configs/p1l/main/M1-L4-s17.json` only
  in `name`, `writer`, `train.steps` and `eval.depths` (checked by `tests/test_llm_oracle_codes.py`).

## Arms
Let g_e be P1-L's unit random entity codes (`OracleWriter.global_codes`, seed 17; e = 0…1023 plus the
insufficient-information row), and Π_ρ, ρ = 0…3, one fixed orthogonal matrix per relation: the Q factor of a
Gaussian r × r matrix seeded by `derive_seed(17, "oracle", "rotation", ρ)` (float32 draw, float64 QR, the columns of
Q scaled by sign(diag R) so that diag R > 0). A fact (h, ρ, t) is written, in list order, by the unchanged decay-first
delta rule with λ = β = 1 (exact overwrite of the key just written); invalid fact slots never address a table and
are never written.
- **O-rot** (`writer: oracle_rot`): key = normalize(Π_ρ g_h), value = g_t. All codes are fixed. Because Π_ρ is
  orthogonal and g is unit, key(h, ρ) = Π_ρ g_h: the map from a bridge entity's value code to the key of its next
  fact is the fixed linear map Π_ρ (P1-L: an unrelated random key).
- **O-shared** (`writer: oracle_shared`): a trainable table C ∈ R^{1025 × 32}, initialized to g; key = normalize(Π_ρ
  C_h), value = normalize(C_t). C is a writer parameter trained with the reader on the same loss: AdamW at the same
  lr and warmup, **weight decay 0** (the normalization removes scale, so decay would only shrink rows), gradient
  clipping jointly with the reader parameters at 1.0. C is counted separately from the reader's adapter parameters
  (reader counts unchanged: M1 at r = 32 has 101,376 trainable and 103,426 stored parameters; C adds 32,800).
  Rows of names that appear in no training fact (the unseen pool and the insufficient row) receive no gradient and
  stay at g. In any later study the table is trained identically in every reader run that uses it.

## Pass criterion and selection (stated before results; one cell per arm)
- An arm **passes** if, at its final checkpoint (after step 5,666; no early stopping, no checkpoint selection), its
  hop-1 validation full-vocabulary exact-match at L = 4 (memory-only, 1,000 episodes, eval seed 4242) is ≥ 30%.
- Selection reads that cell only (hop 2/3 and other depths are reported, never used):
  1. O-rot passes → select O-rot (a fixed code, closest to the registered oracle).
  2. else O-shared passes → select O-shared.
  3. else stop: P1L-S is not run; the paper reports that at rank 32 the LM path did not learn these oracle codes
     within this budget (one seed, one lr, 5,666 steps; not a proof of impossibility).
- Machine-readable decision: `scripts/run_wp_l6.py decide` → `reports/wp-l6/decision.json`.

## Run rules
- One run per arm, no second attempt, no hyperparameter change after seeing a result. A non-finite loss, evaluation
  value or gradient norm ends that arm as failed (it does not pass). An infrastructure failure (node, OOM, I/O) is
  rerun with the unchanged configuration in a new output directory.
- Order: tests (incl. regression of the existing oracle tests) → 10% time-only smoke (567 steps per arm, derived
  seed and validation seed; scores never read) → `size` (both arms in one 1-node job only if 1.5 × (summed expected
  seconds + 90 s) ≤ 1 h, else one job per arm) → pilot → `decide`.
- Reported for both arms, not gating: all 32 (L, hop) cells, the training log, code drift of C from g (cosine to
  initialization by pool), parameter counts, step times, peak memory, writer and checkpoint hashes.

## Interpretation
- O-rot passes: the unrelated value→key transitions were what the r = 32 reader could not learn. Hop 1 passing says
  the reader uses the memory at all; it says nothing yet about composition.
- O-rot fails, O-shared passes: the random maps at the LM's two ends (question state → key, value code → name) are
  the r = 32 bottleneck; codes the reader can shape remove it.
- Neither passes: stop as above.
- No arm is compared with B4/B8 here. A pass is a design pilot, not evidence for or against multiple reads.

## Limits
Validation only; seed 17; one run per arm. Expected cost from P1-L's measured r = 32 M1-L4 step times (0.188–0.197
s/step): about 1,065–1,115 s of training plus evaluation (≈ 60 s, 32 cells; the smoke measures it) per arm, so
≈ 0.65 node·h for the pilot, ≈ 0.1 node·h for the smoke and ≈ 0.5 node·h for the tests, ≈ 1.3 node·h in all; every
job ≤ 1 node × 1 h (no `--scale-up`); node·h recorded per job.
Executed by a Claude Sonnet-5.5 subagent. Report: `docs/reports/WP-L6.md`.
