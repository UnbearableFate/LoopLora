# P5L pre-registration — second-scale confirmation of the P1-L r = 128 oracle result on Qwen3-1.7B-Base (v1, registered 2026-10-29 05:19 JST)

Status: **registered** by the orchestrator on 2026-10-29 05:19 JST, before any P5L score exists. Drafted by sn-p1l
(WP-P5L step 1, `docs/tasks/WP-P5L_second_scale.md`, wp/p1l d998f29); the orchestrator's decisions on the draft are in
"Decisions at registration" at the end. Nothing below depends on any P5L score: none exists, and none
was computed. The 1.7B timing smokes (`docs/reports/P5L-state.md`) expose only step times and peak memory
(`smoke=True` never evaluates; no loss or accuracy field was read).

## Why and what is claimed
P5 of the plan asks for a confirmation at a second model scale. The result that stands from P1-L
(`reports/decisions/P1L.md`) is the `oracle_random` r = 128 result on Qwen3-0.6B-Base: B7 passes, M1-L4@4 − B4@1 on
hops {2, 3} is +21.55 pp [20.38, 22.76] at equal parameters (Amendment B6 met), G1-L fails for B8 (+0.98 pp) and G2-L
passes. P5L asks whether that result holds on a larger backbone of the same family. The learned-writer line has
stopped (`docs/reports/DPFP.md`; no P3), so P5L is the oracle condition only. The claim, if it holds, has the limits of
P1-L Amendment B7 rule 3: oracle memory, the 768 seen names, trained hops; no equivalence with the symbolic oracle, no
unseen-name claim.

## Information seen before registration
- P1-L confirmation (test [100000, 102000)) and P1L-C (test [102000, 104000), 0.6B) results: `reports/decisions/P1L.md`,
  `docs/reports/P1L-C.md`. No P5L design choice uses them (the design is P1-L's, with the backbone replaced). P5L
  confirms on a test range no study has used or registered (below), so no episode of its confirmation set was scored
  at either scale.
- Timing and memory only at 1.7B: 100-step smokes of all five readers (seed 17, r = 128) and 1,700-step (10%) smokes of
  B8 and M1-L4; no score. Numbers in "Training budget".
- Qwen3-1.7B-Base tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`) have the same
  blake2b digests as Qwen3-0.6B-Base's, so the pinned name pool (`qwen3_names.v1`, built with the 0.6B tokenizer) is
  valid unchanged; `validate_name_pool` still runs against the 1.7B tokenizer at load.

## Design: P1-L's r = 128 oracle condition, except
- **Backbone** `Qwen/Qwen3-1.7B-Base` @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (cached in
  `/work/xg24i002/x10041/hf_home`), bf16, sdpa: 28 layers, hidden 2048, 16 heads × 128, vocab 151,936. The module is
  at layer 14 `o_proj` (the same relative depth as the 0.6B's 28 layers); its sizes are read from the loaded model:
  **in 2048 (16 × 128) → out 2048** (0.6B: 2048 → 1024). Adapters and states stay fp32.
- Selected by the config field `backbone: {id, revision}` (`configs/p5l/main-r128/*.json`); absent = the pinned 0.6B
  default, unchanged (tests prove the off path: recorded P1-L training config hashes and the exact `from_pretrained`
  calls). Training uses `scripts/run_p5l.py`; P1-L's `fit` is reused unchanged.
- **Writer** `oracle_random` only (oracle seed 17, identical memory across readers and seeds, as P1-L). **Rank**
  r = r_mem = 128 only.
- **Readers** B3-L4, B4, M1-L4, B8, B5-L4, seeds 17/29/43 (15 trainings). B0 (frozen base, no training) is evaluated.
  **B1 and B2 are not run**: P1-L reported them (r = 32 only, as context-retained controls) and used them for no gate.
- **Steps** S = 17,000 for every reader, as P1-L, so the two scales are comparable. Batch 64, warmup 20, weight decay
  0.01, clip 1.0, constant lr, `prefetch_workers` 2, task `t2_chain_end` (E = 12, chains (8, 4), head sharing 0.3,
  n_rel 4, train hops {1, 2, 3}; evaluation hops 1–8), depths [4] (B4 L = 1), all as P1-L's r = 128 configs.
- **Learning rates** carried from the P1-L r = 128 lock (`reports/p1l/r128/lr-S-lock.json`): **0.003 for B8, M1-L4, B4
  and B5-L4; 0.001 for B3-L4**. There is **no lr pilot**; this is a carried choice, not a selection at 1.7B. If a
  reader fails to learn at the carried lr, that is part of the result (B7), not a reason to retune.
- **Confirmation set**: split `test`, episodes [104000, 106000) (disjoint from P1-L's [100000, 102000), P1L-C's
  [102000, 104000) and P1L-S's registered [110000, 112000)), eval seed
  4242, 2000 episodes, memory-only protocol, hops 1–8, primary and `t2_typed` settings as P1-L. It is evaluated only
  after all 15 trainings have finished and the committed authorization names it (below).
- **Intervals** 97.5% (1 − 0.05/2; two gated contrasts, B6's and G1-L's), paired episode bootstrap with 2000 draws and the
  P1-L bootstrap seeds scheme; per-seed values reported.

## Endpoints and decision rules (P1-L's, per rank 128 only)
All on the confirmation set, memory-only, full-vocabulary EM, hops {2, 3} unless stated.
1. **B7 memory-use check (rule 1)**: hop-1 EM of B4@1, B8@4 and M1-L4@4 each minus B3-L4@4, 97.5% CI lower bound > 0 for
   all three.
2. **B6 equal-parameter contrast**: M1-L4@4 − B4@1 ≥ 3 pp, 97.5% CI > 0, positive in every seed (M1-L4 and B4 have the
   same adapter parameters). Hop-1 (M1-L4@4 − B4@1 ≥ −1 pp, point) and the same contrast on `t2_typed` are reported, not
   part of the decision, as P1-L reported them for M1-L4.
3. **G1-L** (B8@4 − B4@1 ≥ 3 pp, CI > 0, every seed; hop 1 ≥ −1 pp; typed replication CI > 0) and **G2-L** (B8@4 above
   B5-L4@4 and B3-L4@4, CI > 0), reported as in P1-L; they do not enter the decision below.
4. Secondary, descriptive: M1-L4 against B5-L4 and B3-L4, M1-L4 depth scan L = 1…8, walk specificity, B8 − M1-L4,
   `swap_successors`, per-hop tables, adapter parameter counts and inference time per question (as P1-L). Cross-scale
   (descriptive, no test): the 1.7B contrasts next to P1-L's 0.6B r = 128 contrasts. The per-step readout (J-lens) and
   the P1L-C causal analyses are not part of P5L.

**Decision use** (the task's, applied after the confirmation):
- B7 fails: "the readers did not learn to use the oracle memory at 1.7B"; the other endpoints are reported, not
  interpreted.
- B7 passes and B6 passes: the equal-parameter result holds at a second scale (oracle condition, seen names, trained
  hops; B7 rule 3 limits apply).
- B7 passes and B6 fails: LP4 is narrowed to 0.6B.
No claim is made about B8 beyond G1-L/G2-L as reported; no learned writer, no unseen names, no P3.

## Training budget (measured at 1.7B; one 1-node job per run)
Step times (steady state, step > 5) from the smokes (`docs/reports/P5L-state.md`), peak GPU memory
12.77 GB (B8, M1-L4), 12.73 GB (B4), 12.76 GB (B5-L4), 12.27 GB (B3-L4); no growth between 100 and 1,700 steps.

| Reader | s/step | 17,000 steps | × 1.5 walltime |
| --- | ---: | ---: | ---: |
| B8 | 0.2359 (1,700-step smoke; 0.2388 at 100) | 1.11 h | 1.67 h |
| M1-L4 | 0.2321 (1,700-step smoke; 0.2382 at 100) | 1.10 h | 1.64 h |
| B4 | 0.1791 (100) | 0.85 h | 1.27 h |
| B5-L4 | 0.2025 (100) | 0.96 h | 1.43 h |
| B3-L4 | 0.1813 (100) | 0.86 h | 1.28 h |

The largest reader fits one 2 h job (1.67 h < 2 h), so the pipeline needs no interrupted-run/resume path (none exists:
checkpoints are written at the end of a run and `train` refuses an existing output directory). Each job trains one
reader at one seed (`scripts/submit.py --scale-up docs/prereg/P5L.md`, ≤ 1 node × 2 h, at most 6 jobs at once). A job
killed at its walltime is rerun from scratch under a new output directory with the same config and seed, at the same
source commit, and the event is recorded; equality of a rerun with an uninterrupted run is not assumed or claimed.
**Projected main training: 14.6 node·h** (sum of the measured step times × 17,000 × 3 seeds) plus about 0.2 node·h
start-up and checkpointing over 15 jobs = **14.8 node·h**.

## Confirmation budget (projection; evaluation is not measured at 1.7B)
P1-L's r = 128 confirmation cost 0.94 node·h for the five reader groups at 0.6B (B8 0.24, M1-L4 0.61, B4/B5-L4/B3-L4
0.03 each), plus about 0.05 for B0. Forward-only cost at 1.7B is bounded by about 3.2× (non-embedding parameters 1.4 B
vs 0.44 B) and was about 1.2× for training steps (the adapter loop probably dominates them; not measured): **projected 1.2–3.2 node·h**.
Pre-confirmation checks, the evaluation-time guard and finalizers cost about 0.1 node·h. **Projected total after
registration: 16.1–18.1 node·h**, plus the 0.36 node·h already spent on step 1 (tests and smokes, ≤ 0.5 budget).

## Provenance and order
1. The orchestrator registers this file on master. `reports/p5l/r128/lr-S-lock.json` (S 17,000 and the carried lrs
   above, `status: locked`) is committed before the first main training; `scripts/run_p5l.py main` checks that the
   lock, the prereg file and the configs agree. Jobs are pinned to a source commit (`submit.py --commit`).
2. All 15 trainings finish and are verified (complete records, checkpoints, the unchanged oracle memory hash).
3. A confirmation authorization (condition `p5l-r128`, 97.5%, window `p5l` = test [104000, 106000)) is committed;
   the evaluator and finalizer are the P1-L ones, extended (not changed) for the study profile, the named window and
   the 1.7B backbone read from each checkpoint's config. No confirmation episode is generated before that commit.
4. Finalizer outputs, decision record `reports/decisions/P5L.md` and report `docs/reports/P5L.md` follow.

## Decisions at registration (orchestrator, 2026-10-29 05:19 JST)
- **B6 reading accepted** as written: the decision contrast is M1-L4@4 − B4@1 on hops {2, 3}, ≥ 3 pp, 97.5% CI > 0,
  positive in every seed; its hop-1 and `t2_typed` values are reported, not required. This is how B6 was applied in
  `reports/decisions/P1L.md` §3, so the two scales are judged by the same rule.
- **97.5% kept** from P1-L for comparability (two gated contrasts, B6's and G1-L's).
- **Confirmation window changed** from the draft's [102000, 104000) (P1L-C's, already scored at 0.6B) to the unused
  [104000, 106000); `CONFIRMATION_WINDOWS` gets `p5l: 104000`.
- **Smoke deviation accepted**: the 10% (1,700-step) smoke ran for B8 and M1-L4 only, and 100-step timing smokes for the
  other three readers, to stay within the 0.5 node·h step-1 budget; the 100- and 1,700-step step times of B8 and M1-L4
  agree within 2%. Step 1 used 0.36 node·h.
- **No resume path**: the slowest reader requests 1.67 h, inside one 2 h job; a run killed at walltime is rerun from
  scratch as stated above.
- **Step 2**: sn-p1l builds the evaluator/finalizer extension (study profile, window, B0, backbone from the checkpoint
  config) with tests, and may time a validation-only evaluation on the smoke checkpoints to replace the confirmation
  projection; neither reads a confirmation episode.
