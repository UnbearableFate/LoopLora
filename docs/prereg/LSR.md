# Study LSR: label-free stop rules on the r = 128 M1-L4 checkpoints at two backbones, pre-registered

Registered 2026-10-29 13:09 JST by the orchestrator, before any LSR code, run, validation score or test episode. Drafted by
sn-core (Claude Sonnet 5.5; WP-L9 step 1, `docs/tasks/WP-L9_stop_rules.md`, wp/core `e206189`) on the login node from code and
committed files only. The user approved G1–G4 on 2026-10-29 13:00 JST ("OK,启动接下来的G1-G4"; `plan/followup_2026-10-29.md` §4).
Proposal R1 of `docs/review/rv-fable_2026-10-29.md`. Evaluation only: no training, no gradient step, no new checkpoint. The
orchestrator's decisions are in "Decisions at registration" at the end.

## Question
The (hop, L) matrix of the M1-L4 checkpoint peaks at L ~ h + 1 to h + 2 for hop h and decays afterwards
(fable §1.3, 0.6B: hop 1 is 64.2% at L = 2 and 47.9% at L = 4; hop 4 is 7.2% at L = 4 and 16.6% at L = 6). The deployed L = 4
under-reads deep hops and over-reads hop 1. Can a per-question stop rule that uses no labels raise hops 4-8 over the fixed
L = 4 without a large loss on hop 1? "Falsified" means: no rule does it at either backbone.

## Information seen before registration
- The (hop, L) matrix of M1-L4 (seed mean) on the **P1-L confirmation range** test [100000, 102000)
  (`reports/p1l/r128/accuracy.csv`, tabulated in fable §1.3; 1.7B only its best cells). Nothing of LSR's test range
  [106000, 108000) was generated or scored by anyone. No LSR validation episode was scored either.
- The symbolic stopping results (`docs/paper/evidence_symbolic.md` H1-H6; `docs/reports/P4S-stopping.md`; `docs/reports/CVH.md`):
  relative-change stopping helped long hops of the symbolic readers and kept a hop-1 loss; trained halting heads failed. They
  fix conventions below (first qualifying post-update state, read norm normalised by the initial read) but nothing here
  depends on a symbolic number.
- Timing only: the `primary/timing.json` files of the P1-L and P5L confirmation runs (seconds per depth). No accuracy field of
  any confirmation output was opened by this draft.

## Fixed design (the task's; deviations are flagged in "Flagged deviations")
- **Checkpoints** (r = 128, `oracle_random`, oracle seed 17, S = 17,000, lr 0.003, trained at L = 4, read from the committed training
  manifests; each directory holds `checkpoint.pt`, `config.json`, `record.json`):
  - Qwen3-0.6B-Base @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` (P1-L, `runs/p1l/main/r128/training-manifest.json`), bf16,
    sdpa, layer 14 `o_proj` (2048 -> 1024):
    `runs/p1l/main/r128/main/M1-s{17,29,43}/M1-L4-s{17,29,43}`.
  - Qwen3-1.7B-Base @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (P5L, `reports/p5l/r128/training-manifest.json`), layer 14
    `o_proj` (2048 -> 2048): `runs/p5l/main/M1-L4-s{17,29,43}/M1-L4-s{17,29,43}`.
  - All under `/work/xg24i002/x10041/LoopLora/`. Six checkpoints, seeds 17, 29, 43. The evaluator checks `record.json`
    `status == complete` and `checkpoint_hash` against the file (blake2b), as `evaluate_p1l_causal.check_runs` does, and that the
    config names the registered backbone. The training manifests are the authority; nothing is retrained.
- **Task and episodes**: `t2_chain_end`, E = 12, chains (8, 4), head sharing 0.3, n_rel 4, memory-only protocol, one query per
  hop 1-8 per episode (`stratify_queries`), so 2,000 episodes give 2,000 questions per hop. Evaluation config = the P1-L
  confirmation template (`configs/p1l/main/eval-confirmation.json`) with its splits changed, so the task spec and every episode
  seed equal P1-L's (as `evaluate_p1l_causal.evaluation_config` asserts). Oracle memory is identical across readers and seeds.
- **Validation set (thresholds are chosen here only)**: split `val`, episodes **[1000, 3000)**, eval seed 4242, 2,000 episodes,
  16,000 questions per checkpoint, via `generate(spec, "val", 1000, 2000, 4242)`. [0, 1000) is the range P1-L and P5L used for
  their lr selection and end-of-training validation, so it is not reused.
- **Test set**: split `test`, episodes **[106000, 108000)**, eval seed 4242, 2,000 episodes, 16,000 questions per checkpoint. New
  named window `lsr`: `CONFIRMATION_WINDOWS["lsr"] = 106000` in `src/looplora/llm/config.py`; the entries `p1l` 100000, `p1l-s` 110000
  and `p5l` 104000 stay as they are. The range is disjoint from every used or registered range ([100000, 106000), P1L-C's
  [102000, 104000), P1L-S's [110000, 112000), WP-L10's [108000, 110000)). It is read once per checkpoint, after this prereg is
  registered and after `reports/lsr/thresholds.json` is committed (below).
- **Depth cap**: L_max = 8 (the deepest depth P1-L scanned; the M1-L4 depth scan used L = 1..8).

## What is read and where it comes from (all hooks exist; the new code only composes them)
Notation as P1L-C: a row is one scored question; its **final position** is its last real question token (`p1lc.last_positions(
prepared.question_mask)`). The module reads the fixed memory M at every question token. u is the adapter's projection A x of the
layer-14 `o_proj` input at that token. The stop is decided from the final position's trajectory.
- **z^s**, s = 0..8: z^0 = u; z^1..z^8 from `looplora.llm.p1lc.read_states(adapter, u, memory, 8)`, the loop of
  `IterativeQueryAdapter._read` (`z <- u + rho * relu(R z + _query(z))`) whose last state equals `adapter.read` bit for bit
  (`tests/test_p1lc.py`). Rows and memory are the arguments of `adapter._read(u, M_prev, loop_steps)` while one scoring call runs; the
  capture uses the patch pattern of `p1lc.read_control` (`process_sequence`, `_masked_branch`, `_read` patched for one call,
  restored on exit, one cursor to pick the final position). Because the memory snapshot and u do not depend on L (Gate 0 of
  `llm.evaluate`), z^s of a question does not depend on the depth the pass runs at.
- **M Q z^s**: `adapter._query(z^s, memory, adapter.Q, adapter.U)` = U M q(Q z^s); `q` is the identity for these M1 checkpoints
  (`unit_query` False, `dpfp_nu` 0). The P4-S convention is "UMQz" (`core.stopping._read`), so the norm includes U, the vector that
  enters the next update. r^s = that vector, s = 0..7 (r^0 from z^0 = u).
- **Logit lens**: the answer-position top-1 token of the deployed model when every position reads s times,
  `experiment.score(prepared, s)[0].argmax(-1)` for s = 0..8 (full vocabulary, as the primary metric). This is the same quantity
  as the lens of `scripts/readout_p1l.prefix_reader` + `score_answer_branches`, which that script asserts equal to the deployed
  logits at s = 4 (it restricts the argmax to the 1,024 name tokens for its readout table; here the full vocabulary, so that the
  lens top-1 is the deployed prediction). p^s denotes it.
- **One pass per batch** (batch 16 episodes, as P1-L): `score(prepared, s)` for s = 0..8 (plain), plus the trace of the L = 8
  pass at the final rows. Audits that raise on failure: the traced z^8 reproduces the plain L = 8 logits exactly (as
  `p1lc.bridge_patch_samples` audits its capture), and the memory/x/u hashes are unchanged across depths.

## Rules (per question, from the final-position trajectory; the stop depth s* in {1..8}; s = 8 always fires)
All three return the state z^{s*} and its prediction p^{s*}; "stop at s*" means every question position reads s* times
(the question is scored at fixed depth s*).
1. **R1 relative state change**: fires at s if ‖z^s - z^{s-1}‖ / ‖z^{s-1}‖ <= tau1. Cost s* reads.
2. **R2 logit-lens stability**: fires at s >= 2 if p^s = p^{s-1} = p^{s-2} (two consecutive unchanged steps; p^0 is the L = 0
   prediction). No threshold. Cost s* reads and s* + 1 passes through layers 15-28 (one per lens evaluation, L = 0..s*).
3. **R3 read norm**: fires at s if ‖r^s‖ / ‖r^0‖ <= tau2. Cost s* + 1 reads (the read at z^{s*} is paid), as P4-S.

By construction the stopped prediction of a question is p^{s*}, read from the fixed-depth scan, so every rule's EM is exactly
computable from one scan plus the trace; there is no separate stopped scoring pass for the statistics (the latency audit below
checks the equivalence).

## Threshold selection (objective stated before any validation score exists)
Applies to R1 (tau1) and R3 (tau2); R2 has no free parameter. Done once per (rule, backbone) on the pooled validation questions
of the backbone's three checkpoints (**one threshold shared by the three seeds**, as P4-S and CVH did), offline from the stored
validation traces by numpy.
- **Grid**: tau in {2^(k/2) : k = -20, -19, ..., 4} (25 values, about 0.001 to 4), fixed here for both rules.
- Let EM_h(rule, tau) be the validation EM of hop h, pooled over the three seeds, and EM_h(L4) the fixed L = 4's.
- **Feasible**: tau is feasible if EM_1(rule, tau) - EM_1(L4) >= -8 pp pooled, and >= -10 pp in each seed. (The decision bound is
  10 pp; the 2 pp margin on the pooled value is about 1.6 paired standard errors at 2,000 questions and protects the test.)
- **Objective**: J(tau) = mean over hops 4-8 of EM_h(rule, tau) (pooled, equal weight per hop).
- **Choice**: tau* = argmax of J over feasible tau; ties on J (exact) go to the lower mean reads, then to the smaller k.
- **No feasible tau**: the rule fails for that backbone. Its test statistics are never computed (the CVH precedent).
- The same objective and constraint pick the descriptive control **L_c**, the best *fixed* depth in 1..8 on validation, so the
  report can say whether adaptivity adds anything over choosing another fixed depth.
- Nothing about hops 2-3 enters the choice. Validation labels are used for the choice only; the deployed rule is label-free.
- The six thresholds (and R2's null entry), the grid, the objective, the validation EMs that led to them and every grid value's
  validation numbers are written to `reports/lsr/thresholds.json` (`status: locked`, with the source commit and blake2b of the
  validation trace files) and committed. The orchestrator checks that commit. No test episode is generated before it.

## Test evaluation and endpoints
Per backbone and checkpoint on test [106000, 108000): one scan (s = 0..8) plus the trace, then the three rules at their locked
thresholds, L = 4 (the primary comparator), L = 8, and L_c.
- **Rule statistic** (per backbone): for hop set H, d_e = mean over seeds and h in H of (correct of the rule - correct of fixed
  L = 4) on episode e. The pooled difference is mean_e d_e, in pp.
- **Intervals**: paired episode bootstrap (whole-episode joint resampling over the three seeds, hops and rules), **10,000 draws**,
  seed derive_seed(2718, "lsr", backbone), level **1 - 0.05/6 = 99.17%** (two-sided percentile; 3 rules x 2 backbones, as CVH's
  six comparisons). Per-seed values are reported with their own 99.17% intervals; descriptive tables carry 95% intervals.
- **A rule passes at a backbone** if all hold:
  1. hops 4-8: pooled gain over fixed L = 4 is >= +3 pp and the 99.17% interval is above 0;
  2. hop 1: pooled change against fixed L = 4 is >= -10 pp (point estimate; the interval is reported);
  3. per seed: the hops 4-8 gain is > 0 and the hop-1 change is >= -10 pp in every one of seeds 17, 29, 43;
  4. a feasible validation threshold existed (R2: no condition).
- Reported, not gating: hops 1-3 pooled change against L = 4 (and hops 2 and 3 separately); EM by hop at the stopped depth for every
  rule and for L = 1..8 (the full test (hop, L) matrix per backbone and per seed, which the paper needs); stop-depth distribution by hop;
  mean reads per question overall and by hop; passes through layers 15-28 for R2; rule minus fixed L = 8 and rule minus L_c on hops
  4-8 and hop 1.
- **Oracle stops (labels used, descriptive only, never a comparator in a decision)**: (O1) best fixed L per hop chosen on validation,
  its test EM by hop (pooled hops 1-3 and 4-8); (O2) best cell per hop of the test matrix, the optimistic ceiling in fable §1.3; (O3)
  per-question "any L in 1..8 correct", an upper bound inflated by chance agreement.
- **Latency** (descriptive, compared within one job only, no latency claim; H2 of the symbolic evidence found fewer reads did not mean
  faster): in the validation job, 128 validation episodes (validation [1000, 1128)), per checkpoint, wall seconds per question of
  fixed L = 4, fixed L = 8 and a two-phase stopped pass for R1 and R3 (phase 1 traces the final positions to L = 8 and fixes s*;
  phase 2 scores each question at fixed depth s* with rows grouped by s*), 2 warmups and 5 repeats, CUDA-synchronised. The same batches
  check that the phase-2 argmax equals the assembled p^{s*}; more than 0.2% disagreeing questions (numerical ties only are expected)
  stops the job for the orchestrator. R2's early-exit is not timed: each step needs a pass through layers 15-28; its pass count is reported.

## Decision rule and use of results (stated before results)
- The prediction recorded from fable R1 (not a gate): R3 gives hops 4-6 >= 12% and hop 1 >= 55% at 0.6B. As a check on how much room the
  gate leaves: at 0.6B the fixed L = 4 reaches about 5.1% pooled on hops 4-8 and the best fixed L per hop about 12.6% (fable §1.3, P1-L
  range, so +7.5 pp is the ceiling of a per-hop depth choice); the +3 pp gain needs about 40% of it.
- **A rule passes at a backbone** by the three conditions above. Outcomes are per (rule, backbone):
  - passes at one backbone: "a label-free per-question stop at deployed-checkpoint level raised hops 4-8 over fixed L = 4 at that
    backbone in the oracle-memory condition (r = 128, seen names, the checkpoints' trained task); evaluation only". The report states the cost in
    reads and that L_c, the best fixed depth under the same constraint, is the fair alternative (the gain over L_c is reported with its interval).
  - passes at both backbones: additionally "at both scales".
  - a rule that passes against L = 4 but does not exceed L_c is reported as a better *depth* rather than as an adaptive stop.
- **No rule passes at either backbone**: fable's R1 is falsified for these rules on these checkpoints. The paper keeps the descriptive
  (hop, L) matrix and the oracle stop row, states that none of the three label-free signals closed the gap, and that the missing stop
  remains open (symbolic trained halting heads also failed: CVH, HL). No further stop-rule study is registered by this one.
- No gate of P1-L, P5L or P1L-C depends on LSR. Limits carried from P1-L Amendment B7 rule 3 apply: oracle memory, the 768 seen names,
  no unseen-name claim, no learned writer, no training claim. The intervals condition on the three trained seeds per backbone and one
  oracle seed and layout; they do not cover new training seeds.

## Flagged deviations and interpretations (the task's fixed design is exact where not listed)
1. **R1 look-back, not look-ahead.** The task writes ‖z^{s+1} - z^s‖ / ‖z^s‖ < tau1 at step s. Implemented as the registered P4-S rule
   (`core.stopping.halting_steps`, "first qualifying post-update state"): ‖z^s - z^{s-1}‖ / ‖z^{s-1}‖ <= tau1, stopping at s, no extra
   read. The two fire on the same quantity, shifted by one state. Reason: reuse of the validated convention, equal cost accounting
   (s* reads), and the s = 8 cap needs no z^9. `<=` as in P4-S (differs from "below" only on exact ties).
2. **R3 normalised and with U.** The task writes ‖M Q z^s‖ < tau2. Implemented as ‖r^s‖ / ‖r^0‖ with r = U M Q z, the P4-S
   `read_norm` (UMQz^L / UMQz^0), cost s* + 1 reads. Reason: the absolute norm depends on the scale of u at each question's token and on U, which has
   no meaning separate from the update; the ratio is scale-free and keeps the grid usable.
3. **Per-question stop from the final position.** The module reads at every question token. The stop depth is decided from the
   final position's trajectory and then applied to all positions of the question (the question is scored at fixed depth s*), which is
   what one fixed-L scan with a per-question L means and keeps the comparison to fixed L exact.
4. **R2 detail.** "Top-1 unchanged for two consecutive steps" is read as three equal predictions p^{s-2} = p^{s-1} = p^s with p^0 the
   L = 0 prediction (patience 2 of the P4-S `output_stability` rule); the earliest stop is s = 2; no threshold is selected. Top-1 is the
   full-vocabulary argmax of the deployed model at depth s. Its true cost includes s* + 1 upper-layer passes, which the read count hides.
5. **Selection details the task does not fix**: grid, objective J, the 8 pp validation margin, one threshold per (rule, backbone)
   shared across seeds, infeasible rules not scored on test, the validation range [1000, 3000). Reasons above.
6. **Decision-rule details the task does not fix**: gain >= +3 pp (P1-L's margin; fable's text has no number), the interval (99.17%,
   Bonferroni over 3 rules x 2 backbones), hop 1 judged on the point estimate and in every seed (the task states a point loss; the
   P1-L G1-L hop-1 condition is also a point estimate), per-seed condition = gain > 0 and hop-1 loss <= 10 pp in each seed.
   Bonferroni rather than 97.5% because six one-sided claims are tested; the cost is power.
7. **10,000 draws** instead of the 2,000 of P1-L and CVH: the 0.42% tail of a 99.17% interval has about 8 draws at 2,000.
8. **Descriptive additions** not in the task: L = 8 and L_c comparisons, the O1-O3 oracles (the task asks only for the per-hop best L),
   the test matrix, hops 1-3 split, R2's pass count.
9. **Existing test to update.** `tests/test_p1ls_pipeline.py::test_confirmation_windows_are_named_registered_and_disjoint` compares the
   whole `CONFIRMATION_WINDOWS` dict, which any added window breaks by design. WP-L10 (registered on master 3c9c396) adds `b8w: 108000`
   to the same dict. Per the orchestrator's instruction, this study's code and tests assert each registered entry individually
   (`p1l` 100000, `p1l-s` 110000, `p5l` 104000, `lsr` 106000) and never compare the whole dict; the existing whole-dict assertion is edited to the
   same per-entry form, plus pairwise disjointness of the named windows and of [102000, 104000). The two branches then conflict only on the
   dict literal, which the orchestrator resolves at merge.

## Order of operations
1. The orchestrator registers this file on master. Step 2 on wp/core: `src/looplora/llm/lsr.py` (trace capture, rules, selection,
   pure numpy, torch only in the capture), `scripts/evaluate_lsr.py`, `scripts/analyze_lsr.py`, the `lsr` window with its test,
   `configs/lsr/*.json`, tests (below). The design lock `reports/lsr/design.json` records this prereg's blake2b, the grid, the
   checkpoints (with `checkpoint_hash`), the validation and test ranges.
2. Tests and smoke on compute (submit.py with `--commit`): unit tests (rules on synthetic traces; the traced z^8 and r^s equal
   `read_states`/`_query` on a random adapter; the stopped assembly equals the per-question fixed-depth prediction; the grid and tie-breaks;
   the window and disjointness; the off path: `evaluate_p1l_main`/`p1lc` outputs and every existing config hash unchanged), then a
   smoke on val [0, 64) per backbone for timing and the audits (its numbers are never used).
3. Validation job per backbone (3 checkpoints, val [1000, 3000)) with the latency audit; `analyze_lsr.py select` writes
   `reports/lsr/thresholds.json`. STOP: the orchestrator commits it (the lock).
4. A committed test authorization names the prereg and thresholds hashes, the window `lsr`, k = 6 and 99.17%. The evaluator refuses to
   generate any test episode without it, and refuses a second read of a checkpoint's test range (output directory must not exist).
5. Test job per backbone, then `analyze_lsr.py` (endpoints, bootstrap, tables) and `docs/reports/LSR.md`, `docs/reports/LSR-state.md`.

## Cost estimate (measured P1-L / P5L evaluation times; nothing here has run)
Measured, `primary/timing.json` of the confirmation runs, M1-L4, 2,000 episodes x 8 hops, depths 1-8 scanned in one run
(`evaluation_seconds`, of which reader / writer): 0.6B 102.2 / 100.4 / 102.0 s (reader 75.3 / 73.9 / 75.2; writer 4.3); 1.7B 111.2 / 111.0 /
110.9 s (reader 84.6 / 84.1 / 83.7; writer 4.2-4.4). Reader seconds per depth: 0.6B 8.4 (L = 1, with warm-up), 7.4 (L = 2), rising about 0.73 s
per depth to 11.8 (L = 8); 1.7B 9.5, 8.4, rising about 0.75 s to 13.0. The rest of `evaluation_seconds` (about 21-23 s) is generation,
tokenisation and the Gate 0 hashing. P1L-C ran three readers per job in about 150 s on 2,000 episodes at L = 4.
- Per checkpoint and split: this scan + the L = 0 pass (about 6 s at 0.6B, 7 s at 1.7B, from the slope) + the trace (up to 5 s, assumed:
  final rows only) = about **113 s (0.6B), 123 s (1.7B)**.
- Per job (3 checkpoints, one split) with about 60 s for model and checkpoint load (assumed): 0.6B 400 s, 1.7B 430 s, so 0.11 and 0.12 node·h.

| Step | Jobs (1 node each, <= 1 h) | node·h |
| --- | --- | ---: |
| Tests and smoke (val [0, 64)) | 1-2 | 0.10 |
| Validation scan + latency audit (128 episodes, 6 checkpoints, about 0.04) | 2 | 0.27 |
| Test scan | 2 | 0.23 |
| Analysis (bootstrap, CPU work in a job) | 1 | 0.02 |
| **Projected total** | 6-7 | **0.62** |

Cap: **0.8 node·h in total, tests included**, each job 1 node and <= 1 h (requested walltime = measured x 1.5, about 10-12 min per scan
job), no `--scale-up`. Contingency of 0.18 covers one failed or rerun scan job. If the smoke's measured times project above 0.8, the
orchestrator decides before the validation jobs. node·h are recorded per job in `docs/reports/LSR-state.md`. Raw outputs under
`runs/lsr/`, committed summaries under `reports/lsr/` (thresholds, per-hop and per-depth tables as CSV, `results.json`).

## Provenance and rules
- Every digest uses the repository's blake2b helpers (`digest`, `tensor_hash`, `stable_hash`); no SHA-256. Seeds by `derive_seed`.
- Evaluation only: no optimizer, no `requires_grad` change, no checkpoint is written. The new module patches adapter methods inside a
  context manager and restores them (`p1lc.read_control`'s contract); a test shows the plain evaluation path is bitwise unchanged.
- Executed by sn-core (Claude Sonnet-5.5 subagent) on wp/core; every job through `scripts/submit.py --commit <full hash>`.

## Decisions at registration (orchestrator)
1. **Flags 1–9 are accepted as written.** They cover R1 look-back, R3 as ‖UMQz^s‖/‖UMQz^0‖, the stop decided at the final
   position and applied to the whole question, R2 with patience 2, the selection details, the decision-rule details,
   10,000 draws, the descriptive additions, and the per-entry window test.
2. **Gate thresholds** (open question 1): accepted.
   - The hops 4–8 gain is at least +3 pp, with the 99.17 % interval above 0.
   - The 8 pp pooled margin applies on validation.
   - Hop 1 is judged on its point estimate (≥ −10 pp pooled and in every seed), and its interval is reported.
3. **R2 cost** (open question 2): accepted. R2's pass count is reported and R2 is not timed. The report states that R2 needs
   s* + 1 passes through layers 15–28 per question. No latency or efficiency statement is made for R2, and none is made for
   any other rule beyond the descriptive within-job timing.
4. **Window.** Assert each registered `CONFIRMATION_WINDOWS` entry individually; WP-L10 (`b8w` 108000, B8W registered on
   master `3c9c396`) does the same. The orchestrator resolves the dict-literal conflict at merge by keeping every entry.

## Clarification C1 (2026-10-29 13:30 JST, before the threshold lock and before any test episode; from sn-core's step-2 smoke)
1. **Latency-audit stop condition.** The condition "more than 0.2 % disagreeing questions stops the job" is replaced, as
   follows.
   - The job records three things: the number and fraction of questions whose grouped stopped-pass argmax differs from
     the assembled p^{s*}, and the largest top-2 logit gap among them.
   - It stops for the orchestrator only if a disagreeing question has a top-2 gap above 1.0.

   Reason: the smoke on val [0, 64) found 1.4–2.2 % disagreement for R1 and about 0 for R3. Every disagreeing question had
   a top-2 gap of at most 0.125. The answer logits are bf16, and they depend on how the batch is grouped: a plain
   fixed-depth pass on half the batch also flips 2 of 128 questions, with a maximum logit difference of 0.27. With a
   constant depth, the grouped pass is bitwise equal to the plain pass (unit test).

   Every registered statistic uses p^{s*} from the fixed-depth scan. Only the descriptive latency audit is affected, and the
   report gives the disagreement fraction.
2. **Where the latency audit runs.** It moves from the validation job to the test job, because it needs the locked
   thresholds. It still runs on validation episodes [1000, 1128), never on test episodes.
3. **Information seen before the threshold lock.** The smoke was on val [0, 64), which is P1-L's validation range and not
   LSR's selection range [1000, 3000). It showed two things, and neither changes the grid, the objective or any rule:
   - the hop × L shape, which looks like P1-L's matrix;
   - the medians of the R1 and R3 signals at both backbones. At 1.7B the read ratio grows to about 10, and the relative
     state change stays at 0.5–0.7.
4. **Infeasible rules.** These stay as registered. A rule with no feasible validation threshold fails at that backbone,
   and its test statistics are never computed. Nothing else is added. R2 has no threshold and is always evaluated.
5. **Cost.** The smoke projects about 165 s per 2,000-episode scan, against the 113–123 s assumed in the cost section.
   The projected total is about 0.45 node·h, inside the 0.8 cap.
