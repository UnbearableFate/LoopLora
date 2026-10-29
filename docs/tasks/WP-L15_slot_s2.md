# WP-L15: SLOT-LOOP-v1 S2 addendum draft (2026-10-30 01:25 JST)

**Worker.** wk-slot, a codex worker (GPT-6.1-sol, reasoning effort high), starting with a fresh context. It works in worktree `LoopLora-wt/slot` on branch `wp/slot`. S1A is done and merged (master 650dc3e).

**Authorization.** The user authorized S1B, S2 and S3 "按计划推进": each stage starts only through its entry gate.
- **S2 entry.** The continuation rule at the end of §5 of `docs/prereg/SLOT-LOOP-v1-S1B.md`.
- **This task.** It drafts the S2 addendum while S1B runs on wp/loop (wk-loop, amendment A1).
- **Nothing runs yet.** No S2 code, data or job comes before S1B's verdict and the orchestrator's registration of this addendum.

## Read first
1. `docs/README.md`, `docs/tasks/_common_rules.md`, and the row wk-slot in `docs/workers.md`.
2. `docs/prereg/SLOT-LOOP-v1.md`: "Addenda for S1B, S2, S3" and the task cautions there.
3. `docs/prereg/SLOT-LOOP-v1-S1B.md` and `docs/prereg/SLOT-LOOP-v1-S1B-A1.md`.
4. `plan/looplora_diagnosis_and_execution_plan_2026-10-29.md`:
   - §9, all of it: S2;
   - §5 and §11–§13: data, statistics and interfaces;
   - the execution record.
5. The S1A result: `reports/slot-loop-v1/decision_s1a.md`.
6. The S1B transfer diagnosis: `reports/slot-loop-v1/s1b/diagnosis-6855048/` on wp/loop, which also states its limits in `docs/reports/WP-L14.md`. Read it with `git show wp/loop:<path>`.
7. Code:
   - `src/looplora/data/relations.py` (`t2_typed`, `n_distract`) and `src/looplora/data/render.py`;
   - the slot read code;
   - the S1B loop code on wp/loop, `src/looplora/llm/s1b_*.py`, read with `git show`.

## Step 1 (login node only): the addendum draft, then STOP
Draft `docs/prereg/SLOT-LOOP-v1-S2.md` on wp/slot. Plan §9 is the reference and the preregs' rules govern. Settle these points with evidence from the code and records, and flag every deviation from the plan.

1. **Entry and backend by S1B outcome.** Pre-specify every S1B continuation outcome, so that no S2 choice depends on S2 scores:
   - only Matrix qualifies;
   - both qualify and C_storage passes;
   - both qualify and C_storage fails;
   - only SoftSlot qualifies;
   - none qualifies, which means no S2.
2. **Factor ladder** (§9.1): Base-old, Arch, Arch+Role, +Transition, +Null.
   - Which contrasts are confirmatory, their family and multiplicity correction, and which are descriptive.
   - Which rungs run in parallel and which wait on a gate.
3. **Task** (§9.4).
   - At least two relations, with the relation to follow named in each question.
   - Distractor edges; training hops 1–3, extrapolation to hops 4–8; control of the fact count.
   - Check `t2_typed` and `n_distract` against §9.4, and lock the layout digest.
   - **Question wording.** S1B's diagnosis found that the T1-trained query fails on new question wording: final-token key top-1 was 15 % on T2, against 89 % on T1. State how S2's query formation handles this. Plan §9.2 builds p0 from the start-entity span through a mention encoder; say whether that is enough and what is checked before training.
4. **Positions** (§9.1). `last_valid_question` applies to all arms, and the S-RE parent was trained on all tokens. State the consequence and the check.
5. **State and parameters** (§9.2).
   - Dimensions, initialization and parameter counts per arm.
   - A budget-matched control if the parameter counts exceed Q/R/U/B.
6. **Per-transition objective** (§9.3).
   - Loss, negatives, temperature and weights; free rollout.
   - The rule for runs shorter than h.
7. **NULL and stopping** (§9.5).
   - NullHead.
   - Labels: on the visible graph only, with no rollout-misaligned labels.
   - Soft training and a hard 0.5 threshold.
   - h + 1 reads and `budget_exhausted`.
8. **Mechanism acceptance** (§9.6).
   - The metrics.
   - The causal intervention protocol and its two controls.
9. **Data.**
   - S2's own seed namespace and sets, disjoint from S0, S1A and S1B.
   - The inherited registry, sealing, `--manifest` and open-record rules.
10. **Statistics.** Primary contrasts, the paired episode bootstrap, thresholds, the decision table and the continuation to S3.
11. **Process and cost.**
    - Preflight and lock, smoke, pilot, release and STOP for approval, as in S1A and S1B.
    - Estimate each job from measured step times (S1A: 0.159 s/step at T1 L1; S1B's pilot will add T2 L4). Each job is ≤ 16 node·h; written budgets do not bind.

End with `STOP: addendum <full hash>`. Write no code before the orchestrator registers the addendum on master.

## Rules
- **Merging.** Merge master into wp/slot once now. It brings the S1B addendum, A1 and the records. Do not merge wp/loop.
- **Stay blind to S1B outcomes.** Do not read S1B pilot, dev, confirmation or unseen outputs, or S1B training records under `runs/slot-loop-v1/s1b/`, while drafting. The calibration and the diagnosis may be read.
- **Commits.** Commit on wp/slot only. Do not push.

## Registration (2026-10-30 01:55 JST)
The addendum is registered on master as `docs/prereg/SLOT-LOOP-v1-S2.md`, blake2b-128 `a5262fc4ecf1ffdaab4613321b0ff2b0`. Its "Decisions at registration" 1–7 change the draft in four places:
- the ladder runs in two waves (wave 1: Base-old, Base-budget, Arch, Arch+Role and +Transition; wave 2: +Null);
- intervention control (1) is a donor a-only swap, reported without a threshold; question isolation becomes an engineering gate;
- an S3 at fixed read budgets is open when H2 passes without H3;
- implementation and tests may start before S1B's verdict.

**Next, Step 2 (implementation; S1B still running on wp/loop).**
1. **Merges.** Merge master into wp/slot once. Then merge wp/loop at exactly `258f5f5a24eae684f640a9fd62679bb643470a76`, the A1 lock commit, not wp/loop's tip. At that commit wp/loop holds only S1B's preflight, lock, stopped calibration and diagnosis.
2. **Implement S2 as registered.** This covers both backends, all six arms including +Null, the wrapper, spans, losses, trajectory capture, interventions and the analysis. In shared files, keep S0/S1A/S1B registered behavior bitwise unchanged, with tests.
3. **Tests on compute only.** Use synthetic fixtures, the real S-RE parent and the S2 smoke namespace (train [0,64)). Generate no calibration, pilot, dev, training, confirmation or unseen data. No preflight or lock: they bind S1B's branch evidence and wait for its verdict.
4. **Report.** Update `docs/reports/WP-L15.md` with state and ledger.
5. **Stop.** End with `STOP: review <full hash>`. That review is the one cross-review of the S2 core code. Fixes after it are checked by tests only.

**Rules for Step 2.**
- **Blindness.** Stay blind to S1B outcomes: do not read wp/loop commits after `258f5f5`, or anything under `runs/slot-loop-v1/s1b/` or `reports/slot-loop-v1/s1b/` that `258f5f5` does not contain.
- **Jobs.** At most 6 wk-slot jobs queued or running, each ≤ 16 node·h, walltime = measured × 1.5. End a turn with `WAIT <ids>`; do not poll.
- **Commits.** Commit on wp/slot only. Do not push.

## Before the review (02:50): complete the job entry points
`STOP: review 72d30b3` is not the end of Step 2. The S2 science modules exist, but the code that runs and guards them does not. S1B's one review covered `s1b_run.py`, `s1b_preflight.py` and the release builder, and S2's one review must cover their counterparts. Code written after the review would go unreviewed.

1. **Add the S2 entry points**, modeled on S1B and bound to the registered prereg:
   - a CLI with `--manifest` for smoke, the calibration audit, the LR pilot, main training with dev checkpoints, the frozen dev selection, and, after `open.json`, the confirmation, unseen, sweep, probe and intervention evaluations;
   - the real-parent preflight and lock builder, binding the S1B branch evidence (§11);
   - the candidate and ready release builders with CLI reload checks;
   - the approval and open-record checks;
   - wave-2 release extension handling.
2. **Test them on compute** with fixtures and the S2 smoke namespace. The preflight and lock may be tested with a synthetic branch record; the real ones still wait for S1B's verdict.
3. **Stop.** Update `docs/reports/WP-L15.md`, then end with `STOP: review <full hash>`. The one cross-review covers all S2 core code at that commit.

**Jobs.** While S1B's twelve main runs hold most of the project's 16 running slots, keep at most 4 wk-slot jobs queued or running.

## Review (03:30): fixes before the preflight
The one cross-review of `a727669` is running. rv-code's report is in: `docs/review/static_2026-10-29/s2-rv-code.md` (3 High, 3 Medium). The orchestrator checked C01 and C02 in the code. rv-opus's report comes later. Its findings that rv-code does not cover will be added here as further items.

Fix these on wp/slot. Each fix gets a test that fails without it.
1. **S2-C01 (branch evidence).** `backends()` must stop on missing or contradictory S1B evidence, never default to "not qualified".
   - Require the exact continuation and contrast schema implied by the bound S1B release branch (advance: M and S rows, all three contrasts).
   - Reconcile `qualified` with the recorded loop contrast and sweep screen.
   - Check the parser against the verdict schema that S1B's own frozen analysis (`s1b_analysis.py` at `258f5f5`) writes. Build the test fixture with that code on synthetic inputs, not as a hand-written dict.
2. **S2-C02 (lock coverage).** Hash every regular file under `src/` except bytecode, as the inherited `source_digest` does, plus the S2 science scripts and config. Add a mutation test on the name asset, including a consistent name/token permutation. Earlier stages keep their own digests.
3. **S2-C03 (payload provenance).** Write origin metadata into every evaluation, trajectory, probe and intervention payload at production time: label, arm, backend, seed, checkpoint digest, release and config. Analysis reconciles each payload with the frozen selected mapping and its evaluation record. Add a negative test that swaps intact payloads between labels. Replace the reuse of one payload for all labels in `tests/test_s2_workflow.py` with distinct payloads.
4. **S2-C04 (untrained interface audit).** Add the §3 smoke-only audit before any optimization: per relation, untrained first-query key top-1 and margins, plus raw-read and frozen-value-head diagnostics on hop 1. The preflight requires it to be complete; it is descriptive, with no quality threshold. The calibration audit records it as well.
5. **S2-C05 (+Null diagnostics).** Capture diagnostics for both hard and soft inference at budgets 4 and 9. Emit the §7 and §10 short-hop and h4–8 stopping summaries with explicit denominators. Collection of a +Null run requires these artifacts.
6. **S2-C06 (walltime accounting).** Measure every phase of a run:
   - scale the per-checkpoint export and I/O cost to the eleven main evaluations;
   - include support verification and the final audit and reload.

   Add a fixture with distinguishable costs that checks the main total and the over-cap STOP.

Then run all S2 tests plus the S0, S1A and S1B test modules on compute, update `docs/reports/WP-L15.md`, and end with `STOP: fixed <full hash>`. There is no second review round. The fixes are checked by tests only (user, 2026-10-29).

## Review (03:55): rv-opus items 7-11
rv-opus's report is in: `docs/review/static_2026-10-29/s2-rv-opus.md` (0 High, 4 Medium, 2 Low). F1 repeats item 2 (S2-C02). F2 is item 4 (S2-C04) with one addition. The rest are new. The orchestrator checked F3 in the code (`s2_analysis.py:107`).

7. **F2 addition to item 4.** The audit also checks that changing only the named relation changes the question text and c. The calibration job saves the audit: trajectory diagnostics at budget 1 over hops 1-3, split by relation, and a decode of the raw read through the parent's frozen value head. Capture that value head before `prepare()` drops it.
8. **F3 (exact thresholds).** Compare the terminal screens, coverage screens and every other registered threshold exactly from integer counts: integer cross-multiplication or `fractions.Fraction`, never averaged floats. §10 requires this. Test the exact-boundary cases, 5 % and 80 %.
9. **F4 (registered descriptive outputs).** Add these to the locked code:
   - **+Null pilot and dev:** save a soft budget-4 sample file next to the hard one.
   - **Relation-switch probe:** paired changes over all queries, plus the byte-identical question re-render control.
   - **`analyze_s2.py` tables:** per-hop and per-seed EM and NLL, per-relation counts and metrics, the UNKNOWN prediction rate, soft-hard differences, and anchor versus free-rollout NULL head accuracy.
10. **F5 (N/A, not 0).** For Base-old and Base-budget, semantic metrics are N/A: bridge, value-to-query and p interventions. Never write them as zeros.
11. **F6 (preflight checks that can fail).**
    - Perturb labels before `prepare()`.
    - Compare B = 0 logits with the parent's adapter-free frozen base forward.
    - Report question isolation as N/A for base arms, or run it for them; never hard-code True.
    - Have `verify_trace` recompute c and p0 from the captured x and spans.

Fix items 7-11 together with items 1-6, in one pass, before `STOP: fixed <full hash>`.
