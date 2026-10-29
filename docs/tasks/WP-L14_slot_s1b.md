# WP-L14: SLOT-LOOP-v1 S1B: storage effect against loop effect (2026-10-29 22:25 JST)

**Worker.** wk-loop is a codex worker (GPT-6.1-sol, reasoning effort high). It works in worktree `LoopLora-wt/loop` on branch `wp/loop`, which starts from master 2f7abbc: the S0 code and results. It runs in parallel with wk-slot, which runs S1A on `wp/slot`.

**Authorization.** The user authorized S1B, S2 and S3 "按计划推进": each stage starts only through its entry gate.
- The S1B entry rule is the prereg's "Decisions at registration" 2. An S1B addendum may follow any S1A that is valid on its dev with the confirmation set isolated. S1A's verdict decides which arms S1B has.
- This task prepares S1B while S1A runs. S1B runs only after S1A is done and the orchestrator has registered the addendum.

## Read first
1. `docs/README.md`.
2. `docs/tasks/_common_rules.md`, including "Step 2 on compute".
3. Row wk-loop in `docs/workers.md`.
4. `docs/prereg/SLOT-LOOP-v1.md`, all of it, especially "Decisions at registration" and "Addenda for S1B, S2, S3".
5. `plan/looplora_diagnosis_and_execution_plan_2026-10-29.md`: §8 (S1B) and the execution record; §5 and §11-§13 for data, statistics and interfaces.
6. `docs/reports/SLOT-LOOP-v1.md` and `docs/reports/SLOT-LOOP-v1-state.md`: the S0 result, the `MemoryBackend` semantics, the lock and runbook patterns.
7. `docs/tasks/WP-L13_slot_loop.md`: the stage cycle and "Rules repeated here because they bit us before".
8. The existing loop-depth code (P1-L L4 runs) and `src/looplora/data/relations.py`.

## Step 1 (login node only): the addendum draft, then STOP
Draft `docs/prereg/SLOT-LOOP-v1-S1B.md` on wp/loop. Plan §8 is the reference; the prereg's rules govern. Settle these points with evidence from the code and records, and flag every deviation from the plan:
1. **Arms by S1A outcome** (decision 2). Pre-specify all three branches so nothing is chosen after S1A's verdict is known:
   - ADVANCE: the 2×2 of M1, M4, S1 and S4.
   - NOT ADVANCED, with the ablation not run or not flagged: Matrix-L1 against Matrix-L4 only.
   - NOT ADVANCED, with the ablation flagged: the rule for slot arms with the `A_query` copy.
2. **Task and layout.** Plan §8.1 names T2 chain-end: E12, two chains, relation fixed at 0, training on hops 1-3, testing on hops 1-8.
   - S-RE was trained on T1 (E16, `n_rel` 4, hops [1]); see `runs/p1l/sq2/arms/S-RE/record.json`. Check whether its writer is valid on the T2 layout, and how the old loop studies handled this.
   - Choose with reasons and lock the task/layout digest. An alternative such as E16 needs the same lock.
3. **Common start.** All arms start from the same writer/reader state. The writer is frozen and Q/R/U/B are trained. The data stream, position rules and number of updates are equal across arms.
4. **Calibration.** g and τ are calibrated once, on S1B's own calibration set, with the L1 parent and the S0 grid. L1 and L4 of the same backend share them, and no formal checkpoint is re-tuned.
5. **Decision rule for the loop effect** (H2).
   - The primary contrast: L4 against the independently trained L1 of the same backend, on the registered hops, with the inherited paired episode bootstrap.
   - The continuation condition of plan §8.3.
   - The same-checkpoint depth sweep at L = 1/2/4/8 is descriptive.
6. **Mechanism controls.** Fixed-query L4 and warm-start untied L4 (plan §8.2) run only after a loop gain appears. State their trigger.
7. **Cost accounting** (plan §8.2): `loop_depth`, logical reads (= valid read positions × L), backend calls, and measured kernel and step time.
8. **Data.**
   - S1B gets its own seed namespace and sets: calibration, dev, confirmation and optional unseen. They must be disjoint from S0 and S1A, and S1A's confirmation set is never reused.
   - Inherit the registry, sealing, `--manifest` and open-record rules of S1A.
9. **Preflight and lock.** S1B code changes `src/`, so S1B needs its own preflight and lock, bound to its source digest. Matrix-L4 must replay the existing loop path bitwise where that path exists.
10. **Job estimate.** Estimate each job from measured step times; S1A's pilot will measure the slot read step. Each job is ≤ 16 node·h. Written budgets do not bind (task WP-L13, "Budget").

End with `STOP: addendum <hash>`. Write no code before the orchestrator registers the addendum on master.

## Later steps (after registration)
- Implement and test on compute. This is core code, so one cross-review follows, by codex rv-code and an Opus subagent. Fixes are checked by your tests and get no further review round.
- Preflight and lock.
- Wait for S1A's verdict, then run the branch the addendum names.

## Rules
- **Merging.** Do not merge wp/loop into master, and do not merge master into wp/loop without the orchestrator's word. wp/slot is locked on its own `src/` for S1A, so the two lines stay apart until S1A is done.
- **Shared code.** Do not change S1A's registered behavior in shared files. Where S1B needs a change in shared code, make it explicit and keep S1A's path bitwise unchanged, with a test.
- Commit on wp/loop only. Do not push.

## Registration (2026-10-29 22:45 JST)
The addendum is registered on master as `docs/prereg/SLOT-LOOP-v1-S1B.md`, blake2b-128 `6f2e776f2c6cfde131380fb343854784`. Its "Decisions at registration" 1-6 add three things to the draft:
- the seed and layout check;
- a hop-1 sanity stop after calibration;
- the branch source and the process.

**Next, Step 2.**
1. Merge master into wp/loop once now. It brings docs only: the registered addendum, the plan record and this section.
2. Implement S1B as registered and run its tests on compute.
3. Update `docs/reports/WP-L14.md` (state and ledger).
4. End with `STOP: review <full hash>`.

That review is the one cross-review of the S1B core code. After it come the fixes (checked by tests), then the preflight and the lock. S1B formal runs wait for S1A's decision record.

## Review (2026-10-29 23:50 JST): fixes before the preflight
The one cross-review of the S1B core code, both at 67deeee:
- `docs/review/static_2026-10-29/s1b-rv-code.md`: 1 High, 4 Medium.
- `docs/review/static_2026-10-29/s1b-rv-opus.md`: 1 High, 1 Medium, 4 Low.

No further review round follows; your tests check the fixes. Every item changes `src/`, so all of them land before the preflight and the lock.
1. **Exact bootstrap** (RVC-01 = F1, High). Resample the integer sums. Take the quantile and the `> 0` test in exact arithmetic; convert to pp only for reporting. Regression: rv-code's zero-boundary case, including the continuation choice. Show it failing on 67deeee and passing after the fix.
2. **Reload against the original smoke** (RVC-02, F3). Record the smoke checkpoint's IDs, predictions, correctness and NLL from the in-memory instance before closing it. Compare both the first load and the CLI reload with that record. Test: a deterministic altered reload fails.
3. **Checkpoint provenance** (RVC-03). Bind each dev row to its export's digest and step, and check that the selected file matches the winning row. Keep the scheduled candidate exports until the selection audit; each is a few MB.
4. **Trajectories** (RVC-04). Save the final-valid-token q/m/z at each loop for each evaluated depth, ordered by episode, hop and depth, with provenance. Test the saved artifact.
5. **Confirmation report** (RVC-05 = F2). The frozen analysis writes the registered tables, labeled descriptive and kept apart from the decisions:
   - EM, counts and NLL per arm, seed and hop;
   - per-depth sweep EM;
   - S1−M1 and the Memory×Depth interaction.

   Test it with a passing and a failing outcome.
6. **Branch records** (F4). Accept an ablation record only when S1A's dev outcome triggered the ablation. All S1A records must carry the same release digest.
7. **Calibration** (F5). Exclude and record a nonfinite candidate, as the inherited rule does, instead of aborting.
8. **Preflight coverage** (F6).
   - T1 identity at L1, L2 and L4.
   - The owner check at every depth.
   - Assert the [B,12,768] and [B,12,128] capture shapes.
9. **Real training stream** (rv-opus). Before the pilot, a compute test must run `BatchStream` on T2 (`token_fact_positions`). Every current test passes `source=`.

**Then:**
1. Run the full S1B set and the legacy slot shard green.
2. Run the preflight and commit the lock (§8).
3. Update `docs/reports/WP-L14.md`.
4. End with `STOP: locked <full hash>`.

Calibration and all later steps wait for S1A's decision record (decision 4).

## Calibration stop (2026-10-30 00:56 JST): diagnose before any amendment
Calibration job 3729493 hit the registered sanity stop (decision 3). On S1B calibration, the untrained L1 parent reached these numbers (from `runs/slot-loop-v1/s1b/release-prep-bd4e24f/calibration/calibration.json`):

| backend | setting | hop-1 EM | hop-1 NLL |
| --- | --- | --- | --- |
| Matrix | best (g=2) | 5.7 % | ≈ 10 nats |
| Matrix | selected (g=1) | 5.1 % | ≈ 10 nats |
| SoftSlot | selected | 1.3 % | 8.5 nats |

On T1 the same parent's step 0 is about 73 % with NLL 1.42. Decision 3 reads such a stop as a broken transfer, prompt or position rule, not a reading limit. Find which before anything else runs.

**Scope.** S1B calibration data and non-sealed T1 data only. No pilot, dev or confirmation. Put the diagnostic code under `scripts/`, so `src/` and the lock bd4e24f stay unchanged.

Run one job of at most 30 min on debug-g that reports:
1. **The S1B evaluation path on T1.** The parent's own L1 read on a non-sealed T1 set, for example the S1A pilot shard, which has a recorded step 0: Matrix 733/1000, NLL 1.4211. A mismatch means an S1B code bug.
2. **Prompts and targets.** Five rendered T2 hop-1 calibration prompts and five T1 prompts, each with its target tokens and the model's top-5 next tokens at the answer position.
3. **Where hop 1 fails on T2** (calibration, hop 1, Matrix and SoftSlot at their selected settings):
   - key top-1 and margin of the actual query against the 12 written keys, at the read position(s) used;
   - an oracle read (the true next value as m, like S0's clean condition): EM and NLL;
   - memory off: EM and NLL.

Then update `docs/reports/WP-L14.md` and end with `STOP: diagnosis <full hash>`. Include a recommendation with its evidence, one of:
- (a) a code bug: fix it, then a new preflight, lock and calibration;
- (b) a prompt or position mismatch;
- (c) a writer transfer failure.

Do not change the registered design; the orchestrator decides any amendment.

## Amendment A1 (2026-10-30 01:20 JST): implement, then the release
Your diagnosis (STOP 272cd24, recommendation (b)) agrees with the orchestrator's reading. Amendment A1 is registered on master 4178808: `docs/prereg/SLOT-LOOP-v1-S1B-A1.md`, blake2b-128 `341d3c71f9334a1384a102b097e7ad75`. It keeps the chain-end task and lets the trained Q adapt to its wording. Its changes:
- the stop is cleared;
- the reads come from S1A's ready release (Matrix g 1; SoftSlot g 1, τ 0.1);
- N = 18,000, with dev every 1,800 steps and training streams train [0, 576,000);
- everything else is unchanged.

**Steps.**
1. Merge master into wp/loop once. It brings A1, the S1A merge and docs. Check that `src/` outside the `s1b_*` files is still identical to master.
2. Implement A1 in `src/`:
   - bind A1's digest the way the addendum's is bound;
   - take the read settings from the S1A release by digest;
   - set N, the dev schedule and the training window, and extend the registry and disjointness checks to it;
   - take decision 3's sanity rule out of the gating path;
   - keep the stopped calibration as descriptive evidence cited by the release (blake2b-128 `6e4f7346376cb010820a84832f6d7d91`).
3. Run the full S1B set and the legacy slot shard green on compute. No review round.
4. Run a new real-parent preflight and commit the lock.
5. Run smoke and reload with the S1A reads, then the LR pilot (200 steps, as registered), the candidate release, the reload checks and the ready release. Size walltime from the pilot's measured step time × 1.5, and keep each job at most 16 node·h. If one main run cannot fit, STOP.
6. Update `docs/reports/WP-L14.md` and end with `STOP: release <full hash>` for `approval.json`.

## Approval (02:42): twelve main runs, then the dev selection
The A1 ready release `d1ccbb7ff4c31fa53abfd7a3cc9ad572` is approved: `reports/slot-loop-v1/s1b/a1/approval.json` on wp/loop (0b5973b; scope amended in 547a5e4 to allow all twelve jobs at once). Do not edit it.
1. **Submit the twelve main jobs**, M1, M4, S1 and S4 × seeds 17, 29 and 43, all from science commit `258f5f5` with `scripts/submit.py --commit <full hash>`.
   - One node each, regular-g. Walltimes: M1 01:30:00, M4 02:40:00, S1 01:40:00, S4 03:00:00.
   - All twelve may be queued at once.
   - End each turn with `WAIT <ids>`.
2. **Check each finished run.** It needs:
   - exit 0 and complete status;
   - the exact release digest in its record;
   - freeze and finiteness checks;
   - checkpoints 1,800 to 18,000.

   A failure stops for diagnosis; do not resubmit a changed run.
3. **When all twelve are complete,** run the frozen dev selection with the frozen analysis script. End with `STOP: selected <full hash>`. The independent re-run and the orchestrator's `s1b-open.json` come next. Do not open confirmation or unseen data.
4. **Keep `docs/reports/WP-L14.md` current** (state and node·h ledger).

## Opened (05:18): confirmation, the depth sweeps and the frozen verdict
`STOP: selected fe5320e` is accepted. rv-code's independent re-run (job 3730976) reproduced the selection: `docs/review/static_2026-10-29/s1b-selection-rerun.md` on master (fee4bfa).

**The open record** is `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/open-root/reports/slot-loop-v1/s1b/open.json`, blake2b-128 `b698f7ce5e31b2b43da5c210134588b0`. The same bytes are committed on wp/loop at `reports/slot-loop-v1/s1b/open.json` (9f68575). The open root also holds read-only copies of `scripts/analyze_s1b.py` and the three preregistration files from `258f5f5`, at their registered digests. Pass `--open-root /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/open-root`. Do not edit the open root or the record.

**Authorized now,** from science commit `258f5f5a24eae684f640a9fd62679bb643470a76` with `scripts/submit.py --commit <full hash>`:
1. **The entries file:** each of the twelve labels mapped to its `selected.pt`.
2. **Confirmation evaluation** (`--mode evaluate --set confirmation`): the twelve selected checkpoints on the S1B confirmation set, 2,000 episodes, hops 1-8. The mode also runs the registered L = 1/2/4/8 sweep of M4 and S4. Unseen stays off, because the release fixed it false.
3. **The descriptive dev depth sweep** (`--set dev`). It changes no decision.
4. **The frozen confirmation analysis:** `scripts/analyze_s1b.py --stage CONFIRMATION` with the selection, the open root and the confirmation records. It writes `verdict.json` and `verdict.md`.

**Jobs.** Size walltime as the measured evaluation time × 1.5. Use regular-g unless the job fits in 30 min. Each job stays at or below 16 node·h. End each turn that waits on jobs with `WAIT <ids>`. A failure stops for diagnosis; do not resubmit a changed job.

**Then:**
- Archive the verdict on wp/loop.
- Update `docs/reports/WP-L14.md`: the state, the verdict tables and the node·h ledger.
- End with `STOP: verdict <full hash>`.

**Not yet.** Do not train the §6 mechanism controls. They run only for a backend with a confirmed loop gain, and only after the orchestrator reads the verdict and records a release extension.
