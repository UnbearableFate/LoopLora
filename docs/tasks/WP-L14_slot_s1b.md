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
