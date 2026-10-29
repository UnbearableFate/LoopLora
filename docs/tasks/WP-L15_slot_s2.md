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
