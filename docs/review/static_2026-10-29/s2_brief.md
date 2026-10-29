# SLOT-LOOP-v1 S2: cross-review brief

This is the **one** cross-review of the S2 core code (addendum §11; registration decision 7). The user asked on 2026-10-29 for no repeated review rounds ("不要搞这么多次review,核心代码没问题就行"): the worker's compute tests check the fixes, and no second round follows. Report what would make S2 wrong now.

Two reviewers review the same commit independently:
- **rv-code:** codex GPT-6-Astra with reasoning effort xhigh, in its own herdr tab.
- **rv-opus:** an Opus-5.5 subagent of the main session with effort xhigh.

Neither reviewer reads the other's file. Each gets this brief, the code and the spec, but not the orchestrator's view.

## What to review

| item | value |
| --- | --- |
| commit | `wp/slot` at `a727669467e1870a675681308a13bbe881561c94` (full hash), checked out read-only and detached in `/work/xg24i002/x10041/LoopLora-wt/review-s2` |
| diff | `git -C /work/xg24i002/x10041/LoopLora-wt/review-s2 diff 10f213e a727669467e1870a675681308a13bbe881561c94`: S2's changes only. 10f213e merges master `14c1fae` (which registered the addendum) and the S1B A1 lock commit `258f5f5`; the S1B code it brings was reviewed separately |
| authority | `docs/prereg/SLOT-LOOP-v1-S2.md`, blake2b-128 `a5262fc4ecf1ffdaab4613321b0ff2b0`, all of it. Its "Decisions at registration" 1-7 govern where they differ from the text above them. Inherited rules come from `docs/prereg/SLOT-LOOP-v1.md`, `SLOT-LOOP-v1-S1B.md` and `SLOT-LOOP-v1-S1B-A1.md` |
| task | `docs/tasks/WP-L15_slot_s2.md` |
| state | `docs/reports/WP-L15.md` in the review checkout: the worker's ledger |
| reference | `plan/looplora_diagnosis_and_execution_plan_2026-10-29.md` §9, as a reference only. Where the plan and the addendum differ, the addendum governs |

**Blindness.** S1B is still running. Do not read S1B pilot, dev, confirmation or training outputs: anything under `runs/slot-loop-v1/s1b/`, or wp/loop commits after `258f5f5`. The review needs none of them.

The next steps, after S1B's verdict, are the S2 preflight and lock, the calibration audit, smoke, LR pilot and ready release, then wave 1 (Base-old, Base-budget, Arch, Arch+Role, +Transition) and, if gated in, wave 2 (+Null). Look first for what would make those runs invalid or a registered decision come out wrong:
1. **Entry and branch** (§1). K and the primary backend come mechanically from S1B's frozen verdict. No S2 score changes them; missing or inconsistent S1B records stop.
2. **Task and data** (§2, §9).
   - The TaskSpec and composite layout digests.
   - Evaluation stratifies the raw graph before the relation swap; the swap applies `r XOR b` to every fact and question relation; both IDs are recorded.
   - The relation-switch probe and the question wording are as registered.
   - S2 sets are disjoint from S0, S1A, S1B (with A1's 576,000-episode prefixes) and the old studies, checked with each entry's own layout.
   - Confirmation and unseen are sealed and open only through `open.json`.
3. **Positions and query formation** (§3).
   - `last_valid_question`: other positions get exactly zero adapter branch.
   - With B = 0, every arm's logits equal the frozen base bitwise.
   - c is the frozen standardized u at the last question token; p0 is the start-entity span mean through f_entity. No ID lookup or oracle code enters ordinary forward.
4. **State, parameters and initialization** (§4).
   - The registered parameter counts per arm, including Base-budget.
   - One module-named common initialization; no warm start between rungs.
   - The exact trainable and frozen sets; the writer, buffers, backbone and support stay unchanged.
5. **Losses** (§6).
   - Role: InfoNCE with deduplicated in-batch identity negatives, the role grouping, stop-gradient canonical targets.
   - Step: free rollout, path-intent targets, the registered mix of cosine and InfoNCE.
   - Weights .1/1/.1, temperature .1; the short-budget CE mask.
6. **NULL and stopping** (§7). Visible-graph anchor labels with detached inputs and class-balanced BCE; no rollout NULL labels; the exact soft survival update; hard 0.5 at inference; h+1 reads; absorbing rows do no logical reads; `budget_exhausted` emits Unknown; strict and raw outputs reported separately.
7. **Waves, gates and selection** (§5; decisions 3 and 6).
   - N = 18,000 with dev every 1,800; the LR pilot rule; selection on hop 1-3 dev NLL (for +Null the raw hard-stop convention) with the registered tie rule.
   - The stability rule, the Role target-space audit and the NULL entry gate.
8. **Mechanism metrics and interventions** (§8; decision 4). Next-address and bridge top-1 among all visible entities; the confidence rule; terminal instability D; the donor selection rule; the a-only swap (descriptive) and same-entity re-encoding control (gating); question isolation as an engineering gate.
9. **Statistics** (§10). `d_i` with seeds averaged first, hops 2-3; 10,000 whole-episode resamples with the registered seed; quantiles α/(2m), m = 6K; exact count thresholds; the six contrasts, including the C_Ntask noninferiority rule and C_Nstable on D; the decision table and the S3 rule of decision 5.
10. **Release, lock, reload, approval and open** (§9, §11). The lock binds the `src/` digest, science scripts and S1B branch evidence; the release carries the registered fields; reload reproduces smoke; a candidate never trains; approval and open records are checked; every job is ≤ 16 node·h.
11. **Shared code.** S0, S1A and S1B registered paths stay bitwise unchanged.
12. **Tests.** Check whether the tests catch the realistic failures above. A test that only mirrors the implementation is not evidence.

## Rules

- **Login node only.** Do not import torch, run pytest, train, evaluate or submit PBS jobs.
  - Allowed: reading, `git`, `grep`, and small stdlib checks with `/usr/bin/python3 -B`.
  - Also allowed: the project venv interpreter (`/work/xg24i002/x10041/LoopLora/.venv/bin/python -B`) for numpy-only checks.
  - Set `PYTHONPYCACHEPREFIX` to a scratch directory and do not combine it with `-I`, so no bytecode lands in a checkout.
- **Writes.** Write only your review file. Change no tracked file, branch or commit, and commit nothing.
- **Hashing.** Use blake2b only.

## Output

Write the file `docs/review/static_2026-10-29/s2-<reviewer>.md` in the main checkout `/work/xg24i002/x10041/LoopLora`, where `<reviewer>` is `rv-code` or `rv-opus`. It has these parts:
1. **Header:** the reviewer, the date, the commit, the scope and the checks run.
2. **Verdict:** a few lines.
3. **Findings table:** id, severity, `file:line`, the concrete failure (inputs or state → wrong number, wrong decision, opened set or crash), the evidence, a suggested fix, and when it must be fixed (before the preflight, or before the formal runs).
   - **High:** changes a registered number or decision, opens a sealed set early, or invalidates a run.
   - **Medium:** can stop a stage, waste compute, or leave a record that cannot support its claim.
   - **Low:** robustness or clarity, with no effect on results.
   - Report only findings with a concrete failure mode. Style preferences and unsupported suspicions are not findings.
4. **Not checked:** what you did not check, and why.

Your final reply: the file path and the number of findings by severity.
