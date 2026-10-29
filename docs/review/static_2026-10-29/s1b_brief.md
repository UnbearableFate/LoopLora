# SLOT-LOOP-v1 S1B: cross-review brief

This is the **one** cross-review of the S1B core code (addendum §8). The user asked on 2026-10-29 for no repeated review rounds ("不要搞这么多次review,核心代码没问题就行"): the worker's compute tests check the fixes, and no second round follows. Report what would make S1B wrong now.

Two reviewers review the same commit independently:
- **rv-code:** codex GPT-6-Astra with reasoning effort xhigh, in its own herdr tab.
- **rv-opus:** an Opus-5.5 subagent of the main session with effort xhigh.

Neither reviewer reads the other's file. Each gets this brief, the code and the spec, but not the orchestrator's view.

## What to review

| item | value |
| --- | --- |
| commit | `wp/loop` at `67deeee5db78146d5eb9a9b55c93b081399f1c3a` (full hash), checked out read-only and detached in `/work/xg24i002/x10041/LoopLora-wt/review-loop` |
| diff | `git -C /work/xg24i002/x10041/LoopLora-wt/review-loop diff a9178d5...67deeee5db78146d5eb9a9b55c93b081399f1c3a`: S1B's changes only. a9178d5 is the master commit that registered the addendum; wp/loop merged it at 5d5bec5 |
| authority | `docs/prereg/SLOT-LOOP-v1-S1B.md`, blake2b-128 `6f2e776f2c6cfde131380fb343854784`, all of it, including "Decisions at registration" 1-6. Inherited rules come from the parent `docs/prereg/SLOT-LOOP-v1.md`, blake2b-128 `56e8c611c01e0ef9b7ae32db97b0e334` |
| task | `docs/tasks/WP-L14_slot_s1b.md` |
| state | `docs/reports/WP-L14.md` in the review checkout: the worker's ledger and plan |
| reference | `plan/looplora_diagnosis_and_execution_plan_2026-10-29.md` §8, as a reference only. Where the plan and the addendum differ, the addendum governs |

The next steps are the S1B preflight and lock, then calibration, smoke, LR pilot and the ready release, then 6 or 12 main runs. Look first for what would make those runs invalid or a registered decision come out wrong:
1. **Data isolation.** S1B uses its own namespace and sets, disjoint from S0 and S1A. S1A's confirmation set is never read. Sealed S1B sets open only through the inherited open-record rules.
2. **Branch.** The arms come mechanically from S1A's frozen decision record (§1 table, decision 4). No S1B score selects a branch. `A_query = clone(A)` trains only in the flagged branch, and writer `A` always stays frozen.
3. **Identity and isolation** (§8 table).
   - S1A's registered path stays bitwise unchanged.
   - Matrix-L4 at g=1 equals the native tensor loop bitwise.
   - Disabled capture leaves outputs unchanged.
   - Every read uses only its own episode's memory.
4. **Common start and training** (§1, §4).
   - The parent digest and config are checked.
   - Only Q/R/U/B (and `A_query` in the flagged branch) train.
   - Batches and order are identical across arms within a seed; N=6000.
   - The LR pilot rule and the dev schedule are as registered.
   - The checkpoint is selected on hop 1-3 dev NLL only, with the registered tie rule.
5. **Calibration** (§4, decision 3). Calibrated once per backend at L1 on S1B calibration, with the registered grid and tie rule. L1 and L4 share the result. A calibrated Matrix-L1 hop-1 EM below 10 % stops before the pilot.
6. **Statistics** (§5).
   - `d_i`: seeds averaged first, hops 2-3.
   - Resampling: 10,000 whole-episode resamples with the registered seed.
   - Quantiles: .025/.975 for one contrast, 1/120 and 119/120 for three.
   - Thresholds are compared exactly from counts.
   - The loop-gain, C_storage and continuation (depth sweep) rules are as registered.
7. **Release, lock and reload** (§4, §8).
   - The lock is bound to the `src/` digest.
   - The release carries the registered fields.
   - Reload reproduces smoke.
   - A `pending_reload` candidate never trains.
   - Every job is ≤ 16 node·h.
8. **Cost accounting** (§7): depth, logical reads, backend calls and timings are recorded as registered.
9. **Tests.** Check whether the tests catch the realistic failures above. A test that only mirrors the implementation is not evidence.

## Rules

- **Login node only.** Do not import torch, run pytest, train, evaluate or submit PBS jobs.
  - Allowed: reading, `git`, `grep`, and small stdlib checks with `/usr/bin/python3 -B`.
  - Also allowed: the project venv interpreter (`/work/xg24i002/x10041/LoopLora/.venv/bin/python -B`) for numpy-only checks.
  - Set `PYTHONPYCACHEPREFIX` to a scratch directory and do not combine it with `-I`, so no bytecode lands in a checkout.
- **Writes.** Write only your review file. Change no tracked file, branch or commit, and commit nothing.
- **Hashing.** Use blake2b only.

## Output

Write the file `docs/review/static_2026-10-29/s1b-<reviewer>.md` in the main checkout `/work/xg24i002/x10041/LoopLora`, where `<reviewer>` is `rv-code` or `rv-opus`. It has these parts:
1. **Header:** the reviewer, the date, the commit, the scope and the checks run.
2. **Verdict:** a few lines.
3. **Findings table:** id, severity, `file:line`, the concrete failure (inputs or state → wrong number, wrong decision, opened set or crash), the evidence, a suggested fix, and when it must be fixed (before the preflight, or before the formal runs).
   - **High:** changes a registered number or decision, opens a sealed set early, or invalidates a run.
   - **Medium:** can stop a stage, waste compute, or leave a record that cannot support its claim.
   - **Low:** robustness or clarity, with no effect on results.
   - Report only findings with a concrete failure mode. Style preferences and unsupported suspicions are not findings.
4. **Not checked:** what you did not check, and why.

Your final reply: the file path and the number of findings by severity.
