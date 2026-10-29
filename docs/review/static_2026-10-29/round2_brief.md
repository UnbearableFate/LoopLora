# SLOT-LOOP-v1 review round 2: cross-review brief

Two reviewers review the same commit independently, as the user asked on 2026-10-29: "代码审查则使用单独的codex
(gpt-6-astra xhigh) 和 opus-5.5 xhigh subagent进行交叉评审".
- **rv-code:** codex GPT-6-Astra with reasoning effort xhigh, in its own herdr tab.
- **rv-opus:** an Opus-5.5 subagent of the main session with effort xhigh.

Neither reviewer reads the other's file. Each gets this brief, the code and the spec, but not the orchestrator's view.

## What to review

| item | value |
| --- | --- |
| commit | `wp/slot` at `24385991af6755b687f46d46ba5351a196902b8c` (full hash), checked out read-only and detached in `/work/xg24i002/x10041/LoopLora-wt/review-slot` |
| diff | `git -C /work/xg24i002/x10041/LoopLora-wt/review-slot diff master...24385991af6755b687f46d46ba5351a196902b8c`: the study's changes only, since master was merged into wp/slot |
| authority | `docs/prereg/SLOT-LOOP-v1.md`: registered at master 5f04a5c, blake2b-128 `56e8c611c01e0ef9b7ae32db97b0e334`, including "Decisions at registration" 1-9 |
| task | `docs/tasks/WP-L13_slot_loop.md`. Its section "Review round 1: fixes before the preflight" lists the fixes the worker had to make |
| reference | `plan/looplora_diagnosis_and_execution_plan_2026-10-29.md` (§5-§7, §11-§13), as a reference only. Where the plan and the prereg differ, the prereg governs |
| state | `docs/reports/SLOT-LOOP-v1-state.md` in the review checkout: the job ledger and the worker's plan |
| round 1 | `docs/review/static_2026-10-29/slot-data-analysis.md` and `slot-model-training.md` (head ccdcc96, before the fixes) |

Work in this order:
1. Review the whole diff against the prereg as if no earlier review existed.
2. Check each fix-list item.

The next steps are the preflight, the lock and S0 (≤ 1.0 node·h), then the S1A release (≤ 4.5 node·h). Look first for what would make those runs invalid or a registered decision come out wrong:
1. **Data isolation.** Sealed and confirmation sets stay unopened until their registered time, and S1B never reuses S1A's confirmation sets. Seeds, ids and the layout digest are as registered.
2. **Identity.** Matrix must be bitwise identical to Native before any calibration or dev data is read, and R0 must stop the run otherwise. The off path of every changed existing file must stay bitwise unchanged.
3. **S0.** Conditions, temperatures and scales, metrics, the recency-matched SoftSlot, and `decision_s0.md` must follow the prereg's rules.
4. **S1A.**
   - The writer is frozen, all arms start from the same point, and steps and LR are matched.
   - The LR pilot uses the registered stream and excludes non-finite candidates.
   - Pre-clip gradient norm and the clipped fraction are logged.
   - Each run's checkpoint is selected on dev NLL.
   - The `S1A-ABL` ablation and its flag are as specified.
5. **Analysis.**
   - The decision rules are as registered.
   - The paired episode bootstrap is correct.
   - The open-record schema is followed.
   - Main records cannot be ablation records.
   - A missing reference file is refused.
6. **Release and lock.**
   - The lock is bound to a blake2b digest of `src/`.
   - The release builder's self-checks cover config hashes, steps, LR, g/τ, walltime and digests.
   - `reload_check` compares the config with the release.
   - `--open-root` is required and absolute.
7. **Tests.** Check whether the tests catch the realistic failures above. A test that only mirrors the implementation is not evidence.

## Rules

- **Login node only.** Do not import torch, run pytest, train, evaluate or submit PBS jobs.
  - Allowed: reading, `git`, `grep`, and small stdlib checks with `/usr/bin/python3 -B`.
  - Also allowed: the project venv interpreter (`/work/xg24i002/x10041/LoopLora/.venv/bin/python -B`) for numpy-only checks.
  - Set `PYTHONPYCACHEPREFIX` to a scratch directory and do not combine it with `-I`, so no bytecode lands in a checkout.
- **Writes.** Write only your review file. Change no tracked file, branch or commit, and commit nothing.
- **Hashing.** Use blake2b only.

## Output

Write the file `docs/review/static_2026-10-29/round2-<reviewer>.md` in the main checkout `/work/xg24i002/x10041/LoopLora`, where `<reviewer>` is `rv-code` or `rv-opus`. It has these parts:
1. **Header:** the reviewer, the date, the commit, the scope and the checks run.
2. **Verdict:** a few lines.
3. **Findings table:** id, severity, `file:line`, the concrete failure (inputs or state → wrong number, wrong decision, opened set or crash), the evidence, and a suggested fix.
   - **High:** changes a registered number or decision, opens a sealed set early, or invalidates a run.
   - **Medium:** can stop a stage, waste compute, or leave a record that cannot support its claim.
   - **Low:** robustness or clarity, with no effect on results.
   - Report only findings with a concrete failure mode. Style preferences and unsupported suspicions are not findings.
4. **Fix list:** one line per item: done, partial or not done, with evidence.
5. **Not checked:** what you did not check, and why.

Your final reply: the file path and the number of findings by severity.
