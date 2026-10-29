# SLOT-LOOP-v1 review round 3: check of the round-2 fixes

The same two reviewers check the same commit independently, as in round 2:
- **rv-code:** codex GPT-6-Astra xhigh.
- **rv-opus:** an Opus-5.5 xhigh subagent.

Neither reads the other's round-3 file.

## What to review

| item | value |
| --- | --- |
| commit | `wp/slot` at `8c283340ad67a7a8eead70c30ebff7b0e9b6a8ed`, checked out read-only and detached in `/work/xg24i002/x10041/LoopLora-wt/review-slot` |
| diff | `git -C /work/xg24i002/x10041/LoopLora-wt/review-slot diff f1ec035 8c283340ad67a7a8eead70c30ebff7b0e9b6a8ed`: the worker's round-2 changes, 9 files. f1ec035 is the merge of master into wp/slot, and 2438599 was the round-2 commit |
| fix list | `docs/tasks/WP-L13_slot_loop.md`, section "Review round 2: fixes before the preflight", fixes 1-4 |
| findings | `docs/review/static_2026-10-29/round2-rv-code.md` and `round2-rv-opus.md`. You may read both now: round 2 is closed |
| tests | The new tests were first run on the pre-fix code at 29b171a, in job 3728243 (`logs/pbs/slot-r2-red-20261029-212940-3263592.log`, exit 1). They passed on the fixed code in jobs 3728256 and 3728257 (`logs/pbs/slot-r2-meta-20261029-213408-3287227.log`, `logs/pbs/slot-r2-exp-20261029-213408-3287235.log`) |

Check three things:
1. **The fixes.** For each fix, judge whether the defect is gone on every path that reaches it, not only on the path the new test takes.
   - Fix 1: the real smoke → `smoke.json` → builder → JSON release → reload path.
   - Fix 2: every label kind (main, reference, unseen) and both the evaluation and the analysis side.
   - Fix 3: the open record, the ready release and the script file.
   - Fix 4: every runbook command in the state file, run exactly as written.
2. **The tests.** Does each new test fail on the pre-fix code for the stated reason? Use the red log, not only the test's name. Would it catch the defect if it came back?
3. **New defects.** Did the fix diff introduce one, or change behavior that the round-1 and round-2 reviews found correct? Include the prereg's rules and the sealed-set guards.

The next steps are the preflight, the lock and S0. A `src/` change after the preflight forces a new preflight.

## Rules and output

Same rules as `round2_brief.md` ("Rules"): login node only, no torch, pytest or jobs, and write only your file.

Write the file `docs/review/static_2026-10-29/round3-<reviewer>.md` in the main checkout. It has these parts:
1. **Header.**
2. **Verdict:** may the preflight start?
3. **Per fix:** done, partial or not done, with evidence.
4. **Findings table:** in the round-2 format and with the round-2 severity scale. Report only findings with a concrete failure mode.
5. **Not checked.**

Your final reply: the file path and the number of findings by severity.
