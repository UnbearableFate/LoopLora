# Rules for every subagent task (2026-10-27)

- Until the orchestrator says step 2 may start (after the Miyabi maintenance, 2026-10-28 22:00 JST), work on the
  login node only: edit files and run static checks (`python3 -m py_compile`, `pyflakes`/`ruff` if present).
  Never import torch, run pytest, train or evaluate on a login node. Compute work goes through
  `scripts/submit.py` (see `docs/reports/P1L-state.md` for the pattern) and only in step 2.
- Each job ≤ 16 node·h, nodes × walltime (user, 2026-10-29: "不必在意文档中描述的预算, 保持每单个run在16 node * h即可";
  `scripts/submit.py` enforces it). Budgets written in plans and preregs no longer cap jobs or stages. Walltime =
  measured × 1.5. Record node·h per job.
- Dependencies through uv; do not change `pyproject.toml`/`uv.lock` without saying why in your report.
- Hashes: blake2b only, never SHA-256. Seeds via `derive_seed(base, label…)`.
- Work in your own worktree and branch; commit there; do not merge into master, do not push, do not rewrite
  history. Preserve unrelated changes.
- Every new study gets a pre-registration draft in `docs/prereg/<NAME>.md` on your branch. The orchestrator
  reviews it and commits it to master before any run. Decision rules come from `plan/followup_2026-10-27.md`;
  flag any deviation instead of silently changing it. Never read test/confirmation data before its registered
  time.
- Keep the off path bitwise unchanged when you add an option (add a test that proves it).
- Final reply: short — commits, files, what was verified and how, open questions. Details go in the report file.

## Step 2 on compute (from 2026-10-28 23:40 JST)
- You run on the login node (`/usr/bin/python3 -I /home/x10041/.claude/skills/miyabi-pbs-cmd/scripts/context.py
  --target-system Miyabi-G` shows the plane). torch, pytest, training and evaluation run only inside PBS jobs submitted
  with `scripts/submit.py --commit <full hash>`; the commit must contain your branch head named in the runbook.
- A short job (a few minutes) may use `submit.py --wait` inside one Bash call (timeout ≤ 600000 ms).
- For a longer wait, do not sleep or poll in a loop. End your turn with one line `WAIT <job ids> | <next action>`.
  The orchestrator arms a watcher and messages you once every listed job has left the queue. Then read the logs and
  outputs: exit status 0 alone is not success.
- End your turn with a line starting `STOP:` for anything the orchestrator decides: a lock commit, a failure that the
  prereg and its clarifications do not cover, any deviation from a registered rule (registered budget caps excepted: they no
  longer bind), a job above 16 node·h.
- Keep at most 6 of your jobs queued or running at once; four workers share the project's 16 running jobs.
- Record every job (ID, node·h, outcome) in your WP report as you go, and commit on your branch.
