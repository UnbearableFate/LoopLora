# Workers: long-lived codex agents with fixed roles (AGENTS.md)

Policy (from 2026-10-26 21:02): each worker keeps one role across tasks, in its own herdr tab, worktree and
branch. The orchestrator reuses it for every task in that role and does not close it after a task. When its
context is low or stale (repeated compaction, or before a large unrelated task), the orchestrator clears it
(`/new` in the codex TUI; record the new session id here) and re-injects context with one message:
"Read docs/README.md, your role row in docs/workers.md, the listed context docs, then docs/tasks/<task>.md".

Monitoring (user rule, 2026-10-26 23:45): the orchestrator checks every worker every 20 minutes and sends
"continue" to a worker that stopped unexpectedly (model/stream error, interrupted turn, idle mid-task without a
done report, armed PBS wakeup or live job). Idle after "WP-… done", idle with a wakeup armed, and a pending
question are expected states.

Model: sessions opened or resumed from 2026-10-26 19:36 use GPT-6.1-sol xhigh; wk-sym still runs its original
GPT-6-Astra xhigh session until its next clear.

Open or resume (new tab, worktree cwd):
`herdr tab create --workspace w1 --cwd <worktree> --label <name> --no-focus`, then
`herdr agent start <name> --kind codex --pane <pane> -- [resume <session-id>] -m gpt-6.1-sol -c model_reasoning_effort=xhigh`.

| worker | role (owns) | worktree, branch | context to inject | session | state |
| --- | --- | --- | --- | --- | --- |
| wk-sym | symbolic mechanism experiments: `src/looplora/sym`, `configs/p1s`, runs under `runs/p1s` | LoopLora-wt/sym, wp/sym | `docs/prereg/P1S.md`, `docs/reports/WP-D.md`, `docs/reports/P1S-pilot.md` | 01a177a0-f742-7c63-8800-0983d801fc19 | idle (WP-H done) |
| wk-llm | LLM module experiments: `src/looplora/llm`, `configs/p1l`, runs under `runs/p1l` | LoopLora-wt/llm, wp/llm | `docs/prereg/P1L.md`, `docs/reports/WP-E.md`, `docs/reports/WP-G.md` | 01a1779d-c123-75a1-9458-ae849ee0fcde (resumed, GPT-6.1-sol) | active: WP-L2 |
| wk-theory | theory and analytical checks: `docs/theory.md`, `tests/test_theory_examples.py` | LoopLora-wt/theory, wp/theory | `docs/theory.md`, `docs/reports/WP-T.md` | 01a177c4-57fb-7f42-a920-533e74680115 (resumed 21:47, GPT-6.1-sol) | idle (tab w1:t14) |
| wk-core | core adapters, memory, accounting: `src/looplora/core` | LoopLora-wt/core, wp/core | `docs/notation.md`, `docs/reports/WP-A.md` | 01a1779d-747b-7df1-8492-b517a9be8c6a (resumed 22:13, GPT-6.1-sol) | idle (WP-C4 done; WP-C5 queued after WP-L2 merges; tab w1:t15) |
| wk-data | data generators, rendering, real-task preprocessing: `src/looplora/data` | LoopLora-wt/data, wp/data | `docs/reports/WP-B.md`, `docs/tasks/WP-B_contract.md` | 01a1779d-96b3-7431-8fb7-0458e768764e (resumed 21:19, GPT-6.1-sol) | idle |
| wk-symf | symbolic follow-up studies (capacity/rank, later P2-S budget matching): runs under `runs/p1s/f1`, `reports/p1s-f1` | LoopLora-wt/symf, wp/symf | `docs/prereg/P1S.md`, `docs/prereg/P1S-F1.md`, `docs/reports/P1S-pilot.md` | 01a17828-31b5-75f0-89b3-6bfbcb4d89a5 (GPT-6.1-sol, opened 21:24) | idle (WP-F2 done; tab w1:t13) |

## Task history
| task | worker | merged into master |
| --- | --- | --- |
| WP-A core | wk-core | b367d3a |
| WP-B data | wk-data | 678ab7c |
| WP-E LLM injection | wk-llm | c88485f |
| WP-D symbolic pipeline | wk-sym | c86d72b |
| WP-T theory | wk-theory | d897f02 |
| WP-G P1-L pipeline | wk-llm | 2bf5d6d |
| WP-D2 P1-S pilots | wk-sym | 633f37c |
| WP-H P1-S main runs | wk-sym | 758c06a |
| WP-L1 P1-L Pilot-1 | wk-llm | 9eb0e10 |
| WP-L2 P1-L writer diagnosis | wk-llm | (running) |
| WP-B2 swap_successors counterfactual | wk-data | bbe2a24 |
| WP-B3 MuSiQue data and retention suite (T-08) | wk-data | ba2b2e8 |
| WP-T2 terminal-set shortcut and writer designs | wk-theory | 0c63cdc |
| WP-F1 P1-S rank follow-up | wk-symf | 7155dc1 |
| WP-C2 factorized writer code for F2 | wk-core | 132fc71 |
| WP-F2 P1-S factorized-writer follow-up | wk-symf | 4bafd71 |
| WP-C3 P4-S stopping pilot | wk-core | 5bdd0b6 |
| WP-C4 P4-S stopping on r = 32 learned codes | wk-core | ecf397f |
