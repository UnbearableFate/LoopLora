# Workers: long-lived codex agents with fixed roles (AGENTS.md)

**Handover (2026-10-27 07:27 JST, user: "提前启动交接过程,关闭所有的codex worker").** All seven codex workers are
closed. Each role passes, unchanged (same worktree, branch and context docs), to a Claude Sonnet-5.5 subagent of
the main session, spawned on demand with its role row, the task file and the WP state file; subagents report by
committed documents plus a short final reply. Do not reopen codex workers unless the user asks. Codex sessions
below are kept for provenance only (their herdr tabs were closed at 07:28 on the user's request); wk-p1l's handover is `docs/reports/P1L-state.md` on wp/p1l (aa5005c, all codex
wake-ups cancelled).

| subagent | role | worktree, branch | state |
| --- | --- | --- | --- |
| sn-llm | wk-llm: LLM module experiments (learned writer line) | LoopLora-wt/llm, wp/llm | WP-L5 step 1 done (8eb24ab); WP-L7 step 1 done (9588998); RFX and LSQ registered on master (077bb81, 64f3e54); WP-L7 step 1b done (17b27c2: code to the registered rules, tests unrun); static review: no defect (docs/review/static_2026-10-27/rfxlsq.md); step 1c done (0a3048a: entity-EM guard, stdlib release scripts); second review (docs/review/static_2026-10-27/release.md): no defect; step 1d done (b79e0f8: restricted argmax breaks ties as the full argmax; smoke reload check before a release); WP-L7 step 2: tests fixed (efadc36, fixture rank), DPFP release 1dd1869, RFX verdicts store+query/store+query/none (379fccd, report faacd37), LSQ not triggered; merged d4de948; DPFP E16 arms 39.0/58.0/60.8% -> registered E8 retrain of D128-3 (orchestrator lock 72ed469) 73.0% < 80%: **stop**, no writer, P1L-D not run; zero/shuffle audit, overrun record (2b75fcf); merged 711d1ac; WP-L7 step 2 done; WP-L8 done (DPFP-X, `docs/reports/DPFP-X.md`): X256 66.2 %, X17 59.3 % (−1.5 pp against D128-3, not distinguished), neither reaches 0.80, learned-writer line stays stopped; 3.05 node·h; merged e86bdfe; idle |
| sn-p1l | wk-p1l: P1-L main study | LoopLora-wt/p1l, wp/p1l (wp/p1l merged into master as ca85a40) | WP-P1LC step A done (wp/p1l c5de5c2: B7 check in finalize, P1L-C code and tests, unrun); static review: two low findings (docs/review/static_2026-10-27/p1l.md); step A2 done (02fc561: master merged, one source in the P1L-C audit, launcher deadline removed, interrupted-run rule); step B done (395e309: confirmation, B7 r32 fail / r128 pass, G1-L fail, G2-L pass, B6 met at r128; `reports/decisions/P1L.md`); step C done (2d9bb1d: P1L-C, no causal-bridge claim); merged 16e4fca, 28e122f; WP-P5L done (755201e: Qwen3-1.7B-Base, B7 pass, B6 met +25.78 pp, equal-parameter result holds at a second scale; 16.12 node·h); merged c6b5275 |
| sn-oracle | new (2026-10-27): LLM oracle code designs, WP-L6 and P1L-S | LoopLora-wt/oracle, wp/oracle | WP-L6 step 1 done (wp/oracle 3a788b1; preregs WP-L6 and P1L-S registered 4ac2410); static review: one low finding plus P1L-S traps (docs/review/static_2026-10-27/oracle.md); step 1b done (7175e09: runner executable, P1L-S implementation plan); step 1c done (4dcc6b3: P1L-S pipeline and tests on top of wp/p1l 02fc561, P1-L defaults unchanged; pilot stays at 7175e09); step 2 done (98d048c): registered decision **stop** (hop-1 EM O-rot 0.1%, O-shared 11.8% < 30%), P1L-S not run; merged ddfcbc2 |
| sn-sym | wk-sym/wk-symf: symbolic studies | LoopLora-wt/sym, wp/sym | WP-S9 step 1 done (c443e01); IVD, CVH, TSM, UNS, NCR registered on master (40ddf08); WP-S9 step 1b done (be2a972); TSM Clarifications C1 (3477526) and C2 (d488b78); steps 1c and 1d done (87746bf); static review: F1 Medium, others Low (docs/review/static_2026-10-27/tsm.md); TSM Clarification C3 (master d335fe7, wp/sym 3ca4d89); step 1e done (09be205: F1-F8 and C3 in the TSM scripts); second review (docs/review/static_2026-10-27/tsm2.md) -> TSM Clarification C4 (master d6ceb6a, wp/sym 76a00b3); step 1f done (d32463e: C4 in the TSM scripts, stopping child and heartbeat notes); step 2 done (70bfdfc; locks IVD 2a8d063, CVH 8a13c45, CVH theta 19c5613, UNS a836928, TSM 1bf689b; about 1.8 node·h); step 3 done (39b5875: reports IVD, CVH, TSM, UNS, NCR); merged into master d689235 |
| sn-paper | wk-theory/T-tasks: paper and claim contract | LoopLora-wt/paper, wp/paper | WP-T9 done, merged 1cc0e16; WP-T10 parts 1, 2, 3a, 3b, 4 merged (8a1f14a, cfb50a9, 172edba, 047fe8f, f4cd105); active: part 5 (DPFP-X descriptive rows) |
| (on demand) | wk-core, wk-data, wk-theory | — | no pending work |
| rv-fable | external reviewer, Claude Fable 5.1 (max), herdr tab w1:t17 (user request 2026-10-27) | — | done (c370e22); tab closed after the report |
| rv-astra | external reviewer, Codex GPT-6-Astra (max, web search), herdr tab w1:t18 (user request 2026-10-27) | — | done 08:18 (6179f3a); codex closed, tab closed |
| rv-fable (round 2) | external reviewer, Claude Fable 5.1 (max), herdr tab w1:t19 (user request 2026-10-29) | — | done (9f04490); tab closed; integrated in `plan/followup_2026-10-29.md` |
| rv-astra (round 2) | external reviewer, Codex GPT-6-Astra (max, web search), herdr tab w1:t1A (user request 2026-10-29) | — | done (e198ad7); tab closed; integrated in `plan/followup_2026-10-29.md` |

Tasks follow `plan/followup_2026-10-27.md`; rules shared by all subagent tasks: `docs/tasks/_common_rules.md`.

Policy (from 2026-10-26 21:02): each worker keeps one role across tasks, in its own herdr tab, worktree and
branch. The orchestrator reuses it for every task in that role and does not close it after a task. When its
context is low or stale (repeated compaction, or before a large unrelated task), the orchestrator clears it
(`/new` in the codex TUI; record the new session id here) and re-injects context with one message:
"Read docs/README.md, your role row in docs/workers.md, the listed context docs, then docs/tasks/<task>.md".

Monitoring (user rule, 2026-10-26 23:45; narrowed by the user 2026-10-29): the 20-minute check applied to the herdr
codex workers. In-session subagents get no timed check: each stop reaches the orchestrator as a notification, and the
orchestrator arms a background watcher (`wait_jobs.sh`) for every `WAIT <job ids>` line and resumes the subagent when
it exits. A subagent that stopped unexpectedly (model/stream error, interrupted turn, idle mid-task without a done,
WAIT or STOP line) gets "continue".

Model: sessions opened or resumed from 2026-10-26 19:36 use GPT-6.1-sol xhigh; wk-sym still runs its original
GPT-6-Astra xhigh session until its next clear (switched by resume at 03:46 on 2026-10-27; all workers now GPT-6.1-sol).

Open or resume (new tab, worktree cwd):
`herdr tab create --workspace w1 --cwd <worktree> --label <name> --no-focus`, then
`herdr agent start <name> --kind codex --pane <pane> -- [resume <session-id>] -m gpt-6.1-sol -c model_reasoning_effort=xhigh`.

| worker | role (owns) | worktree, branch | context to inject | session | state |
| --- | --- | --- | --- | --- | --- |
| wk-sym | symbolic mechanism experiments: `src/looplora/sym`, `configs/p1s`, runs under `runs/p1s` | LoopLora-wt/sym, wp/sym | `docs/prereg/P1S.md`, `docs/reports/WP-D.md`, `docs/reports/P1S-pilot.md` | 01a177a0-f742-7c63-8800-0983d801fc19 (resumed 03:46 2026-10-27, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-llm | LLM module experiments: `src/looplora/llm`, `configs/p1l`, runs under `runs/p1l` | LoopLora-wt/llm, wp/llm | `docs/prereg/P1L.md`, `docs/reports/WP-E.md`, `docs/reports/WP-G.md` | 01a1779d-c123-75a1-9458-ae849ee0fcde (resumed, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-theory | theory and analytical checks: `docs/theory.md`, `tests/test_theory_examples.py` | LoopLora-wt/theory, wp/theory | `docs/theory.md`, `docs/reports/WP-T.md` | 01a177c4-57fb-7f42-a920-533e74680115 (resumed 21:47, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-core | core adapters, memory, accounting: `src/looplora/core` | LoopLora-wt/core, wp/core | `docs/notation.md`, `docs/reports/WP-A.md` | 01a1779d-747b-7df1-8492-b517a9be8c6a (resumed 22:13, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-data | data generators, rendering, real-task preprocessing: `src/looplora/data` | LoopLora-wt/data, wp/data | `docs/reports/WP-B.md`, `docs/tasks/WP-B_contract.md` | 01a1779d-96b3-7431-8fb7-0458e768764e (resumed 21:19, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-symf | symbolic follow-up studies (capacity/rank, later P2-S budget matching): runs under `runs/p1s/f1`, `reports/p1s-f1` | LoopLora-wt/symf, wp/symf | `docs/prereg/P1S.md`, `docs/prereg/P1S-F1.md`, `docs/reports/P1S-pilot.md` | 01a17828-31b5-75f0-89b3-6bfbcb4d89a5 (GPT-6.1-sol, opened 21:24) | closed 2026-10-27 07:27 (handover) |
| wk-p1l | P1-L main study (Pilot-2, main runs, confirmation, analysis): `configs/p1l` run sets, runs under `runs/p1l/main`, `reports/p1l` | LoopLora-wt/p1l, wp/p1l | `docs/prereg/P1L.md`, `docs/reports/WP-C7.md`, `docs/reports/WP-C6.md`, `configs/p1l/README.md` | 01a17999-7c19-7ac2-a6d2-cf6aee2583c0 (opened 04:05 2026-10-27, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |

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
| WP-L2 P1-L writer diagnosis | wk-llm | 44b27cf |
| WP-L3 P1-L writer fix screen | wk-llm | 39f7f0b (with WP-L4) |
| WP-L4 P1-L read fix (last writer screen; Amendments A1, A2: parallel r = 128 arms) | wk-llm | 39f7f0b (no writer selected) |
| WP-C7 oracle writer in the LLM pipeline | wk-core | d057720 |
| WP-P4L learned halting on frozen readers (prereg `docs/prereg/P4S-L.md`) | wk-sym | merged after WP-T6 |
| WP-P2S symbolic budget matching (P2-S) | wk-symf | 0f84084 |
| WP-P2S-F1 sharing at matched depth; shortcut-free budget contrasts | wk-symf | f7d2aed |
| WP-T3 retrieval interference with learned keys (theory) | wk-theory | merged after WP-P2S-F1 |
| WP-C6 LLM support for B8 and fixed-depth training | wk-core | merged after WP-T3 |
| WP-T4 symbolic claims–evidence table (T-10 part 1) | wk-theory | 056a542 |
| WP-T5 paper draft, symbolic part (T-10 part 2) | wk-theory | merged after WP-T4 (`docs/paper/draft.md`) |
| WP-IS memory load × read depth (prereg `docs/prereg/IS.md`) | wk-symf | 641911d |
| WP-P1L P1-L main study (oracle_random condition first; Amendments B2: S = 17,000, B3: parallel r = 128 oracle condition) | wk-p1l, sn-p1l | 16e4fca (confirmation), 28e122f (P1L-C) |
| WP-JS Jacobian spectral-radius regularization (prereg `docs/prereg/JS.md`) | wk-symf | 1e40833 |
| WP-T6 IS into evidence/draft; latent-reasoning and Jacobian literature check | wk-theory | merged after WP-IS |
| WP-T7 P4-S-L and JS into evidence/draft; same-rank comparison framing | wk-theory | 8783a1b |
| WP-T8 LLM section part 1 (writer screens, P1-L protocol), manuscript wording | wk-theory | 97cd06c |
| WP-B2 swap_successors counterfactual | wk-data | bbe2a24 |
| WP-B3 MuSiQue data and retention suite (T-08) | wk-data | ba2b2e8 |
| WP-T2 terminal-set shortcut and writer designs | wk-theory | 0c63cdc |
| WP-F1 P1-S rank follow-up | wk-symf | 7155dc1 |
| WP-C2 factorized writer code for F2 | wk-core | 132fc71 |
| WP-F2 P1-S factorized-writer follow-up | wk-symf | 4bafd71 |
| WP-C3 P4-S stopping pilot | wk-core | 5bdd0b6 |
| WP-C4 P4-S stopping on r = 32 learned codes | wk-core | ecf397f |
| WP-C5 exact fast write-only scan | wk-core | 7f68a3c |
| WP-B4 background batch prefetch for LLM training | wk-data | merged (see git log) |
| WP-S9 symbolic follow-ups IVD, CVH, TSM, UNS, NCR (`plan/followup_2026-10-27.md` F8–F11) | sn-sym | d689235 |
| WP-L6 LLM oracle codes (registered stop; P1L-S not run) | sn-oracle | ddfcbc2 |
| WP-L7 RFX (LSQ not triggered); WP-L5 DPFP code and release | sn-llm | d4de948 |
| WP-L5 DPFP arms, E8 retrain and audit (registered stop; learned-writer line ends) | sn-llm | 711d1ac |
| WP-T10 part 1 follow-up results into the paper | sn-paper | 8a1f14a |
| WP-T10 part 2 (P1-L, P1L-C) and part 3a (LP5 readouts) | sn-paper | cfb50a9, 172edba |
| WP-T10 part 3b (DPFP negative result; P1L-D not run) | sn-paper | 047fe8f |
| WP-P5L second-scale confirmation (Qwen3-1.7B-Base; B6 met, holds at a second scale) | sn-p1l | c6b5275 |
| WP-T10 part 4 (P5L into the LLM ledger LP14–LP20 and the draft §8.5) | sn-paper | f4cd105 |
| WP-L8 DPFP-X (X256 r = 256, X17 Qwen3-1.7B-Base; registered stop, neither reaches 0.80) | sn-llm | e86bdfe |
| Merged master full test suite after DPFP-X (job 3723646, exit 0, 2717 passed, 1 expected skip) | orchestrator | e86bdfe |
| Merged master full test suite after P5L (job 3722062, exit 0, 2676 passed, 1 expected skip) | orchestrator | c6b5275 |
| Merged master full test suite (job 3721356, exit 0, 1 expected skip) | orchestrator | d4de948 |
