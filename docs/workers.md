# Workers: long-lived codex agents with fixed roles (AGENTS.md)

**Codex reopened (2026-10-29 20:45 JST, user: "在本次静态评审结束后,将worker任务重新分派给在 /herdr 新tab里启动的codex
(gpt-6.1-sol high) , 代码审查则使用单独的codex (gpt-6-astra xhigh) 和 opus-5.5 xhigh subagent进行交叉评审").**
- Worker tasks go to codex workers (GPT-6.1-sol, reasoning effort high), each in its own herdr tab, worktree and
  branch, one per role and reused across tasks (policy below). The active role moved first: sn-slot → wk-slot
  (table below). An idle Sonnet role moves to a codex worker when it gets its next task.
- Code reviews are cross-reviews of the same commit by two independent reviewers: a codex reviewer (GPT-6-Astra,
  reasoning effort xhigh, own herdr tab, read-only) and an Opus-5.5 subagent of the main session (effort xhigh). Each
  gets the code and the spec, not the other's findings or the orchestrator's view. The orchestrator compares the two
  reports, commits both, and sends the worker the fixes.
- Review scope (user, 2026-10-29 21:53: "不要搞这么多次review,核心代码没问题就行"):
  - one cross-review per change of core code (models, data, training, analysis rules, sealed-set guards);
  - fixes, runbooks and docs get no further review round: the worker's tests check fixes, and the orchestrator reads the diff.
- The weekly-quota rule still holds: below 20 % left, roles go back to Sonnet subagents after the current task.

**Handover (2026-10-27 07:27 JST, user: "提前启动交接过程,关闭所有的codex worker").** All seven codex workers are
closed. Each role passes, unchanged (same worktree, branch and context docs), to a Claude Sonnet-5.5 subagent of
the main session, spawned on demand with its role row, the task file and the WP state file; subagents report by
committed documents plus a short final reply. Do not reopen codex workers unless the user asks. Codex sessions
below are kept for provenance only (their herdr tabs were closed at 07:28 on the user's request); wk-p1l's handover is `docs/reports/P1L-state.md` on wp/p1l (aa5005c, all codex
wake-ups cancelled).

| subagent | role | worktree, branch | state |
| --- | --- | --- | --- |
| sn-llm | wk-llm: LLM module experiments (learned writer line) | LoopLora-wt/llm, wp/llm | WP-L5 step 1 done (8eb24ab); WP-L7 step 1 done (9588998); RFX and LSQ registered on master (077bb81, 64f3e54); WP-L7 step 1b done (17b27c2: code to the registered rules, tests unrun); static review: no defect (docs/review/static_2026-10-27/rfxlsq.md); step 1c done (0a3048a: entity-EM guard, stdlib release scripts); second review (docs/review/static_2026-10-27/release.md): no defect; step 1d done (b79e0f8: restricted argmax breaks ties as the full argmax; smoke reload check before a release); WP-L7 step 2: tests fixed (efadc36, fixture rank), DPFP release 1dd1869, RFX verdicts store+query/store+query/none (379fccd, report faacd37), LSQ not triggered; merged d4de948; DPFP E16 arms 39.0/58.0/60.8% -> registered E8 retrain of D128-3 (orchestrator lock 72ed469) 73.0% < 80%: **stop**, no writer, P1L-D not run; zero/shuffle audit, overrun record (2b75fcf); merged 711d1ac; WP-L7 step 2 done; WP-L8 done (DPFP-X, `docs/reports/DPFP-X.md`): X256 66.2 %, X17 59.3 % (−1.5 pp against D128-3, not distinguished), neither reaches 0.80, learned-writer line stays stopped; 3.05 node·h; merged e86bdfe; WP-L12 (G5) SQ2 done (does not advance: S-RC − S-DE +4.1 pp, read-error ratio 0.554; best cell S-RE 72.3 %) and LSQT done (best ridge arm LT-RG001 46.1 %, band 40 % to < 48.5 %: line open, W3 not triggered); merged 1e28bed; LSQT-U control done (LT-U 36.0 %; RG001 over R-r128 +12.6 pp splits into +2.5 pp unit query and +10.1 pp store; one seed, descriptive; 1.452 node·h), merged b60f144; idle |
| sn-p1l | wk-p1l: P1-L main study | LoopLora-wt/p1l, wp/p1l (wp/p1l merged into master as ca85a40) | WP-P1LC step A done (wp/p1l c5de5c2: B7 check in finalize, P1L-C code and tests, unrun); static review: two low findings (docs/review/static_2026-10-27/p1l.md); step A2 done (02fc561: master merged, one source in the P1L-C audit, launcher deadline removed, interrupted-run rule); step B done (395e309: confirmation, B7 r32 fail / r128 pass, G1-L fail, G2-L pass, B6 met at r128; `reports/decisions/P1L.md`); step C done (2d9bb1d: P1L-C, no causal-bridge claim); merged 16e4fca, 28e122f; WP-P5L done (755201e: Qwen3-1.7B-Base, B7 pass, B6 met +25.78 pp, equal-parameter result holds at a second scale; 16.12 node·h); merged c6b5275; WP-L10 done (B8W, `reports/decisions/B8W.md`, c04a608): untying keeps the solution, Delta untied@4 - shared@4 on hops 2-3 +0.90 pp [0.09, 1.71], no seed loses; B8 gets no equal-parameter status, G1-L stays failed; shared arm +5.09 pp over the start (reader not converged); 1.004 node·h; merged 6f3511d; idle |
| sn-oracle | new (2026-10-27): LLM oracle code designs, WP-L6 and P1L-S | LoopLora-wt/oracle, wp/oracle | WP-L6 step 1 done (wp/oracle 3a788b1; preregs WP-L6 and P1L-S registered 4ac2410); static review: one low finding plus P1L-S traps (docs/review/static_2026-10-27/oracle.md); step 1b done (7175e09: runner executable, P1L-S implementation plan); step 1c done (4dcc6b3: P1L-S pipeline and tests on top of wp/p1l 02fc561, P1-L defaults unchanged; pilot stays at 7175e09); step 2 done (98d048c): registered decision **stop** (hop-1 EM O-rot 0.1%, O-shared 11.8% < 30%), P1L-S not run; merged ddfcbc2; WP-L11 done (CTC, `docs/reports/CTC.md`, 7b69f60): candidate flagged, arm CE, p = 1.0 (12,000 of 12,000 facts), M1-L4@4 hop 1 50.7 %, +24.05 pp [21.8, 26.3] over B4@1 on hops 2-3, equal to the oracle reference within 0.2 pp; entities read at their own tokens and the relation is constant, so it nearly restates the oracle result (not a learned writer); confirmation only by user decision; 1.028 node·h; merged 525ab32; idle |
| sn-sym | wk-sym/wk-symf: symbolic studies | LoopLora-wt/sym, wp/sym | WP-S9 step 1 done (c443e01); IVD, CVH, TSM, UNS, NCR registered on master (40ddf08); WP-S9 step 1b done (be2a972); TSM Clarifications C1 (3477526) and C2 (d488b78); steps 1c and 1d done (87746bf); static review: F1 Medium, others Low (docs/review/static_2026-10-27/tsm.md); TSM Clarification C3 (master d335fe7, wp/sym 3ca4d89); step 1e done (09be205: F1-F8 and C3 in the TSM scripts); second review (docs/review/static_2026-10-27/tsm2.md) -> TSM Clarification C4 (master d6ceb6a, wp/sym 76a00b3); step 1f done (d32463e: C4 in the TSM scripts, stopping child and heartbeat notes); step 2 done (70bfdfc; locks IVD 2a8d063, CVH 8a13c45, CVH theta 19c5613, UNS a836928, TSM 1bf689b; about 1.8 node·h); step 3 done (39b5875: reports IVD, CVH, TSM, UNS, NCR); merged into master d689235 |
| sn-paper | wk-theory/T-tasks: paper and claim contract | LoopLora-wt/paper, wp/paper | WP-T9 done, merged 1cc0e16; WP-T10 parts 1, 2, 3a, 3b, 4, 5 merged (8a1f14a, cfb50a9, 172edba, 047fe8f, f4cd105, 15f9ec8); WP-T11 done and merged 8b2ed97 (G1 revision a12ed8f; LaTeX manuscript `docs/paper/tex/`, PDF `docs/paper/paper.pdf`, 54 pages, d6dab2d); WP-T12 part 1 done (a472386: ledger LS1-LS4, BW1-BW3, CT1-CT3; §3.4, §4.4, App. I; PDF 60 pages), merged; part 2 done (6cec11f, fix ea95759: ledger SQ1-SQ3, LQ1-LQ3; §4.2 and App. I; PDF 64 pages), merged dc5cbdf; part 3 done (5019752: ledger LQ4, LQ3 open -> narrowed; §4.2 U paragraph, Tables 12-13 and 41; PDF 66 pages), merged b932ef4; idle |
| sn-core | wk-core: stopping and readers (2026-10-29: G2) | LoopLora-wt/core, wp/core | WP-L9 done (LSR, `docs/reports/LSR.md`, 302aa4a; prereg 63da9c62 after C1): R2 logit-lens stability passes at 0.6B and 1.7B (hops 4-8 +3.90 / +5.68 pp over L = 4, hop 1 +6.3 / +4.7; hop 2 -10.8 / -12.3, below L = 8 on hops 4-8, 5.6-5.7 reads, no efficiency claim); R1, R3 infeasible on validation; 0.50 node·h; merged 975bb6a; idle |
| sn-slot | new (2026-10-29): SLOT-LOOP-v1 (plan/looplora_diagnosis_and_execution_plan_2026-10-29.md), S0 to S3 by gates | LoopLora-wt/slot, wp/slot | prereg registered 5f04a5c; WP-L13 step 2 done (ccdcc96: implementation, compute tests, state file; 0.503 node·h); static review round 1: no high (`docs/review/static_2026-10-29/slot-*.md`, 6c21e19); handed over to codex wk-slot 2026-10-29 20:45 |
| (on demand) | wk-data, wk-theory | — | no pending work |
| rv-fable | external reviewer, Claude Fable 5.1 (max), herdr tab w1:t17 (user request 2026-10-27) | — | done (c370e22); tab closed after the report |
| rv-astra | external reviewer, Codex GPT-6-Astra (max, web search), herdr tab w1:t18 (user request 2026-10-27) | — | done 08:18 (6179f3a); codex closed, tab closed |
| rv-fable (round 2) | external reviewer, Claude Fable 5.1 (max), herdr tab w1:t19 (user request 2026-10-29) | — | done (9f04490); tab closed; integrated in `plan/followup_2026-10-29.md` |
| rv-astra (round 2) | external reviewer, Codex GPT-6-Astra (max, web search), herdr tab w1:t1A (user request 2026-10-29) | — | done (e198ad7); tab closed; integrated in `plan/followup_2026-10-29.md` |
| rv-code | code reviewer, Codex GPT-6-Astra xhigh, herdr tab w1:t1C, session 01a1878b-98c6-7800-abe2-9d68a5c81790 (user request 2026-10-29); kept for later review rounds | main checkout, read-only; reviews a detached worktree (`LoopLora-wt/review-slot`) | SLOT-LOOP-v1 round 2 done 21:15 (wp/slot 2438599: 1 High, 2 Medium; `docs/review/static_2026-10-29/round2-rv-code.md`); round 3 done 21:51 (8c28334: 1 Medium; `round3-rv-code.md`); S1B review done 23:44 (wp/loop 67deeee: 1 High, 4 Medium; `s1b-rv-code.md`); S1A selection re-run done 00:11 (job 3729280: reproduced byte for byte; `s1a-selection-rerun.md`) |
| rv-opus | code reviewer, Opus-5.5 xhigh subagent of the main session (user request 2026-10-29), one per review round | as rv-code | SLOT-LOOP-v1 round 2 done 21:24 (2 Medium, 2 Low; `round2-rv-opus.md`); round 3 done 21:50 (same subagent resumed; 1 Low; `round3-rv-opus.md`); S1B review done 23:49 (new subagent; 1 High, 1 Medium, 4 Low; `s1b-rv-opus.md`) |

Tasks follow `plan/followup_2026-10-27.md`; rules shared by all subagent tasks: `docs/tasks/_common_rules.md`.

Policy (from 2026-10-26 21:02): each worker keeps one role across tasks, in its own herdr tab, worktree and
branch. The orchestrator reuses it for every task in that role and does not close it after a task. When its
context is low or stale (repeated compaction, or before a large unrelated task), the orchestrator clears it
(`/new` in the codex TUI; record the new session id here) and re-injects context with one message:
"Read docs/README.md, your role row in docs/workers.md, the listed context docs, then docs/tasks/<task>.md".

Monitoring (user rule, 2026-10-26 23:45; narrowed by the user 2026-10-29): no timed checks. In-session subagents
reach the orchestrator by notification at each stop. Each codex worker gets one background `herdr agent wait <name>`
after every prompt; it returns on idle, done or blocked, and the orchestrator reads the pane's last lines. For every
`WAIT <job ids>` line the orchestrator arms a background watcher (`wait_jobs.sh`) and prompts the worker when it
exits. A worker that stopped unexpectedly (model/stream error, interrupted turn, idle mid-task without a done, WAIT
or STOP line) gets "continue". A blocked approval or question UI goes to the user.

Model: workers opened from 2026-10-29 use GPT-6.1-sol with reasoning effort high (user), reviewers GPT-6-Astra
xhigh. The sessions closed on 2026-10-27 ran GPT-6.1-sol xhigh (wk-sym: GPT-6-Astra xhigh until 03:46 on 2026-10-27).

Open or resume (new tab, worktree cwd; a reviewer's cwd is the main checkout):
`herdr tab create --workspace w1 --cwd <worktree> --label <name> --no-focus`, then
`herdr agent start <name> --kind codex --pane <pane> -- [resume <session-id>] -m gpt-6.1-sol -c model_reasoning_effort=high`
(reviewers: `-m gpt-6-astra -c model_reasoning_effort=xhigh`). Codex 0.163.0 opens with an update prompt (0.165.0):
the orchestrator skips it (esc) and leaves updating to the user.

| worker | role (owns) | worktree, branch | context to inject | session | state |
| --- | --- | --- | --- | --- | --- |
| wk-sym | symbolic mechanism experiments: `src/looplora/sym`, `configs/p1s`, runs under `runs/p1s` | LoopLora-wt/sym, wp/sym | `docs/prereg/P1S.md`, `docs/reports/WP-D.md`, `docs/reports/P1S-pilot.md` | 01a177a0-f742-7c63-8800-0983d801fc19 (resumed 03:46 2026-10-27, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-llm | LLM module experiments: `src/looplora/llm`, `configs/p1l`, runs under `runs/p1l` | LoopLora-wt/llm, wp/llm | `docs/prereg/P1L.md`, `docs/reports/WP-E.md`, `docs/reports/WP-G.md` | 01a1779d-c123-75a1-9458-ae849ee0fcde (resumed, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-theory | theory and analytical checks: `docs/theory.md`, `tests/test_theory_examples.py` | LoopLora-wt/theory, wp/theory | `docs/theory.md`, `docs/reports/WP-T.md` | 01a177c4-57fb-7f42-a920-533e74680115 (resumed 21:47, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-core | core adapters, memory, accounting: `src/looplora/core` | LoopLora-wt/core, wp/core | `docs/notation.md`, `docs/reports/WP-A.md` | 01a1779d-747b-7df1-8492-b517a9be8c6a (resumed 22:13, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-data | data generators, rendering, real-task preprocessing: `src/looplora/data` | LoopLora-wt/data, wp/data | `docs/reports/WP-B.md`, `docs/tasks/WP-B_contract.md` | 01a1779d-96b3-7431-8fb7-0458e768764e (resumed 21:19, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-symf | symbolic follow-up studies (capacity/rank, later P2-S budget matching): runs under `runs/p1s/f1`, `reports/p1s-f1` | LoopLora-wt/symf, wp/symf | `docs/prereg/P1S.md`, `docs/prereg/P1S-F1.md`, `docs/reports/P1S-pilot.md` | 01a17828-31b5-75f0-89b3-6bfbcb4d89a5 (GPT-6.1-sol, opened 21:24) | closed 2026-10-27 07:27 (handover) |
| wk-p1l | P1-L main study (Pilot-2, main runs, confirmation, analysis): `configs/p1l` run sets, runs under `runs/p1l/main`, `reports/p1l` | LoopLora-wt/p1l, wp/p1l | `docs/prereg/P1L.md`, `docs/reports/WP-C7.md`, `docs/reports/WP-C6.md`, `configs/p1l/README.md` | 01a17999-7c19-7ac2-a6d2-cf6aee2583c0 (opened 04:05 2026-10-27, GPT-6.1-sol) | closed 2026-10-27 07:27 (handover) |
| wk-slot | SLOT-LOOP-v1, S0 to S3 by gates: `src/looplora/llm/slot_*.py`, `src/looplora/core/memory_backend.py`, `configs/slot_loop_v1`, runs under `runs/slot-loop-v1` (took over from sn-slot at ccdcc96) | LoopLora-wt/slot, wp/slot | `docs/prereg/SLOT-LOOP-v1.md`, `docs/tasks/WP-L13_slot_loop.md`, `docs/reports/SLOT-LOOP-v1-state.md` (wp/slot), `plan/looplora_diagnosis_and_execution_plan_2026-10-29.md` | 01a18778-254b-7613-983f-7fe00f6ab7da (opened 20:42 2026-10-29, GPT-6.1-sol high, herdr tab w1:t1B) | review round 1 fixes done (b782450; fixture fix ca2855a after job 3728079; tests 3728078, 3728107 pass; 0.117 node·h); STOP: review 2438599; round 2: four fixes (11ff92a; red 3728243, green 3728256 and 3728257), STOP: review 8c28334; round 3 closed by the user (no more review rounds); preflight 3728339 passed, lock 2f22aba, S0 3728354, decision R2 (3475ecc; 0.778 node·h), merged 2f7abbc; S1A: release 30e8dab approved (approval 1114287), six runs (1.75 node·h), dev no_trigger (03c6046), open record 7a113ae, verdict ADVANCE (81e8996), merged 650dc3e; idle |
| wk-loop | SLOT-LOOP-v1 later stages in parallel with wk-slot: S1B addendum and loop-depth code on the slot backends (then S2/S3 preparation) | LoopLora-wt/loop, wp/loop (from master 2f7abbc) | `docs/tasks/WP-L14_slot_s1b.md`, `docs/prereg/SLOT-LOOP-v1.md`, `docs/reports/SLOT-LOOP-v1.md`, plan §8 | 01a187d7-72f7-7510-878f-f7a10f0cd053 (opened 22:26 2026-10-29, GPT-6.1-sol high, herdr tab w1:t1D) | addendum draft 73250eb, registered a9178d5; S1B code 67deeee (one cross-review); fixes 1-9 4a76e16 (red 3729223, green 3729273: 179 passed); preflight 3729312, lock bd4e24f (src bb60b89c); active: advance branch (calibration, smoke, pilot, release, then STOP for approval.json) |

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
| WP-T10 part 5 (DPFP-X into the LLM ledger DP4–DP7 and the draft §8.5) | sn-paper | 15f9ec8 |
| WP-T11 G1 paper revision and readable LaTeX/PDF manuscript (tectonic, /write; user request) | sn-paper | 8b2ed97 |
| WP-L9 LSR label-free stop rules (G2; R2 passes at both backbones, R1/R3 infeasible) | sn-core | 975bb6a |
| WP-L10 B8W warm start (G3; untying keeps the solution, non-inferiority met) | sn-p1l | 6f3511d |
| master test after WP-L9 and WP-L10 merges (job 3724308 at 6f3511d: 2801 passed, 1 skipped) | orchestrator | 6f3511d |
| WP-L11 CTC classify-then-code (G4; candidate flagged at p = 1, nearly the oracle result) | sn-oracle | 525ab32 |
| WP-L12 G5 SQ2 (does not advance) and LSQT (band open, line neither closed nor triggered) | sn-llm | 1e28bed |
| master test after the WP-L12 merge (job 3725533 at 1e28bed: 2957 passed, 1 skipped) | orchestrator | 1e28bed |
| WP-T12 part 2 (G5 SQ2 and LSQT into the ledger, LaTeX manuscript and PDF; 64 pages) | sn-paper | dc5cbdf |
| WP-L12 LSQT-U control (descriptive; U 36.0 %, split +2.5 pp unit query / +10.1 pp store) | sn-llm | b60f144 |
| master test after the cleanup of finished smoke/pilot weights (job 3727237 at 177a4fe: 2957 passed, 1 skipped; `docs/review/cleanup_2026-10-29.md`) | orchestrator | 177a4fe |
| WP-T12 part 3 (LSQT U control in the ledger, manuscript and PDF; LQ4, LQ3 narrowed) | sn-paper | b932ef4 |
| master test after the LSQT-U merge (job 3727608 at 37abbfc: 2960 passed, 1 skipped; +3 = `tests/test_analyze_lsqt_u.py`) | orchestrator | 37abbfc |
| resource rule (user, 2026-10-29): written budgets no longer bind, each job ≤ 16 node·h in `scripts/submit.py` (job 3728309: `tests/test_submit_cap.py` 6 passed) | orchestrator | 07cf11a |
| WP-L13 SLOT-LOOP-v1 S0 (slot backends, data, analysis; preflight with Matrix bitwise equal to Native; lock; decision R2) | sn-slot, wk-slot | 2f7abbc |
| master test after the S0 merge (jobs 3728584 and 3728585 at 2f7abbc, split in two debug-g jobs: 3257 passed, 1 skipped) | orchestrator | 2f7abbc |
| S1B cross-review (rv-code and rv-opus at wp/loop 67deeee) and the fix list in WP-L14 | orchestrator | cba968b |
| WP-L13 SLOT-LOOP-v1 S1A (release 523d11d5; six runs; dev no_trigger, re-run reproduced; confirmation ADVANCE, +3.73 pp [+2.70, +4.82]) | wk-slot | 650dc3e |
| master test after the S1A merge (jobs 3729456, 3729457, 3729458 at 650dc3e, three debug-g shards: 3270 passed, 1 skipped) | orchestrator | 650dc3e |
| Merged master full test suite after DPFP-X (job 3723646, exit 0, 2717 passed, 1 expected skip) | orchestrator | e86bdfe |
| Merged master full test suite after P5L (job 3722062, exit 0, 2676 passed, 1 expected skip) | orchestrator | c6b5275 |
| Merged master full test suite (job 3721356, exit 0, 1 expected skip) | orchestrator | d4de948 |
