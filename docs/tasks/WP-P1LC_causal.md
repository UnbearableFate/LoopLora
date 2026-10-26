# WP-P1LC — P1-L confirmation with Amendment B7, and study P1L-C (sn-p1l; LoopLora-wt/p1l, wp/p1l)

Read first: `docs/tasks/_common_rules.md`, `docs/reports/P1L-state.md` (the full handover from the closed codex
worker wk-p1l — its warnings are binding, e.g. never rerun `scripts/split_p1l_packed_jobs.py`, never target the
closed codex session), `docs/prereg/P1L.md` (all amendments, especially B5–B7), `docs/prereg/P1L-C.md`,
`plan/followup_2026-10-27.md` (§2, F1, F7).

## Step A (now, no compute)
1. B7 memory-use check in the P1-L analysis: per rank, hop-1 full EM on the confirmation set of B4@1, B8@4 and
   M1-L4@4 minus B3-L4@4, pooled paired episode bootstrap (2000 draws) with 97.5% CIs and per-seed values; a
   per-rank flag `memory_use_passed`. Add it to the analysis outputs and results tables without changing any gate
   computation; tests (run in step B).
2. P1L-C implementation as new code that leaves the confirmation path unchanged (the confirmation evaluates the
   pinned source commit recorded in `P1L-state.md`):
   - analysis A: deployed reads vs memory only at the final question position (other question positions run with
     M = 0);
   - analysis B: worlds A/B/C (only the question's first edge differs; b_B and c drawn with a seed derived from the
     episode index), patch z_1 at the final question position (B-patch, C-patch, random direction of norm
     ‖z_1^B − z_1^A‖), secondary all-position patch; shift statistic and CIs as registered;
   - any hook added to the module/injection code is off by default and bitwise neutral (test);
   - tests on a tiny model: masking, patch mechanics, world construction (both continuations present, t_B ≠ t_A,
     c has no outgoing fact).
3. Step-B and step-C commands (P1L-C: ≤ 1 node × 1 h per job, test [102000, 104000), seed 4242).
Report: `docs/reports/P1L-C-state.md`. Done line in your final reply: `WP-P1LC step A done`.

## Step B (only when the orchestrator says so, after 2026-10-28 22:00)
Finish the three remaining P1-L trainings, then the confirmation evaluation and analysis exactly as
`P1L-state.md` and the prereg say, including the B7 check; draft `reports/decisions/P1L.md`.

## Step C
P1L-C, only if r = 128 passes the memory-use check. Report `docs/reports/P1L-C.md`.
