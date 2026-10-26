# WP-P1L — P1-L main study (oracle_random condition now; learned condition if WP-L4 selects a writer)

Read first: `docs/prereg/P1L.md` (v2 final for `oracle_random`: design, Pilot-2, endpoints, per-step readout,
decision use), `docs/reports/WP-C7.md` and `configs/p1l/README.md` (oracle writer usage), `docs/reports/WP-C6.md`
(B8 in the LLM pipeline), `docs/reports/P1L-writer-fix.md` (why the learned writer is pending),
`docs/prereg/P1S.md` (gate statistics conventions). Branch/worktree: `wp/p1l` (from `master`). Outputs:
`reports/p1l/` (design lock, lr/S lock, run manifest with blake2b hashes, accuracy by condition/reader/hop/L/seed
with counts, endpoints with CIs and per-seed values, per-step readout tables, figures), report
`docs/reports/P1L.md`.

## Steps
1. Design-lock commit before any training: every run config (oracle_random condition: B8, M1-L4, B4, B5-L4 ×
   seeds 17/29/43; B3-L4 × 3 seeds; B0; B1 and B2 context-retained × 3 seeds), the `t2_typed` replication set, the
   per-step readout and J-lens direction definitions.
2. 10% smoke for time and memory (no scores), then Pilot-2 on validation (lr ∈ {1e-3, 3e-3} at S/3 steps), and
   lock lrs and S in a commit before main training.
3. Main training: jobs ≤ 8 nodes × 2 h through `scripts/submit.py --scale-up docs/prereg/P1L.md` (pack runs per
   node when they fit); every job must end before 2026-10-28 09:00 JST. Use miyabi-pbs-goal-wakeup for waits.
4. Confirmation evaluation only when the prereg's conditions hold; ask the orchestrator whether WP-L4 has decided
   (it fixes the CI level) before evaluating. Endpoints and readout exactly as registered; report per-seed values
   and any departure from the prereg.
5. If the orchestrator adds the learned-writer addendum, run that condition with the same procedure.

## Limits
P1-L total ≤ 16 node·h; record node·h per job. No test-set use before step 4. Final line:
`WP-P1L done: docs/reports/P1L.md`.
