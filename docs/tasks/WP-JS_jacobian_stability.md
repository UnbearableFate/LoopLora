# WP-JS — Jacobian spectral-radius regularization of the reader (study JS)

Read first: `docs/prereg/JS.md` (registered design, endpoints, rules), `docs/reports/P2S-F1.md` and
`docs/reports/IS.md` (your training/evaluation code and checkpoints), `docs/reports/P4S-stopping.md` (relative-change
rule), `src/looplora/core/adapters.py` (reader step), `docs/theory.md` Props. 3 and 7. Branch/worktree: `wp/symf`
(merge `master` first). Outputs: `reports/js/` (design lock, ρ̂ implementation choice and its test, run manifest
with blake2b hashes, accuracy by writer/condition/hop/L/seed with counts, endpoints with CIs and per-seed values,
ρ(J) tables, figure: accuracy vs L per hop for RS, RS+J and the references), report `docs/reports/JS.md`.

## Steps
1. Design lock commit before training; a compute test that ρ̂ and its gradient match the exact r × r Jacobian's
   eigenvalues on random states.
2. 10% smoke (time, memory, penalty finite and active; no scores), fix the ρ̂ method, then the 12 runs.
3. Validation only for the stopping threshold; confirmation (eval seed 999003) after every run has finished;
   endpoints exactly as registered; report per-seed values and any departure from the prereg.

## Limits
≤ 1.5 node·h; each job ≤ 8 nodes × 1 h; record node·h per job. Final line: `WP-JS done: docs/reports/JS.md`.
