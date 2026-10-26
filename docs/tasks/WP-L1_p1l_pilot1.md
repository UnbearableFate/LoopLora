# WP-L1 — P1-L Pilot-1: LLM writer feasibility (validation only)

Read first: `docs/prereg/P1L.md` (Pilot-1 section, binding), `docs/reports/WP-G.md`, `configs/p1l/README.md`.
Branch/worktree: `wp/llm` (merge `master` first). Report: `docs/reports/P1L-pilot1.md` (≤ 1 page + tables).
This pilot does not depend on P1-S results; it runs in parallel with P1-S to save wall-clock time.

## Runs (stage `writer`, seed 17, r = 32, layer 14, L = 1; val split, 1000 episodes, eval seed 4242)
1. Throughput (one short job): writer-stage batch ∈ {8, 16, 32} and procs per GPU ∈ {1, 2}; pick the setting
   with the highest training episodes/s that fits memory. It becomes the P1-L batch for every stage (record it).
2. Policy × lr at E = 16, chains (8,4,2,2) layout as in P1-S: `write_policy` ∈ {`fact_final`, `all_tokens`} ×
   lr ∈ {3e-4, 1e-3, 3e-3}. Steps S: the same for all runs, chosen from step 1 so that one run plus its eval
   takes ≤ 30 min (report S and training episodes). Select (policy, lr) by val single-hop EM (full vocabulary);
   ties within 0.5 pp → `fact_final`, then the smaller lr.
3. Capacity at the selected (policy, lr): E ∈ {8, 12, 24} (E = 16 from step 2).

## Report
Val EM and candidate accuracy for every run; loss curves (converged or still falling); timing; node·h per job;
the selection and E*_L by the prereg rule. If EM < 0.95 at every E, say so plainly; do not tune further.

## Do not
Run reader/joint/baseline stages, `test`/`confirm` splits or other seeds. Each PBS job ≤ 8 nodes × 1 h.
