# P1-L oracle-random main study

The authoritative science is `docs/prereg/P1L.md`, including B2. Design and
per-step/J-lens-inspired name-gradient probe definitions are in `design-lock.md`, including A1. No confirmation
evaluation is authorized yet. S17000; Pilot-2 5666steps, seed17, lr.001/.003.
The900-step timing-only smokes remain valid by the orchestrator's B2 instruction.

Raw root: `/work/xg24i002/x10041/LoopLora/runs/p1l/main` (called R below).
The shared pinned venv and model are unchanged. All scientific commands below
must run on Miyabi-G via scripts/submit.py, from a committed source snapshot.
`launch_p1l_main.py` and `p1l_job_ledger.py` are stdlib control-plane utilities.

1. Timing: preflight3715826; initial sizing3715842 failed because the20-step
   preflight logged every second step. The reader now accepts the actual8
   post-warmup observations; full smokes log every step. Corrected sizing3715847;
   B1 sizing3715850. B2 supersedes the B1 total cap and locks S17000.
2. Smokes3715854/3715855:900steps for all7 trained variants, training-only
   development seed; only timing/memory fields are consumed. Sourcefb62c99.
3. `scripts/analyze_p1l_main.py size --root R --out ABS/reports/p1l/smoke-sizing.json`
   reads both smoke subdirectories and verifies B2's S against the slowest path.
   Commit the sizing before Pilot-2; do not read smoke losses or accuracies.
4. On login: `scripts/launch_p1l_main.py pilot --root R --sizing reports/p1l/smoke-sizing.json`.
   Seven jobs each train the two lr candidates sequentially and evaluate val
   hops1/2/3 on1000 episodes. Single model process; prefetch2 per active run.
5. On compute: `scripts/analyze_p1l_main.py select --root R/pilot --sizing
   ABS/reports/p1l/smoke-sizing.json --out ABS/reports/p1l/lr-S-lock.json`.
   Commit that lock and update the22 config files before any main run.
6. On login: `scripts/launch_p1l_main.py main --root R --sizing reports/p1l/smoke-sizing.json`.
   B4/B3 share a job sequentially; other variants have one run/job. Three seeds
   each, plus the untrained B0. R/training-manifest.json maps every artifact.
7. Hold confirmation until all runs finish AND the orchestrator supplies WP-L4's
   decision and writer-condition count. Record this in a committed authorization
   file with statusallowed, writer_conditions1/2, and orchestrator_decision text.
   `evaluate_p1l_main.py` checks every main checkpoint against the locked matrix
   before generating any test episode. Evaluate confirmation/typed once, with
   the cached registered J-lens-inspired name-gradient probe for M1/B8; preserve all original samples.
8. Finalize on compute with `scripts/finalize_p1l_main.py --manifest R/training-manifest.json
   --evaluations EVAL-MANIFEST --authorization AUTH --out ABS/reports/p1l`.
   This rechecks exact full-vocabulary labels, sample-to-table reproduction,
   oracle code hashes and support snapshots, then computes all registered CIs.

Readout/oracle regression tests3715857:33passed onmg0022. J-lens-inspired name-gradient probe timing
trial3715862 passed onmg0032; full cache3715870 uses only the fixed512 validation
questions and the frozen model. Gates/authorization tests have a separate job.
The logit-lens final-step replay must reproduce the deployed logits exactly.

Refresh node-hours on login with
`/usr/bin/python3 scripts/p1l_job_ledger.py --out reports/p1l/jobs.json`.
Every job goes through submit.py; each request includes measured1.5margin,
never exceeds2hours, and must finish before2026-10-28 09:00JST. Long waits use
the Miyabi wakeup skill in Codex message mode (no Goal exists in this session).
