# WP-H reusable pipeline — 2026-10-26

Ready for WP-F1 reuse. Implemented samples-only confirmation with explicit global
release/finalization, full B4 checkpoint depth sweeps without changing its L1
training, removal and A5 successor-swap diagnostics, pooled walk specificity,
strict final-v1 gate predicates, main matrix/runner/barrier, artifact audits,
statistics and figures. Interfaces: WP-H-main-interface.md,
WP-H-evaluation-notes.md, WP-H-stats-notes.md, WP-H-analysis-notes.md.

Validation: `scripts/submit.py --name t-p1s-h-pipe --queue regular-g --nodes 1
--walltime 00:10:00 --worktree . --wait -- -c '... pytest -q -o addopts=
 tests/test_sym_*.py ...'`: **335 passed in 40.32s**, exit 0, PBS **3712766**.
All execution was on Miyabi-G compute. Includes GPU tiny-run collection,
confirmation embargo, retry retention, matrix/release invariants, synthetic
specificity/gate boundaries, diagnostic pairing, all 29 figure render paths,
and CPU/CUDA bit-for-bit oracle-code, memory and reader-output equivalence.

Orthogonal codebook staging now batches host/device transfers; the original
per-episode QR seeds, code values and assignments are preserved exactly. New
finalizer unit fixtures use synthetic seed900001, outside the confirmation set.
No main matrix training or confirmation aggregate has run at this checkpoint.
The measured original throughput triggers the preregistered 15000-step rule;
optimized throughput is being measured before main job sizing.
