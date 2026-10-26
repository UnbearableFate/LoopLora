# P2-S artifacts

Binding protocol: `docs/prereg/P2S.md`; task: `docs/tasks/WP-P2S_budget_matching.md`.
All execution uses `scripts/submit.py` on Miyabi-G. Source hashes, allocations,
elapsed node seconds and exit statuses are recorded in `jobs.csv`.

The study has eight new families × seeds 17/29/43 = 24 main runs, each 15000
steps, batch 64. B6 uses r=r_mem=64/128 under joint/oracle_random; B7 uses
r=32, r_mem=64/128 under joint; B8 uses four unshared reads at r=r_mem=32 under
both writers. `selected_lrs.csv` was committed at `be6dd16` before main training.
It uses validation-only 5000-step pilots; integer correct counts break ties
exactly, then the smaller LR wins. `pilot_grid.csv` retains every candidate.

After the complete 24-stage training barrier, **84 checkpoints** are evaluated
on test seed 888001, 2000 episodes, one query at each hop 1–8. These comprise
24 new models, 24 WP-H references (M1/B4/B5/B3 at r32) and 36 F1 references
(M1/B5/B3 at r64/r128), under joint/oracle_random and all three seeds. B8 is
defined only at L4; all other checkpoints use L0/1/2/4/8/16. Original reference
directories and checkpoints remain read only. All outputs go to `runs/p2s/`.

- `design.json`, `training_manifest.csv`, `checkpoint_manifest.csv`: fixed
  design, training identities and evaluated checkpoint fingerprints.
- `smoke.csv`, `smoke_timing.json`: all-eight-family end-to-end acceptance and
  measured training cost. The latency smoke's raw audit remains in runs/.
- `endpoints.csv`: E1/E2/E3 on hops 2–3, accompanying hop-1 and hops 4–8
  differences, both 95% and 97.5% intervals, and every per-seed value. Differences
  are fractions. Paired bootstrap draws resample 2000 complete episodes jointly
  across the three training seeds (seed 2718); uncertainty conditions on these
  seeds. All 84 sample files must be released before any confirmation aggregation.
- `tables.md`: readable E1–E3 intervals and per-seed differences, selected E4
  points and practical eligibility, formatted from the summary CSVs.
- `accuracy.csv`: three-seed pooled candidate/full-vocabulary/chance accuracy
  by family, hop and evaluated L, including L0/16 where defined.
- `cost.csv`, `cost_per_seed.csv`: E4 at L1/2/4/8 where defined, with accuracy
  on hops 1–3 and 1–8, actual unique trainable parameter counts, persistent FP32
  reader state bytes **per episode**, and analytic read-update FLOPs per query.
  The latter count R/Q/Mq/U MACs ×2, cache the fixed-query read for B5, and omit
  scalar/nonlinear/normalization work. Output projection and candidate decoding
  are tabulated separately. Unused B3 memory is zero persistent reader state.
- `cost_plane.{pdf,png}`: GPU latency versus accuracy. Crosses label F1's
  **20000-step descriptive-only** points; they never enter E1–E3 or the primary
  practical decision. Extra reads of L1-trained B4/B6/B7 are extrapolations;
  L8 also exceeds the mixed-depth reference models' training maximum L4.
- `practical.csv`, `decision.json`: unrounded Gate 3 decisions. Capability
  requires all six E1/E2 reachable-hop lower bounds >0. Primary practical checks
  compare WP-H M1/r32 at L1/2/4/8 against all **15000-step L1** points within the
  same writer and batch size; equal accuracy at equal or lower mean GPU latency
  matches M1. The practical claim requires capability and at least one unmatched
  eligible M1 point; the supported writer, depth and batch size are named.
- `audit.json`, `budget.json`: sample/checkpoint/benchmark receipts and final
  costs. Raw plans, logs, checkpoints, sealed samples, per-hop timings and full
  bootstrap metadata remain under `runs/p2s/`, outside git.

Latency follows WP-C3: FP32, five warmups and 20 repeats, CUDA events plus
synchronized wall clocks, prepared query batches of 64 and 1. Batch 64 selects
one query per episode cycling hops 1–8; batch 1 averages eight separate queries,
one per hop. Context/writer preparation and input projection are excluded;
candidate-restricted decoding is included and checked against the evaluator.
Both clocks retain mean/median/p95. Batch-one median/p95 columns average the
eight query-level summaries; pooled columns then average the three seeds.
All timings are collected serially on one GPU without concurrent evaluations.

Reproduction uses `scripts/run_p2s.py`, `scripts/benchmark_p2s.py` and
`scripts/analyze_p2s.py`, with explicit source hashes from `jobs.csv`. The main
training LR lock and reference budgets must remain unchanged.
