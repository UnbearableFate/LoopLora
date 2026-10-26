# WP-L2 — P1-L writer diagnosis (validation only)

Read first: `docs/reports/P1L-pilot1.md` (no admissible E*_L: val EM ≤ 0.3%, candidate accuracy 2.6–9.0% ≈ chance,
loss plateau ≈ 7.3–7.5 after ~200 steps, S = 1500 steps × batch 32), `docs/prereg/P1L.md` (Pilot-1 rule: writer
diagnosis before any T2 run), `docs/reports/WP-E-interface.md`, `src/looplora/llm/inject.py`. Branch/worktree:
`wp/llm` (merge `master` first). Report: `docs/reports/P1L-diagnosis.md`.

## Question
Which link fails: (a) entity/relation information at the module input (layer-14 `o_proj` input, the 2048-d
attention output) at write and read positions; (b) the path from an r = 32 read through `B` and the frozen upper
layers to the answer token; (c) the training budget (S = 1500 is 6.7× fewer episodes than the symbolic T1 writer
needed for 85%); (d) key–query matching. Fix the first failing link only after it is identified.

## Diagnostics (T1 as in Pilot-1: n_rel = 4, head sharing 0.3, E = 16; seed 17; val 1000 episodes, eval seed 4242)
- D1 probes (frozen features, no adapter): held-out accuracy of linear probes for the tail name (1024-way) and
  the head name and relation at the write positions (`fact_final` position and the tail-name token), and for the
  start name and relation at the question's read position(s). Feature sites: layer-14 `o_proj` input (current),
  layer-14 residual-stream input, and the residual stream at layers 7 and 21. Report full-rank and rank-32
  probes (what a 32-d `u` could carry). Train probes on train-split episodes, test on val.
- D2 output-path oracle: bypass writer and memory; at the read position set z = c(answer) from a learned
  r = 32 code table over the 1024 names, train only the table and `B` (α, ρ as P1-L), same S, batch and lr grid
  {3e-4, 1e-3, 3e-3}. Report val EM and loss, with the frozen model (B0, memory-only) loss for reference. If
  it fails, the injection site, rank or scale is the bottleneck.
- D3 budget: the Pilot-1 selected configuration (fact_final, lr 3e-4) and lr 1e-3 for the longest step count one
  ≤ 1 h job allows; log the loss every 50 steps and val EM at 1500, half and final steps.
- D4 (only if D1 shows the identity is linearly available in the residual stream but not at the `o_proj` input,
  or D2 passes while D3 fails): add a minimal tested option for the module's input site (u from the layer-14
  residual stream at the same positions; output still added at `o_proj`) and rerun the D3 configuration.

## Report and limits
For each diagnostic: what was run, numbers, and the conclusion about links (a)–(d); then one recommended P1-L
writer configuration (input site, steps, batch, lr) with evidence. Val only; no `test`/`confirm`, no other seeds,
no reader/joint/T2 runs. Each PBS job ≤ 1 node × 1 h; WP-L2 total ≤ 4 node·h; record node·h per job.
