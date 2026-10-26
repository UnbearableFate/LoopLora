# P2-S pre-registration — budget-matched comparisons, symbolic (v1, 2026-10-27)

Registered before any P2-S training or any evaluation on the P2-S confirmation set (eval seed 888001).

## Question (plan §8.2, Gate 3 §11.6)
In the P1-S task, can a single read with more rank (B6) or more state (B7) do what several reads of a smaller
module do (M1 at L = 4)? Is weight sharing across reads needed (B8)? Where does M1 sit on the accuracy–latency
plane against single-step alternatives? Parameters are matched by construction for M1 vs B4 (depth adds none);
P2-S adds the comparisons in which the single-step model gets more rank or state than M1.

## Fixed design (as `docs/prereg/P1S.md` final v1 unless listed)
- Task `t2_chain_end`, E = 12, chains (8, 4), train hops {1, 2, 3}, evaluation hops 1–8; ρ = 0.25, α = 1;
  batch 64; **15000 steps** (the WP-H stage length); seeds 17, 29, 43.
- Writers: `joint` (P1-S primary learned condition) and `oracle_random` (the oracle condition available at
  r = 32, 64, 128). Oracle writers require r = r_mem, so B7 runs only with `joint`.
- New training runs (24):
  - B6, single read (`iterq`, L = 1) at r = r_mem ∈ {64, 128}, both writers;
  - B7, single read at r = 32, r_mem ∈ {64, 128} (state 16 / 64 KiB vs 4 KiB), `joint`;
  - B8, unshared reader (separate R, U, Q per read), r = 32, L = 4, both writers.
- Reference checkpoints (no retraining): WP-H r = 32 (`runs/p1s/main/20261026/`: M1, B4, B5, B3 under `joint` and
  `oracle_random`); F1 r = 64, 128 (`/work/xg24i002/x10041/LoopLora-wt/symf/runs/p1s/f1/`: M1, B5, B3; trained
  20000 steps, used only in the descriptive cost curves, labelled as such).
- Pilot (validation only, seed 17, 5000 steps, lr ∈ {3e-4, 1e-3, 3e-3}) per (variant, size, writer) by val
  accuracy on hops 1–3 at the variant's trained L; ties → smaller lr.
- Confirmation set: split `test`, **eval seed 888001**, 2000 episodes, one query per hop 1–8; evaluated only
  after every P2-S run has finished training. Every compared model, references included, is evaluated on it, so
  contrasts are paired by episode.

## Endpoints (hops {2, 3}, accuracy at the stated L; pooled paired episode bootstrap, 2000 resamples, 97.5% CI,
## per-seed estimates reported)
- E1 rank: M1(r = 32)@L4 − B6(r)@L1, r ∈ {64, 128}, per writer.
- E2 state: M1(r = 32)@L4 − B7(r_mem)@L1, r_mem ∈ {64, 128}, `joint`.
- E3 sharing: M1(r = 32)@L4 − B8(r = 32)@L4, per writer.
- Reported with each: hop-1 difference and hops 4–8 means.
- E4 cost (descriptive): for every evaluated (variant, r, L ∈ {1, 2, 4, 8}), accuracy on hops 1–3 and 1–8 vs
  trainable parameters, state bytes (FP32), analytic read FLOPs per query and measured GPU latency per query batch
  (batch 64 and 1; warmups and repeats as WP-C3).

## Decision use (Gate 3, symbolic)
- Capability claim holds if E1 and E2 are positive (CI > 0) at every tested size and writer: more rank or more
  state does not let one read do what four reads do.
- Practical claim needs, in addition, an M1 point that no single-step point of equal or lower measured latency
  matches on hops 1–3 (E4). Otherwise claims narrow to the capability statement.
- Any E1/E2 contrast with CI ≤ 0 → report that a larger single read matches M1 at that budget; the depth claim is
  restricted to the matched-parameter and matched-state comparisons of P1-S.
- E3 decides whether sharing is required (B8 ≥ M1: sharing is a cost saving, not a requirement).

## Budget
≤ 3 node·h (pilot ≤ 1); every job one node, ≤ 1 h.
