# P2-S follow-up F1 — sharing at matched training depth; budget contrasts with shortcut-free codes
(pre-registered 2026-10-27, before any run below; P2-S results are known: `reports/decisions/P2S.md`)

## Questions
Q1. At the same training depth (L = 4 only), does sharing parameters across reads (M1) cost accuracy against
step-specific reads (B8)? Q2. How much of the B8 − M1 gap is M1's mixed-depth training (L ∈ {1, 2, 4})?
Q3. With shortcut-free learned codes (`joint_fact`), can a single read with more rank or state match M1?

## Design (everything not listed is as in `docs/prereg/P2S.md`)
- New runs, seeds 17/29/43, 15000 steps, batch 64:
  - M1-L4: `iterq` ReLU, r = 32, trained at L = 4 only, writers `joint`, `oracle_random`, `joint_fact` (9 runs);
  - with `joint_fact`: B6 r = r_mem ∈ {64, 128} (L = 1), B7 r = 32, r_mem ∈ {64, 128} (L = 1), B8 r = 32 (L = 4)
    (15 runs).
- References, evaluated unchanged: P2-S B8 (`joint`, `oracle_random`), WP-H M1 (`joint`, `oracle_random`), F2 M1
  and B4 (`joint_fact`).
- Pilot as P2-S (validation, seed 17, 5000 steps, lr ∈ {3e-4, 1e-3, 3e-3}, hops 1–3 at the trained L); lrs locked
  in a commit before main training.
- Confirmation: the P2-S set (split `test`, eval seed 888001, 2000 episodes), evaluated after every F1 run has
  finished training; contrasts paired by episode.

## Endpoints (hops {2, 3} unless stated; 97.5% pooled paired bootstrap, per-seed values reported)
- F-E1 (Q1): B8@4 − M1-L4@4, per writer; also hop 1 and hops 4–8.
- F-E2 (Q2): M1-L4@4 − M1(mixed)@4, per writer (`joint_fact` mixed = F2 M1); also hop 1.
- F-E3 (Q3): M1(mixed, `joint_fact`)@4 − B6(r)@1 and − B7(r_mem)@1, r, r_mem ∈ {64, 128}.

## Decision use
- Reader carried into P1-L: B8 if F-E1 > 0 (CI) for `joint` and `joint_fact`; otherwise M1. Training depth: fixed
  L = 4 if F-E2 > 0 (CI) for the carried writer, else mixed. Both recorded in the P1-L prereg before Pilot-2.
- Capability claim with shortcut-free learned codes holds if every F-E3 contrast has CI > 0; otherwise it is
  restricted to oracle codes.

## Budget
≤ 2.5 node·h (pilot ≤ 0.5); one node per job, ≤ 1 h.
