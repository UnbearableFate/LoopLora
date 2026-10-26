# Study JS — a locally stable chain terminal via Jacobian spectral-radius regularization, pre-registered

Registered 2026-10-27 04:05 JST, before any run below. Known then: overshoot in every shortcut-free family
(F2 joint_fact M1 hop-1 −35.6 pp at L = 4 vs 1; P4-S label-free stopping fails; 0/96000 contraction certificates),
fixed-L4 training helps at L = 4 (P2-S-F1), IS (depth gain shrinks with load). Literature: STARS (arXiv 2605.26733,
Jacobian spectral-radius regularization with random loop sampling in looped LMs).

## Question
The reader step is f(z) = u + ρ φ(R z + U M Q z), with Jacobian J(z) = ρ diag(φ′(·)) (R + U M Q) (r × r, r = 32).
A walk between equidistant entity codes cannot be a global contraction, but the chain terminal can be a locally
attracting fixed point (Prop. 3's terminal stability). Does penalizing the spectral radius of J at the final read
state make extra reads harmless (no overshoot), make label-free stopping work, and let longer chains be walked
with more reads?

## Design
- Writers `oracle_random` and `joint_fact`, r = r_mem = 32, M1 (shared iterq, ReLU), task and data as P2-S-F1
  (E = 12, chains (8, 4), train hops {1, 2, 3}), seeds 17/29/43, 15000 steps, batch 64, lr 3e-3 (the locked M1-L4 lr
  for both writers; no new search).
- New conditions (12 runs):
  - **RS**: random loop sampling, L ~ Uniform{4, …, 8} per batch, target = chain terminal; no penalty.
  - **RS+J**: RS plus λ_J · mean over episodes of max(0, ρ̂(J(z_L)) − γ)², γ = 0.9, λ_J = 1, ρ̂ = spectral radius
    of the exact r × r Jacobian at the final state (eigenvalues, or power iteration checked against them in a
    test; the choice is fixed in the smoke, before any score is read).
- References evaluated unchanged: M1-L4 (P2-S-F1, fixed L = 4) and M1 mixed-depth (WP-H / F2), same writers.
- Confirmation: split `test`, **eval seed 999003**, 2000 episodes, one query per hop 1–8 (never evaluated before),
  L = 1…16, plus relative-change stopping (WP-C3 rule; threshold per family selected on validation, eval seed
  4242, and locked before confirmation). Measured ρ(J(z_L)) reported at L = 4, 8, 16.
- Statistics: pooled paired episode bootstrap, 2000 draws, 97.5% CIs (two writers); per-seed values reported.

## Endpoints (per writer; hops 1–3 unless stated)
- JS-E1 (primary, overshoot): S = Acc(L = 16) − Acc(L = 4); contrast ΔS = S(RS+J) − S(RS).
- JS-E2 (cost at the trained depth): Acc(L = 4) of RS+J − RS.
- JS-E3 (longer walks): hops 4–8 at L = 16, RS+J − RS.
- JS-E4 (label-free stopping): relative-change-stopped accuracy on hops 1–8, RS+J − RS.
- Descriptive: RS − M1-L4 and RS − M1-mixed (effect of random loop sampling), hop-1 by L, ρ(J) distributions.

## Use of the results (stated before results)
- RS+J removes overshoot for a writer if ΔS > 0 with CI > 0 and S(RS+J) ≥ −1 pp; the paper then states local
  terminal stability as a remedy (with its conditions), Prop. 7 is restated in the local form, and later LLM
  stages may adopt the penalty through a new prereg. Otherwise the negative result is reported with the
  ρ(J) measurements.
- No gate depends on JS; no follow-up is registered from it.

## Limits
≤ 1.5 node·h; each job ≤ 8 nodes × 1 h; node·h recorded per job.
