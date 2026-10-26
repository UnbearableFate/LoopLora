# WP-T3 — retrieval interference of the delta-rule memory with learned keys (theory note)

Read first: `docs/theory.md`, `src/looplora/core/memory.py` (`delta_update`: decay-first, normalized keys,
scalar λ, β), `src/looplora/core/adapters.py` (`IterativeQueryAdapter`: z = u + ρ φ(R z + U M Q z), output α B z),
`docs/reports/P1L-diagnosis.md`, `docs/tasks/WP-L3_p1l_writer_fix.md`. Branch/worktree: `wp/theory` (merge
`master` first). Output: a new section in `docs/theory.md` plus `docs/reports/WP-T3.md` (≤ 1 page).

## Observations to explain (LLM T1, E = 16 facts, r = r_mem = 32, ρ = 0.25, λ₀ = 0.99, β₀ = 0.1)
- With oracle addressing (the read returns the queried fact's stored value), val EM rises slowly: 4 / 12 / 22%
  at 1200 / 2400 / 4800 steps, 34% at 6580.
- With input standardization plus an InfoNCE addressing loss (τ = 0.1 over the episode's keys) and a tail-decoding
  loss on written values, both auxiliary losses become small (≈ 0.65 and ≈ 0.72 at 9000 steps), yet the answer
  loss stays ≈ 5.4 and EM ≈ 8%. The actual read is linear in the key–query inner products.
- The symbolic writer (random isotropic entity embeddings, E = 12) learned single-hop retrieval within ~200 steps.

## Deliver
1. Signal-to-interference of one read M q after E delta-rule writes (λ, β, normalized keys with pairwise
   cosines c_ij, query cosines q·k_j, value norms): exact expression for small β and for general β, and the
   condition under which the target value dominates the read. State what an InfoNCE loss at temperature τ does
   and does not guarantee about this linear read.
2. The scale of the memory term ρ φ(R z + U M Q z) relative to u in z, and its effect on the output path
   (B = 0 initialization; u varies with the question) — why oracle addressing can still learn slowly.
3. Ranked, concrete recommendations that keep the V2 definition (e.g. an auxiliary loss on the actual read
   output, key decorrelation, β or ρ initialization within the plan's candidates ρ ∈ {0.1, 0.25, 0.5}), each with
   the predicted effect and how to falsify it in a ≤ 1 h validation run.
Small numerical checks are fine (run them on compute via `scripts/submit.py` if they import torch). ≤ 0.2 node·h.
Final line: `WP-T3 done: docs/reports/WP-T3.md`.
