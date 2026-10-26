# WP-B2 — counterfactual `swap_successors` (same terminal set, different answer)

Read first: `docs/reports/WP-B.md`, `src/looplora/data/relations.py` (`counterfactual`), `docs/theory.md` Prop. 5
(identical-observation pair). Branch/worktree: `wp/data` (merge `master` first). Report: `docs/reports/WP-B2.md`.

## Why
Validation pilots (`docs/reports/P1S-pilot.md`) show learned writers scoring ~45–50% on every hop with flat depth
profiles, consistent with guessing among the episode's chain terminals. A counterfactual that changes the correct
terminal while keeping the set of terminals (and every other global count) unchanged separates walking from
terminal guessing directly.

## Specification
Add kind `swap_successors` to `counterfactual()` for relation-chain tasks (`t2_chain_end`, `t2_typed`; reject
others with a clear error). Per original query (same row layout and metadata as the path edits): let the query's
relation-0 path be s → … → T1 (chain A). Choose, with the derived per-row RNG, another chain B of the same episode
and relation with terminal T2 ≠ T1, a uniformly random edge (a_i → a_{i+1}) on the query's path, and a uniformly
random edge (b_j → b_{j+1}) of chain B. Swap their tails: a_i → b_{j+1} and b_j → a_{i+1}. The new answer is T2;
`q_hop_true` is the new path length. If the episode has no second chain in that relation, raise (callers only use
layouts with ≥ 2 chains).

## Tests (compute node via `scripts/submit.py`)
- The multiset of relation-0 heads, the set of terminals and the set of chain starts are unchanged; the result is
  acyclic, every node keeps out-degree ≤ 1 per relation, and the new answer equals the walk on the new facts.
- The new answer differs from the original for every row; `q_hop_true` matches the new walk length.
- Deterministic for a fixed seed; inputs not mutated; `t2_typed` distractors untouched.

## Deliverable boundary
Code, tests, a short report. Do not change existing counterfactual behaviour or generator outputs.
