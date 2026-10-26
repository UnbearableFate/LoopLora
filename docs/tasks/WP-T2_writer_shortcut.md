# WP-T2 — terminal-set shortcut of linear writers and writer designs that remove it

Read first: `docs/theory.md` (Props 3–7), `docs/reports/P1S-pilot.md` (learned writers: 41–50% on every hop, flat
in L, B4 included), `src/looplora/sym/backbone.py`, `src/looplora/sym/writers.py`, `src/looplora/sym/model.py`
(`prepare`, `read`). Branch/worktree: `wp/theory` (merge `master` first). Output: a new proposition section in
`docs/theory.md`, deterministic checks in `tests/test_theory_examples.py`, report `docs/reports/WP-T2.md`.

## Setting (as implemented)
Learned writer: fact input x = [e_h; e_r; e_t] (frozen random d_e = 64 embeddings), query input
[e_s; e_r + e_hop; 0], u = A x for both, k = Ku/‖Ku‖, v = Vu, delta-rule memory, reader V2 (iterq/fixedq), output
αBz scored against candidate entity embeddings. Primary layout `t2_chain_end`, all chain facts in relation 0,
chains (8,4); replication layouts (8,1) typed and (8,8,2).

## Questions
1. Orchestrator hypothesis to prove or refute: because the value sees both e_h and e_t and the key and query
   share the relation component, one read can return ≈ Σ_edges P(e_t − e_h) = Σ_chains P(e_terminal − e_start)
   for any query (telescoping), so a single read ranks the relation's terminal set first and scores 1/c on
   every hop (c = number of chains). State exact conditions (key/query overlap, embedding geometry, d_e, r, ρ, φ)
   and the margin with random Gaussian embeddings at d_e = 64; give an explicit parameter construction.
2. Factorized writer (key and query from [e_h; e_r] / [e_s; e_r + e_hop] only, value from e_t only): show what a
   single read can and cannot compute (is a terminal-set readout impossible; what is chance), and what iterated
   generic reads can compute (e.g. depth filtrations "nodes at depth ≥ s"; when they isolate terminals —
   equal-length chains — and when not, e.g. (8,4) with L ≤ 4). Do the A3 walk-specificity criterion and the
   `swap_successors` diagnostic detect each remaining shortcut?
3. Any other minimal writer change that removes the shortcut while keeping k and v from one u (e.g. constraints
   on V A restricted to the head block, or a value-orthogonality penalty); compare with factorization.
4. LLM analog (P1-L, `docs/reports/WP-E-interface.md` write policies `fact_final`/`all_tokens`): when do values
   carry head information there (attention mixing at layer 14), and which factorized analog is coherent (e.g.
   key from the position before the tail name, value from the tail position or a lower layer)? Design options
   with predictions only; no implementation.
5. Quantitative predictions for WP-H/F1 learned-writer rows: per-hop accuracy of B4 and of each reader under the
   shortcut, the `swap_successors` follow rate versus the unswapped rate, by layout.

## Checks
Deterministic numeric examples of (1) and (2) in `tests/test_theory_examples.py`, run on compute via
`scripts/submit.py` (CPU is fine). Keep claims proved, constructed, or measured, and label each.
