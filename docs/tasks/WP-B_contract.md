# WP-B implementation choices

Companion to `WP-B_data.md`; records underspecified details for implementation and tests.

- Public `TaskSpec(name, E=24, Q=4, V=1024, n_rel=4, seen=(0,768),
  unseen=(768,1024), head_sharing=0.0, chain_lengths=None, hops=(1,2,3),
  base=None, n_distract=0, n_updates=2, update_query_rate=0.5, missing_rate=0.5)`.
  Pools are half-open bounds. `base` is `t1` or `t2_chain_end` for updates
  (default `t1`), and `t1` or `t2_walk` for missing (default `t2_walk`).
  Absent chain lengths resolve to one chain containing all base edges.
  For typed tasks sum(lengths)=E-n_distract; for chain updates it is E-n_updates.
- The public arrays and generate signature are exactly those in the task. Metadata
  includes a JSON-compatible spec dictionary, version, split, base seed and start.
  Every generated fact is valid; counterfactual removal replaces facts rather than padding.
- Hop sampling is uniform over `hops`. T2 chooses an eligible chain uniformly;
  walk positions are uniform within that chain. Heads/tails use only the split pool.
- Seen splits have disjoint fact sets by construction: the minimum fact entity
  relative to the pool start has residue 0/1/2 modulo 3 for train/val/test.
  After random renaming, that minimum moves down by at most two ids; a swap with
  an unused local node preserves injectivity. Sampling starts at pool start plus
  the split residue, requiring that many extra pool entities. This small label
  restriction prevents collisions even in tiny graph spaces without a shared
  rejection cache. The unseen pool is already disjoint.
- Update contexts contain E-n_updates final base facts and n_updates obsolete facts.
  Obsolete tails are fresh leaves. Each obsolete fact precedes its final fact.
  For chain updates, an updated query means its required path includes an updated
  key. Update construction reserves one eligible chain for updated queries and,
  when update_query_rate<1, another sufficiently long chain for untouched queries.
  Infeasible specifications raise ValueError rather than silently changing rates.
- Missingness is sampled per episode: remove a randomly chosen required edge of
  query 0, replace it with an unrelated fresh-key distractor, and recompute every
  query. Thus query 0 has the requested missing rate; overlapping queries may also
  become insufficient. q_hop_true retains the requested pre-removal hop.
- Path counterfactuals fork each (episode,query) independently, returning N*Q
  episodes with Q=1, plus answer_cf of shape [N*Q,1] (also arrays_cf.answer).
  Metadata records source_episode and source_query. This avoids conflicting edits
  for questions sharing a context. They require answerable source queries;
  otherwise ValueError. Replacement edits a required edge (last edge for fixed-hop
  walks) to a fresh entity; removal removes all historical versions of that key
  and replaces each with a fresh-key distractor, preserving E and latest semantics.
  Terminal-query q_hop_true is recomputed after counterfactual edits or swaps;
  fixed-hop queries retain the requested hop even when information is missing.
- swap_memory preserves N and Q; meta['pairing'][i] names the source facts for i.
  The pairing is a derangement for N>=2; N=1 raises ValueError. Oracle labels are
  recomputed and returned; score the memory-swap control against the original
  arrays.answer, as required by the task. Inputs are never mutated.
- `INSUFFICIENT` is spec.V (a spec property); module constant 1024 is the default.
  Only train/val/test/test_unseen splits are accepted.
