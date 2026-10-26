# WP-B independent test and review notes

- `tests/test_relations.py` uses a presentation-order dictionary and independent
  bounded/terminal walks, without production solver or generation helpers. All
  six tasks and both update/missing bases are covered, including defaults,
  zero updates, one-edge contexts, and all-updated contexts.
- Checks cover shapes/dtypes/metadata, functional keys and chain topology,
  terminal distances, nonterminal walk answers, typed distractors, shuffling,
  uniform hops, temporal updates and sampled update/missing rates, independent
  slices and seeds, disjoint fact sets, unseen/custom pools, counterfactual
  labels and input immutability, and global Python/NumPy RNG isolation.
- Invalid inputs cover infeasible chains/hops, pool bounds and capacity, update
  categories, rates, unsupported counterfactuals and insufficient source paths.
  Statistical checks use fixed seeds, thousands of queries and broad tolerances.
- The independent read-only review found that distinct seeds alone cannot
  guarantee disjoint fact sets for small pools. The parent added an injective
  renaming partition based on the minimum **presented fact** entity. Four
  regressions cover tiny single-edge pools, shared heads, missing edges and
  nonzero pool lower bounds. Review also aligned the contract's eligible-chain
  wording. Both findings are resolved.
- Review inspected local RNG use, cached-layout mutation, missing-path overlap,
  pair ordering, zero/all/mixed updates, fresh-entity counterfactuals, historical
  key removal, and terminal versus fixed-hop relabeling. No other concrete
  correctness issue was found; this inspection is distinct from runtime tests.
- Initial compute suite: 101 cases passed, job `3711357.opbs`, host `mg0255`.
  Final expanded suite: **105 cases passed**, job `3711448.opbs`, host `mg0852`,
  exit 0 (2026-10-26). Both used
  `scripts/submit.py ... -- -m pytest -q tests/test_relations.py`.
- Test authoring and static review occurred on login/control `miyabi-g2` without
  importing NumPy/Torch or executing pytest. Runtime validation was parent-run
  on Miyabi-G compute. No dependency files were changed.
