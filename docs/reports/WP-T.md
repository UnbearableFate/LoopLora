# WP-T — completed 2026-10-26

Branch: `wp/theory`. Theory/checks commit: `9d2100b`.

Delivered [docs/theory.md](../theory.md): nine propositions covering all seven
task topics, each with assumptions, proof, counterexample, and experimental
variables tied to P1-S or P4. Includes the sequence-memory Jacobian witness,
exact chain-terminal construction, random/delta interference bounds, candidate-set
Bayes analysis, ReLU cone obstruction, contraction/stopping, exact elimination,
and first-/second-order double-BA initialization analysis. Standard versus local
derivations are identified; no empirical success or novelty is claimed.

Added [tests/test_theory_examples.py](../../tests/test_theory_examples.py):
16 deterministic FP64 cases, including core reader/writer correspondence and
counterexamples. Two independent proof reviews were incorporated.

Validation command:

```bash
PYTHONUNBUFFERED=1 scripts/submit.py --name t-wpt --queue regular-g \
  --walltime 00:05:00 --worktree . --wait -- \
  -m pytest -q tests/test_theory_examples.py tests/test_contraction.py
```

**Result: 36 passed, exit 0**, Miyabi-G compute host `mg0856`, job
`3711845.opbs`; application log interval 19:46:49–19:47:00 JST.
Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/t-wpt-20261026-194543-1622398.log`.
The queued debug job `3711797.opbs` was cancelled before execution because its
pool was full; the replacement used the same one-node, five-minute request.
No numerical checks ran on login. `git diff --cached --check` passed before the
implementation commit; no dependencies or production code changed.

Necessary qualifications to the task wording:

- The `1/(N−2)` multihop ceiling requires an aligned, exact first-hop observation
  and exchangeable labels without graph side information. Unrestricted learned
  `Q` can encode more information in one read; a universal nontrivial ceiling
  for implemented B4/B5 is not established. Actual candidate chance is `1/N`,
  with `N=20/29/14` for the proposed/fallback layouts.
- The exact chain construction requires `rho*gamma>1`; it stabilizes by
  nilpotence despite failing the Euclidean contraction test.
- ReLU double-BA has zero first derivative at `B=0`, but generally no classical
  Hessian there. Hidden bias alone does not remove the cone obstruction.

No implementation blockers remain. P1-S training, G1/G2 confirmation, and P4
stopping experiments were outside this theory task and were not run. The note
does not amend the preregistration.
