# WP-T2 — completed 2026-10-26

Branch `wp/theory`; fast-forwarded `master` through `d243aec` before work.
Theory/checks commit: `8818756`.

Delivered [Proposition 10](../theory.md): all five requested topics, with explicit
assumptions, proofs, counterexamples, and separate constructed/measured claims.
It includes shared-u linear/ReLU parameter constructions, Gaussian score-margin
laws at d_e=64 and effective rank p, factorized-writer limitations and root-depth
filtrations, hard/soft writer constraints, P1-L design options, and quantitative
per-hop/A3/successor-swap predictions for all three layouts. Two independent
analytical reviews were incorporated. No production writer or preregistration changed.

Main conclusions:

- Common-key delta storage is an EMA, not an exact edge sum. Telescoping is
  approximated with near-unit lambda and small beta; the note gives its error
  bound and a depth-flat r=32, p=15 construction for both activations. The
  reported pretrained beta≈0.565 does not satisfy that construction's regime.
- Tail-only values remove direct head subtraction but do not universally prevent
  terminal inference. A3 rejects the specified generic filtration policies;
  neither A3 nor positive paired swap follow identifies every internal algorithm.
- Ideal invariant terminal guessing predicts 50%/50%/33.33% by layout, flat in
  hop/depth, with zero paired swap improvement. Ideal walking follows iff the
  **new** path fits L; explicit tables account for swapped paths longer than 8.

Validation:

```bash
PYTHONUNBUFFERED=1 scripts/submit.py --name t-wpt2 --queue regular-g \
  --walltime 00:05:00 --worktree . --wait --tail 80 -- \
  -m pytest -q -s -o addopts= tests/test_theory_examples.py
```

**36 passed in 5.60s, exit 0**, including 16 original and 20 new cases; Miyabi-G
compute host `mg0851`, job `3712948.opbs`. Log:
`/work/xg24i002/x10041/LoopLora/logs/pbs/t-wpt2-20261026-215041-2374593.log`.
Checks cover actual shared writer/reader parameters, EMA and error bounds,
query/depth invariance, ideal terminal decoding, large-beta failure, factorized
filtrations, swap invariants, and exhaustive cut counts. Constructed Gaussian
measurements use 256 validation episodes only: at p=15, top-one terminal selection
was 219/256 and both-terminals-first was 126/256. Full moments are in the log
and summarized in the note. No numerical checks ran on login; diff checks passed.

Qualifications: the proposed shortcut and universal claims about factorization
required the restrictions above. The trained WP-H/F1 mechanism remains unmeasured
by this task; no confirmation metrics were inspected. P1-L options are predictions
only, as requested. No required work or checks remain outstanding.
