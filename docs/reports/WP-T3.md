# WP-T3 — completed 2026-10-27

Branch `wp/theory`; merged `master` by fast-forward to required `b4f1ad4`.
Theory/checks commit: `628ba19`.

Delivered [Proposition 11](../theory.md), covering all three task items:
exact general-beta ordered-product and triangular-Gram read coefficients;
small-beta expansion with rigorous remainder; value-Gram SIR and sufficient
dominance/decoder margins; InfoNCE guarantees and counterexamples; residual
scale, zero-B gradient routing, and the distinction between OA and direct-z D2.
Two independent analytical reviews were incorporated.

Key conclusions:

- Correct cosine ranking and low clean-value CE do not certify the actual
  delta read. At the specified E=16/r=32/default gates, a constructed example
  has addressing CE **0.648229**, value-head CE **0.006266**, but read-head CE
  **6.479258** and selects the newest distractor. Larger beta can reverse the
  target coefficient despite unchanged good key ranking.
- Normalized InfoNCE does not constrain raw query gain. A 1000× query reduction
  preserves its CE while reducing the actual memory-specific residual 1000×.
- OA bypasses interference, beta, decay and query norm, so those cannot explain
  OA's own slow learning. Its contextual value encoder, U/ReLU/rho, question
  residual, token coverage and output learning remain different from D2.

Ranked follow-ups preserve the deployed V2 equation: audit exact reads/scales
first; supervise actual Mq with a noncollapse safeguard; test matched OA rho
0.25/0.5; decorrelate distinct-address keys before beta tuning; consider a
training-only deployed-output teacher bridge if reads already decode well.
Each has predicted outcomes, explicit falsifiers, and a timed, validation-only
one-node ≤1 h design. These are proposals, not changes to WP-L3's selection rule.

Validation:

```bash
PYTHONUNBUFFERED=1 scripts/submit.py --name t-wpt3 --queue debug-g \
  --walltime 00:03:00 --worktree . --wait --tail 90 -- \
  -m pytest -q -s -o addopts= tests/test_theory_examples.py
```

**42 passed in 5.72s, exit 0**, including six new cases; PBS `3715090.opbs`,
compute host `mg0005`. PBS reports 13 s walltime; its start/end interval is
19 s (<0.006 node·h). The request itself caps usage at 0.05 node·h, below 0.2.
Log: `/work/xg24i002/x10041/LoopLora/logs/pbs/t-wpt3-20261027-022056-713492.log`.
All numerical execution used compute; staged diff checks passed.

No production code, dependencies, or running experiments changed. WP-L3 source
was inspected read-only at `wp/llm c5879d3`; its supplied observations were not
independently reproduced. No real-checkpoint diagnosis, training or confirmation
evaluation was run. The analysis establishes possible bottlenecks, not a unique
cause of the observed EM curves. No required work or checks remain outstanding.
