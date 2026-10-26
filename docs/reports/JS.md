# JS — Jacobian spectral-radius regularization (2026-10-27)

**Regularization reduces overshoot but does not remove it for either writer
under the registered criterion. Longer-walk and stopped accuracy decrease.**
[Oracle figure](../../reports/js/accuracy_depth_oracle_random.png) /
[joint_fact figure](../../reports/js/accuracy_depth_joint_fact.png); PDFs accompany both.

Merged master `f5bb486`. The [design lock](../../reports/js/design.json),
commit `852d293`, preceded training and score inspection. All **12 runs**
finished **15000 steps**, batch64, lr0.003, seeds17/29/43: oracle_random and
joint_fact, RS versus RS+J, M1 ReLU, r=r_mem=32, E12, chains(8,4), train
hops1–3. Both conditions share data/depth RNG streams; one uniform L4–8 is
sampled per batch. RS+J uses gamma0.9, lambda1 and episode/query-mean penalty.
[Exact 32×32 eigenvalues](../../reports/js/method.json) on CPU were selected
by timing before scores; the Jacobian uses the next-step gate at final z_L,
with gradients through reader parameters and trainable memory. The 10% smoke
read timing/memory and verified finite, active penalties without reading scores.

All training passed before validation. Eight WP-C3 relative-change thresholds
were selected on **validation seed4242**, 2000 episodes, and committed in
**`0c99cc1` before confirmation**. Fresh **test seed999003**, 2000 episodes,
one query per hop1–8, evaluated all models at L1–16 and stopped at cap16;
12 reference checkpoints remained unchanged. All **24 traces** preceded
the single confirmation aggregation.

Effects below are percentage points, with **97.5% paired episode-bootstrap
CIs**, 2000 draws, conditional on three fixed training seeds. DeltaS resamples
all four accuracy terms jointly. [All 36 contrasts and per-seed values](../../reports/js/tables.md)
and [counted accuracies](../../reports/js/accuracy.csv) are retained.

| Endpoint | oracle_random | joint_fact |
| --- | --- | --- |
| S(RS), L16−L4, hops1–3 | −11.267 [−12.356,−10.050] | −3.017 [−3.789,−2.272] |
| S(RS+J), L16−L4, hops1–3 | −2.833 [−3.800,−1.806] | −1.772 [−2.484,−1.061] |
| **E1: DeltaS** | **+8.433 [7.483,9.418]** | **+1.244 [0.222,2.200]** |
| E2: RS+J−RS, L4, hops1–3 | +0.789 [0.261,1.295] | −0.728 [−1.567,0.067] |
| E3: RS+J−RS, L16, hops4–8 | −5.547 [−6.200,−4.890] | −1.443 [−1.990,−0.933] |
| E4: RS+J−RS, stopped, hops1–8 | −1.040 [−1.538,−0.544] | −0.677 [−1.227,−0.115] |

DeltaS is positive with CI>0 for both, but **S(RS+J)<−1 pp** for both;
neither meets the remedy rule. Primary per-seed effects are
8.367/8.617/8.317 pp and 0.817/1.050/1.867 pp. Stopped RS+J accuracy is
18.704%/15.879%, using 15.938/15.028 reads on average.
[Radius distributions](../../reports/js/rho.csv) at z4/z8/z16 show lower
L16 means with regularization: **1.307→1.053** oracle and **0.861→0.834**
joint_fact. Nevertheless only **22.24%/92.70%** of RS+J states have radius<1
at L16. These are pointwise Jacobians at reached states, not fixed-point,
global-contraction or decoding certificates. No gate depends on JS.

Validation: `pytest -q -o addopts= tests/test_js.py` — **16 passed**, PBS
**3716003**, `mg0035`; includes exact Jacobian/gradient and finite-difference
checks, deployed-trace equivalence, paired inference and release/seal tests.
Final PBS **3716104**, `mg0048`, exited0; [audit](../../reports/js/audit.json)
passed **6,528,000 accuracy observations**, **1,152,000 radius measurements**,
paired labels, immutable inputs/memory and checkpoint seals. Both figures
were inspected. No scientific protocol deviation or open check remains.

[Cost](../../reports/js/budget.json): **1.011/1.5 node-hours**, all **10 jobs**
included, one node each, requests≤25 minutes via `scripts/submit.py`.
Two pretraining development checks failed (mask dtype and test-helper call),
were corrected before smoke/main, and remain in the [ledger](../../reports/js/jobs.csv).
[Artifacts](../../reports/js/README.md) contain compact tables; raw checkpoints,
traces and full statistics remain in `runs/js/`.
