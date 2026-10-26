# P1-L writer fix — completed 2026-10-27

**No writer selected.** A1 W5 finished at **14.4% EM**, below the 0.80 writer
and 0.30 partial-learning gates. Diagnostic OA reached 70.3% and is ineligible.
Scope: seed 17, T1 E16, four relations, head sharing 0.3, rank 32, layer-14
o_proj, L1, batch 32/Q4, lr 0.003, memory-only; 1000 val episodes, seed 4242.
No T2, reader/joint, test/confirmation or additional-seed runs occurred.

Built OA and W0–W5 with fact-aligned writes. W1 standardizes inputs using pooled
statistics from 256 training episodes; W2 adds address CE (weight 1, temperature
0.1); W3 adds clean-value CE through a training-only affine 1024-name head
(weight 1); W5 combines them. W4 uses separate pre-tail key/tail-token value
maps, one write per fact; the reader shares A_k. Contextual values can retain
head information. OA supplies the true fact's raw value at every valid question
token. Auxiliary labels exceed answer-only supervision; exports exclude the
auxiliary head and optimizer.

Four fixed pre-tail probes used 4096 train/1000 val episodes, 1200 fitting steps
and training-only standardization, before screening. Separate probes establish
available information, not joint learnability through one A. All seven 4800-step
arms used source `3fbc504`, including WP-C5. A1 was recorded before this worker
read validation scores; its top-two ranking selected OA/W5, with no third arm.
After WP-B4 merged, both reran fresh for 32,900 steps (1,052,800 episodes each),
source `0713cfb`, scan_final/prefetch2, same initialization and data stream.

W5's final-token key top1 reaches 82.4%, yet answer EM is 14.4% (candidate 39.3%,
tail prediction 23.4%). OA reaches 70.3% EM/candidate 95.7%, showing substantial
clean-value/output learning with residual error. OA also bypasses attenuation
and interference, so its advantage cannot isolate addressing. Key ranking and
small auxiliary losses do not establish a useful raw read; mixture quality,
gain and downstream conditioning remain unresolved. No feasible writer is
recommended. WP-L3's amended stop is reached; the separately authorized WP-L4
follow-up is conditional on this negative decision. This is single-seed
validation evidence, without unseen confirmation or multi-hop claims.

Validation: 70 original tests (3714689); 68 A1 tests and exact 100-step
workers0/2 final-parameter/logged-loss equality (3714886). Final audit 3715599,
mg0844, reproduced all sample metrics/paired labels, checked provenance and
normalization, and reloaded both head-free exports with identical 1000
predictions. All runtime used submit.py on Miyabi-G; none ran on login.
**Cost: 12,701 s = 3.52806/4 node·h**, all jobs exit 0. Reproduction, locks and
raw-artifact location: [README](../../reports/p1l-writer-fix/README.md);
[loss curves](../../reports/p1l-writer-fix/loss_curves.png),
[EM curves](../../reports/p1l-writer-fix/em_curves.png),
[metrics](../../reports/p1l-writer-fix/runs.csv),
[cost ledger](../../reports/p1l-writer-fix/jobs.json).

**Pre-tail probes (16000 fact examples each)**

| Target | Full linear | Rank32 |
| --- | ---: | ---: |
| Head | 83.975% (13436) | 95.6125% (15298) |
| Relation | 99.99375% (15999) | 99.9875% (15998) |

**All validation checkpoints.** Percent columns are EM, candidate accuracy,
prediction among episode tails and key top1; cosine columns are signed means.
OA's key metrics are structural diagnostics of a bypassed matrix.

| Stage | Arm | Step | EM% | NLL | Cand% | Tails% | Key1% | Margin | Pair cos |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Screen | OA | 1200 | 4.2 | 5.067 | 38.2 | 6.3 | 6.4 | -0.219 | 0.385 |
| Screen | OA | 2400 | 12.0 | 3.948 | 62.2 | 13.7 | 6.3 | -0.255 | 0.308 |
| Screen | OA | 4800 | 22.1 | 3.297 | 74.4 | 23.5 | 6.5 | -0.290 | 0.276 |
| Screen | W0 | 1200 | 0.2 | 7.290 | 3.4 | 1.1 | 7.8 | -0.124 | 0.931 |
| Screen | W0 | 2400 | 0.0 | 7.239 | 3.6 | 0.8 | 5.3 | -0.108 | 0.727 |
| Screen | W0 | 4800 | 0.0 | 7.232 | 3.6 | 2.3 | 6.9 | -0.079 | 0.578 |
| Screen | W1 | 1200 | 0.1 | 7.268 | 3.4 | 0.9 | 8.7 | -0.323 | 0.895 |
| Screen | W1 | 2400 | 0.2 | 7.239 | 3.6 | 1.9 | 8.6 | -0.092 | 0.949 |
| Screen | W1 | 4800 | 0.0 | 7.238 | 3.8 | 2.3 | 6.6 | -0.073 | 0.884 |
| Screen | W2 | 1200 | 0.1 | 7.429 | 3.2 | 1.7 | 58.1 | 0.033 | 0.064 |
| Screen | W2 | 2400 | 0.0 | 7.196 | 4.0 | 1.0 | 66.7 | 0.084 | 0.058 |
| Screen | W2 | 4800 | 0.5 | 6.873 | 8.8 | 1.9 | 78.9 | 0.139 | 0.049 |
| Screen | W3 | 1200 | 0.3 | 7.239 | 4.2 | 2.0 | 6.1 | -0.373 | 0.598 |
| Screen | W3 | 2400 | 0.4 | 7.095 | 4.8 | 4.7 | 5.8 | -0.386 | 0.586 |
| Screen | W3 | 4800 | 0.6 | 6.944 | 5.5 | 15.2 | 6.1 | -0.495 | 0.584 |
| Screen | W4 | 1200 | 0.1 | 7.224 | 4.4 | 1.7 | 8.7 | -0.143 | 0.982 |
| Screen | W4 | 2400 | 0.1 | 7.189 | 3.4 | 1.6 | 8.7 | -0.161 | 0.978 |
| Screen | W4 | 4800 | 0.3 | 6.793 | 8.3 | 3.6 | 26.4 | -0.094 | 0.494 |
| Screen | W5 | 1200 | 1.6 | 6.628 | 14.5 | 5.2 | 59.2 | 0.040 | 0.100 |
| Screen | W5 | 2400 | 4.8 | 5.979 | 24.1 | 11.3 | 69.1 | 0.084 | 0.125 |
| Screen | W5 | 4800 | 7.8 | 5.537 | 31.1 | 17.2 | 75.9 | 0.114 | 0.132 |
| A1 | OA | 6580 | 33.7 | 2.614 | 84.6 | 34.8 | 5.9 | -0.298 | 0.277 |
| A1 | OA | 13160 | 55.9 | 1.701 | 91.7 | 56.7 | 5.0 | -0.315 | 0.218 |
| A1 | OA | 19740 | 65.2 | 1.296 | 95.3 | 65.7 | 5.7 | -0.306 | 0.130 |
| A1 | OA | 26320 | 71.0 | 1.024 | 96.9 | 71.6 | 5.8 | -0.325 | 0.114 |
| A1 | OA | 32900 | 70.3 | 1.075 | 95.7 | 70.9 | 5.8 | -0.266 | 0.129 |
| A1 | W5 | 6580 | 8.5 | 5.304 | 33.6 | 17.6 | 78.8 | 0.134 | 0.140 |
| A1 | W5 | 13160 | 10.3 | 5.137 | 36.0 | 20.1 | 75.8 | 0.142 | 0.133 |
| A1 | W5 | 19740 | 12.0 | 4.961 | 37.3 | 21.8 | 81.9 | 0.164 | 0.140 |
| A1 | W5 | 26320 | 14.2 | 4.841 | 40.8 | 22.7 | 84.1 | 0.173 | 0.130 |
| A1 | W5 | 32900 | 14.4 | 4.852 | 39.3 | 23.4 | 82.4 | 0.177 | 0.099 |

**Allocated costs**

| PBS job | Role | Seconds | Node·h |
| --- | --- | ---: | ---: |
| 3714689 | original tests | 32 | 0.00889 |
| 3714703 | pre-tail probes | 33 | 0.00917 |
| 3714716 | screen OA/W0 | 1498 | 0.41611 |
| 3714717 | screen W1/W4 | 1587 | 0.44083 |
| 3714718 | screen W2/W3 | 1644 | 0.45667 |
| 3714719 | screen W5 | 1030 | 0.28611 |
| 3714886 | A1 tests/audit/timing | 290 | 0.08056 |
| 3714914 | A1 OA/W5 rerun | 6550 | 1.81944 |
| 3715599 | final sample/export audit | 37 | 0.01028 |
