# P1-L read fix — completed 2026-10-27

**P1-L stops; no writer selected.** The best learned arm, R at rank 128,
finished at **33.5% EM**, below both the 0.80 writer gate and 0.50 E8 trigger.
Diagnostic OA at rank 128 reached 84.0% but is ineligible. No E8 retrain occurred.

Built R (W5 plus raw-read error divided by target squared norm + 1e-6, with the
entire target and denominator stopped), Rρ (ρ = 0.5), RK (R plus squared key cosine
over distinct addresses), and diagnostic OAρ. A2 adds W5/R/OA at r = r_mem = 128.
Auxiliary weights are 1; W5 retains pooled training-input standardization,
address InfoNCE and clean-value CE. These labels exceed answer-only supervision;
inference exports discard the auxiliary head and optimizer. OA supplies the true
learned value at every valid question token, distinct from the random oracle.

Scope: T1 E16, seed 17, four relations, head sharing 0.3, layer-14 o_proj, L1,
B32/Q4, lr 0.003, scan_final/prefetch2, val 1000/seed 4242. Rank 32 science uses
`0b03d60`; rank 128 uses tested/smoked `be07f7f`. LR was unchanged at rank 128
without a new search, as A2 requires. All three rank 128 arms were submitted at
04:56:45 JST before any WP-L4 validation score was read. Every budget was fixed
from timing, at the largest multiple of 5 satisfying the 1.5× margin within 2 h.
Smokes were 10% of provisional budgets; expanded schedules reuse those timings
(rank 128 W5/R 9.74%, OA 7.70%). No WP-L4 quality score selected a budget.

The no-training audit of [WP-L3's final OA/W5](P1L-writer-fix.md) exactly
reproduced all 1000 predictions with zero repeat-logit drift. Maximum relative
Gram reconstruction error was 3.74e-7/1.41e-7; b/a/d vectors were independently
reconstructed. Zero/shuffle changed actual deployed memory while preserving
question inputs. All four train minibatches (128 episodes, zero optimizer steps)
would clip. A1 had only OA/W5; the task's third rerun arm did not exist.

Rank 32 arms trained with read supervision ended near 0.67 error and 12% EM (A1 W5
error: 26.635). RK ended at key Gram RMS 0.302 and 12% EM. Rank 128 R reached
33.5% versus W5's 29.9%, yet clean/read head decoding remained 93.3%/19.7%.
This is consistent with unresolved deployed-read distortion; clean-head transfer does
not measure all information in the read. OAρ at matched 6580 steps reached 43.2%
versus A1 OA 33.7% (NLL 2.241 vs 2.614). Larger rank/gain helped diagnostics but
did not establish a feasible learned writer. Evidence is single-seed validation
with unequal fixed step budgets; no scientific T2, reader/joint, test,
confirmation or additional-seed runs.

Validation: corrected 55 tests (3715437), A1's 57 (3715659), A2's **69 passed**
(3716019, mg0851): via submit.py, `-m pytest -q tests/test_llm_read_fix.py
tests/test_llm_writer_fix.py tests/test_llm_prefetch.py`. Initial 3715417 had two
test/export failures, fixed before science and counted. Final audit 3716541 on
mg0859 reproduced every sample metric, paired label and frozen schedule, then
reloaded all seven head-free exports with identical 1000 predictions each.
All runtime used submit.py on compute; none ran on login. Compute ended at
06:47 JST. **Cost: 20,642 s = 5.73389 node·h across 16 jobs**, each ≤1 node×2 h;
A2 removed the study total cap.

**Baseline norms, means over 1000 queries.** OA uses its true value as the
deployed read; its matrix/coefficient diagnostics describe the bypassed store.

| Arm | ‖q‖ | ‖v_t‖ | ‖read‖ | ‖u‖ | ‖Ru‖ | ‖Uread‖ | ‖gM‖ | ‖BgM‖ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OA | 37.686 | 27.765 | 27.765 | 42.343 | 189.097 | 124.632 | 21.916 | 245.236 |
| W5 | 19.012 | 30.750 | 159.350 | 42.333 | 233.930 | 211.721 | 45.565 | 515.622 |

| Arm | b_t | a_t | d_t | Value SIR | Parallel interference | Gram RMS | Mean max abs cos | Clean/read head EM% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OA | 1.021 | 0.771 | 0.074 | 0.0455 | 12.210 | 0.3549 | 0.8089 | unavailable |
| W5 | 15.086 | 12.752 | 2.641 | 0.5379 | 0.1086 | 0.3307 | 0.8542 | 85.4 / 10.5 |

| Arm | Original EM% / NLL | Zero EM% / NLL | Shuffle EM% / NLL |
| --- | --- | --- | --- |
| OA | 70.3 / 1.075 | 0.0 / 13.702 | 0.0 / 20.339 |
| W5 | 14.4 / 4.852 | 0.1 / 8.525 | 0.2 / 9.721 |

Full [audit](../../reports/p1l-read-fix/audit.csv),
[age strata](../../reports/p1l-read-fix/audit-age.csv),
[gradient evidence](../../reports/p1l-read-fix/audit-summary.json),
[reproduction and artifact paths](../../reports/p1l-read-fix/README.md).

**All validation checkpoints.** Error is mean normalized squared deployed-read
error. Head columns use the training-only clean-value decoder. Oracle error is
zero by construction; its key metrics concern an unused store and are omitted
here. † is the predeclared matched-step diagnostic, excluded from selection.

| r | Arm | Step | EM% | NLL | Read error | Key1% | Clean/read head% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | R | 4718 | 5.9 | 5.790 | 0.677 | 77.1 | 72.9 / 4.8 |
| 32 | R | 9436 | 9.4 | 5.455 | 0.675 | 79.7 | 80.2 / 5.9 |
| 32 | R | 14154 | 10.1 | 5.235 | 0.687 | 77.5 | 80.9 / 5.6 |
| 32 | R | 18872 | 12.9 | 5.074 | 0.695 | 81.4 | 83.1 / 6.9 |
| 32 | R | 23590 | 12.0 | 5.045 | 0.666 | 83.2 | 84.5 / 8.8 |
| 32 | Rρ | 4844 | 5.2 | 5.707 | 0.660 | 78.3 | 73.1 / 5.9 |
| 32 | Rρ | 9688 | 9.9 | 5.296 | 0.671 | 78.8 | 78.2 / 7.6 |
| 32 | Rρ | 14532 | 10.8 | 5.170 | 0.670 | 80.4 | 80.7 / 6.4 |
| 32 | Rρ | 19376 | 10.2 | 5.090 | 0.683 | 82.2 | 82.7 / 6.5 |
| 32 | Rρ | 24220 | 12.2 | 4.997 | 0.663 | 84.7 | 84.2 / 8.8 |
| 32 | RK | 4396 | 6.2 | 5.654 | 0.669 | 76.5 | 73.9 / 6.7 |
| 32 | RK | 8792 | 9.2 | 5.353 | 0.700 | 79.5 | 77.8 / 8.5 |
| 32 | RK | 13188 | 10.8 | 5.075 | 0.681 | 83.5 | 83.2 / 8.4 |
| 32 | RK | 17584 | 12.4 | 5.068 | 0.681 | 83.7 | 82.5 / 8.2 |
| 32 | RK | 21980 | 12.0 | 4.993 | 0.669 | 82.3 | 84.9 / 7.7 |
| 32 | OAρ | 5408 | 38.4 | 2.428 | 0.000 | — | — |
| 32 | OAρ | 6580† | 43.2 | 2.241 | 0.000 | — | — |
| 32 | OAρ | 10816 | 59.4 | 1.540 | 0.000 | — | — |
| 32 | OAρ | 16224 | 67.7 | 1.173 | 0.000 | — | — |
| 32 | OAρ | 21632 | 66.1 | 1.156 | 0.000 | — | — |
| 32 | OAρ | 27040 | 74.7 | 0.922 | 0.000 | — | — |
| 128 | W5 | 4927 | 16.0 | 4.903 | 6.828 | 77.4 | 85.1 / 12.3 |
| 128 | W5 | 9854 | 21.9 | 4.332 | 2.555 | 83.5 | 91.7 / 14.7 |
| 128 | W5 | 14781 | 26.5 | 3.963 | 1.335 | 84.1 | 91.9 / 16.1 |
| 128 | W5 | 19708 | 30.4 | 3.819 | 1.155 | 85.4 | 95.5 / 20.8 |
| 128 | W5 | 24635 | 29.9 | 3.648 | 0.894 | 88.3 | 94.2 / 20.9 |
| 128 | R | 4929 | 16.3 | 4.814 | 0.677 | 79.3 | 86.9 / 10.9 |
| 128 | R | 9858 | 22.4 | 4.255 | 0.674 | 81.6 | 90.9 / 12.6 |
| 128 | R | 14787 | 27.6 | 3.864 | 0.630 | 89.1 | 91.8 / 16.5 |
| 128 | R | 19716 | 31.0 | 3.639 | 0.623 | 87.8 | 93.1 / 18.8 |
| 128 | R | 24645 | 33.5 | 3.531 | 0.631 | 88.5 | 93.3 / 19.7 |
| 128 | OA | 6235 | 58.4 | 1.537 | 0.000 | — | — |
| 128 | OA | 12470 | 71.6 | 1.036 | 0.000 | — | — |
| 128 | OA | 18705 | 77.8 | 0.774 | 0.000 | — | — |
| 128 | OA | 24940 | 83.4 | 0.580 | 0.000 | — | — |
| 128 | OA | 31175 | 84.0 | 0.578 | 0.000 | — | — |

[EM curves](../../reports/p1l-read-fix/em_curves.png),
[all objective loss curves](../../reports/p1l-read-fix/loss_curves.png),
[full metrics](../../reports/p1l-read-fix/runs.csv),
[immutable selection](../../reports/p1l-read-fix/selection.json),
[seven export checks](../../reports/p1l-read-fix/summary.json).

**Allocated costs.** All jobs used one node. Queue wait is excluded; failed
tests are included. [Machine-readable ledger](../../reports/p1l-read-fix/jobs.json).

| PBS job | Role | Seconds | Node·h | Exit |
| --- | --- | ---: | ---: | ---: |
| 3715417 | initial tests, two failures | 33 | 0.00917 | 1 |
| 3715437 | corrected tests, 55 passed | 34 | 0.00944 | 0 |
| 3715461 | r32 smoke R/Rρ | 218 | 0.06056 | 0 |
| 3715462 | r32 smoke RK/OAρ | 230 | 0.06389 | 0 |
| 3715504 | r32 smoke artifact/timing audit | 20 | 0.00556 | 0 |
| 3715621 | no-training A1 checkpoint audit | 64 | 0.01778 | 0 |
| 3715659 | A1 tests, independent coefficient audit, release | 43 | 0.01194 | 0 |
| 3715707 | r32 science R/Rρ | 4759 | 1.32194 | 0 |
| 3715708 | r32 science RK/OAρ | 4483 | 1.24528 | 0 |
| 3716019 | A2 tests, 69 passed | 191 | 0.05306 | 0 |
| 3716039 | r128 smoke W5/R | 510 | 0.14167 | 0 |
| 3716040 | r128 smoke OA | 412 | 0.11444 | 0 |
| 3716145 | r128 artifact/export audit, timing release | 34 | 0.00944 | 0 |
| 3716159 | r128 science W5/R | 4920 | 1.36667 | 0 |
| 3716160 | r128 science OA | 4612 | 1.28111 | 0 |
| 3716541 | selection and seven export reloads | 79 | 0.02194 | 0 |
