# RFX report: read-failure factorial on the frozen WP-L4 checkpoints (WP-L7)

Registered design: `docs/prereg/RFX.md` (2026-10-27 11:05 JST). Run 2026-10-29 JST by sn-llm at `efadc36436e4d29514a4ea570596925b779b8b2d`
(wp/llm; contains the step-1b rule, the step-1c entity-EM guard and the step-1d tie rule). Records: `reports/p1l-rfx/` (copies of
`runs/p1l/rfx/science/`; the per-row `*-samples.npz` stay under `runs/`). Every table below is printed by
`python3 scripts/analyze_rfx.py reports/p1l-rfx` from those records and nothing else.

## Result
| checkpoint | verdict (first rule that applies) | triggers |
| --- | --- | --- |
| R-r128 | **store+query**: ridge/true/fit rescues (and ridge/true/asis); no other non-control cell does | nothing |
| R-r32 | **store+query**: the same | nothing |
| W5-r32 | **none** | nothing |

`lsq_arms` is empty and `rfx_eta_by_rank` / `eta_by_rank` are empty (no storage verdict at any rank): under `docs/prereg/LSQ.md`
no LSQ arm runs and no LSQ release exists (orchestrator decision after this summary). The registered rule is applied as written;
the rest of this page is descriptive.

What the tables show, in the registered order of the rule:
- **Deployed read** (delta/raw): normalized error 0.614 (R-r128), 0.655 (R-r32), 0.714 (W5-r32, with the fitted gain; 25.4 as is)
  against a zero read at 1.0, with D1 EM 33.8 / 11.6 / 14.9 % (baseline EM_b = the better of as-is and gain-fitted).
- **Storage alone** (ridge/raw/fit and ridge/unit/fit): error 0.57 to 0.60 at the two R checkpoints and 0.69 at W5-r32; EM 38.8 %
  against EM_b 33.8 % at R-r128 (+5.0 pp), 13.5 to 13.7 % against 11.6 % at R-r32 (+1.9 to +2.1 pp), and 9.1 to 9.2 % at W5-r32,
  below its EM_b of 14.9 % (the as-is ridge read, 15.7 %, is +0.8 pp). None halves the error and gains 10 pp, so the exact store with the
  deployed or the unit query does not rescue. Unit normalization of the query on the delta store changes the error by about -0.005
  and the EM by at most +0.2 pp, so there is no normalization verdict either.
- **Query alone** (delta/true/fit): error 0.498 / 0.584 / 0.566, EM 39.9 / 14.0 / 9.4 %. At R-r128 that is +6.1 pp over EM_b with
  the error ratio 0.81; it does not rescue.
- **Exact store and true key together** (ridge/true): error 3e-10 to 5e-8, i.e. the written value is read back exactly (the
  clean control, which uses the same value), and EM 84.2 % at R-r128 and 49.0 % at R-r32 equal the clean control (84.4 / 48.8 %).
  At W5-r32 the clean control itself is 17.6 %, so nothing can rescue there (its clean-value head accuracy is 84.6 %).
- **Decoder**: the refit linear decoder (D3) on the delta/raw reads reaches 18.4 / 6.8 / 8.1 %, below EMe_b at every checkpoint. No
  decoder verdict. The clean-value head (D2) reads the clean value at 93.0 / 84.9 / 84.6 % and the deployed read at 18.9 / 7.1 /
  11.1 %.

Reading guide (rv-fable E3; descriptive, not a rule): delta/true/fit, the deployed store with the true key, reaches 39.9 % at
R-r128, far from the OA level (84.0 %) and much closer to the checkpoint's own EM (33.8 %); at R-r32 14.0 % against its own 11.6 %
(the clean control 48.8 % is the ceiling there). By the guide, write interference in the delta store dominates at both ranks and
the query path is not the bottleneck (a descriptive reading of one seed): the unit true key on the learned delta state still reads a heavily cross-talked value
(error 0.50 to 0.58 against 3e-10 for the exact store). The exact store becomes the OA level only when the true key is also
given, which is why the registered rule calls it store+query. The learned key Gram is badly conditioned (log10 condition number
median 2.4 at r = 128, 3.6 at r = 32, with E = 16 keys), and the ridge store needed eta = 1 (a regularized, no longer
interpolating store) to be best with the learned queries but eta = 1e-6 with the true key.

## Cost
| job | purpose | wall | node·h | outcome |
| --- | --- | --- | ---: | --- |
| 3720943 | `rfx_run --smoke` (64 + 64 episodes, train split) | 1m05s | 0.018 | guard passed, max `full_hit_entity_miss` 0 over 42 cells |
| 3721047 | `rfx_run` science (three checkpoints) | 4m52s | 0.081 | guard passed, no instrumentation failure |

Total 0.10 node·h, below the registered 0.15 to 0.2 (fit 16 to 18 s and evaluation about 74 s per checkpoint). One node, no
`--scale-up`, no training, no test or confirmation data read. The tests and smokes of this work package are recorded in
`docs/reports/WP-L5-state.md` and `docs/reports/DPFP.md`.

## Registered checks
Entity-EM guard (step 1c) passed on the smoke and the science run; with the step-1d tie rule (the restricted argmax breaks ties
like the full-vocabulary one) `full_hit_entity_miss` is 0 in every cell, as a correct glue must give. The checkpoint digests
equal `rfx_run.DIGESTS` (the job refuses otherwise). The four instrumentation checks (identity of the re-inserted read, recompute
of M q, rebuild of the delta state, permutation invariance of ridge) passed at every checkpoint; see the last table.

## Tables
### Verdicts and baselines

| label | rank | steps | verdict | e_b = err delta/raw/fit | EM_b (%) | EMe_b (%) | D3 delta/raw (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R-r128 | 128 | 24645 | `store+query` | 0.6145 | 33.8 | 33.9 | 18.4 |
| R-r32 | 32 | 23590 | `store+query` | 0.6552 | 11.6 | 11.6 | 6.8 |
| W5-r32 | 32 | 32900 | `none` | 0.7136 | 14.9 | 14.9 | 8.1 |

Triggered LSQ arms (`lsq_arms`): none.

### Rescue flags

| cell | R-r128 | R-r32 | W5-r32 |
| --- | ---: | ---: | ---: |
| `clean` | **yes** | **yes** | no |
| `zero` | no | no | no |
| `delta/raw/asis` | no | no | no |
| `delta/raw/fit` | no | no | no |
| `delta/unit/asis` | no | no | no |
| `delta/unit/fit` | no | no | no |
| `delta/true/asis` | no | no | no |
| `delta/true/fit` | no | no | no |
| `ridge/raw/asis` | no | no | no |
| `ridge/raw/fit` | no | no | no |
| `ridge/unit/asis` | no | no | no |
| `ridge/unit/fit` | no | no | no |
| `ridge/true/asis` | **yes** | **yes** | no |
| `ridge/true/fit` | **yes** | **yes** | no |
| `decoder_refit` | no | no | no |

### Cells: R-r128 (rank 128, 24645 steps)

| cell | err (SE) | D1 EM % (SE) | D1 EMe % (SE) | D1 NLL | D2 head acc % | D3 acc % | D3 NLL | rescues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | 0 (0) | 84.4 (0.6) | 84.4 (0.6) | 0.801 | 93.0 | 91.7 | 0.513 | yes |
| `zero` | 1 (7.2e-13) | 21.9 (0.7) | 21.9 (0.7) | 4.305 | 0.1 | 0.0 | 6.727 |  |
| `delta/raw/asis` | 0.6145 (0.0025) | 33.8 (0.7) | 33.9 (0.7) | 3.416 | 18.9 | 18.4 | 4.522 |  |
| `delta/raw/fit` | 0.6145 (0.0026) | 33.8 (0.7) | 33.8 (0.7) | 3.419 | 19.0 |  |  |  |
| `delta/unit/asis` | 0.9942 (2.8e-05) | 22.3 (0.7) | 22.3 (0.7) | 4.279 | 0.4 | 18.7 | 4.500 |  |
| `delta/unit/fit` | 0.6097 (0.0026) | 34.0 (0.7) | 34.0 (0.7) | 3.420 | 19.1 |  |  |  |
| `delta/true/asis` | 0.9913 (2.9e-05) | 22.4 (0.7) | 22.4 (0.7) | 4.264 | 0.5 | 28.1 | 3.760 |  |
| `delta/true/fit` | 0.4979 (0.0023) | 39.9 (0.8) | 39.9 (0.8) | 2.990 | 22.8 |  |  |  |
| `ridge/raw/asis` | 3268 (23) | 25.7 (0.7) | 25.8 (0.7) | 4.216 | 27.3 | 24.1 | 4.085 |  |
| `ridge/raw/fit` | 0.5765 (0.0028) | 38.8 (0.8) | 38.8 (0.8) | 3.185 | 27.4 |  |  |  |
| `ridge/unit/asis` | 0.6191 (0.0019) | 38.4 (0.8) | 38.4 (0.8) | 3.202 | 27.3 | 24.5 | 4.054 |  |
| `ridge/unit/fit` | 0.5712 (0.0028) | 38.8 (0.8) | 38.8 (0.8) | 3.189 | 27.4 |  |  |  |
| `ridge/true/asis` | 3.761e-10 (9.1e-12) | 84.3 (0.6) | 84.3 (0.6) | 0.801 | 93.0 | 91.7 | 0.513 | yes |
| `ridge/true/fit` | 3.22e-10 (8.3e-12) | 84.2 (0.6) | 84.2 (0.6) | 0.801 | 93.0 |  |  | yes |

Selected eta (fitting set) and scalar gain g, by query: raw: eta 1, gain ridge 0.01126, gain delta 1.029; unit: eta 1, gain ridge 1.505, gain delta 136.8; true: eta 1e-06, gain ridge 1, gain delta 117.4.

Fitting-set mean normalized error after the gain fit, by eta:

| query | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.8833 | 0.8832 | 0.8826 | 0.877 | 0.8361 | 0.6992 | 0.5793 | 0.597 |
| unit | 0.8824 | 0.8824 | 0.8817 | 0.876 | 0.8346 | 0.6949 | 0.5739 | 0.5928 |
| true | 3.342e-10 | 3.336e-08 | 3.286e-06 | 0.0002851 | 0.01189 | 0.1069 | 0.3027 | 0.4523 |

### Cells: R-r32 (rank 32, 23590 steps)

| cell | err (SE) | D1 EM % (SE) | D1 EMe % (SE) | D1 NLL | D2 head acc % | D3 acc % | D3 NLL | rescues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | 0 (0) | 48.8 (0.8) | 48.8 (0.8) | 2.576 | 84.9 | 80.6 | 0.932 | yes |
| `zero` | 1 (8.1e-12) | 7.5 (0.4) | 7.5 (0.4) | 5.406 | 0.1 | 0.0 | 6.727 |  |
| `delta/raw/asis` | 0.6586 (0.0036) | 11.6 (0.5) | 11.6 (0.5) | 4.975 | 7.1 | 6.8 | 5.431 |  |
| `delta/raw/fit` | 0.6552 (0.0032) | 11.6 (0.5) | 11.6 (0.5) | 4.979 | 7.1 |  |  |  |
| `delta/unit/asis` | 0.9557 (0.00025) | 8.2 (0.4) | 8.2 (0.4) | 5.343 | 0.9 | 6.8 | 5.411 |  |
| `delta/unit/fit` | 0.6508 (0.0031) | 11.5 (0.5) | 11.5 (0.5) | 4.977 | 7.1 |  |  |  |
| `delta/true/asis` | 0.947 (0.00025) | 8.2 (0.4) | 8.2 (0.4) | 5.315 | 1.2 | 11.3 | 4.999 |  |
| `delta/true/fit` | 0.584 (0.0032) | 14.0 (0.5) | 14.0 (0.5) | 4.744 | 10.5 |  |  |  |
| `ridge/raw/asis` | 59.44 (0.51) | 6.6 (0.4) | 7.6 (0.4) | 6.062 | 11.1 | 9.8 | 5.128 |  |
| `ridge/raw/fit` | 0.6015 (0.0033) | 13.5 (0.5) | 13.5 (0.5) | 4.808 | 10.6 |  |  |  |
| `ridge/unit/asis` | 0.6157 (0.0026) | 13.1 (0.5) | 13.1 (0.5) | 4.839 | 10.3 | 9.5 | 5.106 |  |
| `ridge/unit/fit` | 0.5974 (0.0033) | 13.7 (0.5) | 13.7 (0.5) | 4.809 | 10.6 |  |  |  |
| `ridge/true/asis` | 4.803e-08 (1.6e-09) | 49.0 (0.8) | 49.0 (0.8) | 2.575 | 84.9 | 80.6 | 0.932 | yes |
| `ridge/true/fit` | 4.424e-08 (1.5e-09) | 49.0 (0.8) | 49.1 (0.8) | 2.576 | 84.9 |  |  | yes |

Selected eta (fitting set) and scalar gain g, by query: raw: eta 1, gain ridge 0.07586, gain delta 0.9165; unit: eta 1, gain ridge 1.27, gain delta 15.34; true: eta 1e-06, gain ridge 1, gain delta 15.43.

Fitting-set mean normalized error after the gain fit, by eta:

| query | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.9138 | 0.9132 | 0.9076 | 0.8726 | 0.7779 | 0.6655 | 0.6035 | 0.6375 |
| unit | 0.9131 | 0.9125 | 0.9069 | 0.8713 | 0.7745 | 0.66 | 0.5984 | 0.6329 |
| true | 4.182e-08 | 4.089e-06 | 0.0003361 | 0.01184 | 0.0962 | 0.2677 | 0.4306 | 0.5479 |

### Cells: W5-r32 (rank 32, 32900 steps)

| cell | err (SE) | D1 EM % (SE) | D1 EMe % (SE) | D1 NLL | D2 head acc % | D3 acc % | D3 NLL | rescues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | 0 (0) | 17.6 (0.6) | 17.6 (0.6) | 4.589 | 84.6 | 82.2 | 0.875 |  |
| `zero` | 1 (8.7e-12) | 6.8 (0.4) | 6.8 (0.4) | 5.512 | 0.1 | 0.0 | 6.727 |  |
| `delta/raw/asis` | 25.44 (0.28) | 14.9 (0.6) | 14.9 (0.6) | 4.817 | 11.1 | 8.1 | 5.369 |  |
| `delta/raw/fit` | 0.7136 (0.0036) | 8.5 (0.4) | 8.5 (0.4) | 5.346 | 10.9 |  |  |  |
| `delta/unit/asis` | 0.7728 (0.0019) | 7.8 (0.4) | 7.8 (0.4) | 5.420 | 9.7 | 8.4 | 5.333 |  |
| `delta/unit/fit` | 0.7093 (0.0036) | 8.6 (0.4) | 8.6 (0.4) | 5.343 | 10.8 |  |  |  |
| `delta/true/asis` | 0.6723 (0.0017) | 8.2 (0.4) | 8.2 (0.4) | 5.387 | 13.1 | 13.6 | 4.794 |  |
| `delta/true/fit` | 0.5664 (0.0035) | 9.4 (0.5) | 9.4 (0.5) | 5.269 | 15.0 |  |  |  |
| `ridge/raw/asis` | 69.95 (0.68) | 15.7 (0.6) | 15.7 (0.6) | 4.774 | 13.1 | 9.7 | 5.195 |  |
| `ridge/raw/fit` | 0.6906 (0.0036) | 9.1 (0.5) | 9.1 (0.5) | 5.311 | 13.2 |  |  |  |
| `ridge/unit/asis` | 0.6954 (0.0029) | 8.7 (0.4) | 8.7 (0.4) | 5.340 | 13.2 | 9.9 | 5.155 |  |
| `ridge/unit/fit` | 0.6856 (0.0036) | 9.2 (0.5) | 9.2 (0.5) | 5.308 | 13.1 |  |  |  |
| `ridge/true/asis` | 4.843e-08 (1.6e-09) | 17.5 (0.6) | 17.5 (0.6) | 4.590 | 84.6 | 82.2 | 0.876 |  |
| `ridge/true/fit` | 4.442e-08 (1.5e-09) | 17.7 (0.6) | 17.7 (0.6) | 4.590 | 84.6 |  |  |  |

Selected eta (fitting set) and scalar gain g, by query: raw: eta 1, gain ridge 0.06225, gain delta 0.09695; unit: eta 1, gain ridge 1.21, gain delta 1.88; true: eta 1e-06, gain ridge 1, gain delta 1.986.

Fitting-set mean normalized error after the gain fit, by eta:

| query | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.9686 | 0.9683 | 0.9661 | 0.9513 | 0.8973 | 0.782 | 0.6915 | 0.7083 |
| unit | 0.9692 | 0.969 | 0.9668 | 0.952 | 0.8972 | 0.7781 | 0.686 | 0.7032 |
| true | 4.295e-08 | 4.201e-06 | 0.0003466 | 0.0124 | 0.1009 | 0.2711 | 0.4376 | 0.5599 |

### Write age (0 = most recent of the 16 writes): error / EM %


R-r128

| age | n | `delta/raw/asis` | `delta/raw/fit` | `ridge/raw/fit` | `delta/true/fit` | `ridge/true/fit` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 253 | 0.574 / 32.0 | 0.571 / 32.0 | 0.55 / 36.0 | 0.455 / 39.1 | 2.95e-10 / 83.8 |
| 1 | 270 | 0.581 / 33.7 | 0.579 / 33.7 | 0.551 / 35.9 | 0.459 / 40.7 | 2.77e-10 / 82.6 |
| 2 | 244 | 0.583 / 35.7 | 0.582 / 35.2 | 0.549 / 40.2 | 0.457 / 40.6 | 3.11e-10 / 86.9 |
| 3 | 262 | 0.595 / 27.9 | 0.595 / 28.2 | 0.57 / 34.7 | 0.467 / 36.3 | 3.78e-10 / 83.6 |
| 4 | 237 | 0.608 / 28.3 | 0.608 / 28.3 | 0.575 / 32.5 | 0.475 / 35.9 | 2.92e-10 / 82.3 |
| 5 | 230 | 0.599 / 38.3 | 0.598 / 38.7 | 0.566 / 44.8 | 0.461 / 43.9 | 3.19e-10 / 87.4 |
| 6 | 249 | 0.603 / 41.4 | 0.602 / 40.2 | 0.57 / 42.6 | 0.474 / 47.0 | 3.16e-10 / 85.5 |
| 7 | 263 | 0.607 / 35.7 | 0.607 / 35.7 | 0.582 / 40.3 | 0.488 / 40.7 | 3.54e-10 / 89.0 |
| 8 | 248 | 0.605 / 35.5 | 0.605 / 36.7 | 0.583 / 41.5 | 0.479 / 43.5 | 4.04e-10 / 86.3 |
| 9 | 255 | 0.597 / 40.8 | 0.596 / 40.8 | 0.564 / 46.7 | 0.492 / 45.1 | 3.52e-10 / 85.9 |
| 10 | 236 | 0.604 / 35.2 | 0.603 / 34.3 | 0.563 / 42.8 | 0.497 / 41.5 | 3.4e-10 / 88.1 |
| 11 | 254 | 0.619 / 33.9 | 0.62 / 34.3 | 0.573 / 40.2 | 0.526 / 40.6 | 3.71e-10 / 85.4 |
| 12 | 244 | 0.626 / 29.1 | 0.628 / 29.5 | 0.585 / 36.1 | 0.524 / 32.4 | 3.52e-10 / 81.6 |
| 13 | 243 | 0.64 / 39.5 | 0.642 / 38.7 | 0.579 / 44.0 | 0.533 / 45.3 | 3.28e-10 / 84.4 |
| 14 | 254 | 0.657 / 32.3 | 0.657 / 32.3 | 0.599 / 38.6 | 0.553 / 36.6 | 2.7e-10 / 81.1 |
| 15 | 258 | 0.732 / 22.5 | 0.736 / 22.5 | 0.664 / 25.2 | 0.624 / 29.5 | 1.97e-10 / 74.0 |

R-r32

| age | n | `delta/raw/asis` | `delta/raw/fit` | `ridge/raw/fit` | `delta/true/fit` | `ridge/true/fit` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 253 | 0.585 / 16.6 | 0.594 / 17.0 | 0.549 / 17.4 | 0.515 / 19.4 | 3.29e-08 / 51.4 |
| 1 | 270 | 0.62 / 11.9 | 0.622 / 11.5 | 0.573 / 13.3 | 0.542 / 13.3 | 4.66e-08 / 49.3 |
| 2 | 244 | 0.614 / 16.4 | 0.615 / 15.6 | 0.569 / 20.9 | 0.542 / 18.4 | 5e-08 / 54.5 |
| 3 | 262 | 0.637 / 9.9 | 0.635 / 9.5 | 0.601 / 12.2 | 0.56 / 12.2 | 4.45e-08 / 52.7 |
| 4 | 237 | 0.649 / 10.1 | 0.647 / 9.7 | 0.6 / 11.0 | 0.566 / 11.0 | 3.91e-08 / 46.4 |
| 5 | 230 | 0.621 / 10.9 | 0.621 / 11.3 | 0.564 / 13.5 | 0.549 / 14.8 | 4.82e-08 / 57.4 |
| 6 | 249 | 0.634 / 12.4 | 0.633 / 12.0 | 0.58 / 14.9 | 0.564 / 16.1 | 5.42e-08 / 56.6 |
| 7 | 263 | 0.661 / 9.9 | 0.659 / 10.6 | 0.621 / 11.0 | 0.588 / 13.3 | 4.26e-08 / 52.1 |
| 8 | 248 | 0.647 / 12.5 | 0.645 / 12.9 | 0.596 / 15.7 | 0.575 / 14.5 | 5e-08 / 51.6 |
| 9 | 255 | 0.654 / 17.6 | 0.652 / 17.6 | 0.595 / 16.1 | 0.58 / 18.0 | 4.22e-08 / 52.5 |
| 10 | 236 | 0.651 / 12.3 | 0.651 / 11.9 | 0.583 / 16.9 | 0.59 / 14.4 | 3.91e-08 / 52.5 |
| 11 | 254 | 0.685 / 11.4 | 0.677 / 11.8 | 0.622 / 13.0 | 0.62 / 14.6 | 4.19e-08 / 48.0 |
| 12 | 244 | 0.704 / 9.4 | 0.691 / 9.0 | 0.644 / 10.2 | 0.633 / 11.1 | 5.12e-08 / 44.3 |
| 13 | 243 | 0.735 / 10.3 | 0.72 / 10.7 | 0.634 / 14.0 | 0.66 / 13.2 | 4.94e-08 / 48.1 |
| 14 | 254 | 0.684 / 8.3 | 0.679 / 8.3 | 0.61 / 10.6 | 0.61 / 12.2 | 4.48e-08 / 38.6 |
| 15 | 258 | 0.756 / 6.2 | 0.741 / 5.8 | 0.68 / 5.4 | 0.65 / 7.0 | 3.2e-08 / 29.8 |

W5-r32

| age | n | `delta/raw/asis` | `delta/raw/fit` | `ridge/raw/fit` | `delta/true/fit` | `ridge/true/fit` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 253 | 21.3 / 20.9 | 0.609 / 11.1 | 0.667 / 11.1 | 0.408 / 12.3 | 3.75e-08 / 23.7 |
| 1 | 270 | 21.4 / 18.5 | 0.628 / 9.6 | 0.679 / 10.7 | 0.441 / 11.5 | 4.83e-08 / 21.1 |
| 2 | 244 | 22.9 / 20.1 | 0.628 / 9.8 | 0.67 / 10.2 | 0.449 / 11.9 | 4.35e-08 / 20.9 |
| 3 | 262 | 25.4 / 17.2 | 0.665 / 7.3 | 0.698 / 7.3 | 0.496 / 7.6 | 5.06e-08 / 15.3 |
| 4 | 237 | 26.1 / 15.6 | 0.68 / 11.8 | 0.695 / 12.2 | 0.521 / 13.5 | 3.8e-08 / 20.3 |
| 5 | 230 | 24.6 / 19.1 | 0.673 / 9.6 | 0.669 / 10.4 | 0.507 / 11.3 | 4.55e-08 / 20.9 |
| 6 | 249 | 26 / 19.3 | 0.697 / 10.0 | 0.692 / 10.0 | 0.543 / 9.6 | 5.59e-08 / 20.1 |
| 7 | 263 | 24.4 / 14.8 | 0.693 / 9.5 | 0.679 / 9.9 | 0.571 / 10.3 | 5.83e-08 / 18.3 |
| 8 | 248 | 25.3 / 16.1 | 0.709 / 7.7 | 0.679 / 8.5 | 0.571 / 9.3 | 4.21e-08 / 15.7 |
| 9 | 255 | 25 / 14.9 | 0.726 / 9.4 | 0.683 / 11.4 | 0.575 / 10.6 | 3.55e-08 / 20.8 |
| 10 | 236 | 24.3 / 11.0 | 0.719 / 6.4 | 0.657 / 7.2 | 0.597 / 5.9 | 4.64e-08 / 14.8 |
| 11 | 254 | 26.3 / 12.2 | 0.77 / 7.9 | 0.712 / 7.9 | 0.647 / 7.9 | 3.89e-08 / 14.6 |
| 12 | 244 | 30.3 / 9.0 | 0.774 / 5.3 | 0.714 / 6.6 | 0.664 / 5.7 | 4.22e-08 / 14.8 |
| 13 | 243 | 30.5 / 13.6 | 0.819 / 8.6 | 0.721 / 8.6 | 0.691 / 7.8 | 4.28e-08 / 15.6 |
| 14 | 254 | 24.7 / 9.4 | 0.79 / 5.5 | 0.687 / 6.7 | 0.674 / 7.9 | 3.91e-08 / 12.6 |
| 15 | 258 | 28.7 / 6.2 | 0.84 / 7.0 | 0.746 / 7.0 | 0.712 / 7.0 | 4.49e-08 / 13.6 |

### Write order (four per-episode random permutations, deployed q through the LM path)

| label | order | err | EM % | agreement with natural % | normalized sq. diff of read |
| --- | ---: | ---: | ---: | ---: | ---: |
| R-r128 | natural | 0.6145 | 33.8 |  |  |
| R-r128 | perm 0 | 0.6143 | 33.7 | 97.8 | 7.99e-05 |
| R-r128 | perm 1 | 0.6143 | 33.8 | 97.8 | 8.09e-05 |
| R-r128 | perm 2 | 0.6144 | 34.0 | 97.9 | 8.04e-05 |
| R-r128 | perm 3 | 0.6143 | 33.8 | 98.0 | 7.89e-05 |
| R-r32 | natural | 0.6586 | 11.6 |  |  |
| R-r32 | perm 0 | 0.6577 | 11.7 | 95.3 | 0.0005 |
| R-r32 | perm 1 | 0.6578 | 11.6 | 95.1 | 0.000511 |
| R-r32 | perm 2 | 0.6577 | 11.6 | 95.3 | 0.000502 |
| R-r32 | perm 3 | 0.6578 | 11.5 | 95.0 | 0.000492 |
| W5-r32 | natural | 25.44 | 14.9 |  |  |
| W5-r32 | perm 0 | 25.17 | 14.8 | 77.2 | 0.0573 |
| W5-r32 | perm 1 | 25.24 | 14.8 | 77.7 | 0.0585 |
| W5-r32 | perm 2 | 25.12 | 14.5 | 77.4 | 0.0568 |
| W5-r32 | perm 3 | 25.27 | 14.4 | 77.2 | 0.0564 |

Error and EM % by permuted write age (mean over the four permutations):

| permuted age | R-r128 | R-r32 | W5-r32 |
| --- | ---: | ---: | ---: |
| 0 | 0.6 / 35.6 | 0.636 / 11.8 | 23.3 / 20.5 |
| 1 | 0.611 / 36.3 | 0.642 / 11.4 | 23.5 / 18.2 |
| 2 | 0.608 / 33.5 | 0.647 / 11.9 | 23.8 / 17.5 |
| 3 | 0.616 / 33.0 | 0.656 / 9.5 | 24.5 / 15.1 |
| 4 | 0.607 / 36.3 | 0.645 / 14.2 | 24.3 / 18.4 |
| 5 | 0.621 / 34.2 | 0.664 / 10.0 | 25.3 / 15.4 |
| 6 | 0.61 / 37.8 | 0.659 / 12.2 | 25.2 / 15.3 |
| 7 | 0.613 / 31.3 | 0.662 / 11.0 | 24.9 / 13.9 |
| 8 | 0.607 / 32.6 | 0.654 / 11.7 | 24.8 / 13.4 |
| 9 | 0.623 / 33.6 | 0.671 / 9.8 | 25.4 / 10.7 |
| 10 | 0.61 / 33.9 | 0.653 / 12.7 | 25.3 / 13.9 |
| 11 | 0.623 / 32.4 | 0.675 / 11.9 | 26.5 / 13.1 |
| 12 | 0.628 / 31.9 | 0.675 / 12.3 | 27.5 / 12.5 |
| 13 | 0.62 / 31.5 | 0.663 / 10.2 | 27.3 / 10.5 |
| 14 | 0.611 / 33.6 | 0.663 / 12.4 | 26.3 / 12.6 |
| 15 | 0.624 / 33.3 | 0.66 / 12.3 | 25.5 / 12.5 |

### Conditioning (log10 condition number over the 4,000 evaluation episodes)

| label | matrix | eta | median | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R-r128 | K^T K |  | 2.39 | 2.63 | 2.85 | 3.17 |
| R-r128 | K^T K + eta I (raw) | 1 | 0.66 | 0.72 | 0.77 | 0.82 |
| R-r128 | K^T K + eta I (true) | 1e-06 | 2.39 | 2.63 | 2.85 | 3.17 |
| R-r128 | K^T K + eta I (unit) | 1 | 0.66 | 0.72 | 0.77 | 0.82 |
| R-r32 | K^T K |  | 3.55 | 3.85 | 4.09 | 4.50 |
| R-r32 | K^T K + eta I (raw) | 1 | 0.71 | 0.77 | 0.82 | 0.87 |
| R-r32 | K^T K + eta I (true) | 1e-06 | 3.55 | 3.85 | 4.09 | 4.50 |
| R-r32 | K^T K + eta I (unit) | 1 | 0.71 | 0.77 | 0.82 | 0.87 |
| W5-r32 | K^T K |  | 3.55 | 3.84 | 4.12 | 4.33 |
| W5-r32 | K^T K + eta I (raw) | 1 | 0.71 | 0.77 | 0.82 | 0.90 |
| W5-r32 | K^T K + eta I (true) | 1e-06 | 3.55 | 3.84 | 4.12 | 4.33 |
| W5-r32 | K^T K + eta I (unit) | 1 | 0.71 | 0.77 | 0.82 | 0.90 |

### Normalization and gain effects

| label | unit - raw, delta: err | EM pp | unit - raw, ridge: err | EM pp | fit - asis, delta/raw: err | EM pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R-r128 | -0.0048 | +0.18 | -0.0053 | -0.05 | -1.21e-05 | -0.03 |
| R-r32 | -0.0044 | -0.05 | -0.0041 | +0.18 | -3.43e-03 | -0.05 |
| W5-r32 | -0.0043 | +0.10 | -0.0050 | +0.05 | -2.47e+01 | -6.35 |

### Instrumentation checks and run

| label | identity max abs logit diff | raw recompute max rel | delta rebuild max rel | ridge permutation max abs | fit / evaluate seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| R-r128 | 0.0 | 0.0 | 0.0 | 2.73e-12 | 18 / 73 |
| R-r32 | 0.0 | 0.0 | 0.0 | 1.71e-13 | 16 / 74 |
| W5-r32 | 0.0 | 0.0 | 0.0 | 1.71e-13 | 16 / 74 |

Entity-EM guard: {'passed': True, 'slack': 0.005}; `full_hit_entity_miss` over every cell: 0. Instrumentation failures: {'R-r128': [], 'R-r32': [], 'W5-r32': []}. Job 3721047.opbs at `efadc36436e4d29514a4ea570596925b779b8b2d`, source dirty False, NVIDIA GH200 120GB, torch 2.14.0+cu132.


## Caveats (from the registration, still binding)
A substituted read is a distribution shift for the LM path after the read; D1 decides the storage and query verdicts. The true-key
cells use the label and are not deployable. One seed (17) and one evaluation set of 4,000 val queries (EM standard error 0.5 to
0.8 pp). "zero" here zeroes the final-position read only, not WP-L4's zero-memory control. The W5-r32 as-is error of 25.4
(delta/raw/asis) is a scale artifact that the fitted gain removes (0.714), which is why the baseline is the better of the two.
