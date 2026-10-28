# DPFP report: DPFP keys and queries for the learned LLM writer (WP-L5)

Registered design: `docs/prereg/DPFP.md` (2026-10-27 07:22 JST, Amendment A1 08:51 JST). Run 2026-10-29 JST by sn-llm at
`efadc36436e4d29514a4ea570596925b779b8b2d` (wp/llm). Arms: D32-3 (r = 32, nu = 3), D128-1 (r = 128, nu = 1), D128-3 (r = 128,
nu = 3); T1 as WP-L4 (E = 16, seed 17, layer 14, L = 1, batch 32 / Q 4, lr 3e-3), one 1-node 2 h job per arm
(`--scale-up docs/prereg/DPFP.md`), step counts from the 10% smokes by the registered rule (release `reports/p1l-dpfp/release.json`,
1dd1869). Records are copied to `reports/p1l-dpfp/arms/` and, for the two WP-L4 R references, `reports/p1l-dpfp/refs/`; every table
below is printed by `python3 scripts/analyze_dpfp.py reports/p1l-dpfp` from them.

## Result and final registered decision
**Final decision (orchestrator-confirmed 2026-10-29): stop. No DPFP writer is selected, P1L-D is not run, and the study is a negative result.**
No E = 16 arm reaches val EM 0.80 (best D128-3: 608 of 1,000, 60.8 %). That is at least 0.50, so the registered rule triggered one retrain of
D128-3 at E = 8 (flagged; audited lock `reports/p1l-dpfp/e8-lock.json`, 72ed469), accepted only at val EM 0.80. It reached **730 of 1,000
(73.0 %)** at its released 26065 steps (job 3721535), so it is not accepted. Per `docs/prereg/DPFP.md` the paper reports the negative result
and no further writer screen follows from this study; LSQ ran no arm (RFX triggered none), so the learned-writer line stops.

| arm | final val EM | candidate order | 0.80 |
| --- | ---: | --- | --- |
| D32-3 | 39.0 % | 1 | no |
| D128-1 | 58.0 % | 2 | no |
| D128-3 | 60.8 % | 3 | no |
| D128-3 at E = 8 (retrain, flagged) | 73.0 % | | no |

Every arm completed its released steps (`status` complete, not a smoke, source clean, `actual_steps` = released steps; E = 16, and E = 8
for the retrain; the logs have no error and exit 0). The WP-L4 references (R at r = 32: 12.0 %, at r = 128: 33.5 %) are used as registered, not rerun,
because the DPFP-off reproduction test (`tests/test_llm_dpfp_repro.py`, job 3720637 `t-dpfp-repro`, passed) matches WP-L4's first 200
logged steps exactly at both ranks.

## Reading (descriptive; one seed; a validation EM from 1,000 episodes has a standard error of about 1.5 pp)
- DPFP raises val EM at both ranks: r = 32 from 12.0 % to 39.0 % (+27 pp), r = 128 from 33.5 % to 58.0 % (nu = 1) and 60.8 %
  (nu = 3). At r = 128 the higher key dimension (768 against 256) adds 2.8 pp, about 1.3 standard errors of the difference, so the two are not distinguished.
  Against the 0.80 gate the best arm is 19 pp short.
- Curves: D32-3 and D128-1 flatten (D32-3 +1.4 pp over the last interval, D128-1 -0.8 pp: 58.8 % at the fourth checkpoint, 58.0 % at the
  fifth); D128-3 gained 0.6 pp in the last interval after +7.9 pp before it. None is still rising fast, so the fixed budget is not
  what separates these arms from 0.80, but this study has no longer run to test that.
- The read distortion is lower but not removed: normalized read error 0.39 / 0.43 / 0.36 against 0.67 (R-r32) and 0.63 (R-r128);
  the read-value head accuracy rises from 8.8 / 19.7 % to 30.2 / 46.0 / 51.0 % while the clean-value head stays at 83 to 93 % (R: 85 / 93
  %), so the clean/read gap narrows (e.g. 93 to 20 for R-r128, 91 to 51 for D128-3) without closing. Value SIR is 5.5 to 8.0 against 0.5.
- Key overlap in the original space is **not** lower: Gram RMS 0.37 / 0.37 / 0.32 and mean max |cos| 0.91 / 0.87 / 0.83 for the DPFP arms
  (the `*_base` columns) against 0.33 / 0.29 and 0.87 / 0.77 for R. The phi-space numbers (RMS 0.20 / 0.18 / 0.19) are not comparable
  with them: phi features are nonnegative, so phi-space cosines have a positive floor (see `docs/reports/WP-L5-state.md`). What changes
  is the addressing geometry and the read, not a decorrelation of the learned projections K.
- Normalization confound (Amendment A1): DPFP also makes the query unit-norm (query norm 1.0, against 16.5 and 131.6 in R). RFX
  measured the unit-query effect separately on the frozen WP-L4 checkpoints (`docs/reports/RFX.md`, evaluation only, not a
  training ablation): on the delta store it changes the error by about -0.005 and the EM by at most +0.2 pp, with the fitted gain. The
  EM gains above therefore cannot be explained by a substituted unit query on the old state; whether a unit query changes training
  is not tested here (the LSQ arm U, which would, did not run: RFX triggered no LSQ arm).
- RFX at the same time says why the old read failed: exact store plus true key reads the value exactly (84 % at r = 128), the deployed
  store with the true key stays near the checkpoint's own EM (write interference). DPFP lowers that interference (parallel interference
  and read error above) but not far enough here.

## The E = 8 retrain (registered rule, flagged)
The runner and release scripts took the audited lock through `--e8-decision reports/p1l-dpfp/e8-lock.json` (`dpfp_release.eligible_e8`
accepts it only for `retry_e8` on that arm). Sequence run: E8 smoke 3721493 (2,400 steps; 0.180 s per step), release
`reports/p1l-dpfp/release-e8.json` (E 8, D128-3, 26065 steps, evaluations at 5213, 10426, 15639, 20852, 26065), reload check 3721520 (config
equal, reload identical, 0 mismatches), science job 3721535 (2 h cap with `--scale-up docs/prereg/DPFP.md`, 78 min used).
D128-3 at E = 8: val EM 56.9, 64.1, 70.7, 72.4, 73.0 % at the five checkpoints (still +0.6 pp over the last interval), normalized read error
0.269, read-value head 62.9 % (clean-value head 89.1 %), value SIR 28.5, against 60.8 %, 0.363, 51.0 % and 8.0 at E = 16 for the same arm.
Halving the number of facts per episode therefore raises EM by 12.2 pp and lowers the read error further, but ends 7.0 pp below the gate;
under the registration that retrain is the last attempt of this study.

## Memory interventions at the final checkpoints (registered descriptive diagnostic; jobs 3721521 and 3721773)
`scripts/audit_dpfp.py` (no training; tiny-model test `tests/test_audit_dpfp.py` passed in job 3721495) reloads each final export and scores the same
1,000 validation episodes (eval seed 4242) with the deployed memory, a zeroed memory and the memory of another episode (a one-step roll of a
seeded permutation; no episode keeps its own). Sources: `reports/p1l-dpfp/audit-e16.json` and `audit-e8.json`. The original condition reproduces the record's final
validation exactly for every arm (same correct count and NLL), repeated forwards are bit-identical, and the question input is unchanged
under both interventions (asserted).

| arm | original EM % (NLL) | zero memory EM % (NLL) | shuffled memory EM % (NLL) |
| --- | ---: | ---: | ---: |
| D32-3 | 39.0 (3.501) | 0.0 (15.087) | 0.0 (12.746) |
| D128-1 | 58.0 (2.277) | 0.0 (12.736) | 0.2 (13.635) |
| D128-3 | 60.8 (2.129) | 0.0 (17.440) | 0.2 (13.336) |
| D128-3 at E = 8 | 73.0 (1.316) | 0.1 (11.612) | 0.2 (14.502) |

Reading (descriptive): every DPFP arm's answers depend entirely on the episode's own memory: with zero or another episode's memory the EM is at
most 0.2 % (2 of 1,000) and the NLL is 11.6 to 17.4 against 1.3 to 3.5 with the deployed memory, so the writer's memory is used and is
episode specific. This says the gains in the table are carried by the memory; it does not say the
read is accurate (read error 0.36 to 0.43).

## Not done
No test or confirmation data was read. No P1L-D registration (no writer selected). Nothing remains to run for this study.

## Jobs and cost (node·h = wall hours of a one-node job)
| job | purpose | wall | node·h | outcome |
| --- | --- | --- | ---: | --- |
| 3720636 `t-dpfp` | tests (b79e0f8) | 6m19s | 0.105 | 3 failures in `tests/test_llm_dpfp.py`: fixture rank 4 gave a zero read (see state doc) |
| 3720637 `t-dpfp-repro` | DPFP-off reproduction of WP-L4 R, ranks 32 and 128 | 2m13s | 0.037 | passed |
| 3720748, 3720796, 3720880 | debug of the fixture (3720748 could not see its script) | 7 s, 25 s, 27 s | 0.015 | read at rank 4 exactly 0, nonzero at 8 and 16 |
| 3720890 `t-dpfp2` | `tests/test_llm_dpfp.py` at efadc36 | 40 s | 0.011 | 16 passed |
| 3720940, 3720941, 3720942 | 10% smokes, E = 16 (2,400 steps) | 8m54s, 8m40s, 9m09s | 0.446 | complete; timings in the release |
| 3721044, 3721045, 3721046 | reload checks, E = 16 | 23 s each | 0.019 | config equal, reload identical, 0 mismatches |
| 3721087, 3721088, 3721089 | the three E = 16 arms | 80.0, 79.7 and 81.5 min | 4.02 | complete |
| 3721495 `t-audit-dpfp` | test of `scripts/audit_dpfp.py` (tiny model) | 24 s | 0.007 | 2 passed |
| 3721493 `s-dpfp-e8-D128-3` | 10% smoke, E = 8 | 7m56s | 0.132 | complete |
| 3721520 `c-dpfp-e8-D128-3` | reload check, E = 8 | 24 s | 0.007 | passed |
| 3721521 `audit-dpfp-e16` | zero/shuffled-memory audit of the three E = 16 arms | 78 s | 0.022 | original reproduces the records |
| 3721535 `r-dpfp-e8-D128-3` | science run, E = 8 retrain | 78.5 min | 1.309 | complete, 730 of 1,000 |
| 3721773 `audit-dpfp-e8` | audit of the E = 8 arm | 39 s | 0.011 | original reproduces the record |

Total DPFP node·h about 6.14 (the RFX jobs, 0.10, are in `docs/reports/RFX.md`).

**Recorded overrun (deviation from the registered cap).** `docs/prereg/DPFP.md` caps tests plus smokes at 0.5 node·h. They used about 0.78: tests,
reproduction, failed run, rerun and debug 0.17 (the failed test run and the debug jobs are 0.13 of it), the E = 16 smokes 0.45, the E = 8 smoke
0.13, the reload checks 0.03 and the audit test 0.01. Causes: the failed test run (a fixture defect, not a product defect) and its debug and
rerun jobs, and the E = 8 smoke, which the registration's retrain rule requires but the cap did not budget. The three E = 16 arms (4.02) and
the E = 8 science run (1.31) are science jobs under the 2 h `--scale-up` and are not part of that cap. No other registered rule or limit was
exceeded.

## Tables
### Decision table (final checkpoint, 1,000 validation episodes, eval seed 4242)

| arm | rank | nu | key dim | steps | correct of 1,000 | val EM % | >= 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D32-3 | 32 | 3 | 192 | 23015 | 390 | 39.0 | no |
| D128-1 | 128 | 1 | 256 | 23690 | 580 | 58.0 | no |
| D128-3 | 128 | 3 | 768 | 22370 | 608 | 60.8 | no |
| R-r32 (WP-L4, not rerun) | 32 |  |  | 23590 | 120 | 12.0 |  |
| R-r128 (WP-L4, not rerun) | 128 |  |  | 24645 | 335 | 33.5 |  |

Registered decision: **retry D128-3 at E = 8 (flagged)** (best D128-3 with 608 of 1,000; thresholds 800 and 500).

E = 8 retrain of D128-3 (job 3721535.opbs, 26065 steps): 730 of 1,000 correct (73.0 %); final registered decision: **stop: no writer selected, negative result** (threshold 800).

### Validation EM % at the five evenly spaced checkpoints

| arm | checkpoint 1 | 2 | 3 | 4 | 5 (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| D32-3 | 4603: 18.2 | 9206: 30.1 | 13809: 36.5 | 18412: 37.6 | 23015: 39.0 |
| D128-1 | 4738: 35.0 | 9476: 45.6 | 14214: 55.0 | 18952: 58.8 | 23690: 58.0 |
| D128-3 | 4474: 36.0 | 8948: 50.2 | 13422: 52.3 | 17896: 60.2 | 22370: 60.8 |
| D128-3 at E = 8 | 5213: 56.9 | 10426: 64.1 | 15639: 70.7 | 20852: 72.4 | 26065: 73.0 |
| R-r32 (WP-L4) | 4718: 5.9 | 9436: 9.4 | 14154: 10.1 | 18872: 12.9 | 23590: 12.0 |
| R-r128 (WP-L4) | 4929: 16.3 | 9858: 22.4 | 14787: 27.6 | 19716: 31.0 | 24645: 33.5 |

### Diagnostics at the final checkpoint (WP-L4 audit columns; one seed)

| metric | D32-3 | D128-1 | D128-3 | D128-3 E8 | R-r32 | R-r128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val EM % | 39.0 | 58.0 | 60.8 | 73.0 | 12.0 | 33.5 |
| NLL | 3.501 | 2.277 | 2.129 | 1.316 | 5.045 | 3.531 |
| normalized read error | 0.391 | 0.431 | 0.363 | 0.269 | 0.666 | 0.631 |
| clean-value head acc % | 83.3 | 92.8 | 90.6 | 89.1 | 84.5 | 93.3 |
| read-value head acc % | 30.2 | 46.0 | 51.0 | 62.9 | 8.8 | 19.7 |
| value SIR | 5.47 | 6.88 | 8.00 | 28.53 | 0.48 | 0.55 |
| key top-1 % | 75.8 | 81.1 | 84.9 | 89.8 | 83.2 | 88.5 |
| key margin | 0.223 | 0.215 | 0.252 | 0.333 | 0.158 | 0.177 |
| key Gram RMS (phi space for DPFP) | 0.202 | 0.178 | 0.190 | 0.180 | 0.332 | 0.288 |
| key Gram RMS (original space) | 0.374 | 0.366 | 0.321 | 0.313 |  |  |
| mean max |cos| (phi space for DPFP) | 0.834 | 0.733 | 0.692 | 0.541 | 0.871 | 0.767 |
| mean max |cos| (original space) | 0.907 | 0.870 | 0.829 | 0.710 |  |  |
| mean pairwise cosine (as logged) | 0.117 | 0.108 | 0.140 | 0.127 | 0.136 | 0.130 |
| parallel interference | 0.573 | 0.413 | 0.270 | 0.159 | 0.322 | 0.283 |
| query norm | 1.0 | 1.0 | 1.0 | 1.0 | 16.5 | 131.6 |

### Read error and clean/read head accuracy over training (normalized read error / read-value head %)

| arm | checkpoint 1 | 2 | 3 | 4 | 5 (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| D32-3 | 4603: 0.251 / 20.9 | 9206: 0.410 / 23.2 | 13809: 0.371 / 27.6 | 18412: 0.382 / 28.4 | 23015: 0.391 / 30.2 |
| D128-1 | 4738: 0.443 / 28.0 | 9476: 0.416 / 35.4 | 14214: 0.428 / 41.6 | 18952: 0.433 / 44.2 | 23690: 0.431 / 46.0 |
| D128-3 | 4474: 0.444 / 34.8 | 8948: 0.413 / 42.5 | 13422: 0.423 / 43.6 | 17896: 0.362 / 51.0 | 22370: 0.363 / 51.0 |
| D128-3 at E = 8 | 5213: 0.316 / 50.9 | 10426: 0.290 / 55.2 | 15639: 0.274 / 60.6 | 20852: 0.273 / 65.2 | 26065: 0.269 / 62.9 |
| R-r32 (WP-L4) | 4718: 0.677 / 4.8 | 9436: 0.675 / 5.9 | 14154: 0.687 / 5.6 | 18872: 0.695 / 6.9 | 23590: 0.666 / 8.8 |
| R-r128 (WP-L4) | 4929: 0.677 / 10.9 | 9858: 0.674 / 12.6 | 14787: 0.630 / 16.5 | 19716: 0.623 / 18.8 | 24645: 0.631 / 19.7 |

### Jobs, time and memory

| arm | job | source | steps | training min | steady step s | slowest 1,000-episode eval s | peak CUDA GiB | state bytes per episode | inference parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D32-3 | 3721087.opbs | efadc364 | 23015 | 79.6 | 0.205 | 7.6 | 2.92 | 24576 | 103426 |
| D128-1 | 3721088.opbs | efadc364 | 23690 | 79.3 | 0.199 | 7.5 | 3.26 | 131072 | 475138 |
| D128-3 | 3721089.opbs | efadc364 | 22370 | 81.1 | 0.215 | 7.5 | 4.08 | 393216 | 475138 |
| D128-3 (E8) | 3721535.opbs | 72ed4691 | 26065 | 78.1 | 0.178 | 7.0 | 3.87 | 393216 | 475138 |
