# DPFP report: DPFP keys and queries for the learned LLM writer (WP-L5)

Registered design: `docs/prereg/DPFP.md` (2026-10-27 07:22 JST, Amendment A1 08:51 JST). Run 2026-10-29 JST by sn-llm at
`efadc36436e4d29514a4ea570596925b779b8b2d` (wp/llm). Arms: D32-3 (r = 32, nu = 3), D128-1 (r = 128, nu = 1), D128-3 (r = 128,
nu = 3); T1 as WP-L4 (E = 16, seed 17, layer 14, L = 1, batch 32 / Q 4, lr 3e-3), one 1-node 2 h job per arm
(`--scale-up docs/prereg/DPFP.md`), step counts from the 10% smokes by the registered rule (release `reports/p1l-dpfp/release.json`,
1dd1869). Records are copied to `reports/p1l-dpfp/arms/` and, for the two WP-L4 R references, `reports/p1l-dpfp/refs/`; every table
below is printed by `python3 scripts/analyze_dpfp.py reports/p1l-dpfp` from them.

## Result and registered decision
**No arm reaches val EM 0.80, so no writer is selected at E = 16. The best arm, D128-3, has 608 of 1,000 correct (60.8 %), which is
at least 0.50, so the registered rule triggers one retrain of D128-3 at E = 8 with the same schedule rule, accepted only if its val EM
is at least 0.80 (flagged).** The retrain is not submitted: it needs the audited lock (below) and is the orchestrator's decision.

| arm | final val EM | candidate order | 0.80 |
| --- | ---: | --- | --- |
| D32-3 | 39.0 % | 1 | no |
| D128-1 | 58.0 % | 2 | no |
| D128-3 | 60.8 % | 3 | no |

Every arm completed its released steps (`status` complete, not a smoke, source clean, `actual_steps` = released steps, E = 16; the
logs have no error and exit 0). The WP-L4 references (R at r = 32: 12.0 %, at r = 128: 33.5 %) are used as registered, not rerun,
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

## What the E = 8 retrain needs (STOP: orchestrator)
The runner and release scripts take the audited lock through `--e8-decision LOCK.json` and accept it only when
`lock['choice']` is `retry_e8` for that arm (`dpfp_release.eligible_e8`). The registered choice computed from the three records is
`{"action": "retry_e8", "arm": "D128-3", "E": 8, "correct": {"D32-3": 390, "D128-1": 580, "D128-3": 608}}`. Sequence once it
is locked (state doc, "Commands for step 2"): (1) the arm's own E8 smoke, `dpfp_run --arm D128-3 --steps 2400 --smoke --e8-decision LOCK
--out RUNS/smoke-e8`; (2) `prepare_dpfp_release.py --root RUNS/smoke-e8 --out reports/p1l-dpfp/release-e8.json --e8-decision LOCK`
(E 8, one arm); (3) `check_smoke_reload.py` for that arm; (4) the science job, 2 h with `--scale-up`, `--schedule release-e8.json --out RUNS/arms-e8`.
Estimated 0.15 + 0.01 + at most 2.0 node·h (the E = 16 runs took 1.33 node·h each; the E8 step count comes from its smoke).

## Not done
The registered zero/shuffle-memory diagnostics (EM/NLL with zeroed and shuffled memory at the final checkpoints) need a no-training
audit of the exported writers (`controlled_score` already supports the rectangular state; the audit script, a counterpart of
`scripts/audit_p1l_read_fix.py`, is not written). The tables use the diagnostics the training run itself logged at each
validation (WP-L4's audit columns). No test or confirmation data was read.

## Jobs and cost (node·h = wall hours of a one-node job)
| job | purpose | wall | node·h | outcome |
| --- | --- | --- | ---: | --- |
| 3720636 `t-dpfp` | tests (b79e0f8) | 6m19s | 0.105 | 3 failures in `tests/test_llm_dpfp.py`: fixture rank 4 gave a zero read (see state doc) |
| 3720637 `t-dpfp-repro` | DPFP-off reproduction of WP-L4 R, ranks 32 and 128 | 2m13s | 0.037 | passed |
| 3720796, 3720880 | debug of the fixture | about 25 s, 27 s | 0.01 | read at rank 4 exactly 0, nonzero at 8 and 16 |
| 3720890 `t-dpfp2` | `tests/test_llm_dpfp.py` at efadc36 | 40 s | 0.011 | 16 passed |
| 3720940, 3720941, 3720942 | 10% smokes (2,400 steps) | 8m54s, 8m40s, 9m09s | 0.446 | complete; timings in the release |
| 3721044, 3721045, 3721046 | reload checks (`scripts/check_smoke_reload.py`) | 23 s each | 0.02 | config equal, reload identical, 0 prediction mismatches |
| 3721087, 3721088, 3721089 | the three arms | 80.0, 79.7 and 81.5 min | 4.02 | complete |

Total about 4.65 node·h. Tests, reproduction and debug cost 0.17, the smokes 0.45 and the reload checks 0.02, together about 0.63
node·h against the registered cap of 0.5 for tests plus smokes (the failed test run and the debug jobs are 0.13 of it). Flag for the
orchestrator.

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

### Validation EM % at the five evenly spaced checkpoints

| arm | checkpoint 1 | 2 | 3 | 4 | 5 (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| D32-3 | 4603: 18.2 | 9206: 30.1 | 13809: 36.5 | 18412: 37.6 | 23015: 39.0 |
| D128-1 | 4738: 35.0 | 9476: 45.6 | 14214: 55.0 | 18952: 58.8 | 23690: 58.0 |
| D128-3 | 4474: 36.0 | 8948: 50.2 | 13422: 52.3 | 17896: 60.2 | 22370: 60.8 |
| R-r32 (WP-L4) | 4718: 5.9 | 9436: 9.4 | 14154: 10.1 | 18872: 12.9 | 23590: 12.0 |
| R-r128 (WP-L4) | 4929: 16.3 | 9858: 22.4 | 14787: 27.6 | 19716: 31.0 | 24645: 33.5 |

### Diagnostics at the final checkpoint (WP-L4 audit columns; one seed)

| metric | D32-3 | D128-1 | D128-3 | R-r32 | R-r128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| val EM % | 39.0 | 58.0 | 60.8 | 12.0 | 33.5 |
| NLL | 3.501 | 2.277 | 2.129 | 5.045 | 3.531 |
| normalized read error | 0.391 | 0.431 | 0.363 | 0.666 | 0.631 |
| clean-value head acc % | 83.3 | 92.8 | 90.6 | 84.5 | 93.3 |
| read-value head acc % | 30.2 | 46.0 | 51.0 | 8.8 | 19.7 |
| value SIR | 5.47 | 6.88 | 8.00 | 0.48 | 0.55 |
| key top-1 % | 75.8 | 81.1 | 84.9 | 83.2 | 88.5 |
| key margin | 0.223 | 0.215 | 0.252 | 0.158 | 0.177 |
| key Gram RMS (phi space for DPFP) | 0.202 | 0.178 | 0.190 | 0.332 | 0.288 |
| key Gram RMS (original space) | 0.374 | 0.366 | 0.321 |  |  |
| mean max |cos| (phi space for DPFP) | 0.834 | 0.733 | 0.692 | 0.871 | 0.767 |
| mean max |cos| (original space) | 0.907 | 0.870 | 0.829 |  |  |
| mean pairwise cosine (as logged) | 0.117 | 0.108 | 0.140 | 0.136 | 0.130 |
| parallel interference | 0.573 | 0.413 | 0.270 | 0.322 | 0.283 |
| query norm | 1.0 | 1.0 | 1.0 | 16.5 | 131.6 |

### Read error and clean/read head accuracy over training (normalized read error / read-value head %)

| arm | checkpoint 1 | 2 | 3 | 4 | 5 (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| D32-3 | 4603: 0.251 / 20.9 | 9206: 0.410 / 23.2 | 13809: 0.371 / 27.6 | 18412: 0.382 / 28.4 | 23015: 0.391 / 30.2 |
| D128-1 | 4738: 0.443 / 28.0 | 9476: 0.416 / 35.4 | 14214: 0.428 / 41.6 | 18952: 0.433 / 44.2 | 23690: 0.431 / 46.0 |
| D128-3 | 4474: 0.444 / 34.8 | 8948: 0.413 / 42.5 | 13422: 0.423 / 43.6 | 17896: 0.362 / 51.0 | 22370: 0.363 / 51.0 |
| R-r32 (WP-L4) | 4718: 0.677 / 4.8 | 9436: 0.675 / 5.9 | 14154: 0.687 / 5.6 | 18872: 0.695 / 6.9 | 23590: 0.666 / 8.8 |
| R-r128 (WP-L4) | 4929: 0.677 / 10.9 | 9858: 0.674 / 12.6 | 14787: 0.630 / 16.5 | 19716: 0.623 / 18.8 | 24645: 0.631 / 19.7 |

### Jobs, time and memory

| arm | job | source | steps | training min | steady step s | slowest 1,000-episode eval s | peak CUDA GiB | state bytes per episode | inference parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D32-3 | 3721087.opbs | efadc364 | 23015 | 79.6 | 0.205 | 7.6 | 2.92 | 24576 | 103426 |
| D128-1 | 3721088.opbs | efadc364 | 23690 | 79.3 | 0.199 | 7.5 | 3.26 | 131072 | 475138 |
| D128-3 | 3721089.opbs | efadc364 | 22370 | 81.1 | 0.215 | 7.5 | 4.08 | 393216 | 475138 |
