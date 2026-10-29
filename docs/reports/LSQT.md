# LSQT report: trained least-squares stores at r = 128, from scratch (WP-L12, G5)

Registered design: `docs/prereg/LSQT.md` (registered 2026-10-29 13:13 JST, blake2b-128 `94ed7364da460aeec18cbd5da6b086ff`), run 2026-10-29 JST by sn-llm at science commit
`dd5e4c89b40a345289aa53647718da56dc3bbb12` (wp/llm), schedule release `reports/p1l-lsqt/release.json` (ec91b3a, approved by the orchestrator, `reports/p1l-lsqt/approval.json`).
Three arms, each WP-L4's R at r = r_mem = 128 with the unit query, trained from scratch (no checkpoint loaded; WP-L4's data stream `derive_seed(17, 'llm', 'writer-data')`),
E = 16, seed 17, layer 14, L = 1, batch 32 / Q 4, lr 3e-3, 1,000 validation episodes at eval seed 4242, evaluated at five evenly spaced checkpoints:

| arm | store | eta | steps (release) | job (1 node, 2 h, `--scale-up docs/prereg/LSQT.md`) |
| --- | --- | ---: | ---: | --- |
| LT-K4 | four Kaczmarz sweeps (beta = 1 delta scan, repeated) | none | 17,970 | 3724516 |
| LT-RG001 | ridge, float64 Cholesky, recency weights lam^(E-1-j) | 1e-2 | 28,005 | 3724517 |
| LT-RG1 | ridge | 1 | 28,245 | 3724518 |
| LT-U (conditional control, descriptive) | R's deployed decay-first delta store with the unit query | none | 24,560 | 3725694 |

The control LT-U ran after the three arms, as registered (decision 2 at registration): its own smoke, steps frozen by the same rule, at most 2 h, the same audit. It has its own release (`reports/p1l-lsqt-u/release.json`, 71a6f1c; orchestrator approval
`reports/p1l-lsqt-u/approval.json`, a0eaddd) at science commit `c07452bfb04af2e95f3abb07629d5654041b1842`, which differs from `dd5e4c8` in `src` only in the release gating `lsqt_release.scheduled_steps` (study `LSQT-U`); the training path is unchanged.
It is descriptive and enters neither outcome of the registration (section "U control").

Records, the audit and the decision are in `reports/p1l-lsqt/` (U: `reports/p1l-lsqt-u/`); every table below is printed by `python3 scripts/analyze_lsqt.py` (U: `python3 scripts/analyze_lsqt_u.py`) from them (R-r128 and D128-3 are read from `reports/p1l-dpfp/`;
`tests/test_analyze_lsqt.py` checks the rule and re-derives `decision.json`; `tests/test_analyze_lsqt_u.py` re-derives `reports/p1l-lsqt-u/split.json`). Validation only, one seed; no test or confirmation data was read.

## Result and registered decision
**No arm reaches val EM 0.80, so no writer candidate is flagged (outcome 1). The best ridge arm, LT-RG001, reaches 46.1 %: inside the 40 % to < 48.5 % band, which is neither fable's W3 trigger (>= 48.5 %) nor the falsifier
(< 40 %). The trained-LSQ line is neither closed nor triggered; the learned-writer line stays stopped.** The conditional U control was proposed because the best arm is above 40 %; it has run (descriptive) and reached 36.0 %, see "U control". It does not change this decision.

| arm | job | store | steps | correct of 1,000 | val EM % | normalized read error | clean-value head % | read-value head % | >= 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 3724516.opbs | sweep | 17970 | 383 | 38.3 | 0.554 | 92.3 | 27.4 | no |
| LT-RG001 | 3724517.opbs | ridge (eta 0.01) | 28005 | 461 | 46.1 | 0.512 | 93.9 | 26.5 | no |
| LT-RG1 | 3724518.opbs | ridge (eta 1) | 28245 | 426 | 42.6 | 0.499 | 95.9 | 22.3 | no |
| R-r128 (WP-L4, not rerun) |  | delta | 24645 | 335 | 33.5 | 0.631 | 93.3 | 19.7 | no |
| D128-3 (DPFP) |  | delta | 22370 | 608 | 60.8 | 0.363 | 90.6 | 51.0 | no |

Registered decision: **no arm reaches 0.80, no writer candidate**. Best RG arm LT-RG001 with 461 of 1,000 (46.1 %): band **open** (>= 48.5 % W3 trigger; 40 % to < 48.5 % neither; < 40 % the trained-LSQ line closes). Trained-LSQ line: **open**. Best arm LT-RG001 (46.1 %): the conditional U control is proposed (best arm >= 40 %; it has since run, see "U control").

Difference to the references in pp of val EM (descriptive; standard error of a difference about 2.2 pp):

| arm | minus R-r128 (33.5 %) | minus D128-3 (60.8 %) |
| --- | ---: | ---: |
| LT-K4 | +4.8 | -22.5 |
| LT-RG001 | +12.6 | -14.7 |
| LT-RG1 | +9.1 | -18.2 |

Each arm is judged at its own final checkpoint. R-r128 is WP-L4's record, not rerun; it is a valid reference because `tests/test_llm_dpfp_repro.py` (the DPFP-off, trained-from-scratch reproduction of R) passes at the science
commit (job 3725434 at `dd5e4c8`: 29 passed, 0 failed, together with `tests/test_lsqt.py`).

- **All three arms are above R-r128** (+4.8, +12.6 and +9.1 pp; the standard error of a difference of two EMs at 1,000 episodes is about 2.2 pp) **and well below D128-3** (-22.5, -14.7 and -18.2 pp).
- **Ridge against Kaczmarz:** RG001 is +7.8 pp and RG1 +4.3 pp above K4 at their final checkpoints (one seed; 3.5 and 2.0 standard errors of a difference). The arms have different step counts (K4 does four times the sequential write work and ran at 0.262 s per step
  against 0.174 and 0.177 s): at the checkpoints nearest K4's final step count (17,970), RG001 has 42.1 % (16,803) and RG1 35.7 % (16,947) against K4's 38.3 %.
- **The read error is not removed.** The final normalized read error is 0.554, 0.512 and 0.499 (R-r128 0.631, D128-3 0.363), and the read-value head is 22 % to 27 % against a clean-value head of 92 % to 96 %.
  The Gram reconstruction of the read is exact (relative error below 1e-6), so what the exact coefficients describe is the deployed read; the store does what it is built to do and the read is still far from the stored value.
- **Registered predictions (decide nothing): RG >= 50 % with read error <= 0.35: no** (46.1 %, 0.512). **K4 between R-r128 and RG: yes.** **Read error < 0.1 with the read-value head within 5 pp of the clean head: no.**
  The conditional "read error < 0.1 and EM < 0.50 means the fault is in the query path" does not apply: no arm has a read error below 0.1.

### Validation EM % at the five evenly spaced checkpoints (matched fractions of training)
| arm | 20 % | 40 % | 60 % | 80 % | 100 % (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 3594: 14.9 | 7188: 25.9 | 10782: 30.4 | 14376: 37.6 | 17970: 38.3 |
| LT-RG001 | 5601: 23.6 | 11202: 35.7 | 16803: 42.1 | 22404: 46.9 | 28005: 46.1 |
| LT-RG1 | 5649: 23.0 | 11298: 30.1 | 16947: 35.7 | 22596: 36.6 | 28245: 42.6 |
| R-r128 (WP-L4, not rerun) | 4929: 16.3 | 9858: 22.4 | 14787: 27.6 | 19716: 31.0 | 24645: 33.5 |
| D128-3 (DPFP) | 4474: 36.0 | 8948: 50.2 | 13422: 52.3 | 17896: 60.2 | 22370: 60.8 |

The curves are still rising at the end for K4 (+0.7 pp over the last interval) and RG1 (+6.0 pp, from 36.6 % to 42.6 %), and RG001 peaks at 80 % (46.9 %) and ends at 46.1 %. The decision uses the final checkpoint; for RG001 either value
lies in the same band. No number is extrapolated.

### Normalized read error
| arm | 20 % | 40 % | 60 % | 80 % | 100 % (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 0.635 | 0.617 | 0.597 | 0.572 | 0.554 |
| LT-RG001 | 0.619 | 0.604 | 0.418 | 0.350 | 0.512 |
| LT-RG1 | 0.366 | 0.398 | 0.507 | 0.519 | 0.499 |
| R-r128 (WP-L4, not rerun) | 0.677 | 0.674 | 0.630 | 0.623 | 0.631 |
| D128-3 (DPFP) | 0.444 | 0.413 | 0.423 | 0.362 | 0.363 |

The read error does not follow the EM. RG1 has its lowest read error (0.366) at 20 % of training, when its EM is 23.0 %, and the error rises to 0.499 while the EM rises to 42.6 %; RG001's read error falls to 0.350 at 80 % and returns to 0.512 at 100 % while its
EM moves from 46.9 % to 46.1 %. One seed: the checkpoint-to-checkpoint movement is not separated from run noise here.

### Read-value head and clean-value head accuracy %
| arm | 20 % | 40 % | 60 % | 80 % | 100 % (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 16.5 / 83.7 | 17.7 / 89.6 | 22.6 / 91.5 | 25.7 / 92.7 | 27.4 / 92.3 |
| LT-RG001 | 19.0 / 85.9 | 17.6 / 91.1 | 25.2 / 92.8 | 30.6 / 92.5 | 26.5 / 93.9 |
| LT-RG1 | 10.8 / 87.1 | 17.2 / 91.3 | 24.8 / 93.4 | 23.5 / 94.4 | 22.3 / 95.9 |

### Diagnostics at the final checkpoint
| metric | LT-K4 | LT-RG001 | LT-RG1 | D128-3 |
| --- | ---: | ---: | ---: | ---: |
| val EM % | 38.3 | 46.1 | 42.6 | 60.8 |
| NLL | 3.187 | 2.791 | 2.907 | 2.129 |
| normalized read error | 0.554 | 0.512 | 0.499 | 0.363 |
| clean-value head acc % | 92.3 | 93.9 | 95.9 | 90.6 |
| read-value head acc % | 27.4 | 26.5 | 22.3 | 51.0 |
| value SIR | 1.05 | 1.55 | 0.87 | 8.00 |
| key top-1 % | 86.7 | 84.7 | 85.0 | 84.9 |
| key margin | 0.146 | 0.126 | 0.156 | 0.252 |
| key Gram RMS (phi space for DPFP) | 0.276 | 0.336 | 0.307 | 0.190 |
| key Gram RMS (original space) |  |  |  | 0.321 |
| mean max |cos| (phi space for DPFP) | 0.704 | 0.788 | 0.793 | 0.692 |
| mean max |cos| (original space) |  |  |  | 0.829 |
| mean pairwise cosine (as logged) | 0.176 | 0.245 | 0.170 | 0.140 |
| parallel interference | 0.140 | 0.224 | 0.742 | 0.270 |
| query norm | 1.0 | 1.0 | 1.0 | 1.0 |

| metric | LT-K4 | LT-RG001 | LT-RG1 |
| --- | ---: | ---: | ---: |
| Gram reconstruction relative error | 0.000000 | 0.000000 | 0.000000 |
| log10 cond of the key Gram | 1.58 | 1.87 | 2.42 |
| log10 cond of the ridge system |  | 2.03 | 0.70 |

The key Gram condition number (log10, median over the validation queries) is 1.6 (K4), 1.9 (RG001) and 2.4 (RG1); the ridge system's is 2.0 at eta = 1e-2 and 0.7 at eta = 1 (RFX's median for the learned keys of R-r128 was 2.4).
RG1 has the highest parallel interference (0.742), K4 the lowest (0.140); value SIR is 0.9 to 1.6 in all three against 8.0 for D128-3.

### Training-log diagnostics
| metric | LT-K4 | LT-RG001 | LT-RG1 |
| --- | ---: | ---: | ---: |
| raw-read loss | 0.560 | 0.445 | 0.489 |
| main loss | 2.941 | 2.402 | 2.811 |
| log10 cond of the key Gram (median) | 1.582 | 1.823 | 2.425 |
| log10 cond of the ridge system (median) |  | 1.995 | 0.698 |
| gradient norm before clipping | 6.881 | 7.020 | 7.744 |

Fraction of logged steps with the gradient clipped: LT-K4 100 %, LT-RG001 100 %, LT-RG1 100 %.

The global gradient-norm clip (1.0) is active at every logged step in all three arms, with a pre-clip norm of about 7 (mean of the last ten logged steps 6.9, 7.0, 7.7). The same holds for D128-3 and the four SQ2 arms (see `docs/reports/SQ2.md`); it is a property of the setting, and the gradients of the individual loss terms were not logged.

## Registered predictions
| prediction | observed | held |
| --- | ---: | ---: |
| RG >= 50 % with read error <= 0.35 | best RG LT-RG001 46.1 %, read error 0.512 | no |
| K4 between R-r128 and RG | K4 38.3 %, R-r128 33.5 %, RG 46.1 / 42.6 % | yes |
| read error < 0.1 and read-value head within 5 pp of the clean head (at r = 128) | read errors 0.554 / 0.512 / 0.499; read-value minus clean-value head -64.9 pp / -67.4 pp / -73.6 pp | no |
| if read error < 0.1 and EM < 0.50, the fault is in the query path | not applicable: no arm has read error below 0.1 | n/a |

## U control: the unit query in training, with R's delta store (conditional, descriptive; job 3725694)
LT-U is WP-L4's R at r = 128 trained from scratch with the unit query and R's own deployed decay-first delta store. A comparison of the recorded configurations of LT-U and R-r128 shows one model or training difference besides the step count (24,560 against 24,645): `model.unit_query = true`
(the `lsqt` block, the name and the step count are the others). Same data stream as WP-L4 and the three arms (`derive_seed(17, 'llm', 'writer-data')`, data seed 7003109974603132079 in R-r128's record and in all four LSQT records), seed 17, E = 16, layer 14, L = 1, batch 32 / Q 4, lr 3e-3, no checkpoint loaded.
Its registered role (decision 2 at registration): descriptive only, a split of a gain over R-r128 between the store and the unit query; it enters neither outcome 1 nor outcome 2, and a result of 0.80 or more would have been reported for the orchestrator to decide (it did not reach it).

Process, by the same release rule as the arms: smoke of 2,400 steps (job 3725519: 0.191 s per step, 100-episode evaluations 0.74 to 0.92 s, peak CUDA 3.05 GiB allocated, 3.29 GiB reserved); steps frozen at 24,560 (the largest multiple of 5 that fits 7,200 s with margin 1.5, a 90 s reserve and five 1,000-episode evaluations);
reload check at the science commit (job 3725676: configs equal, reload identical, 0 prediction mismatches, quality scores not exposed); the orchestrator's approval committed (a0eaddd) before the science job; one job (regular-g, 02:00:00, `--scale-up docs/prereg/LSQT.md`, `--schedule` the release): 74.3 min of training, 4,487 s of wall time.

### Record check
Verified against `release.json` and `approval.json` (by `scripts/analyze_lsqt_u.py` on the committed copies, which are byte-identical to the runs-directory files):
- 24,560 steps, evaluations at 4,912 / 9,824 / 14,736 / 19,648 / 24,560, exactly the released marks, 1,000 validation episodes each; status complete, not a smoke, source not dirty.
- Config hash `a88fe73a30cec96112ee21f7fa45aa9c` and commit `c07452bfb04af2e95f3abb07629d5654041b1842` equal the release's; the prereg digest in the approval is `94ed7364da460aeec18cbd5da6b086ff`, the digest of `docs/prereg/LSQT.md`.
- 6,834 numbers in the record, the five metric files and 493 train-log rows are finite; no NaN or infinity token anywhere (a strict parse rejects both, and an overflowing number is caught); every logged loss and gradient norm is finite.

### Result (final checkpoint, 1,000 validation episodes, eval seed 4242)
| arm | job | store | steps | correct of 1,000 | val EM % | standard error pp | normalized read error | clean-value head % | read-value head % | >= 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LT-U | 3725694.opbs | delta (unit query) | 24560 | 360 | 36.0 | 1.5 | 0.589 | 93.7 | 21.6 | no |
| LT-K4 | 3724516.opbs | sweep | 17970 | 383 | 38.3 | 1.5 | 0.554 | 92.3 | 27.4 | no |
| LT-RG001 | 3724517.opbs | ridge (eta 0.01) | 28005 | 461 | 46.1 | 1.6 | 0.512 | 93.9 | 26.5 | no |
| LT-RG1 | 3724518.opbs | ridge (eta 1) | 28245 | 426 | 42.6 | 1.6 | 0.499 | 95.9 | 22.3 | no |
| R-r128 (WP-L4, not rerun) |  | delta | 24645 | 335 | 33.5 | 1.5 | 0.631 | 93.3 | 19.7 | no |
| D128-3 (DPFP) |  | delta | 22370 | 608 | 60.8 | 1.5 | 0.363 | 90.6 | 51.0 | no |

**U reaches 360 of 1,000 (36.0 %), 2.5 pp above R-r128 and 10.1 pp below the best ridge arm; it does not reach 0.80.**

**Selected checkpoint.** U is judged at its own final checkpoint (step 24,560), the registered rule for every arm (each arm at its own final checkpoint); no checkpoint was selected among the five.
Its five evaluations are 15.9, 24.5, 30.3, 32.8 and 36.0 %, so the final checkpoint is also its highest and a best-checkpoint reading would give the same number. The same rule is applied to the other terms of the split: LT-RG001 at its final 46.1 % (its 80 % checkpoint, 46.9 %, is not used) and R-r128 at its final 33.5 %.

### Split of RG001's gain over R-r128 (descriptive; one seed; not a registered test)
Differences of val EM at the final checkpoints, in pp, with the unpaired binomial standard error of the difference, sqrt(p1 (1 - p1) / 1000 + p2 (1 - p2) / 1000), in brackets:

| arm | total: arm - R-r128 | unit-query part: U - R-r128 | store part: arm - U |
| --- | ---: | ---: | ---: |
| LT-K4 | +4.8 (2.1) | +2.5 (2.1) | +2.3 (2.2) |
| LT-RG001 | +12.6 (2.2) | +2.5 (2.1) | +10.1 (2.2) |
| LT-RG1 | +9.1 (2.2) | +2.5 (2.1) | +6.6 (2.2) |

RG001's +12.6 pp over R-r128 is, in point estimates, **+2.5 pp for the unit query (U - R-r128, standard error 2.1)** and **+10.1 pp for the store (RG001 - U, standard error 2.2)**; the parts add up to the total exactly (360 - 335 = 25 and 461 - 360 = 101 episodes).
The store part is about 4.6 of its standard errors, the unit-query part about 1.2 of its own (a size reference only). The two parts share the sampling error of U, so their errors are not independent. This is one seed per arm and a descriptive split: it was not registered as a test, and the standard errors
cover only the sampling of the 1,000 validation episodes, not training-seed variation, which was not measured. The arms were scored on the same 1,000 episodes; the unpaired figure does not use that pairing and a paired standard error was not computed.

Differences at the matched fractions of training (pp of val EM; each cell is a difference of two single evaluations of one seed):

| contrast | 20 % | 40 % | 60 % | 80 % | 100 % (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| U - R-r128 | -0.4 | +2.1 | +2.7 | +1.8 | +2.5 |
| LT-K4 - U | -1.0 | +1.4 | +0.1 | +4.8 | +2.3 |
| LT-RG001 - U | +7.7 | +11.2 | +11.8 | +14.1 | +10.1 |
| LT-RG1 - U | +7.1 | +5.6 | +5.4 | +3.8 | +6.6 |

- **RG001 is above U at all five fractions (+7.7 to +14.1 pp), and at checkpoints with fewer steps than U's final 24,560:** RG001 has 35.7 % at 11,202 steps and 46.9 % at 22,404 steps against U's 36.0 % at 24,560. RG001's 14 % extra steps (28,005 against 24,560) do not account for its lead.
- **RG1 is above U at all five fractions (+3.8 to +7.1 pp), but its final lead rests on the last interval:** RG1 has 36.6 % at 22,596 steps (U's final is 36.0 % at 24,560) and 42.6 % at 28,245; its extra 15 % of steps (28,245) are not separated from the store.
- **K4 is within 1.4 pp of U at 20, 40 and 60 %, and +4.8 and +2.3 pp at 80 % and 100 %;** K4 ran 27 % fewer steps (17,970) than U.
- **U against R-r128:** -0.4 pp at 20 % and +1.8 to +2.7 pp at the other four fractions. RFX measured the unit query only on frozen delta stores (EM at most +0.2 pp, evaluation only); the trained contrast here is +2.5 pp (standard error 2.1), which one seed does not separate from zero or from a few pp.

### Validation EM % at the five evenly spaced checkpoints
| arm | 20 % | 40 % | 60 % | 80 % | 100 % (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LT-U | 4912: 15.9 | 9824: 24.5 | 14736: 30.3 | 19648: 32.8 | 24560: 36.0 |
| LT-K4 | 3594: 14.9 | 7188: 25.9 | 10782: 30.4 | 14376: 37.6 | 17970: 38.3 |
| LT-RG001 | 5601: 23.6 | 11202: 35.7 | 16803: 42.1 | 22404: 46.9 | 28005: 46.1 |
| LT-RG1 | 5649: 23.0 | 11298: 30.1 | 16947: 35.7 | 22596: 36.6 | 28245: 42.6 |
| R-r128 (WP-L4, not rerun) | 4929: 16.3 | 9858: 22.4 | 14787: 27.6 | 19716: 31.0 | 24645: 33.5 |
| D128-3 (DPFP) | 4474: 36.0 | 8948: 50.2 | 13422: 52.3 | 17896: 60.2 | 22370: 60.8 |

U is still rising at its last interval (+3.2 pp from 19,648 to 24,560 steps); no number is extrapolated.

### Read path and diagnostics at the final checkpoint (one seed)
| metric | LT-U | LT-RG001 | LT-K4 | LT-RG1 | R-r128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| val EM % | 36.0 | 46.1 | 38.3 | 42.6 | 33.5 |
| NLL | 3.382 | 2.791 | 3.187 | 2.907 | 3.531 |
| normalized read error | 0.589 | 0.512 | 0.554 | 0.499 | 0.631 |
| clean-value head acc % | 93.7 | 93.9 | 92.3 | 95.9 | 93.3 |
| read-value head acc % | 21.6 | 26.5 | 27.4 | 22.3 | 19.7 |
| value SIR | 0.73 | 1.55 | 1.05 | 0.87 | 0.55 |
| key top-1 % | 83.8 | 84.7 | 86.7 | 85.0 | 88.5 |
| key margin | 0.157 | 0.126 | 0.146 | 0.156 | 0.177 |
| key Gram RMS | 0.286 | 0.336 | 0.276 | 0.307 | 0.288 |
| mean max cos | 0.738 | 0.788 | 0.704 | 0.793 | 0.767 |
| parallel interference | 0.398 | 0.224 | 0.140 | 0.742 | 0.283 |
| query norm | 1.0 | 1.0 | 1.0 | 1.0 | 131.6 |
| log10 cond of the key Gram | 2.16 | 1.87 | 1.58 | 2.42 | |
| Gram reconstruction relative error | 7.4e-08 | 5.7e-08 | 1.1e-07 | 4.4e-08 | |

- The normalized read error is 0.589 for U, 0.042 below R-r128's 0.631 (U's five values: 0.615, 0.593, 0.596, 0.596, 0.589; R-r128's: 0.677, 0.674, 0.630, 0.623, 0.631), and 0.077 above RG001's 0.512 and 0.090 above RG1's 0.499. The read-value head is 21.6 % for U (R-r128 19.7 %, RG001 26.5 %) against a clean-value head of 93.7 %.
- The unit query fixes the query norm at 1 (R-r128's learned query has norm 131.6); the key Gram RMS and the mean max |cos| of U are within 0.03 of R-r128's, and the key top-1 is 4.7 pp lower (83.8 against 88.5 %); the parallel interference is 0.398 against 0.283. One seed each: these are not separated from run noise here.
- The Gram reconstruction of the read is exact for U as for the other arms (relative error 7.4e-08; the registered check is below 1e-4).
- Training log (mean of the last ten logged steps): raw-read loss 0.570, main loss 3.161, pre-clip gradient norm 7.21, key Gram log10 condition 2.18; the global gradient clip (1.0) is active at 100 % of the 493 logged steps, as in the three arms and in D128-3 and the SQ2 arms.
- Compute: 0.180 s per step in the run (0.191 in the smoke), 6.7 s for the slowest 1,000-episode evaluation, peak CUDA 3.05 GiB, state [128, 128] float32 = 65,536 bytes per episode, 475,138 inference parameters.

### What the U control does not show
- The split goes through U (delta store, unit query). It takes the effect of the unit query under the ridge store to be the one measured under the delta store; the ridge store with the learned query, trained from scratch, was not run (RFX measured that store swap only on frozen R checkpoints: 38.8 % at R-r128), so an interaction between the two cannot be separated.
- The step counts differ (U 24,560, RG001 28,005, RG1 28,245, K4 17,970, R-r128 24,645), fixed by the registered time rule; the final-checkpoint contrasts include that difference, and the matched-fraction table is the step-aware view.
- R-r128 is WP-L4's record, not a rerun; the reproduction test (`tests/test_llm_dpfp_repro.py`, bitwise for the first 200 steps with every new option off) passed at the LSQT science commit, as above.
- One seed per arm; validation only; no test or confirmation data was read.

## Memory interventions at the final checkpoints (registered descriptive audit; job 3725433, `reports/p1l-lsqt/audit.json`; U: job 3727489, `reports/p1l-lsqt-u/audit.json`)
Each final export (the three arms and, in a second job, U) is reloaded with `lsqt_run.load_inference` and scored on the same 1,000 validation episodes with the deployed memory, a zeroed memory and the memory of another episode (a one-step roll of a seeded permutation).
The original reproduces the record's final validation exactly in every arm, U included; the repeated-logit drift is 0 and no optimizer step is taken.

| arm | original correct / NLL | zero memory correct / NLL | shuffled memory correct / NLL | original equals the record | max repeated-logit drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 383 / 3.187 | 1 / 8.718 | 2 / 12.038 | yes | 0.0 |
| LT-RG001 | 461 / 2.791 | 2 / 9.485 | 4 / 12.302 | yes | 0.0 |
| LT-RG1 | 426 / 2.907 | 1 / 9.627 | 4 / 11.820 | yes | 0.0 |
| LT-U (control) | 360 / 3.382 | 2 / 8.697 | 0 / 10.795 | yes | 0.0 |

Zeroing or shuffling the memory leaves 0 to 4 of 1,000 correct in every arm, U included: the answers come from the stored episode. The U audit (job 3727489, debug-g, 41 s, at wp/llm `a0eaddde8a0d1ad031a5972eef42489971313dcd`, whose `src` equals the science commit's) uses the same script, loader, episodes and shuffle as the three arms;
the audit's source hash `327ae5f213b940290577f1da5ad07dd5` equals the training record's.

## Compute and memory
| arm | job | source | steps | training min | steady step s | slowest 1,000-episode eval s | peak CUDA GiB | state bytes per episode | inference parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 3724516.opbs | dd5e4c89 | 17970 | 79.5 | 0.262 | 9.4 | 3.24 | 65536 | 475138 |
| LT-RG001 | 3724517.opbs | dd5e4c89 | 28005 | 82.1 | 0.174 | 7.0 | 3.00 | 65536 | 475138 |
| LT-RG1 | 3724518.opbs | dd5e4c89 | 28245 | 84.4 | 0.177 | 7.0 | 3.00 | 65536 | 475138 |
| LT-U (control) | 3725694.opbs | c07452bf | 24560 | 74.3 | 0.180 | 6.7 | 3.05 | 65536 | 475138 |

Peak CUDA memory is 3.24 GiB (K4) and 3.00 GiB (ridge); the state is [128, 128] float32 = 65,536 bytes per episode in every arm; 475,138 inference parameters and 132,096 training-only parameters in every arm (state-dict names and shapes equal WP-L4 R's at r = 128).
The smokes (2,400 steps) measured 0.261, 0.168 and 0.166 s per step (U: 0.191); the runs 0.262, 0.174 and 0.177 s (U: 0.180). U has the same state and parameter counts, with peak CUDA 3.05 GiB.

## Jobs and cost (node·h = wall hours of a one-node job, from the job logs)
| job | name | what | wall seconds | node·h |
| --- | --- | --- | ---: | ---: |
| 3724339-41 | lsqt-smoke-LT-* x3 | smokes, 2,400 steps, three arms | 1559 | 0.433 |
| 3724492 | lsqt-hashes | science config hashes | 19 | 0.005 |
| 3724499-501 | lsqt-reload-LT-* x3 | reload checks | 70 | 0.019 |
| 3724516 | lsqt-LT-K4 | arm job LT-K4 (17,970 steps) | 4793 | 1.331 |
| 3724517 | lsqt-LT-RG001 | arm job LT-RG001 (28,005 steps) | 4947 | 1.374 |
| 3724518 | lsqt-LT-RG1 | arm job LT-RG1 (28,245 steps) | 5087 | 1.413 |
| 3725433 | lsqt-audit | audit of the three final exports | 78 | 0.022 |
| 3725434 | lsqt-repro | DPFP-off reproduction test and test_lsqt at the science commit | 204 | 0.057 |
| 3724910, 3725511 | *-analyze-test | tests of the two analysis scripts | 17 | 0.005 |
| | | **LSQT jobs (three arms)** | **16774** | **4.659** |
| | | shared WP-L12 test jobs (new code, unchanged-path checks; counted once, see SQ2.md) | 1578 | 0.438 |
| | | **LSQT (three arms) plus the shared test jobs** | **18352** | **5.098** |
| 3725518 | lsqtu-tests | U release-path tests at c07452b (238 tests, 0 failures) | 135 | 0.038 |
| 3725519 | lsqtu-smoke | U smoke, 2,400 steps | 503 | 0.140 |
| 3725669 | lsqtu-hashes | U science config hash | 19 | 0.005 |
| 3725676 | lsqtu-reload | U reload check | 24 | 0.007 |
| | | **U preparation** | **681** | **0.189** |
| 3725694 | lsqt-LT-U | arm job LT-U (24,560 steps) | 4487 | 1.246 |
| 3727489 | lsqtu-audit | audit of the U final export | 41 | 0.011 |
| 3727496, 3727554 | lsqtu-analyze-test, -test2 | tests of `analyze_lsqt_u.py` (with `analyze_lsqt.py`), before and after the matched-fraction output was added (6 passed each) | 17 | 0.005 |
| | | **U control jobs (preparation, arm job, audit, analysis tests)** | **5226** | **1.452** |
| | | **LSQT jobs including U** | **22000** | **6.111** |
| | | **LSQT including U plus the shared test jobs** | **23578** | **6.549** |
| | | WP-L12 cumulative: SQ2 jobs (6263 s, 1.740) + LSQT jobs including U + the shared test jobs once (1578 s, 0.438) | 29841 | 8.289 |

Prereg estimate: about 4.6 node·h for the three arms with smokes and tests (<= 6.5 reserved), and about 1.5 for the optional U control. The three arm jobs cost 4.12 node·h; the three-arm LSQT jobs 4.66 and, with the shared test jobs, 5.10 node·h (the audit and reload checks stayed far below their 0.8 cap).
The U control cost 1.452 node·h (preparation 0.189, arm job 1.246, audit 0.011, analysis tests 0.005; estimate about 1.5). LSQT including U is 6.111 node·h, and 6.549 with the shared test jobs counted in full (as in SQ2.md), against an estimate of about 6.1 (4.6 + 1.5; the 4.6 included tests).
With SQ2 (1.740), the shared test jobs counted once, the WP-L12 total is 8.289 node·h.

## Attribution (what can and cannot be said)
The unit query is part of every LSQT arm and of none of R-r128, so the gain of an LSQT arm over R-r128 mixes the store and the unit query; the U control (R's delta store with the unit query, from scratch) separates them in point estimates.
For the best ridge arm, +12.6 pp over R-r128 is +2.5 pp (standard error 2.1) from the unit query (U - R-r128) and +10.1 pp (standard error 2.2) from the store (RG001 - U); for RG1 +9.1 pp is +2.5 and +6.6 pp, for K4 +4.8 pp is +2.5 and +2.3 pp.
In point estimates, then, most of the gain of the ridge arms over R-r128 goes with the store and a smaller part with the unit query; the unit-query part is about the size of its standard error. The ridge arms are above K4 (above) and the two etas differ by 3.5 pp (RG001 over RG1).
All of it is one seed per arm and descriptive, not a registered test; the split goes through U and takes the unit query's effect to be the same under the delta and ridge stores (the ridge store with the learned query, trained from scratch, was not run), and the arms have different step counts (see "U control").
Nothing more is attributed here.

## Next
No registered step of LSQT remains. The U control has run and is descriptive; it enters neither outcome, so the registered decision (no candidate; the best ridge arm in the open band; the trained-LSQ line open; the learned-writer line stopped) is unchanged.
Whether anything follows from the open band, or from the split above, is for the orchestrator; fable's W3 trigger (best ridge arm >= 48.5 %) was not met (46.1 %) and W3 is not part of this registration.

## Deviations and notes (none changes the registered rule or any threshold)
1. The U release path (`study: LSQT-U`, the control alone) was added to the LSQT release code after these arms ran (commit c07452b), so that the control can be released under the same rule; it does not touch the three-arm path (tests).
2. The decision is computed by `scripts/analyze_lsqt.py` (stdlib, restates `lsqt_run.choose_t`); both restatements are tested at their exact thresholds.
3. Jobs 3724073 and 3724074 were deleted after a submission at the master commit (the submit script resolves `--commit HEAD` in the main checkout); they ran no science code.
4. The comparisons with R-r128 and D128-3 are descriptive, as registered; validation was used to select D128-3 (a screen only).
5. The U control was released and run at science commit `c07452b` (not `dd5e4c8`), as noted at the top; its frozen step count (24,560) comes from the same rule and from its own smoke (the arms' step counts are 17,970, 28,005 and 28,245; R-r128's 24,645).
   The audit of U ran in a separate job at a later wp/llm commit with the same `src` (same source hash as the training record).

## Not done
No E = 8 retrain (registered: none). No second seed for any arm or for U. The ridge store with the learned query was not trained from scratch (no interaction estimate). No test or confirmation range was read.
