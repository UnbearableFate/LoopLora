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

Records, the audit and the decision are in `reports/p1l-lsqt/`; every table below is printed by `python3 scripts/analyze_lsqt.py` from them (R-r128 and D128-3 are read from `reports/p1l-dpfp/`;
`tests/test_analyze_lsqt.py` checks the rule and re-derives `decision.json`). Validation only, one seed; no test or confirmation data was read.

## Result and registered decision
**No arm reaches val EM 0.80, so no writer candidate is flagged (outcome 1). The best ridge arm, LT-RG001, reaches 46.1 %: inside the 40 % to < 48.5 % band, which is neither fable's W3 trigger (>= 48.5 %) nor the falsifier
(< 40 %). The trained-LSQ line is neither closed nor triggered; the learned-writer line stays stopped.** The conditional U control is proposed (the best arm is above 40 %); see "Next".

| arm | job | store | steps | correct of 1,000 | val EM % | normalized read error | clean-value head % | read-value head % | >= 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 3724516.opbs | sweep | 17970 | 383 | 38.3 | 0.554 | 92.3 | 27.4 | no |
| LT-RG001 | 3724517.opbs | ridge (eta 0.01) | 28005 | 461 | 46.1 | 0.512 | 93.9 | 26.5 | no |
| LT-RG1 | 3724518.opbs | ridge (eta 1) | 28245 | 426 | 42.6 | 0.499 | 95.9 | 22.3 | no |
| R-r128 (WP-L4, not rerun) |  | delta | 24645 | 335 | 33.5 | 0.631 | 93.3 | 19.7 | no |
| D128-3 (DPFP) |  | delta | 22370 | 608 | 60.8 | 0.363 | 90.6 | 51.0 | no |

Registered decision: **no arm reaches 0.80, no writer candidate**. Best RG arm LT-RG001 with 461 of 1,000 (46.1 %): band **open** (>= 48.5 % W3 trigger; 40 % to < 48.5 % neither; < 40 % the trained-LSQ line closes). Trained-LSQ line: **open**. Best arm LT-RG001 (46.1 %): the conditional U control is proposed (best arm >= 40 %).

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

## Memory interventions at the final checkpoints (registered descriptive audit; job 3725433, `reports/p1l-lsqt/audit.json`)
Each final export is reloaded with `lsqt_run.load_inference` and scored on the same 1,000 validation episodes with the deployed memory, a zeroed memory and the memory of another episode (a one-step roll of a seeded permutation).
The original reproduces the record's final validation exactly in every arm; the repeated-logit drift is 0 and no optimizer step is taken.

| arm | original correct / NLL | zero memory correct / NLL | shuffled memory correct / NLL | original equals the record | max repeated-logit drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 383 / 3.187 | 1 / 8.718 | 2 / 12.038 | yes | 0.0 |
| LT-RG001 | 461 / 2.791 | 2 / 9.485 | 4 / 12.302 | yes | 0.0 |
| LT-RG1 | 426 / 2.907 | 1 / 9.627 | 4 / 11.820 | yes | 0.0 |

Zeroing or shuffling the memory leaves 1 to 4 of 1,000 correct in every arm: the answers come from the stored episode.

## Compute and memory
| arm | job | source | steps | training min | steady step s | slowest 1,000-episode eval s | peak CUDA GiB | state bytes per episode | inference parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LT-K4 | 3724516.opbs | dd5e4c89 | 17970 | 79.5 | 0.262 | 9.4 | 3.24 | 65536 | 475138 |
| LT-RG001 | 3724517.opbs | dd5e4c89 | 28005 | 82.1 | 0.174 | 7.0 | 3.00 | 65536 | 475138 |
| LT-RG1 | 3724518.opbs | dd5e4c89 | 28245 | 84.4 | 0.177 | 7.0 | 3.00 | 65536 | 475138 |

Peak CUDA memory is 3.24 GiB (K4) and 3.00 GiB (ridge); the state is [128, 128] float32 = 65,536 bytes per episode in every arm; 475,138 inference parameters and 132,096 training-only parameters in every arm (state-dict names and shapes equal WP-L4 R's at r = 128).
The smokes (2,400 steps) measured 0.261, 0.168 and 0.166 s per step; the runs 0.262, 0.174 and 0.177 s.

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
| | | **LSQT jobs** | **16774** | **4.659** |
| | | shared WP-L12 test jobs (new code, unchanged-path checks; counted once, see SQ2.md) | 1578 | 0.438 |
| | | **LSQT plus the shared test jobs** | **18352** | **5.098** |

Prereg estimate: about 4.6 node·h for the three arms with smokes and tests (<= 6.5 reserved). The three arm jobs cost 4.12 node·h; the LSQT jobs 4.66 and, with the shared test jobs, 5.10 node·h (the audit and reload checks stayed far below their 0.8 cap).
The optional U control is not included.

## Attribution (what can and cannot be said)
The unit query is part of every LSQT arm and of none of R-r128, so a gain over R-r128 cannot yet be split between the store and the unit query: the U control has not run. Within the three arms, the ridge arms are above K4 (above) and the two etas differ by
3.5 pp (RG001 over RG1); both are one seed. Nothing more is attributed here.

## Next (proposal, for the orchestrator)
The best arm is above 40 %, so the registration allows the U control (R's decay-first delta store with the unit query, from scratch, `docs/prereg/LSQT.md` decision 2): its own 2,400-step smoke, steps frozen by the same rule, at most 2 h, the same audit;
descriptive only, it enters neither outcome. Its purpose is to split the +12.6 pp of RG001 over R-r128 (and the gains of the other arms) between the store and the unit query.

## Deviations and notes (none changes the registered rule or any threshold)
1. The U release path (`study: LSQT-U`, the control alone) was added to the LSQT release code after these arms ran (commit c07452b), so that the control can be released under the same rule; it does not touch the three-arm path (tests).
2. The decision is computed by `scripts/analyze_lsqt.py` (stdlib, restates `lsqt_run.choose_t`); both restatements are tested at their exact thresholds.
3. Jobs 3724073 and 3724074 were deleted after a submission at the master commit (the submit script resolves `--commit HEAD` in the main checkout); they ran no science code.
4. The comparisons with R-r128 and D128-3 are descriptive, as registered; validation was used to select D128-3 (a screen only).

## Not done
The U control (above). No E = 8 retrain (registered: none). No test or confirmation range was read.
