# SQ2 report: D128-3 continued with ridge storage on the DPFP features and the coefficient loss (WP-L12, G5)

Registered design: `docs/prereg/SQ2.md` (registered 847b17f; **Clarification C1** a3013ff, 2026-10-29 13:43 JST, before the numerics job: w_c rounds on the log2 scale; blake2b-128 with C1
`9f725a62e918ac668dd70220873038c5`). Run 2026-10-29 JST by sn-llm at science commit `3d543b430a2533df0057647081b07d874de893db` (wp/llm), schedule release
`reports/p1l-sq2/release.json` (aeb988a, approved by the orchestrator, `reports/p1l-sq2/approval.json`). Four arms, each the continuation of the final D128-3 checkpoint
(adapter, training-only value head and AdamW state; file digests and config hash checked) for **N = 6,000 steps on one fresh training stream** `derive_seed(17, 'llm', 'sq2-continuation')`,
constant lr 3e-3, batch 32 / Q 4, E = 16, seed 17, 1,000 validation episodes at eval seed 4242, evaluated at step 0 and at 1,200 / 2,400 / 3,600 / 4,800 / 6,000:

| arm | storage | coefficient loss |
| --- | --- | --- |
| S-DE | delta, as trained (the continued control) | no |
| S-DC | delta | yes (w_c = 1) |
| S-RE | ridge on the DPFP features, eta = 1e-4 | no |
| S-RC | ridge on the DPFP features, eta = 1e-4 | yes (w_c = 1) |

Records, the step-0 check, the numerics, the audit and the decision are in `reports/p1l-sq2/`; every table below is printed by `python3 scripts/analyze_sq2.py` from them
(`tests/test_analyze_sq2.py` checks the decision rule and re-derives `decision.json` from the records). Validation only, one seed; no test or confirmation data was read.

## Result and registered decision
**The primary contrast S-RC against S-DE does not advance: the EM gain is +4.1 pp (needs +10) and the read error is 0.554 of S-DE's (needs at most 0.5); only the stability condition is met.
The learned-writer line stays stopped. No cell reaches val EM 0.80, so no writer candidate is flagged.**

| arm | job | storage | coefficient loss | correct of 1,000 | val EM % | normalized read error | clean-value head % | >= 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S-DE | 3724484.opbs | delta | no | 654 | 65.4 | 0.359 | 93.7 | no |
| S-DC | 3724485.opbs | delta | yes | 630 | 63.0 | 0.353 | 92.9 | no |
| S-RE | 3724486.opbs | ridge (eta 1e-4, DPFP features) | no | 723 | 72.3 | 0.196 | 93.6 | no |
| S-RC | 3724487.opbs | ridge (eta 1e-4, DPFP features) | yes | 695 | 69.5 | 0.199 | 92.9 | no |
| D128-3 (parent, 22,370 steps) |  | delta | no | 608 | 60.8 | 0.363 |  | no |

Registered primary contrast S-RC against S-DE: EM gap +4.1 pp (needs >= +10.0): **not met**; read error ratio 0.554 (needs <= 0.500): **not met**; stable decoding (finite training, clean-value head >= 88.7 % at all five checkpoints; S-RC minimum 91.6 %): **met**.

Decision: **does not advance**; learned-writer line stays stopped.

Interaction (S-RC - S-RE) - (S-DC - S-DE): -0.4 pp of EM (descriptive; standard error of a difference of two EMs about 2.2 pp, of this double difference about 4.4 pp). Cells at 0.80 other than S-RC: none.

Registered prediction (S-RC gains >= 10 pp and halves the read error; neither S-RE nor S-DC alone meets both): **did not hold**.

| single change vs S-DE | EM gain pp | read error ratio | meets conditions 1 and 2 |
| --- | ---: | ---: | ---: |
| S-DC | -2.4 | 0.982 | no |
| S-RE | +6.9 | 0.546 | no |

The registered primary contrast: S-RC 695 against S-DE 654 correct (+41 of 1,000; the standard error of a difference of two EMs at 1,000 episodes is about 2.2 pp, so +4.1 pp is about 1.9 standard errors,
and the registered threshold is +10 pp). The normalized read error is 0.199 against 0.359 (ratio 0.554; the threshold is 0.5, so the read error is not quite halved). Stability holds: every logged loss
and gradient in all four arms is finite, and the clean-value head of S-RC is between 91.6 % and 94.4 % at the five checkpoints, above the floor of 88.7 % (S-DE's final 93.7 % minus 5 pp).

## What the four cells show (descriptive; one seed, standard error of a difference about 2.2 pp)
- **The ridge store alone (S-RE) is the best cell: 72.3 %, +6.9 pp over S-DE, with read error 0.196 (ratio 0.546).** It is not the registered contrast, and it too misses conditions 1 and 2 (it falls 3.1 pp short on EM and 0.046 short on the ratio).
- **The coefficient loss did not help in either storage.** S-DC is -2.4 pp against S-DE and S-RC is -2.8 pp against S-RE; the interaction (S-RC - S-RE) - (S-DC - S-DE) is -0.4 pp (standard error of the double difference about 4.4 pp).
  The read error is unchanged by it under delta (0.353 against 0.359) and under ridge (0.199 against 0.196).
- **Most of the ridge arms' head start is the store swap, before any retraining.** At step 0 (D128-3's weights, nothing retrained) the delta arms are D128-3 exactly (608 correct, read error 0.363, as required); the ridge store on the same weights gives
  **687 correct (68.7 %, read error 0.312)**. Over the 6,000 steps S-DE gains +4.6 pp (608 to 654), S-DC +2.2 pp, S-RE +3.6 pp (687 to 723) and S-RC +0.8 pp (687 to 695).
  Both ridge arms first dip below their step-0 value (64.2 % and 63.9 % at step 1,200) and then recover: they start with AdamW moments from another objective (registered deviation 2; the early checkpoints show the transient).
- **The continued control kept improving.** S-DE gained 4.6 pp over the 6,000 continuation steps (D128-3 had gained 0.6 pp over its last 4,474; the prereg cited that as the reason a longer control was fair). One seed, on the validation set that selected D128-3.
- **Registered prediction (S-RC gains at least 10 pp and halves the read error; neither S-RE nor S-DC alone meets both): did not hold** (S-RC +4.1 pp, ratio 0.554). Neither single change meets both conditions either (S-RE +6.9 pp, ratio 0.546; S-DC -2.4 pp, ratio 0.982).

### Validation EM % at step 0 and at the five checkpoints
| arm | 0 | 1200 | 2400 | 3600 | 4800 | 6000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S-DE | 60.8 | 61.5 | 62.7 | 61.5 | 63.6 | 65.4 |
| S-DC | 60.8 | 61.1 | 63.6 | 62.3 | 64.4 | 63.0 |
| S-RE | 68.7 | 64.2 | 67.1 | 70.7 | 69.3 | 72.3 |
| S-RC | 68.7 | 63.9 | 66.8 | 68.5 | 68.2 | 69.5 |

### Normalized read error
| arm | 0 | 1200 | 2400 | 3600 | 4800 | 6000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S-DE | 0.363 | 0.390 | 0.364 | 0.370 | 0.361 | 0.359 |
| S-DC | 0.363 | 0.388 | 0.385 | 0.364 | 0.364 | 0.353 |
| S-RE | 0.312 | 0.274 | 0.242 | 0.202 | 0.216 | 0.196 |
| S-RC | 0.312 | 0.304 | 0.226 | 0.217 | 0.215 | 0.199 |

### Clean-value head accuracy % (condition 3)
| arm | 0 | 1200 | 2400 | 3600 | 4800 | 6000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S-DE | 90.6 | 93.7 | 93.0 | 93.0 | 91.4 | 93.7 |
| S-DC | 90.6 | 93.2 | 92.9 | 92.1 | 93.9 | 92.9 |
| S-RE | 90.6 | 93.1 | 95.2 | 92.4 | 92.7 | 93.6 |
| S-RC | 90.6 | 93.5 | 91.6 | 94.4 | 92.6 | 92.9 |

### Read-value head accuracy %
| arm | 0 | 1200 | 2400 | 3600 | 4800 | 6000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S-DE | 51.0 | 52.8 | 54.8 | 52.7 | 52.9 | 54.6 |
| S-DC | 51.0 | 49.9 | 54.3 | 47.8 | 53.4 | 50.4 |
| S-RE | 68.1 | 58.2 | 66.0 | 66.4 | 64.9 | 66.2 |
| S-RC | 68.1 | 59.6 | 64.6 | 63.7 | 66.9 | 66.4 |

### Diagnostics at the final checkpoint
| metric | S-DE | S-DC | S-RE | S-RC |
| --- | ---: | ---: | ---: | ---: |
| val EM % | 65.4 | 63.0 | 72.3 | 69.5 |
| NLL | 1.894 | 1.957 | 1.642 | 1.695 |
| normalized read error | 0.359 | 0.353 | 0.196 | 0.199 |
| clean-value head acc % | 93.7 | 92.9 | 93.6 | 92.9 |
| read-value head acc % | 54.6 | 50.4 | 66.2 | 66.4 |
| value SIR | 7.88 | 6.74 | 20.33 | 20.61 |
| key top-1 % | 85.9 | 86.9 | 87.1 | 86.8 |
| key margin | 0.255 | 0.254 | 0.231 | 0.239 |
| key Gram RMS (phi space for DPFP) | 0.179 | 0.172 | 0.203 | 0.200 |
| key Gram RMS (original space) | 0.316 | 0.319 | 0.339 | 0.344 |
| mean max |cos| (phi space for DPFP) | 0.678 | 0.671 | 0.750 | 0.749 |
| mean max |cos| (original space) | 0.821 | 0.825 | 0.847 | 0.851 |
| mean pairwise cosine (as logged) | 0.127 | 0.120 | 0.131 | 0.125 |
| parallel interference | 0.274 | 0.315 | 0.178 | 0.227 |
| query norm | 1.0 | 1.0 | 1.0 | 1.0 |

Under the ridge store the read-value head is 66 % against 50 to 55 % under delta, the value SIR is 20.3 and 20.6 against 7.9 and 6.7, and the parallel interference is 0.18 and 0.23 against 0.27 and 0.32; the coefficient loss does not move these consistently.
The clean-value head (the value decoding of the stored value itself) is 92.9 % to 93.7 % in every arm.

### Training-log diagnostics
| metric | S-DE | S-DC | S-RE | S-RC |
| --- | ---: | ---: | ---: | ---: |
| coefficient loss (weighted, w_c = 1) |  | 0.400 |  | 0.313 |
| raw-read loss | 0.359 | 0.360 | 0.197 | 0.203 |
| direction cosine d_t / ||d|| | 0.798 | 0.804 | 0.854 | 0.850 |
| coefficient cancellation (median) | 1.769 | 1.700 | 1.774 | 1.767 |
| log10 cond of the key Gram (median) | 1.089 | 1.057 | 1.198 | 1.190 |
| log10 cond of the ridge system (median) |  |  | 1.198 | 1.190 |
| query norm | 1.000 | 1.000 | 1.000 | 1.000 |
| gradient norm before clipping | 7.852 | 8.535 | 9.286 | 10.117 |

Fraction of logged steps with the gradient clipped: S-DE 100 %, S-DC 100 %, S-RE 100 %, S-RC 100 %.

The direction cosine d_t / ||d|| that the coefficient loss targets is 0.85 under ridge and 0.80 under delta in the last logged steps, with or without the loss (S-DC 0.804 against S-DE 0.798; S-RC 0.850 against S-RE 0.854):
at w_c = 1 the loss did not change it. The coefficient cancellation median is about 1.7 to 1.8 in all four arms. (Single batches at every 50th step; the table is the mean of the last 10 logged steps.)

## Gradient clipping (descriptive)
The global gradient-norm clip (1.0) is active at **every logged step in all four arms**: 121 of 121 logged steps (step 1, every 50th step, the last) each, and the pre-clip norm is about 8 to 10 (median over the logged steps S-DE 8.4, S-DC 8.8,
S-RE 9.3, S-RC 9.6; range 6.3 to 14.7). **D128-3's own training log shows the same:** 449 of 449 logged steps clipped, pre-clip norm median 8.9, range 3.1 to 15.3, mean of the last ten logged steps 8.8
(`/work/xg24i002/x10041/LoopLora/runs/p1l/dpfp/arms/D128-3/train_log.jsonl`). So clipping at this level is a property of the setting the continuation inherits, not something the ridge store or the coefficient loss introduced. The logged norm is that of the total
objective; the gradients of the individual loss terms were not logged per step. Nothing is claimed here about what the constant clipping does to the effect of w_c or of the other loss weights.

## Memory interventions at the final checkpoints (registered descriptive audit; job 3724865, `reports/p1l-sq2/audit.json`)
Each final export is reloaded with `sq2_run.load_inference` and scored on the same 1,000 validation episodes with the deployed memory, a zeroed memory and the memory of another episode (a one-step roll of a seeded permutation).
The original reproduces the record's final validation exactly in every arm (same correct count and NLL), the repeated-logit drift is 0, and no optimizer step is taken.

| arm | original correct / NLL | zero memory correct / NLL | shuffled memory correct / NLL | original equals the record | max repeated-logit drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| S-DE | 654 / 1.894 | 0 / 15.948 | 1 / 12.799 | yes | 0.0 |
| S-DC | 630 / 1.957 | 0 / 16.434 | 2 / 12.669 | yes | 0.0 |
| S-RE | 723 / 1.642 | 1 / 10.108 | 0 / 12.728 | yes | 0.0 |
| S-RC | 695 / 1.695 | 0 / 10.331 | 1 / 12.877 | yes | 0.0 |

Zeroing or shuffling the memory leaves 0 to 2 of 1,000 correct in every arm: the answers come from the stored episode, under the ridge store as under delta. The zero-memory NLL is lower under ridge (10.1 to 10.3) than under delta (15.9 to 16.4); no verdict is attached to that.

## Numerics (training-only; job 3724336, `reports/p1l-sq2/numerics.json`, committed in 3d543b4 before any smoke output was read)
| eta | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean normalized read error | 0.2930 | 0.2930 | 0.2930 | 0.2930 | 0.2937 | 0.3079 | 0.4816 | 0.8453 |

eta = 0.0001 (the lowest error is at 1e-6; the registered clip to [1e-4, 1] sets 1e-4). Gradient-norm ratio ||grad read|| / ||grad coef|| (delta store): median 0.898, min 0.546, max 1.316, so w_c = 1 (log2 rule, Clarification C1); under the ridge store the median is 0.832 (reported, not used).
Deployed delta read error on these batches 0.347; ridge at eta 1e-4 on the same keys and queries 0.293. Direction cosine 0.806 (delta) / 0.837 (ridge); coefficient cancellation 1.80 / 2.06.

**eta was clipped.** The ridge error is lowest at the grid's smallest value, 1e-6, and the curve is flat up to 1e-3 (0.2930 at 1e-6, 1e-5, 1e-4 and 1e-3, 0.2937 at 1e-2, 0.308 at 1e-1, 0.482 at 1, 0.845 at 10), so the registered
clip to [1e-4, 1] set eta = 1e-4; the clip costs nothing measurable. On these 64 batches, with the checkpoint's own queries and before any training, the ridge store (eta 1e-4) reads with normalized error 0.293 against 0.347 for the deployed delta store
(15.7 % lower), and the direction cosine is 0.837 against 0.806. w_c = 1 is the log2 rounding of the median ratio 0.898 (Clarification C1; the ratio ranges from 0.546 to 1.316). On the validation episodes at step 0 the same checkpoint reads at 0.312 under ridge
and 0.363 under delta, a smaller difference than on the training batches.

## Step-0 check (job 3724338, `reports/p1l-sq2/step0.json`)
S-DE at step 0 reproduces D128-3's recorded final validation exactly: **608 correct, NLL 2.1291537284851074 bit for bit, and all 1,000 predictions identical** to the parent's saved `val-22370/samples.npz`.
The parent's file digests (blake2b-128 `b6acff8b...` and `31d98c01...`) and config hash `2207a847e4967fb8e60d1663237fd979` are the registered ones; the S-DE config equals the parent's config in everything but its name and the `sq2` section (a test on the real checkpoint).

## Compute and memory
| arm | job | source | steps | training min | steady step s | slowest 1,000-episode eval s | peak CUDA GiB | state bytes per episode | inference parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S-DE | 3724484.opbs | 3d543b43 | 6000 | 21.9 | 0.210 | 7.2 | 4.08 | 393216 | 475138 |
| S-DC | 3724485.opbs | 3d543b43 | 6000 | 22.3 | 0.214 | 6.9 | 4.08 | 393216 | 475138 |
| S-RE | 3724486.opbs | 3d543b43 | 6000 | 20.8 | 0.199 | 6.9 | 3.76 | 393216 | 475138 |
| S-RC | 3724487.opbs | 3d543b43 | 6000 | 21.1 | 0.202 | 7.2 | 3.76 | 393216 | 475138 |

Peak CUDA memory is 4.08 GiB (delta arms) and 3.76 GiB (ridge arms); the state is [128, 768] float32 = 393,216 bytes per episode in every arm; the inference parameter count is 475,138 and the training-only head 132,096 in every arm.
The smokes (600 steps) measured 0.203 to 0.213 s per step; the runs 0.199 to 0.214 s.

## Jobs and cost (node·h = wall hours of a one-node job, from the job logs)
| job | name | what | wall seconds | node·h |
| --- | --- | --- | ---: | ---: |
| 3724336 | sq2-numerics | numerics (64 train batches) | 77 | 0.021 |
| 3724338 | sq2-step0 | step-0 check | 29 | 0.008 |
| 3724379-82 | sq2-smoke x4 | smokes, 600 steps, four arms | 669 | 0.186 |
| 3724443 | sq2-hashes | science config hashes | 18 | 0.005 |
| 3724445-48 | sq2-reload x4 | reload checks | 93 | 0.026 |
| 3724484-87 | sq2-S-* x4 | the four arm jobs, 6,000 steps | 5279 | 1.466 |
| 3724865 | sq2-audit | audit of the four final exports | 98 | 0.027 |
| | | **SQ2 jobs** | **6263** | **1.740** |
| 3724075, 3724076, 3724189, 3724190, 3724237, 3724303 | l12-tests-* | tests (new code, unchanged-path checks, continuation; six jobs, two of them for the whole suite) (shared with LSQT) | 1559 | 0.433 |
| 3724024 | hash-lsq | LSQ config hashes at the pre-change commit (the unchanged-path pin) (shared with LSQT) | 19 | 0.005 |
| | | **shared WP-L12 test jobs** | **1578** | **0.438** |
| | | **SQ2 plus the shared test jobs** | **7841** | **2.178** |

Prereg estimate: about 2.1 node·h in total. The arm jobs cost 1.47 node·h (estimate 1.6); with tests and checks the study cost about 2.18 node·h, counting the shared test jobs in full.

## Deviations and notes (none changes the registered rule or any threshold)
1. **Clarification C1** (orchestrator, before the numerics job): the w_c rounding is on the log2 scale. It was implemented that way already; the release records the median ratio and w_c under this rule only.
2. **eta was clipped to the registered lower bound 1e-4** (above): by the rule, not a deviation, flagged because the error minimum lies outside the allowed range.
3. **Product defects found by the tests before any science job** (none reached a result): `sq2_numerics.measure` unpacked two values into three names; `gradient_norm` took `torch.sqrt` of a Python float (both fixed, tests rerun at dd5e4c8 and later). One test expectation was wrong
   (5.7 against the log2 rounding boundary 5.657) and was corrected, with no assertion weakened.
4. **Reload gate extended** (81284e1, after the smokes): `check_sq2.py reload` also recomputes the science config hash at N = 6,000 and the prereg digest and requires both to equal the release's. The science jobs ran at 3d543b4; later wp/llm commits add only checks,
   the LSQT study and this analysis.
5. **Decision computed by `scripts/analyze_sq2.py`** (stdlib, restates the rule) rather than by a compute-side `choose` function as for DPFP-X; `tests/test_analyze_sq2.py` covers the rule at its exact thresholds.
6. The step-0 and per-arm validation, the audit and the tables use the 1,000 validation episodes that selected D128-3 (registered deviation 7): a screen only.
7. Jobs 3724073 and 3724074 were deleted after a submission at the master commit (the submit script resolves `--commit HEAD` in the main checkout); they ran no science code.

## Not done
Nothing is open for SQ2. The primary contrast does not advance, so no three-seed confirmation is flagged. The LSQT study (3724516 to 3724518) is separate.
