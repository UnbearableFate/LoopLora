# Decision record — B8W (B8 warm start from M1-L4; G3), 2026-10-29

Sources: `docs/prereg/B8W.md` (registered at master 3c9c396, blake2b-128 `7414b6e0118d8d53a913467d6e7848b2`), `reports/b8w/results/`
(`results.md`, `endpoints.json`, `accuracy.csv`, `drift.csv`, `train-loss.csv`, `run-manifest.json`), `reports/b8w/checkpoint-lock.json`,
`reports/b8w/continuation-manifest.json`, `reports/b8w/confirmation-authorization.json`, `reports/b8w/evaluation-manifest.json`,
`reports/b8w/jobs.json` (ledger), `docs/reports/B8W.md`, `docs/reports/B8W-state.md`. Qwen3-0.6B-Base, layer 14 `o_proj`, r = r_mem = 128,
`oracle_random`, primary setting, memory-only. Starting points: the registered P1-L M1-L4 checkpoints, seeds 17/29/43. Continuations of N = 2,000
steps (lr 0.003, fresh AdamW, warmup restarted, the P1-L stream read from step 17,000), pinned to `70bb12c`; evaluation and finalizer pinned to
`123a28fac67a7b09774d2fab87f9187cbf320dbf`. All numbers: test [108000, 110000), eval seed 4242, 2,000 episodes, L = 4 (B4 at L = 1), hops 1-8,
full-vocabulary EM, 95% paired episode-bootstrap CIs (2000 draws, seed 2718) unless stated.

## 1. Registered decision
**Untying keeps the solution.** Delta = untied@4 - shared@4 on hops 2 and 3 is **+0.90 pp [0.09, 1.71]** (32.40% vs 31.50%), per seed
**+1.05 / +0.90 / +0.75 pp** (seeds 17 / 29 / 43).
- Every seed's loss is at most 2 pp (no seed has a loss at all): condition 1 met.
- The pooled CI lower bound, +0.09 pp, is above -2 pp: condition 2 (non-inferiority) met.
- Registered wording kept: B8 gains **no equal-parameter status** (trainable 589,824 vs 442,368; adapter 622,594 vs 475,138), **G1-L stays failed**, and
  **nothing is said about why B8 failed from a random start**. "Keeps" means the untied parameterization held and kept training the copied solution for
  2,000 steps from three starting seeds; capacity is not what limits keeping it.

## 2. Equality gate and verification (before any test episode)
- Gate at full scale, every seed (validation split, 64 episodes, depth 4): bf16 logits `torch.equal`, predicted tokens equal, the control (M1-L4 scored
  twice) equal, fp32 adapter read equal, max |difference| 0.0 at tolerance 0. Start hashes c90d23ff / 59e293ee / 37a7baf0; converted 1f0c56ae / 6f62abbe / 9c623e91.
- The six continuations were verified by `make_b8w_manifest.py` (records complete, 21 logged rows ending at step 2000, lr, stream start, start hashes,
  P1-L `data_seed`, identical first-step loss in both arms, configs equal apart from name and variant, one source, not dirty, oracle writer hash, exit-0
  receipts); the authorization check (`evaluate_b8w.py --check-only`, job 3724218) passed before the orchestrator released the evaluation.
- Evaluation job 3724223: all 15 records complete, each with the checkpoint hash of the manifest, window `b8w`, 2,000 episodes, seed 4242, hops 1-8,
  `episode_ranges` = test start 108000, count 2000. The finalizer (3724260) re-audited them (metrics reproducible from samples, 2,000 paired queries per
  cell, identical oracle snapshots across runs).

## 3. Protocol check (prereg: shared arm more than 2 pp below the start)
**Not triggered.** The shared arm is **+5.09 pp [4.18, 6.01]** above the starting M1-L4 on hops 2-3 (31.50% vs 26.41%; per seed +4.03 / +5.47 / +5.78),
so the continuation recipe (fresh Adam state at lr 0.003) did not perturb the solution; the M1-L4 reader was still improving (its P1-L training loss was
falling at step 17,000). The decision is therefore read without a caveat, but note that the reference moved up by 5 pp and untied followed it.

## 4. All reference contrasts (hops 2-3 unless stated; pp [95% CI]; per seed 17 / 29 / 43)
| Contrast | Difference | Accuracies | Per seed |
| --- | --- | --- | --- |
| **Delta = untied@4 - shared@4 (decision)** | **+0.90 [0.09, 1.71]** (97.5%: [-0.04, 1.81]) | 32.40 vs 31.50% | +1.05 / +0.90 / +0.75 |
| untied - shared, hop 1 | +2.77 [1.55, 4.02] (97.5%: [1.30, 4.20]) | 55.47 vs 52.70% | +2.85 / +1.70 / +3.75 |
| untied - start M1-L4@4 | +5.99 [5.07, 6.88] | 32.40 vs 26.41% | +5.08 / +6.38 / +6.53 |
| untied - start, hop 1 | +7.45 [6.00, 8.95] | 55.47 vs 48.02% | +6.40 / +8.75 / +7.20 |
| shared - start | +5.09 [4.18, 6.01] | 31.50 vs 26.41% | +4.03 / +5.47 / +5.78 |
| shared - start, hop 1 | +4.68 [3.33, 6.02] | 52.70 vs 48.02% | +3.55 / +7.05 / +3.45 |
| untied - original B8@4 (P1-L) | +26.07 [24.98, 27.13] | 32.40 vs 6.33% | +29.38 / +20.03 / +28.80 |
| untied - B4@1 | +26.98 [25.87, 28.07] | 32.40 vs 5.42% | +29.88 / +21.38 / +29.70 |
| shared - B4@1 | +26.08 [24.98, 27.19] | 31.50 vs 5.42% | +28.82 / +20.47 / +28.95 |
| start M1-L4@4 - B4@1 | +20.99 [19.97, 22.06] | 26.41 vs 5.42% | +24.80 / +15.00 / +23.18 |

Accuracy by hop 1-8 (seed means; `accuracy.csv`): start 48.0 / 34.7 / 18.2 / 6.8 / 4.6 / 5.1 / 4.9 / 5.1%; shared 52.7 / 41.9 / 21.1 / 5.5 / 3.6 / 4.2 / 3.7 / 4.1%;
untied 55.5 / 40.8 / 24.0 / 4.2 / 3.4 / 4.0 / 3.6 / 3.6%; original B8 18.8 / 6.7 / 6.0 / 5.8 / 5.9 / 5.9 / 6.2 / 6.0%; B4@1 20.0 / 5.6 / 5.3 / 5.1 / 5.6 / 5.3 / 5.5 / 5.1%.
The starting M1-L4 (26.41%), original B8 (6.33%) and B4@1 (5.42%) on hops 2-3 match P1-L's values on its own range (26.91 / 6.34 / 5.36%).

## 5. Drift of the continued tensors (`drift.csv`; from checkpoints, no test data)
Relative Frobenius change from the starting tensor, seeds 17 / 29 / 43. R: shared 0.44 / 0.49 / 0.45; untied copies 0.48-0.55 / 0.53-0.58 / 0.50-0.57.
U: shared 0.22 / 0.23 / 0.22; untied copies 0.26-0.32 / 0.27-0.32 / 0.26-0.31. Q: shared 0.21 / 0.24 / 0.21; untied copies 0.22-0.28 / 0.22-0.29 / 0.22-0.28.
Mean pairwise cosine distance between the four untied copies: R 0.191 / 0.215 / 0.205; U 0.072 / 0.074 / 0.072; Q 0.051 / 0.051 / 0.051. So the copies
separated from each other (most in R) and every tensor moved a lot in both arms; the untied arm did not stay tied in practice.
Training loss at step 2,000 (logged batch, training data only, not a decision input): shared / untied 2.989 / 3.169, 3.781 / 3.595, 2.800 / 2.656 (seeds 17 / 29 / 43);
mean of the last five logged steps 2.789 / 2.736, 3.423 / 3.301, 2.968 / 2.872.

## 6. Claims and non-claims (as registered)
- **Supported:** a copied B8 retains hops-2-3 accuracy within 2 pp of continued M1-L4 over N = 2,000 steps at lr 0.003, in three starting seeds, on one fresh range, in the
  oracle condition with seen names. The exact containment (gate, tolerance 0) holds, and the untied reader trains from it without losing it.
- **Exploratory (not registered, not a claim):** untied is nominally above shared (+0.90 pp; the 97.5% interval [-0.04, 1.81] touches 0; hop 1 +2.77 [1.55, 4.02]). No superiority
  test was registered. Both arms improved over the start by about 5-6 pp, so the starting reader was not converged; this says nothing about untying being better.
- **Not claimed:** that sharing is or is not necessary for learning composition; the cause of B8's random-start failure (optimization, update allocation and the inductive bias
  of sharing are not separated; B8 from a random start remains at 6.33% on hops 2-3); equal-parameter status for B8 (M1-L4 vs B4 stays the only equal-parameter contrast);
  a revised G1-L; anything for r = 32, learned writers, unseen names, other scales or other N; any unseen-hop claim.

## 7. Limits and deviations
Three seeds (the bootstrap conditions on them); one N, one lr, one range; the result is sensitive to N because the shared reference was still improving. No deviation from
the registered rule, N, lr, stream, window, interval or contrasts. Procedure notes: the evaluation ran as one job over the 15 groups (the prereg allowed one or two); the
evaluation manifest was assembled from the 15 verified records by a short inline script (its output is committed); the first test job 3723982 failed one test of mine (a dict of
tensors compared with `==`; test defect, fixed, rerun clean, no assertion weakened).

## 8. Cost
`reports/b8w/jobs.json` (`--study b8w`, `--require-final`): **10 jobs, 1.004 node·h** (budget 1.0-2.0; projection 1.05-1.10): tests and smoke 0.20 (3723982, 3723983, 3724025), continuations
0.704 (3724077 / 3724078 / 3724079: 0.237 / 0.235 / 0.232), pipeline tests 0.012 (3724117), authorization check 0.006 (3724218), evaluation 0.073 (3724223, 4:23 for all 15 groups),
finalizer 0.007 (3724260). Every job exits 0 except 3723982.
