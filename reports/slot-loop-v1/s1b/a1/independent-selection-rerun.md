**Verdict: selection reproduced.** No discrepancies were found in the frozen rerun or the independent checks of all twelve runs.

Reviewer: **rv-code**, independent of selection worker wk-loop. Reviewed 2026-10-30 JST under [the registered rerun brief](s1b_selection_rerun_brief.md), against master `cde4cf881c75cda52c628dc1e472c8d965e44fb4`. The governing selection rule is [S1B preregistration](../../prereg/SLOT-LOOP-v1-S1B.md), lines 77 and 89, with N and schedule replaced by [amendment A1](../../prereg/SLOT-LOOP-v1-S1B-A1.md).

`R = /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587`. Run evidence below is from `R/main/{M1,M4,S1,S4}/s{17,29,43}/`.

**Frozen compute rerun.** Exactly one job was submitted: **3730976.opbs**, one node, `debug-g`, requested walltime `00:05:00`, using the brief's command and science commit `258f5f5a24eae684f640a9fd62679bb643470a76`. The [job log](/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-dev-rerun-20261030-050753-2407459.log) records host `mg0003`, the frozen commit, `--stage DEV`, the registered manifest/release/main-record paths, and output `R/dev-selection-rerun`. The log spans `2026-10-30T05:08:00+09:00` to `05:08:33+09:00`; its final line reports `exit=0`. The [exit marker](/work/xg24i002/x10041/LoopLora/logs/pbs/s1b-dev-rerun-20261030-050753-2407459.log.exit) contains `0`. A subsequent `qstat -H 3730976` reports `FINISH`, one node, elapsed `00:00:34`. The [submission spec](/work/xg24i002/x10041/LoopLora/logs/pbs/specs/s1b-dev-rerun-20261030-050753-2407459.json) preserves the submitted arguments.

The following input identities were independently verified with blake2b-128. The ready release also binds the stated lock, analysis script and full science commit.

| Input | Verified blake2b-128 |
| --- | --- |
| `R/release.json` (`status=ready`, `stage=S1B`) | `d1ccbb7ff4c31fa53abfd7a3cc9ad572` |
| `R/manifest.lock.json` | `2309d5a8ebae91332f32bbd85e5eeb8a` |
| `scripts/analyze_s1b.py` bytes at the frozen science commit | `78f001a232595acce6d38091f0e4895b` |

**Output comparison.** Direct byte comparison of the worker's `R/dev-selection/` and the rerun's `R/dev-selection-rerun/` gives:

| File | Bytes in each | Result | blake2b-128 in both |
| --- | ---: | --- | --- |
| `selection.json` | 771 | Byte-identical | `e3f269db39cd9b431d27e0a5eb528d6e` |
| `source_index.json` | 251 | Byte-identical | `c9e51be2a280962ab47cc6bf335b7336` |

There are no differences to explain. The source index binds the verified release, manifest, frozen script and selection output. The selection contains exactly the twelve expected arm/seed entries and the release's branch/contrast list.

**Independent checks.** A separate [checker](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/dev-selection-rerun/independent_checks.py), written for this review, read the run artifacts directly without importing or executing the analysis implementation or any project selection helper. It used only stdlib and NumPy, with pickle disabled, bounded file sizes and one-thread settings on login host `miyabi-g2`. It completed in approximately 1.1 seconds with exit 0. Its [results](/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1b/a1-8457587/dev-selection-rerun/independent_checks.json) record all twelve record-file digests, every dev score, selected steps, export digests and freeze evidence: `passed=true`, `discrepancies=[]`. Checker blake2b-128: `46ac09ef65ccb7437b6dda9ce43135df`; results blake2b-128: `75d10db7fcf4d25cfeb2587a5be2eb03`.

- **Release, completeness, N and LR — pass, 12/12.** Every `record.json` has `release_blake2b_128=d1ccbb7ff4c31fa53abfd7a3cc9ad572`, the registered lock digest, `status=complete`, `all_finite=true`, `steps=18000`, and `lr=0.003`, equal to the release for every arm. Arm/seed, science-config hash, science commit, read settings and trained depth also match the release. All numeric record values are finite. Each `train_steps.npz` contains exactly 18,000 finite entries in each of `loss`, `grad_norm` and `step_seconds`.
- **Schedule and dev count — pass, 132/132 rows.** Each run has exactly steps `0,1800,3600,5400,7200,9000,10800,12600,14400,16200,18000`. Step 0 is nonselectable; all ten later rows are selectable. The lock's `manifest.sets.dev` registers 1,000 episodes and hops `[1,2,3]`; every row has `n=1000` and exactly those per-hop summaries. All 132 referenced dev sample files have matching recorded file digests, finite numeric arrays, `(1000,3)` NLL/correctness/prediction/answer shapes, and the same ordered 1,000 unique episode IDs. Their stored steps and checkpoint digests match their rows. All records carry the lock's dev-ID digest `17d349b2798bce145a26a34ffc84ad8c`.
- **Selection and stability — pass, 12/12.** Per-hop NLL means recomputed from the stored dev arrays agree with the record summaries within `1e-12`; the equal-weight mean of hops 1–3 agrees with each row's NLL within the same tolerance. Independently minimizing this score over nonzero steps, with the earliest step among candidates within `1e-6` of the minimum, reproduces every selected step and checkpoint. No winning tie occurs: the smallest gap to the next candidate is approximately `0.002358859044`, above `1e-6`. Every selected NLL is strictly below its step-0 NLL.
- **Freeze evidence — pass, 12/12.** Every `record.json` has `freeze.digests.frozen_unchanged=true`, the frozen set `A,K,V,a_beta,a_lambda`, buffers `input_mean,input_std`, and the trainable set `Q,R,U,B` (294,914 frozen and 180,224 trainable parameters). All seven final frozen/buffer digests agree across all twelve runs, as listed below. Every `freeze.support` has `equal=true` and identical `before`/`after` digest `9d20c0f195069058de8e3dcfae38b2f9`.
- **Selected export integrity and uniqueness — pass, 12/12.** Each `selected.pt` was streamed through blake2b-128, without deserialization. Its digest equals the independently selected row's checkpoint digest, `record.selected_checkpoint_blake2b_128`, the selection-map entry, and the separately streamed corresponding `checkpoints/step-NNNNNN.pt`. The twelve digests are distinct.

The following scores are shown to 12 decimal places; selection used unrounded values. The gap is the next-lowest selectable NLL minus the minimum.

| Arm/seed | Selected step | Step-0 NLL | Selected NLL | Next-candidate gap |
| --- | ---: | ---: | ---: | ---: |
| M1/17 | 18,000 | 10.687855265323 | 5.618181126138 | 0.044726657756 |
| M1/29 | 18,000 | 10.687855265323 | 5.618710675836 | 0.007126331046 |
| M1/43 | 18,000 | 10.687855265323 | 5.617209882490 | 0.011674124661 |
| M4/17 | 18,000 | 22.504150654793 | 5.478262632091 | 0.070340662715 |
| M4/29 | 18,000 | 22.504150654793 | 5.506690802579 | 0.004324497495 |
| M4/43 | 16,200 | 22.504150654793 | 5.434525033465 | 0.008151681271 |
| S1/17 | 18,000 | 9.925545333371 | 5.510303842987 | 0.064934491547 |
| S1/29 | 14,400 | 9.925545333371 | 5.521522439079 | 0.002358859044 |
| S1/43 | 16,200 | 9.925545333371 | 5.511117310982 | 0.007133419617 |
| S4/17 | 18,000 | 22.280788203239 | 5.266447814490 | 0.048921886194 |
| S4/29 | 14,400 | 22.280788203239 | 5.282213278130 | 0.003796288461 |
| S4/43 | 18,000 | 22.280788203239 | 5.272094159107 | 0.044465733586 |

Recorded final frozen/buffer digests, common to all twelve runs:

| Tensor/buffer | blake2b-128 |
| --- | --- |
| `A` | `0c566731fe973cd6ddd0ad1bca50908e` |
| `K` | `da979c0681fefbf058d579a1f5bcbb4c` |
| `V` | `f4d888dda54c5b4ef304e0f0691cf608` |
| `a_beta` | `fbfaf77c42f4578e329f9df9c277dbc4` |
| `a_lambda` | `fb77a59a351a8275408931dc1453be4a` |
| `input_mean` | `bd3b0d805ffe99d2500ca9459fc16b66` |
| `input_std` | `05cbee6fbb80bdf8ad1a8b746fe2222e` |

The freeze conclusion is based on the saved training-time freeze assertions and digest evidence. This review did not reload tensors or repeat training; model files were read only as byte streams. No confirmation or unseen samples, graphs or scores were read or generated. No additional compute job, pytest run or login-side torch import was performed.

**Twelve selected checkpoint digests** (`arm/seed: blake2b-128`):

```text
M1/17: ccc6effb3ec8cbd5c7d1413a7a7443bf
M1/29: b1723115c2fb1742a3b8460e3f4395cb
M1/43: e99b56e151f3fa85db7fb337f6265ddc
M4/17: aa8e49d3bb9a9ab15142a029dd5d4ee1
M4/29: 5c8644809202acb95b6b1a273defa41a
M4/43: e74fda78e2211e5b583cc97256d03760
S1/17: 0f3e7d002221df8870135b59b6207a80
S1/29: c919668733bcd33d5e34258eca24ab70
S1/43: f55950e835b0da7ef91c8e9eac6294ca
S4/17: 85f44ffb2253ddb7894028ffd927162d
S4/29: bbd85b3b27941058edb92829cf041831
S4/43: a3220bcd77b86669d592e07f80ab2594
```

The registered independent dev-selection rerun is complete. This report does not open confirmation; the S1B open record remains the orchestrator's subsequent step. No commit was made.
