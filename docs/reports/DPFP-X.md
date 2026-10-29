# DPFP-X report: the best DPFP writer at r = 256 and on Qwen3-1.7B-Base (WP-L8)

Registered design: `docs/prereg/DPFP-X.md` (2026-10-29 09:47 JST, blake2b-128 `54334bca2788c66f5133b90dae1a3d26`), run 2026-10-29 JST by sn-llm
at science commit `9214790db52b977029edf6c1dbd17b1519982aad` (wp/llm), schedule release `reports/p1l-dpfp-x/release.json` (61e2a25).
Arms: **X256** (Qwen3-0.6B-Base, r = r_mem = 256, nu = 3, key dimension 1,536) and **X17** (`Qwen/Qwen3-1.7B-Base` @
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, layer-14 `o_proj` 2048 -> 2048, r = r_mem = 128, nu = 3, key dimension 768). Everything else is as D128-3 (E = 16, seed 17,
layer 14, L = 1, batch 32 / Q 4, lr 3e-3, objective R with lambda = 1, 1,000 validation episodes at eval seed 4242, `prefetch_workers` 2). One 1-node 2 h
job per arm (`--scale-up docs/prereg/DPFP-X.md`). Records, the audit and the decision are in `reports/p1l-dpfp-x/`; every table below is printed by
`python3 scripts/analyze_dpfp_x.py` from them (D128-3 is read from `reports/p1l-dpfp/`).

## Result and registered decision
**Neither arm reaches val EM 0.80. The learned-writer line stays stopped; no writer candidate is flagged.** (`dpfp_run.choose_x` on the two
records, job 3723597, `reports/p1l-dpfp-x/decision.json`; it equals the hand computation.)

| arm | rank | nu | key dim | backbone | steps | correct of 1,000 | val EM % | >= 0.80 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| X256 | 256 | 3 | 1536 | default 0.6B | 22780 | 662 | 66.2 | no |
| X17 | 128 | 3 | 768 | Qwen3-1.7B-Base | 21980 | 593 | 59.3 | no |
| D128-3 (DPFP, reference) | 128 | 3 | 768 | default 0.6B | 22370 | 608 | 60.8 | no |

- **Question A (rank 256):** X256 is 13.8 pp below the gate. Registered outcome: no writer at r = 256. No E = 8 retrain follows (registered).
- **Question B (backbone, diagnostic):** X17 minus D128-3 is -1.5 pp of val EM (593 against 608 of 1,000). The registered threshold is 5 pp, and the standard error
  of a difference of two EMs at 1,000 episodes is about 2.2 pp: **not distinguished**. The read error (0.377 against 0.363) and the read-value head
  (47.3 % against 51.0 %) are in the table below; the registration gives them no threshold, so they are reported without a verdict.
- The orchestrator's expectations (recorded before any run, not a gate): X256 between 55 % and 70 % (66.2 %) and X17 within a few points of D128-3 (-1.5 pp).

Every run completed its released steps (status complete, not a smoke, source clean, `actual_steps` = release, `config_hash` = the release's frozen hash,
`git_commit` = the science commit; exit 0, no error in the logs). No test or confirmation data was read.

## The curves are still rising at the final checkpoint (descriptive fact)
In DPFP the curves had flattened by the final checkpoint (D128-3 gained 0.6 pp over the last interval). Here both arms were still rising at the end of their
budget: **X256 gained 3.5 pp over the last interval (62.7 % to 66.2 %, 627 to 662 correct), X17 gained 2.3 pp (57.0 % to 59.3 %, 570 to 593 correct)**. D128-3 gained 0.6 pp (60.2 % to 60.8 %).
The registered decision uses the final checkpoint and is unchanged by this. No number is extrapolated: the study has no longer run, so it cannot say where
either curve ends. One seed, 1,000 validation episodes (standard error about 1.5 pp per EM, 2.2 pp per difference).

## Curves at the five evenly spaced checkpoints (20, 40, 60, 80 and 100 % of each arm's steps, so they are matched fractions of training)

| arm | 20 % | 40 % | 60 % | 80 % | 100 % (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| X256 | 4556: 41.3 | 9112: 52.6 | 13668: 58.7 | 18224: 62.7 | 22780: 66.2 |
| X17 | 4396: 31.1 | 8792: 41.3 | 13188: 53.8 | 17584: 57.0 | 21980: 59.3 |
| D128-3 | 4474: 36.0 | 8948: 50.2 | 13422: 52.3 | 17896: 60.2 | 22370: 60.8 |

Difference to D128-3 at the same fraction of training (pp of val EM):

| arm | 20 % | 40 % | 60 % | 80 % | 100 % (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| X256 | +5.3 | +2.4 | +6.4 | +2.5 | +5.4 |
| X17 | -4.9 | -8.9 | +1.5 | -3.2 | -1.5 |

Descriptive reading: X256 is above D128-3 at all five fractions (+2.4 to +6.4 pp; +5.4 pp at the end, about 2.5 standard errors of a difference), and its
final EM is still 13.8 pp from the gate. X17 is below D128-3 at the first, second and fourth fraction and above it at the third; its final difference is within
the standard error. The step counts differ slightly (22,780, 21,980 and 22,370), which the matched fractions absorb.

## Diagnostics at the final checkpoint (descriptive; one seed)

| metric | X256 | X17 | D128-3 |
| --- | ---: | ---: | ---: |
| val EM % | 66.2 | 59.3 | 60.8 |
| NLL | 1.753 | 2.100 | 2.129 |
| normalized read error | 0.397 | 0.377 | 0.363 |
| clean-value head acc % | 92.9 | 85.8 | 90.6 |
| read-value head acc % | 50.7 | 47.3 | 51.0 |
| value SIR | 6.24 | 7.68 | 8.00 |
| key top-1 % | 84.5 | 86.2 | 84.9 |
| key margin | 0.229 | 0.249 | 0.252 |
| key Gram RMS (phi space) | 0.169 | 0.169 | 0.190 |
| key Gram RMS (original space) | 0.302 | 0.319 | 0.321 |
| mean max abs cos (phi space) | 0.653 | 0.657 | 0.692 |
| mean max abs cos (original space) | 0.788 | 0.816 | 0.829 |
| mean pairwise cosine (as logged) | 0.119 | 0.116 | 0.140 |
| parallel interference | 0.196 | 0.235 | 0.270 |
| query norm | 1.0 | 1.0 | 1.0 |

Normalized read error and read-value head accuracy over training (normalized read error / read-value head %):

| arm | 20 % | 40 % | 60 % | 80 % | 100 % (final) |
| --- | ---: | ---: | ---: | ---: | ---: |
| X256 | 0.468 / 35.2 | 0.436 / 45.1 | 0.415 / 50.1 | 0.407 / 52.9 | 0.397 / 50.7 |
| X17 | 0.445 / 32.1 | 0.444 / 35.0 | 0.400 / 42.8 | 0.374 / 46.1 | 0.377 / 47.3 |
| D128-3 | 0.444 / 34.8 | 0.413 / 42.5 | 0.423 / 43.6 | 0.362 / 51.0 | 0.363 / 51.0 |

Reading (descriptive): the read is still distorted at both new arms (read error 0.38 to 0.40, read-value head about 47 to 51 %, against a clean-value head of 86 to 93 %). At r = 256 the
read error and the read-value head are not better than D128-3's (0.397 against 0.363; 50.7 % against 51.0 %), so the higher EM of X256 is not accompanied by a lower read error in
these columns; this report does not explain it. Key top-1 is about 85 % for all three arms, and the keys' Gram overlap in the original space is similar (RMS 0.30 to 0.32). X17's
clean-value head is lower than D128-3's (85.8 % against 90.6 %). The query norm is 1.0 by construction (unit queries), as in DPFP.

## Memory interventions at the final checkpoints (registered descriptive diagnostic; job 3723595, `reports/p1l-dpfp-x/audit.json`)
`scripts/audit_dpfp.py` (unchanged from DPFP; its tiny-model test now also takes an X256 export) reloaded each final export and scored the same 1,000 validation episodes with the deployed memory, a
zeroed memory and the memory of another episode (a one-step roll of a seeded permutation). The original condition reproduces the record's final validation exactly for both arms (same correct
count; repeated forwards bit-identical; optimizer steps 0).

| arm | original EM % (NLL) | zero memory EM % (NLL) | shuffled memory EM % (NLL) |
| --- | ---: | ---: | ---: |
| X256 | 66.2 (1.753) | 0.0 (12.030) | 0.2 (13.568) |
| X17 | 59.3 (2.100) | 0.4 (10.240) | 0.1 (12.377) |

Reading (descriptive): as for the DPFP arms, the answers depend entirely on the episode's own memory (at most 0.4 % with zero or another episode's memory; NLL 10.2 to 13.6 against 1.8 to 2.1).
This says the EM is carried by the memory; it does not say the read is accurate.

## Compute and memory
| arm | job | wall | steady step | slowest 1,000-episode eval | peak CUDA allocated | state per episode | inference parameters |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| X256 | 3722795 | 76.4 min training (job 76.8) | 0.199 s | 7.3 s | 7.82 GiB | 1.5 MiB ([256, 1536]) | 1,114,114 |
| X17 | 3722796 | 81.6 min training (job 82.0) | 0.220 s | 7.9 s | 6.90 GiB | 0.375 MiB ([128, 768]) | 606,210 |
| D128-3 (DPFP) | 3721089 | 81.1 min | 0.215 s | 7.5 s | 4.08 GiB | 0.375 MiB | 475,138 |

The inference parameters of X17 (606,210) are the r = 128 adapter on the 2048 -> 2048 `o_proj` of the 1.7B model (shapes come from the loaded model).

## Not done
No test or confirmation data was read. No reader or joint run, no E = 8 retrain (registered: none for DPFP-X), no second seed, no longer run. Nothing remains to run for this study. Per the
decision rule the paper adds both arms as descriptive rows next to DPFP.

## Jobs and cost (node·h = wall hours of a one-node job)
Step 1 (code, tests, smokes, reload checks) and step 2 (science, audit, decision):

| job | purpose | wall | node·h | outcome |
| --- | --- | ---: | ---: | --- |
| 3722584 `t-dpfp-x` | tests at 803076d | 3m18s | 0.055 | 3 failures, all test defects (see the state doc) |
| 3722600 `hash-master` | config hashes of every WP-L4 and DPFP config at master 21254c9 | 19 s | 0.005 | the literals pinned in the test |
| 3722607 `t-dpfp-x2` | tests at 3f150a8 | 3m18s | 0.055 | 1 failure, a test defect |
| 3722634 `t-dpfp-x3` | `tests/test_llm_dpfp_x.py` at 9214790 | 54 s | 0.015 | 14 passed |
| 3722649 `sm-dpfp-X256` | smoke, 1,800 steps | 6m54s | 0.115 | complete; step 0.206 s; peak 8.39 GB |
| 3722650 `sm-dpfp-X17` | smoke, 1,800 steps | 7m09s | 0.119 | complete; step 0.213 s; peak 7.41 GB |
| 3722730 `hash-dpfp-x` | science config hashes at the released steps | 21 s | 0.006 | frozen in the release |
| 3722737, 3722738 | reload checks X256, X17 | 24 s, 25 s | 0.014 | config equal, reload identical, 0 mismatches |
| 3722795 `dpfp-X256` | science, 22,780 steps | 76.8 min | 1.281 | complete, 662 of 1,000 |
| 3722796 `dpfp-X17` | science, 21,980 steps | 82.0 min | 1.366 | complete, 593 of 1,000 |
| 3723595 `audit-dpfp-x` | zero / shuffled-memory audit | 1m00s | 0.017 | original reproduces both records |
| 3723597 `decide-dpfp-x` | `choose_x` on the records | 19 s | 0.005 | no writer; X17 not distinguished |

- **Step 1** (tests, hash jobs, smokes, reload checks): **0.384 node·h**. With the audit and the decision job: **0.406**, inside the registered cap of 0.6 for tests, smokes, reload checks and audits.
- **Step 2** (science 2.647, audit 0.017, decision 0.005): **2.669 node·h**.
- **Total step 1 + step 2: 3.053 node·h** (projected 3.5 to 4.5; the runs used 77 and 82 of the 120 minutes).
- Two smoke jobs submitted with a relative `--out` (3722643, 3722644) were deleted with `qdel` before they started: no log, no cost.

## Deviations and notes
- The smokes ran 1,800 steps, 7.9 % (X256) and 8.2 % (X17) of the released steps, not 10 %, to stay inside the 0.6 node·h cap. The step rule is unchanged; the fraction is recorded in the release
  (`smoke_fraction_of_fixed_steps`). The orchestrator accepted it.
- Four test failures in two runs were test defects, not product defects (state doc); no assertion was weakened.
- The config's `writer_fix.value_head` string (`training_only_{rank}_to_1024_affine`, a label that follows the existing rank rule) says 1024 for X17 as for D128-3; it is a label only (the 1.7B
  `o_proj` output is 2048).
