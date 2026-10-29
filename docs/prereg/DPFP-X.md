# Study DPFP-X: the best DPFP writer at rank 256 (0.6B) and at Qwen3-1.7B-Base (rank 128), pre-registered

Registered 2026-10-29 09:47 JST by the orchestrator at the user's request, before any implementation or run. The request
was: "开始在最好的候选方法上进行r=256的实验 以及1.7B上的学习式写入器诊断". The user made this exception after DPFP's
registered stop. DPFP's rule "no further writer screen follows from this study" (`docs/prereg/DPFP.md`) does not cover it.

## What was known at registration
DPFP (`docs/reports/DPFP.md`, final decision stop):
- val EM at E = 16: D32-3 39.0%, D128-1 58.0%, D128-3 60.8%.
- The flagged E = 8 retrain of D128-3 reached 73.0%.
- R references (WP-L4): 12.0% at r = 32 and 33.5% at r = 128.
- D128-3 at E = 16: normalized read error 0.363; read-value head 51.0% against a clean-value head of 90.6%.
- The curves had flattened by the final checkpoint.
- E = 16 ≤ r, so an exact interpolating store exists at every tested rank (DPFP Amendment A1).

P5L (`reports/decisions/P5L.md`): with an oracle memory the r = 128 readers work on Qwen3-1.7B-Base. B6 is met there:
M1-L4@4 − B4@1 = +25.78 pp.

The best DPFP candidate is D128-3: ν = 3 with L2-normalized φ features. No other learned-writer method is on master.

## Questions
- **A (rank).** Does the best DPFP arm at r = r_mem = 256 reach the writer gate, val EM ≥ 0.80?
- **B (backbone, diagnostic).** At the same rank and settings as D128-3, does the learned writer read better on
  Qwen3-1.7B-Base than on Qwen3-0.6B-Base?

## Design
Everything not listed here is as DPFP's T1 and D128-3:
- E = 16, n_rel 4, head sharing 0.3, seed 17, layer-14 `o_proj`, L = 1;
- batch 32 / Q 4, lr 3e-3 (no new search);
- base objective R with the raw-read loss λ = 1;
- the DPFP change with ν = 3 and L2 normalization;
- validation on 1,000 episodes at eval seed 4242, `scan_final`, `train.prefetch_workers: 2`.

Arms:
- **X256**: Qwen3-0.6B-Base (pinned revision as DPFP), r = r_mem = 256, ν = 3 (key dimension 1,536).
- **X17**: `Qwen/Qwen3-1.7B-Base` @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (layer 14 `o_proj` 2048→2048), r = r_mem =
  128, ν = 3 (key dimension 768).

Steps follow DPFP's rule:
- The step count is the largest multiple of 5 whose 10%-smoke-measured time × 1.5 fits one 1-node 2 h job.
- It is fixed in a release before training.
- val EM is recorded at five evenly spaced checkpoints.
- Step counts may differ between arms and from D128-3's 22,370. The report also gives val EM of each arm and of D128-3
  at matched fractions of training, as a descriptive table.

The DPFP-off path stays bitwise unchanged: the three DPFP configs keep their hashes, and the default backbone stays
0.6B. This is shown by a test on compute.

Descriptive diagnostics at the final checkpoint follow the DPFP report:
- key Gram RMS and mean max |cos| in the original and φ space;
- normalized read error;
- clean-value and read-value head accuracy;
- value SIR and parallel interference;
- EM/NLL with zeroed and shuffled memory (`scripts/audit_dpfp.py`).

## Decision rule (stated before results)
- **X256** reaches val EM ≥ 0.80 at its final checkpoint: it is a writer candidate, flagged.
  - The flag is there because this study follows a registered stop and the validation set was used for DPFP's selection.
  - No E = 8 retrain follows.
  - Any reader study with it needs its own registration. The P1-L oracle reference exists only at r = 32 and r = 128,
    so that registration must say how it compares readers at r = 256.
  - Below 0.80: no writer at r = 256.
- **X17** is a diagnostic.
  - Its val EM, read error and read-value head are compared with D128-3 at 0.6B (60.8%, 0.363, 51.0%), descriptively,
    with one seed.
  - With 1,000 episodes the difference of two EMs has a standard error of about 2.2 pp. A difference of at least 5 pp
    is called higher or lower; a smaller one is called not distinguished.
  - If X17 reaches ≥ 0.80, it is also a flagged writer candidate, on Qwen3-1.7B-Base at r = 128. A reader study with it
    needs its own registration; P5L supplies the oracle reference at that backbone and rank.
- **Neither reaches 0.80:** the learned-writer line stays stopped. The paper adds both arms as descriptive rows next
  to DPFP.

Orchestrator's expectation, recorded before any run and not a gate: X256 between 55% and 70%, because the store's
capacity is not this regime's limit (E = 16 ≤ r). X17 within a few points of D128-3.

## Limits and budget
- Validation only, seed 17. No test or confirmation data is read. No reader or joint runs.
- Each science job ≤ 1 node × 2 h, submitted with `scripts/submit.py --scale-up docs/prereg/DPFP-X.md`. The
  justification is DPFP's arms, which used 78–82 min of the 2 h cap.
- Tests, smokes, reload checks and audits ≤ 0.6 node·h in total. Projected total about 3.5–4.5 node·h.
- Executed by sn-llm, a Claude Sonnet-5.5 subagent, on wp/llm. Reports go to `docs/reports/DPFP-X.md` and
  `docs/reports/DPFP-X-state.md`.
