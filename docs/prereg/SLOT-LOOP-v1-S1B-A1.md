# SLOT-LOOP-v1 S1B, amendment A1: disposition of the calibration stop

**Status: registered 2026-10-30 01:16 JST by the orchestrator.** This amends `docs/prereg/SLOT-LOOP-v1-S1B.md` (blake2b-128 `6f2e776f2c6cfde131380fb343854784`), which stays unchanged. Where they differ, this file governs.

**Data seen so far.** Only S1B calibration and the non-sealed S1A T1 pilot shard have been read. No S1B pilot, dev, confirmation or unseen data were generated or read.

## 1. What stopped, and why

Calibration job 3729493 (lock bd4e24f, branch advance) hit registration decision 3: the calibrated Matrix-L1 parent's hop-1 EM on S1B calibration was **5.1 %** (104/2048), below 10 %.

Diagnosis job 3729676 (`scripts/diagnose_s1b_transfer.py`, output `runs/slot-loop-v1/s1b/diagnosis-6855048/diagnosis.json`) ran on calibration hop 1 only. Final-token key top-1 is the query's top-1 key at the final question token.

| condition | EM | NLL | final-token key top-1 |
| --- | --- | --- | --- |
| T1, S1B code path, Matrix (equals the recorded S1A pilot step 0, per sample) | 73.3 % | 1.42 | 89.4 % |
| T2 chain-end wording, Matrix | 5.1 % | 9.90 | 14.8 % |
| same T2 graphs, T1 wording, Matrix | 31.5 % | 4.75 | 59.5 % |
| T2, oracle read at the final token (S0 clean_final) | 79.5 % | 0.83 | — |
| T2, written value of the true fact decoded by the value head | 83.6 % | — | — |
| T2, memory off | 0.2 % | 13.26 | — |

**Reading.**
- **What transfers.** The S-RE writer's values and the upper read path transfer to T2: an oracle read gives about 80 % EM.
- **What fails.** The query formed from the chain-end question wording finds the right fact at the final token 15 % of the time. With T1 wording on the same graphs it does so 60 % of the time.
- **Cause.** The stop comes from the new question wording acting on the untrained query projection Q. It is not a code defect, and the writer has not broken. Q is one of the matrices every S1B arm trains.

## 2. Amendments

1. **The stop is cleared** for this calibration (blake2b-128 `6e4f7346376cb010820a84832f6d7d91`, recorded in the diagnosis) and lock lineage. The calibration results stay in the record as descriptive evidence. The sanity rule of decision 3 no longer gates S1B.
2. **Read settings come from S1A, not from T2 calibration.**
   - Every S1B arm uses S1A's released reads (ready release `523d11d57a08a2491893ebe7cedc86a7`, field `arms.*.read`): Matrix g = 1; SoftSlot g = 1, τ = 0.1.
   - S1B capacity is 12, and the scope is all valid question tokens, as registered.
   - Reason: with a query that misses the fact 85 % of the time, the NLL rule rewards the flattest read. It chose SoftSlot τ = 0.3, whose hop-1 EM was 1.3 %, below τ = 0.03's 8.9 %. A setting chosen that way would bias C_storage against SoftSlot.
   - The S1A values were calibrated where the read works (S0, T1) and confirmed under training (S1A ADVANCE).
   - L1 and L4 of a backend still share one setting for the whole study, and nothing is retuned.
3. **N = 18,000 updates** for every main arm and seed, replacing 6,000.
   - Dev runs at step 0 (not selectable) and at steps 1,800, 3,600, …, 18,000. The selection rule is unchanged: lowest mean hop 1–3 dev NLL, ties within 1e-6 to the earlier step.
   - Each seed's training stream is train [0, 576,000) under its registered base seed. The LR pilot is unchanged: 200 steps, the same three LRs and its stream.
   - Reason: S1B starts far from a trained T2 reader. P1-L's M1-L4 on this T2 layout (`runs/p1l/main/r128/main/M1-s{17,29}`) was still improving at 17,000 steps: training accuracy was 5–6 % over steps 4,001–6,000 and 29–38 % over steps 16,001–18,000. Dev-NLL selection at N/10 makes a longer budget safe; it costs only compute.
   - Each job stays at or below 16 node·h, with walltime from the pilot's measured step time × 1.5. If one run cannot fit, STOP; do not shrink N.
   - Conditional mechanism controls, if triggered, use the same N.
4. **Unchanged:** everything else in the S1B addendum, including the branch (advance: M1, M4, S1, S4), sets, seeds, analysis, contrasts and continuation rule.

## 3. Consequences for the code and the process

- These are `src/` changes, so they invalidate lock bd4e24f:
  - read settings from the S1A release by digest;
  - this file's digest bound like the addendum's;
  - N and the dev schedule;
  - the larger training window, with its disjointness checks;
  - decision 3's sanity rule removed from the gating path.
- **Tests and review.** Compute tests check the change. Per the user's rule (2026-10-29), it gets no new cross-review round.
- **Then:**
  1. a new real-parent preflight and lock;
  2. smoke and reload with the S1A reads;
  3. the LR pilot;
  4. the ready release, then STOP for `approval.json`.
- **Calibration.** The step is not repeated. The release cites this file and the stopped calibration as descriptive evidence.
