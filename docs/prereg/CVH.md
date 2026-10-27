# CVH (F9) — continuation-value halting heads on frozen symbolic readers, pre-registration draft

Draft by sn-sym, 2026-10-27, on the login node: no run, no evaluation, no result read. Registered at the time the
orchestrator commits this file to master, before any run below. Known then: P4-S-L (`docs/reports/P4S-L.md`,
`docs/prereg/P4S-L.md`): a correctness-label halting head per frozen r = 32 M1 reader qualified for 0 of the 6
primary families; every family kept its reach gain but lost hop 1 (M1: oracle_random −15.70 pp, joint_fact
−33.90 pp, t1_frozen_fact −21.90 pp against L1; reach gains +32.51, +21.58, +9.44 pp); the mean number of executed
updates was 12.74–15.99 (the fallback scans all 16 states) and learned inference was 6.3–8.2 times slower than the
fixed 16-update read, with no measured speedup. Its threshold was chosen by validation accuracy without any hop-1
constraint, and its label was "the candidate argmax at step s is correct". Neither a hop-1-constrained threshold
for that head nor a head trained on the value of reading again has been run (rv-astra proposal 3).

## Question
Can a halting head that sees only features of the current state stop each query without losing hop 1 and while
keeping the multi-hop gain, when (A1) the correctness head's threshold is chosen under a hop-1 constraint, or (A2)
the head is trained to predict whether any later read would still lower the answer's loss by a margin?

## Design (frozen readers; only the A2 heads are trained)
- Readers (no retraining): the nine r = r_mem = 32 M1 checkpoints of P4-S-L, writers `oracle_random`,
  `joint_fact` and `t1_frozen_fact` (families `oracle_random/r32/M1`, `joint_fact/r32/M1`,
  `t1_frozen_fact/r32/M1`) × seeds 17/29/43. Paths and blake2b seals of the nine readers, of their nine exported
  P4-S-L heads and the three locked P4-S-L thresholds (one per family: 0.25, 0.45, 0.25) are read from the committed
  `reports/p4s-l/` files and sealed in the design lock (the lock checks that every head file matches its P4-S-L
  record). `joint` and the `M1-lin` readers are not used.
- States and features: the reader run to L = 16 per episode; per step s the post-update state z^s and the 51
  P4-S-L features [z^s; ‖z^s‖; ‖z^s − z^{s−1}‖/‖z^{s−1}‖; read-norm ratio; one-hot(s)]. The head sees only these.
- Arms per reader family:
  - **A0** (reference, not a decision arm): the existing P4-S-L head at its locked P4-S-L threshold.
  - **A1**: the same head; threshold chosen on validation under the hop-1 constraint below.
  - **A2**: a new head of the same size (51 → 64 ReLU → 1, 3393 parameters) trained with the continuation label;
    threshold chosen under the same constraint.
- A2 label of step s: y_s = 1 ("stop") if no later state s' > s has a candidate cross-entropy of the true answer
  lower than that of state s by at least **0.25 nats**; step 16 always stops. The cross-entropy is over the
  episode's candidate entities, as in the reader's decision.
- A2 training: exactly the P4-S-L recipe — split `train`, generator seed 31337, 20000 episodes, one query per hop
  1–8, all 16 states; BCE; Adam lr 1e-3; batch 256 whole episodes × 16 steps; 2000 steps; fixed, no tuning; head
  initialisation and episode sampler seeded by `derive_seed(seed, "cvh", family, …)`. A test shows that the same fit
  with the P4-S-L label reproduces the P4-S-L head fit exactly.
- Inference (all arms): stop at the first s ∈ [1, 16] with p_s ≥ θ, else at argmax_s p_s (earliest); the fallback
  executes all 16 updates. θ ∈ {0.05, 0.10, …, 0.95}.
- Validation (split `val`, eval seed 4242, 2000 episodes, one query per hop 1–8): per family and arm, the table of
  θ → (validation accuracy over hops 1–8 and the three seeds, hop-1 Δ against L1, mean selected and executed steps).
  - A0: the locked P4-S-L θ; its recomputed validation accuracy must equal the P4-S-L record within 1e-3, otherwise
    the confirmation stage refuses to run (the pipeline would not be the registered one).
  - A1 and A2: feasible θ are those with validation hop-1 Δ ≥ −1 pp (seed-pooled); choose the highest validation
    accuracy, then fewer chosen steps, then the smaller θ. No feasible θ: the arm is infeasible (registered failure).
  - The choices are written to `theta.json`, which the orchestrator commits (the theta lock) before confirmation.
- Confirmation (split `test`, **eval seed 999005**, 2000 episodes, one query per hop 1–8; never evaluated before):
  the three arms' heads and the fixed L = 1, 4, 16 references on the same episodes. A family is traced on this set
  only if A1 or A2 is feasible for it; the stage refuses to run if none is.
- Latency: the P4-S-L protocol (`scripts/run_p4s_learned.py` benchmark: prepared batch of 64, 5 warm-ups, 20
  repeats) for each feasible arm and the fixed 16-update read of the same batches; end-to-end wall time is the primary
  clock, CUDA-event time is reported.
- Order of operations: `lock` → the orchestrator commits the design → `smoke` (seed-17 readers; 2000 training
  episodes, 200 steps, 200 validation episodes; no confirmation data; fixes the walltimes; its thresholds are never
  used) → `train` (A2 heads, validation traces, θ choices) → the orchestrator commits `theta.json` → `confirm` →
  `scripts/analyze_cvh.py`, which audits both Git locks, the seals and the structure before any statistic.
- Statistics: pooled paired episode bootstrap over the three training seeds (whole-episode joint resampling), 2000
  draws, seed 2718, confidence 1 − 0.05/6 (three families × two decision arms); per-seed values reported.

## Endpoints
- **Registered, per family and decision arm (A1, A2), Δ = arm − fixed L1:** reach gain on hops {2, 3} (point
  estimate and interval); hop-1 Δ (point estimate); mean executed updates per query (mean over queries and seeds).
  An arm **passes** if reach gain ≥ 3 pp with the interval above 0, hop-1 Δ ≥ −1 pp and mean executed updates ≤ 8.
- **Latency claim**, per feasible arm: only if the arm's measured end-to-end wall time per batch is lower than the
  fixed 16-update read's.
- Reported, not gating: A0 on the same endpoints; reach gain against L4 and L16; accuracy by hop group {1}, {2, 3},
  {4, 5}, {6, 7, 8} and overall for L1/L4/L16/A0/A1/A2 with 95% intervals; selected and executed steps; the
  validation tables; per-step label rates of A2; the timings.

## Decision rule (as in plan F9)
- Per family and arm: hops 2–3 at least 3 pp above L1 (CI > 0), hop-1 loss ≤ 1 pp, mean executed depth ≤ 8.
- A latency improvement is claimed only if the measured end-to-end stopping time beats fixed L16.
- If no validation threshold preserves hop 1, the arm fails and the confirmation set is not looked at for it.

## Use of the results (stated before results)
- An arm that passes: "a halting head with <label>, <threshold rule>, repairs overshoot for <family> on these frozen
  readers" enters the paper with its interval; the label-free and P4-S-L failures stay reported. A latency claim is
  made only where the registered latency condition holds.
- No arm passes: the paper states that neither label-free rules, nor a correctness-label head (also with a
  hop-1-constrained threshold), nor a continuation-value head repairs overshoot in these readers, and that
  fixed-depth training (B8, M1-L4; P2-S-F1) remains the remedy found. No follow-up halting study is registered.
- No gate depends on CVH.

## Flagged interpretations and additions (the plan does not fix these)
1. The plan lists three heads and one rule. A0 is treated as a reference (its failure is known from P4-S-L) and the
   decision rule is applied to A1 and A2; hence six comparisons in the interval level.
2. The margin is not given in the plan: **0.25 nats** is fixed here before any data (on 14 candidates the chance
   cross-entropy is 2.64 nats). It is not tuned. A degenerate label (for example, every state labelled "stop") would
   be reported as a result, not repaired.
3. The hop-1 constraint is applied to the threshold choice of both A1 and A2, with the 1 pp tolerance of the decision
   rule evaluated on validation (seed-pooled hop-1 Δ against L1).
4. "Mean executed depth" is read as the mean number of executed updates per query, which is the P4-S-L cost measure
   (the fallback executes all 16); this is stricter than the mean selected step, which is also reported.
5. Failure handling is per arm and family: a family whose A1 and A2 are both infeasible is not traced on the
   confirmation set; an infeasible arm's confirmation statistics are never computed. If the A2 head of a family is
   feasible but A1 is not, the family's traces are saved for both heads but only the feasible arm is analysed.
6. The validation set (seed 4242) and the training set are the P4-S-L ones; the P4-S-L thresholds were chosen on that
   validation set, which is why A1 and A2 are chosen on it too. Only the confirmation set is new.
7. Wall time is the primary latency clock (the plan asks for end-to-end stopping time); CUDA time is secondary.

## Limits
- Frozen readers, three training seeds: intervals condition on those seeds. Heads see only current-state features;
  useful later states that are locally unpredictable cannot be exploited (rv-astra proposal 3, risk). The two
  `t1_frozen_fact` and `joint_fact` families share a writer family; they are not independent replications.
- Cost: see `docs/reports/WP-S9-state.md` (estimate 0.1 node·h, upper bound 0.3, from the P4-S-L ledger
  `reports/p4s-l/jobs.json`: 0.144 node·h in total, of which the job that trained and validated the 24 heads took
  351 s and the job for confirmation, timing and analysis 72 s). Each job is one node and at most 1 h.
- Code: `src/looplora/sym/cvh.py`, `scripts/run_cvh.py`, `scripts/analyze_cvh.py`; tests `tests/test_cvh.py`,
  `tests/test_cvh_scripts.py` (written, to be run on a compute node before the lock).
