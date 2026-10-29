# Study B8W — B8 warm start from M1-L4: is sharing an optimization scaffold? (G3; DRAFT v1, 2026-10-29)

Status: **draft**, written by sn-p1l (Claude Sonnet 5.5) for the orchestrator's review (`docs/tasks/WP-L10_b8_warm_start.md`,
`plan/followup_2026-10-29.md` G3; astra proposal 3, fable R2). It is not registered. Nothing below depends on a B8W
result: none exists, no compute job was submitted and no test episode was generated or read. Everything quoted comes
from P1-L records and training logs (training data, no test data) and from the code at master `e43ccda`.

## Question and what the answer can and cannot say
`UnsharedAdapter` (B8, four unshared reads) contains `IterativeQueryAdapter` (M1-L4, one shared read iterated four times):
copying the shared R/U/Q into each of B8's four steps gives the same function. P1-L r = 128 and P5L found B8 far below
M1-L4 from a random start (hops 2-3: 6.34% vs 26.91% at 0.6B; G1-L fails at both scales). **Does B8 keep the M1-L4 solution
when it starts there and its steps are untied?**
- "Keeps": the untied parameterization can hold and keep training the solution for N steps; the exact containment is not
  a capacity limit for this solution.
- "Loses": untying destabilizes the solution (fragility under untied updates at this lr and length); it is not a lack of
  expressivity, since the equality gate shows the function is representable.
- Either way: B8 gains **no equal-parameter status** (B8 has 622,594 adapter parameters, 589,824 trainable, against M1-L4's
  475,138 and 442,368: +31.0% and +33.3%); **G1-L stays failed**; and **nothing is said about why B8 failed from a random
  start** (optimization, update allocation and the inductive bias of sharing are not separated by this design).

## Design (fixed unless flagged under "Flags")
- **Backbone and condition**: Qwen3-0.6B-Base (pinned), layer 14 `o_proj`, r = r_mem = 128, `oracle_random` (oracle seed 17,
  one memory for every run), memory-only protocol, primary setting, task `t2_chain_end` (train hops {1, 2, 3}).
- **Starting points**: the registered P1-L r = 128 M1-L4 checkpoints, seeds 17/29/43
  (`runs/p1l/main/r128/main/M1-s{17,29,43}/M1-L4-s{17,29,43}/checkpoint.pt`; blake2b `c90d23ff…`, `59e293ee…`, `37a7baf0…`,
  as in `reports/p1l/r128/run-manifest.json`; trained S = 17,000, lr 0.003, source `1d1bbfd`). The lock file
  `reports/b8w/checkpoint-lock.json` records the full hashes before any continuation (see "Order").
- **Conversion** (`convert_m1_to_b8`, new, pure function on a checkpoint record; the checkpoint is not modified):
  `adapter` keys `A, B, K, V, a_lambda, a_beta` are copied unchanged; `R`, `U`, `Q` are each cloned into `R.i`, `U.i`, `Q.i`
  for i = 0..3 (the `nn.ParameterList` names of `UnsharedAdapter`); the config is the M1-L4 config with `variant: "B8"`
  (`unshared_steps = 4`, `depths = [4]` follow from the variant exactly as in P1-L's B8 configs, which differ from M1-L4's
  only in `name` and `variant`) and a new `name`; the oracle writer record and signature are copied (the writer does not
  depend on the reader). The result is saved as a normal format-1 checkpoint and loaded through `load_checkpoint`
  (`strict=True`), so the converted B8 is a registered `UnsharedAdapter` and no new adapter class exists. K, V and the gates
  are frozen under the oracle writer and unused there; they are copied anyway so every tensor of the state dict is equal.
- **Equality gate** (per seed, before any continuation step; if it fails for any seed, STOP and report):
  - *Data*: the validation split (never test), 64 episodes from episode 0 with the P1-L evaluation seed 4242 and the same
    batching as `evaluate` (batch 16), at L = 4.
  - *Compared*: the answer-position logits that `LLMExperiment.score` returns, for the original M1-L4 and for the converted
    B8, with deterministic algorithms (`configure_numerics`), the same process and device, and the same backbone object.
  - *Dtype and tolerance*: logits are compared in the dtype `score` returns (the backbone's logits dtype, bf16 on the
    pinned stack); the adapter itself is fp32. **Tolerance 0**: `torch.equal` on every logits tensor, plus the predicted
    token equal everywhere. The max absolute difference (computed in fp32) is recorded and must be 0.0. The reader loops
    are the same sequence of fp32 `F.linear`/ReLU operations with equal operands on identical shapes, so a bitwise
    outcome is expected; a nonzero difference is a failure to explain, not a tolerance to widen.
  - *Control*: the original M1-L4 scored twice must be bitwise equal (it tests that the comparison is deterministic).
  - *Adapter-level check* (fp32, on the device): `read(u, M, 4)` of both adapters on a fixed random `u` and a fixed random
    memory state: `torch.equal`.
  - *Training-level check* (unit test, tiny adapters on CPU): the gradient of the untied copy w.r.t. each step's R/U/Q summed
    over the four steps equals the shared gradient (fp32, `allclose` rtol 1e-5, atol 1e-6; this is a different quantity from
    the forward gate, so it has a tolerance). It documents that the two arms start from the same function and a consistent
    gradient, not that they stay together.
- **Arms** (both, per seed, N continuation steps; same data, lr, schedule, clipping, decay, batch size and a freshly built
  `AdamW` per arm):
  - **shared**: the original M1-L4 checkpoint, loaded with `load_checkpoint`, continues training;
  - **untied**: the converted B8 continues training.
  - AdamW lr 0.003 (the registered M1-L4 lock value, `reports/p1l/r128/lr-S-lock.json`; B8's own lock value is also 0.003),
    weight decay 0.01, clip 1.0 on the trainable adapter parameters, batch 64, **the 20-step linear warmup restarted** from
    the new optimizer, constant lr after it, depths [4], `prefetch_workers` 2. Trainable: A, B, R, U, Q (and the copies).
  - **Data stream**: the P1-L training stream of that seed *continued*: `data_seed = derive_seed(seed, "llm", "train-data")`
    (P1-L's value), `BatchStream(start_step = 17,000, steps = 17,000 + N)`, i.e. train-split episodes 17,000·64 … (17,000+N)·64 − 1
    that no P1-L training step used. Both arms read exactly the same batches. This needs one optional config key
    `train.data_start_step` (absent = 0; outside `DEFAULTS` so resolved configs without it keep their hashes) and `_fit`
    passing it to `BatchStream`; a test shows the off path is bitwise unchanged (same batches, same resolved config hash).
  - Each arm ends with a normal format-1 checkpoint and a training record that names the starting checkpoint hash, N, the
    stream start and the source commit.
- **Continuation length N = 2,000 steps** (11.8% of the original 17,000), with reasons:
  1. It is astra's proposal and the plan's G3 budget: six continuations of 2,000 steps measured at 0.194-0.199 s/step cost
     0.65 node·h of training; 4,000 steps would cost 1.31 and leave no room for a rerun inside the 2.0 cap.
  2. It is long enough for untied copies to separate if the loss pushes them apart: lr 0.003 with Adam moves each element of
     each copy by at most about 0.003 per step, against an initialization scale of 0.044 (R, U) and 0.088 (Q) (std
     σ/√cols with σ = 0.5, 0.5, 1.0, cols = 128), so a copy's own updates could move it by that scale within roughly 15-30
     steps (an upper bound; Adam steps are usually smaller). 2,000 steps is about two orders of magnitude beyond that, and a
     collapse would show well before the end (train loss is logged every 100 steps).
  3. It matches the question: whether the solution is *kept*, not whether B8 can rediscover it. A longer run would turn
     the study into "does B8 train as well as M1 for more steps", which P1-L already answered from a random start.
  4. The shared arm is not converged: P1-L M1-L4 train loss (mean over logged steps) was 4.51 / 3.65 over steps 12-15k and
     15-17k for seed 17 (4.96 / 4.35 for seed 29, 4.33 / 3.58 for seed 43), still falling. So the shared arm will keep
     improving during the continuation and the untied arm is asked to follow a moving reference. This is the right
     reference for "does untying cost anything", but it makes the outcome sensitive to N; N is not adapted to any result.
- **Descriptive references** (evaluated on the same new range in the same evaluation, same episodes; not decision inputs):
  the **starting M1-L4@4**; **B4@1** and the **original B8@4** from P1-L (their registered r = 128 checkpoints, hashes in
  the lock, re-evaluated on the new range so all rows are paired); P1-L's own confirmation numbers (B4 5.36%, B8 6.34%,
  M1-L4 26.91% on hops 2-3, test [100000, 102000)) are quoted for orientation only.

## Evaluation (read once)
- **Range**: split `test`, episodes **[108000, 110000)**, 2,000 episodes, eval seed 4242, memory-only, **L = 4** (B4 at L = 1),
  hops 1-8, primary setting, 2,000 paired queries per hop. A new named window `CONFIRMATION_WINDOWS["b8w"] = 108000`
  (`eval.confirmation: "b8w"`); the existing entries (`p1l` 100000, `p1l-s` 110000, `p5l` 104000, and any entry other tasks
  add) are unchanged, and a test pins each existing value. The range is disjoint from P1-L [100000, 102000), P1L-C
  [102000, 104000), P5L [104000, 106000), LSR's [106000, 108000) and P1L-S's [110000, 112000).
- **Read once**: after (1) registration on master and (2) the committed checkpoint lock and a committed authorization
  (hashes of this prereg, the lock and the continuation manifest), by one evaluator run over the 15 groups (start M1-L4,
  shared, untied, B4, B8 original; 3 seeds each). No test episode is generated by any smoke, test or gate.
- **No typed replication, swap, readout, depth scan or inference timing** in this study (no gate on them); the evaluator
  is `scripts/evaluate_b8w.py` on the P5L/P1-L evaluator's building blocks, `scripts/finalize_b8w.py` reuses
  `pooled_compare_runs`.

## Decision rule (G3 / astra #3, applied exactly)
Contrast **Δ = untied@4 − shared@4**, full-vocabulary EM on **hops {2, 3}**, pooled over the three seeds with equal seed
weights (`pooled_compare_runs`: paired joint episode bootstrap, 2000 draws, seed 2718, percentile intervals;
the interval conditions on the three training seeds).
- **Untying keeps the solution** iff both hold:
  1. in every seed the point estimate of the untied loss is at most 2 pp: Δ_seed ≥ −0.02 for 17, 29 and 43;
  2. the pooled **95% CI lower bound of Δ is above −0.02** (non-inferiority: the interval excludes a loss worse than 2 pp).
- **Otherwise** the report states: *untying loses the solution; fragility of the solution under untied updates, not lack
  of expressivity.*
- Interval level: one gated contrast, so k = 1 and 95% two-sided (equivalently a one-sided 97.5% non-inferiority bound);
  97.5% values are reported next to it. Hop 1 (Δ on hop 1) is reported with its CI and enters no rule.
- Sensitivity of the rule to noise: a per-seed hops 2-3 difference has about 4,000 paired queries; at accuracies near 30%
  its standard error is about 0.7-1.0 pp, so a true Δ of 0 fails the per-seed condition only with probability of the
  order of 0.5% per seed, and the pooled interval half-width is about 0.7 pp (cf. P1-L's G1 interval widths).

Reported, not decision inputs (all with the same pooled CIs where a contrast): untied − start M1-L4 and shared − start M1-L4
(hops 2-3 and hop 1); untied − original B8 and − B4@1 (hops 2-3); accuracy by hop 1-8 per arm and seed; train-loss curves
(every 100 steps) of both arms; per-seed Δ; **protocol check**: if the shared arm is more than 2 pp below the starting
M1-L4 on hops 2-3 (pooled point estimate), the report states that the continuation recipe itself (fresh Adam state at lr
0.003) perturbs the solution and reads Δ with that caveat; the decision rule is unchanged; descriptive drift of the untied
copies (after continuation, for each of R, U, Q: ‖copy_i − start‖_F/‖start‖_F and the mean pairwise cosine distance between
the four copies; computed from checkpoints, no test data).

## Claims and non-claims (registered with the rule)
Allowed after "keeps": a copied B8 retains hops-2-3 accuracy within 2 pp of continued M1-L4 over N = 2,000 steps at lr
0.003, from three starting seeds, on one fresh range, in the oracle condition with seen names. Allowed after "loses":
the copied solution is not stable under untied updates in this protocol. Not claimed in either case: that sharing is
or is not necessary for learning composition; the cause of B8's random-start failure; equal-parameter status for B8
(M1-L4 vs B4 stays the only equal-parameter contrast); a revised G1-L (it stays failed); any statement for r = 32,
learned writers, unseen names, other scales or other N; any unseen-hop claim.

## Cost estimate (measured P1-L step times; 1 node per job, each job ≤ 1 h; budget 1.0-2.0 node·h including tests and smokes)
Measured steady steps at r = 128 (P1-L main, `record.json` `steady_step_seconds`): M1-L4 0.194 / 0.192 / 0.199 s, B8
0.195 / 0.203 / 0.194 s (seeds 17 / 29 / 43; peak memory 7.7 GB); model load about 25 s; P1-L r = 128 evaluation of B4
(three seeds, primary + typed + timing) cost 0.031 node·h.

| Item | Basis | node·h |
| --- | --- | ---: |
| Unit tests (CPU) and a GPU test job (gate on seed 17, converted-checkpoint load) | like P5L's test jobs | 0.05 |
| Validation-only smoke: gate for seed 17 and 100 continuation steps of each arm, no test | 2 × 100 × 0.2 s + load | 0.05 |
| 3 continuation jobs (one per seed: gate, shared arm, untied arm, sequentially) | (2 × 2,000 × 0.196 s + 2 × ~40 s overhead + 25 s load + gate ~60 s) ≈ 15 min each; request 30 min (≥ 1.5×) | 0.75 |
| Authorization check + evaluation of 15 groups, primary only, L = 4, hops 1-8 (one or two jobs) | 15 × ~20 s + load; below P1-L's 0.031 per group with typed + timing | 0.12 |
| Finalizer, ledger | | 0.02 |
| **Total** | | **≈ 1.0** |
| Contingency: one rerun of a continuation job | from scratch, new directory | + 0.25 |
With N = 3,000 the total is about 1.45 and with N = 4,000 about 1.7 (2.0 with one rerun). No job exceeds 1 h, so no
`--scale-up` is needed. The ledger gets a `b8w` study profile (like `p5l`).

## Order and provenance
1. The orchestrator registers this file on master.
2. Step 2 (after the OK): code and tests (`tests/test_b8w.py`: exact conversion on tiny adapters, key mapping, gradient-sum
   identity, rejection of wrong shapes or step counts and of a perturbed copy by the gate, `train.data_start_step` off path,
   the window entry and the existing ones unchanged, decision-rule boundaries), run on compute through `scripts/submit.py
   --commit <full hash>`; a validation-only smoke.
3. `reports/b8w/checkpoint-lock.json` (status `locked`: the nine P1-L checkpoint hashes (M1-L4, B8, B4 × 3 seeds), source
   commits, N, lr, the stream start, the window and this prereg's blake2b) is committed; the orchestrator OKs it.
4. The three continuation jobs run pinned to one source commit; each records the gate. A failed gate stops the study.
   No resume path exists: a job killed or failed is rerun from scratch in a new directory with the same config and
   source, and the event is recorded.
5. All six continuations are verified (complete records, logged steps = N, start-checkpoint hashes, same stream, lr), and a
   confirmation authorization is committed; the orchestrator checks it before any test episode is generated.
6. One evaluation over the 15 groups, the finalizer, `reports/decisions/B8W.md`, `docs/reports/B8W.md`, the state file.
   Hashes are blake2b; seeds come from `derive_seed`; no change to `pyproject.toml`/`uv.lock`.

## Flags: differences from the task text or the fixed design
1. **Interval level 95% (k = 1)**, not P1-L's 97.5% (k = 2): there is one gated contrast (as P1L-S, k = 1). 97.5% is reported too.
2. **"Same data stream"** is read as the P1-L stream continued from step 17,000 for both arms (episodes no training step
   used), not a replay of the first N steps. Needs the optional `train.data_start_step` key (off path tested).
3. **The descriptive references B4@1 and original B8@4 are re-evaluated** on the new range from their registered
   checkpoints (about +0.05 node·h), rather than quoted from [100000, 102000), so all differences are paired on the same episodes.
4. **The evaluation is primary-only at L = 4** (no typed replication, swap, readout or depth scan), since no gate uses them.
5. **Warmup restarts** with the fresh optimizer (same 20 steps as P1-L): with Adam state reset, a full-lr first step would
   otherwise be a large perturbation to a trained solution; it is a protocol choice made before any result.
6. **"Keeps" is worded narrowly** (see "Question"): the task text's "if it does, B8's failure is an optimization problem"
   and its "nothing is said about why B8 failed from a random start" pull against each other; this draft follows the
   second and states the first only as "capacity is not what limits keeping the solution".
7. **Merge hazard**: WP-L9 (wp/core) also adds a `CONFIRMATION_WINDOWS` entry (`lsr`, 106000) and will update the same
   equality assertion in `tests/test_p1ls_pipeline.py` (line 153); the two branches will conflict on that dict line and
   that assertion. The orchestrator can resolve it by merging both entries (the dict and the test list both windows), or I
   can make that test assert the registered windows individually.
8. **Per-seed noise**: the "every seed" condition is on point estimates of three single-seed contrasts; the simplest
   failure mode is one seed missing the −2 pp bar by sampling noise alone (order 0.5% per seed under Δ = 0).
