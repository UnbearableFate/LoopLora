# Study P1L-S — P1-L comparison at r = 32 with the WP-L6 oracle code, pre-registered

Drafted 2026-10-27 by sn-oracle (Claude Sonnet-5.5); reviewed and registered by the orchestrator at its master
commit time (2026-10-27, before any WP-L6 or P1L-S run). Source: `plan/followup_2026-10-27.md` §3 F6 and §6. Written before any WP-L6 run, so the
code design is not chosen here: **W is whatever WP-L6's registered rule selects** (`docs/prereg/WP-L6.md`:
`oracle_rot` if it passes, else `oracle_shared`). The study **runs iff** `reports/wp-l6/decision.json` (committed
before the smoke) has status `selected`; it does not depend on P1-L's r = 128 results. P1-L, P1L-C and their episode
ranges are unchanged. Validation episodes are used for Pilot-2; the confirmation range below is read once, at the end.

Information seen before this draft: P1-L's r = 32 `oracle_random` main-run training logs and Pilot-2 (every memory
reader at chance; Amendment B7); P1-L's measured step times (`reports/p1l/smoke-sizing.json`, used only for the cost
table below). No WP-L6 result and no P1L-S run exists.

## Question
At r = r_mem = 32, with an oracle code under which the readers learn to use the memory, does an LM-internal
multi-read reader beat a single read at the same rank, as the symbolic stage found at that rank?

## Fixed design (P1-L's r = 32 `oracle_random` condition, `docs/prereg/P1L.md` with Amendments B1–B7; one change)
- Change: the oracle writer is W with oracle seed 17 (definitions in `docs/prereg/WP-L6.md`; O-shared trains its table
  C in every memory-reader run, same lr and warmup, weight decay 0, clipping jointly with the reader). Nothing else
  differs: backbone `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, layer-14 `o_proj`, fp32
  adapter and state, ρ = 0.25, α = 1, r = r_mem = 32, fast path, `train.prefetch_workers: 2`; task `t2_chain_end`,
  E = 12, chains (8, 4), n_rel 4, head sharing 0.3, train hops {1, 2, 3}, evaluation hops 1–8; memory-only protocol.
- Readers, seeds 17 / 29 / 43 (12 new runs): **B8** (L = 4), **M1-L4**, **B4** (L = 1), **B5-L4** (fixed query,
  L = 4). **B3-L4** (no memory, L = 4) is not retrained: it never reads the memory, so its training does not depend
  on the writer; P1-L's r = 32 B3-L4 runs (seeds 17/29/43, lr 1e-3, S = 17,000) are evaluated on this study's
  confirmation range. B0, B1 and B2 are not rerun (they do not use the oracle and are not gated).
- Optimization: AdamW, batch 64, warmup 20, weight decay 0.01, clip 1.0, constant lr, **S = 17,000 steps for every
  reader** (the P1-L value; the per-job time rule holds, see Cost).
- Smoke and **own Pilot-2** (for the four retrained readers): a 10% time-only smoke (1,700 steps per reader; no
  score read), then lr ∈ {1e-3, 3e-3} per reader at ⌊S/3⌋ = 5,666 steps, seed 17, validation (split `val`, eval seed 4242, 1,000 episodes), selected by
  full-vocabulary EM pooled over hops 1–3 at the reader's trained L (B4: L = 1), ties → the lower lr. The WP-L6 pilot
  run is not reused as a candidate. lrs and S are locked in a commit before any main run.
- **Confirmation: split `test`, indices [110000, 112000), eval base seed 4242, 2,000 episodes**, a range no earlier
  evaluation used (P1-L: [100000, 102000); P1L-C: [102000, 104000)); the typed replication uses `t2_typed` on the same
  range. Evaluated once, only after all 12 trainings have finished and the lr/S lock is committed.

## Endpoints (memory-only, confirmation set, hops {2, 3} unless stated; pooled paired episode bootstrap, 2,000
## draws, **95% CIs** because exactly one condition is gated; per-seed values reported)
- **G1-L**: B8@4 − B4@1 ≥ 3 pp with CI > 0 and positive in every seed; hop 1: B8@4 − B4@1 ≥ −1 pp (point
  estimate); replication: the same contrast on `t2_typed` (3 same-head distractors, same checkpoints) with CI > 0.
- **G2-L**: B8@4 − B5-L4@4 > 0 and B8@4 − B3-L4@4 > 0 (CI).
- **Equal-parameter rule (Amendment B6)**: a claim that more reads beat one read at equal reader parameters is made
  only from M1-L4@4 − B4@1 (hops {2, 3}) and only if it meets the G1-L criterion (≥ 3 pp, CI > 0, positive in every
  seed). Reader parameters at r = 32: B4 = M1-L4 = 103,426; B8 = 112,642 (+8.9%). Under O-shared every memory reader
  also trains a 32,800-parameter table C, counted separately as writer parameters and equal across them.
- **Memory-use check (Amendment B7)**: on hop 1 (full EM), B4@1, B8@4 and M1-L4@4 must each exceed B3-L4@4 with
  CI > 0. If any fails, the report states that the readers did not learn to use the oracle memory, and G1-L, G2-L and
  the B6 outcome are reported but not interpreted as evidence about multiple reads.
- Secondary, not gating (as P1-L): M1-L4 against the same comparators; M1-L4 depth scan L = 1…8 (Δ_reach and walk
  specificity where hops 6–8 exist); B8 − M1-L4; `swap_successors` follow rates; per-hop tables; the per-step
  readout (logit lens and J-lens directions) for M1-L4 and B8; adapter parameter counts and inference time per
  question for every reader; for O-shared the drift of C from its initialization. 

## Decision use
- B7 passes, G1-L and G2-L pass: with oracle W, the multi-read reader composes inside the LM at r = 32 and beats the
  single read of the same rank; the equal-parameter statement follows the B6 rule. This is the paper's LLM evidence
  at the symbolic stage's rank, stated for W, a closed vocabulary of 1,024 names and the trained depth.
- B7 passes, G1-L fails: negative LLM result at r = 32 for W (`reports/decisions/P1L-S.md`). G1-L passes, G2-L
  fails: report which comparator matches B8; no evolving-query claim.
- B7 fails: "the readers did not learn to use the oracle memory at r = 32 with W"; no multi-read claim at r = 32 and
  no further oracle-code study follows from this plan.
- W is not the symbolic oracle: under O-rot the value→key transition is a fixed linear map; under O-shared the codes
  are shaped by each reader's own training, so memories differ across readers. A pass shows composition for that
  code, not equivalence with the symbolic oracle, and says nothing about unseen names.

## Cost (node·h from P1-L's measured r = 32 step times; B8 0.193, M1-L4 0.197, B4 0.133, B5-L4 0.152, B3-L4 0.135 s/step)
| stage | runs | estimate |
| --- | --- | --- |
| smoke | 4 readers × 1,700 steps | 0.33 node·h |
| Pilot-2 | 4 readers × 2 lr × 5,666 steps (one job per reader, 0.42–0.62 h each, ≤ 1 h with the 1.5 margin) | 2.12 node·h |
| main | 12 runs × 17,000 steps (3.19 node·h per seed) | 9.56 node·h |
| confirmation, typed, swap, readouts | 15 checkpoints (12 new, 3 reused B3-L4) × 2,000 episodes (P1-L timing scaled) | ≈ 2 node·h |
| startup and reserve | per-job model load, export, hashing | ≈ 0.5 node·h |
| **total** | | **≈ 14.5 node·h** (plan: ≈ 20) |

Main jobs use `scripts/submit.py --scale-up docs/prereg/P1L-S.md` (justification: S = 17,000 is P1-L's value, and
B8 at 0.193 s/step needs 3,294 s per run, so 1.5 × (3,294 + reserve) = 1.4 h). Walltimes are 1.5 × the P1L-S smoke
measurement, at most 2 h: B8 ≈ 1.4 h, M1-L4 ≈ 1.4 h, B5-L4 ≈ 1.1 h and B4 ≈ 1.0 h as single-run jobs. Each job ≤ 1 node × 2 h; node·h recorded per job.

## Pipeline work that precedes the smoke (step 3; not part of the science above)
The P1-L scripts hard-code `oracle_random`, `configs/p1l/`, `reports/p1l/`, `docs/prereg/P1L.md` and a confirmation
start of 100,000 (`run_p1l_main.py`, `launch_p1l_main.py`, `evaluate_p1l_main.py`, `finalize_p1l_main.py`,
`src/looplora/llm/evaluate.py`). P1L-S needs them parameterized by writer, config root, lock paths and confirmation
start (110,000; the same mechanism P1L-C needs for [102000, 104000)), a CI level of 95% (k = 1), and, for O-shared,
a finalize check that records one `global_codes`/`rotations` hash for all runs plus each run's trained-table hash and
compares write counts and fact lists (not memory snapshots) across readers. Compute tests precede any run.

Executed by a Claude Sonnet-5.5 subagent after WP-L6. Report: `docs/reports/P1L-S.md`.
