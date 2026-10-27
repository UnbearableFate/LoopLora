# P1-L pre-registration — single module in Qwen3-0.6B-Base (draft v2, 2026-10-27)

Status: **v2 final for the `oracle_random` condition** (2026-10-27 04:03 JST, after WP-C7 passed its tests,
`docs/reports/WP-C7.md`; drafted 03:19, writer conditions revised 03:43). Written after P2-S-F1
(`reports/decisions/P2S.md` §6) and WP-L3 (`docs/reports/P1L-writer-fix.md`), before any P1-L T2 run; no T2 result
exists. The `learned` condition is added by an addendum filled from WP-L4's registered rule before its own
Pilot-2; nothing in it depends on P1-L results.

## Changes from v1 (reasons in `reports/decisions/P2S.md` §6, `docs/reports/P1L-diagnosis.md`, `P1L-writer-fix.md`)
- Primary reader **B8** (step-specific reads, trained at fixed L = 4); M1 trained at fixed L = 4 (M1-L4) is the
  shared variant. B8 has no depth scan, so the endpoints compare separately trained models.
- Two writer conditions, as in P1-S: **`oracle_random`** (fixed random entity codes written from the fact list,
  WP-C7; always run; tests the reader inside the LM: question encoding, repeated reads, output through the LM)
  and **`learned`** (the WP-L4 writer, trained on T1 and frozen; run only if WP-L4 selects one). v1's `joint`
  writer is not carried: the LLM writer learns only with auxiliary losses on T1, and only after WP-L4.
- Rank fixed at r = r_m = 32 (the screened writer); r = 64 is not run.
- Budget rule: jobs ≤ 8 nodes × 2 h under `scripts/submit.py --scale-up docs/prereg/P1L.md` (P2-S-F1 and the
  writer screens justify a scale-up); every job ends before 2026-10-28 09:00 JST.

## Fixed design
- Backbone `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, bf16, sdpa; module at layer 14
  `o_proj` (2048 → 1024), fp32 adapter and state, ρ = 0.25, α = 1, r = r_m = 32; fast path (`scan_final`,
  `train.prefetch_workers: 2`).
- Text: WP-E templates, single-token names, answer = one name token. Primary metric: full-vocabulary greedy EM
  at the answer position; candidate-restricted accuracy secondary.
- Protocol: **memory-only** (support truncated after the write) for every gate; context-retained only for B0,
  B1, B2 (reported, not gating).
- Writers: `oracle_random` (WP-C7; codes are fixed buffers) and, if selected, ⟨WP-L4 arm and E_L⟩ with its T1
  checkpoint (seed 17) frozen for every T2 run. blake2b of writer weights/codes and support snapshots recorded.
- Task: `t2_chain_end`, n_rel 4, head sharing 0.3, E = 12, chains (8, 4) as P1-S (if the learned writer is
  accepted only at E_L = 8, its condition uses E = 8, chains (4, 4), flagged; hops 5–8 absent). Train hops
  {1, 2, 3}; evaluation hops 1–8 where present.
- Readers (trained on T2 per writer condition, seeds 17/29/43): **B8** (L = 4), M1-L4 (L = 4), B4 (L = 1), B5-L4
  (fixed query, L = 4), B3-L4 (no memory, L = 4; writer-independent, trained once). Baselines: B0 (frozen base,
  both protocols, no training); B1 LoRA and B2 single-step nonlinear adapter at the same position, trained
  context-retained, seeds 17/29/43.
- Optimisation: AdamW, batch 64, warmup 20, weight decay 0.01, clip 1.0, constant lr; steps S fixed from the
  Pilot-2 smoke (the largest S whose longest run fits 2 h / 1.5), identical for every reader.
- Episode sets: val = split `val`, eval seed 4242, 1000 episodes; **confirmation = split `test` [100000, 102000),
  evaluated for a writer condition only after all of its primary runs, B3-L4 and B0–B2 have finished training,
  and only once WP-L4 has decided (which fixes the CI level below)**.
- Order: the `oracle_random` condition starts when WP-C7 passes; the learned condition follows if selected.

## Pilot-2 (validation only, seed 17)
10% smoke for time and memory (no scores read), then lr ∈ {1e-3, 3e-3} per (writer condition, reader) at S/3
steps; select by val EM on hops 1–3 at the trained L (B4: L = 1; ties → lower lr); lrs and S locked in a commit
before main training.

## Endpoints (memory-only, confirmation set, hops {2, 3} unless stated; pooled paired episode bootstrap, 2000
## draws, 97.5% CIs if both writer conditions run, else 95%; per-seed values reported)
Per writer condition:
- **G1-L (depth/capability)**: B8@4 − B4@1 ≥ 3 pp with CI > 0 and positive in every seed; hop 1: B8@4 − B4@1
  ≥ −1 pp (point estimate); replication: the same contrast on `t2_typed` (E as above, 3 same-head distractors,
  evaluated on the same checkpoints) with CI > 0.
- **G2-L (evolving query and memory)**: B8@4 − B5-L4@4 > 0 and B8@4 − B3-L4@4 > 0 (CI).
- Secondary, not gating: M1-L4 against the same comparators; M1-L4 depth scan L = 1…8 on the same x/u/M
  (Δ_reach and walk specificity as P1-S where hops 6–8 exist); B8 − M1-L4; `swap_successors` follow rates;
  B0/B1/B2 in both protocols; per-hop accuracy tables.
- Per-step readout (descriptive; latent-reasoning check, motivated by the J-lens / J-space work): for M1-L4 and
  B8, decode each intermediate state z_s (s = 1…4) (a) through the trained output path and the frozen top layers
  (logit lens at the answer position) and (b) by projection onto J-lens directions of the 1024 names (each name's
  logit gradient w.r.t. the layer-14 residual at the answer position, averaged over 512 validation questions);
  report the rank and top-1 rate of the hop-s entity at step s against the other steps and a random-direction
  baseline, by hop.

## Decision use
- `oracle_random` passes G1-L and G2-L: the multi-read reader composes inside the LM when the memory is clean;
  with no learned writer, the LLM limit is the writer (stated as such).
- `learned` passes G1-L and G2-L: the composition result transfers to one LLM module end to end in the
  memory-only protocol; P3 may be planned with B8 (after the 2026-10-28 maintenance; new prereg).
- A condition fails G1-L: negative LLM result for that condition in `reports/decisions/P1L.md`; P3 is not run
  unless the learned condition passes.
- G1-L passes and G2-L fails: report which comparator matches B8; no evolving-query claim for that condition.

## Budget
Each job ≤ 8 nodes × 2 h (`--scale-up docs/prereg/P1L.md`); totals per writer condition as in Amendment B1; node·h recorded per job.

## Amendment B1 — budget and S (2026-10-27 04:12 JST, before any Pilot-2 run or score)
Information seen: timing-only preflight 3715826 (all 7 trained paths fit memory; B1/B2 context-retained ≈ 0.28
s/step, oracle readers ≈ 0.13–0.20 s/step); no score. The fixed-S rule ("largest S fitting 2 h / 1.5") and the
16 node·h total conflict: 21 main + 14 Pilot-2 trainings at that S exceed 16 node·h. The per-job limit (≤ 8 nodes
× 2 h) is the user's rule; the total is this study's own budget. Change:
- The `oracle_random` condition, including B0–B2, B3-L4, smoke, Pilot-2, confirmation and readout, has ≤ 16 node·h.
  The `learned` condition, if added, has its own ≤ 8 node·h in its addendum.
- S = the largest common S, rounded down to a multiple of 500, that satisfies both (a) the slowest run, timed under
  the packing actually used (several runs per node where measured throughput justifies it), × 1.5 fits one 2 h
  job, and (b) projected Pilot-2 + main node·h ≤ 13, keeping 3 node·h for smoke, startup, confirmation, J-lens
  directions and failures. If that S is below 6000, the orchestrator decides before Pilot-2.
- Pilot-2 still uses S/3; S stays identical for every reader.

## Amendment B2 — total budget relaxed (2026-10-27 04:18 JST, before any Pilot-2 run or score)
The user clarified (2026-10-27) that the resource rule applies to a single job (≤ 8 nodes × 2 h with
`--scale-up`) and that the total budget may be relaxed. Amendment B1's 16 / 13 node·h totals are therefore
dropped and the v2 S rule is restored: S = the largest common S, rounded down to a multiple of 500, whose slowest
run under the packing actually used × 1.5 fits one 2 h job (preflight 3715826 and sizing
`reports/p1l/preflight-B1-sizing.json` on wp/p1l: job cap 17,413 steps → **S = 17,000**, Pilot-2 at ⌊S/3⌋).
Projected Pilot-2 + main ≈ 24 node·h plus smoke, confirmation and readout; node·h recorded per job. The learned
condition, if added, uses the same rule. Information seen: timing only; no score.

## Amendment B3 — r = 128 replication of the oracle condition (2026-10-27 04:32 JST, user request "并行验证 r = 128";
## before any P1-L validation or confirmation score is read)
- New condition `oracle_random` at **r = r_mem = 128**, run in parallel: B8, M1-L4, B4, B5-L4, B3-L4 × seeds
  17/29/43, everything else as the r = 32 condition (same oracle seed, task, protocol, readout). Its own Pilot-2
  (lr ∈ {1e-3, 3e-3} at ⌊S/3⌋); S = 17,000 if the per-job rule holds at r = 128, otherwise the largest S that does
  (flagged, since cross-rank contrasts then differ in steps). B0–B2 are not repeated.
- Endpoints: G1-L and G2-L per rank condition. Added, secondary: the LLM analog of P2-S E1 — B8(r = 32)@4 −
  B4(r = 128)@1 and M1-L4(r = 32)@4 − B4(r = 128)@1 on hops {2, 3} — and B8(r = 128)@4 − B8(r = 32)@4, descriptive.
- CI level for every gate: 1 − 0.05/k, where k is the number of gated conditions (oracle r = 32, oracle r = 128,
  plus any learned condition), fixed before the first confirmation evaluation. Confirmation stays held until
  every condition's runs have finished training.

## Amendment B4 — no comparison with a larger-rank single read (2026-10-27 04:46 JST, user decision; before any
## P1-L confirmation score and before any r = 128 score is read)
- Criterion (user): a multi-read reader only has to beat the single read at the same rank. M1-L4 has exactly B4's
  parameters; B8 has 9,216 more adapter parameters at r = r_mem = 32 (+8.9%, its per-step R/U/Q; `param_count`).
  The extra read latency is accepted as the cost and reported, not tested.
- Removed from Amendment B3: B8(r = 32)@4 − B4(r = 128)@1 and M1-L4(r = 32)@4 − B4(r = 128)@1. No P1-L claim
  compares against a larger-rank single read. B8(r = 128)@4 − B8(r = 32)@4 stays descriptive.
- Added, descriptive: adapter parameters and measured inference time per question for every reader at each rank.
- Gates, conditions, runs and the CI level are unchanged.

## Amendment B5 — WP-L4 decision and CI level (2026-10-27 07:00 JST; before any P1-L confirmation score is read)
- WP-L4 selected no learned writer (`reports/p1l-read-fix/selection.json`, action `stop`; best learned arm R at
  r = 128, val EM 33.5%, below the 0.80 gate and the 0.50 E8 trigger; merged 39f7f0b). The `learned` condition is
  not run; P3 is not run (it required the learned condition to pass).
- Gated conditions: `oracle_random` at r = 32 and at r = 128, so k = 2 and every gate uses **97.5%** intervals
  (1 − 0.05/2). This is fixed now, before the first confirmation evaluation.
- Schedule: normal operation ended 2026-10-27 08:00 (HPC Challenge, then maintenance until 2026-10-28 22:00).
  Runs that could not finish by then (r = 32 B1/B2 seed 43, r = 128 B4 seed 43) and the confirmation evaluation
  run after 2026-10-28 22:00. No design change.

## Amendment B6 — equal-parameter claim rule (2026-10-27 08:18 JST; before any P1-L confirmation score is read)
Prompted by an external review (`docs/review/rv-astra_2026-10-27.md`). Gates G1-L/G2-L are unchanged.
- Parameters (`param_count`, adapter only): B4 = M1-L4 = 103,426 at r = 32 and 475,138 at r = 128; B8 = 112,642
  (+8.9%) and 622,594 (+31.0%). B8 − B4 is a same-rank contrast, not an equal-parameter one.
- A claim that more reads beat one read **at equal parameters** is made only from the already registered secondary
  contrast M1-L4@4 − B4@1 (hops {2, 3}, per rank), and only if it meets the G1-L criterion: ≥ 3 pp, 97.5% CI > 0,
  positive in every seed. Otherwise the paper states the B8 result with its parameter difference and makes no
  equal-parameter claim for the LLM.

## Amendment B7 — memory-use check and oracle-code interpretation (2026-10-27 08:51 JST; before any P1-L
## confirmation score is read)
Prompted by the external reviews (`docs/review/rv-fable_2026-10-27.md` C1, `docs/review/rv-astra_2026-10-27.md`
§5) and by a check of the main runs' training logs. Runs, gates, confirmation set and the 97.5% level are unchanged.
The reviews' proposal to change the oracle keys before Pilot-2 is moot: main training is complete.

Information seen (training logs and Pilot-2 validation only; no test or confirmation data):
- Training logs at step 17,000 (mean of the last five logged batches, hops 1–3 mixed, trained L): at r = 32 every
  memory reader is at chance (batch accuracy 0.4–1.6%, loss 7.07–7.18; B3-L4, no memory: 0.3–0.4%, 7.15–7.18),
  while B1/B2 (context-retained) reach 99.4–100%. At r = 128: B4 10.0–10.8%, B5-L4 10.3–13.5%, B8 11.3–12.8%,
  M1-L4 28.1–41.2%; B3-L4 0.3–0.6%. B4 seed 43 at r = 128 and B1/B2 seed 43 have not run.
- Pilot-2 validation at r = 32 (S/3, seed 17): full EM ≤ 0.3% and candidate accuracy 7.2–9.1% (chance 7.1%) on
  hops 1–3 for every reader and lr (`docs/reports/P1L-state.md` on wp/p1l noted it; no design change followed).
- Capacity check (`scripts/analyze_oracle_capacity.py`, numpy, `reports/p1l-c1/capacity.json`): the oracle's keys
  (`pair_codes[e, 0]`) and values (`global_codes[e]`) are independent random tables, so every read after the first
  needs the reader to map the bridge's value code to its unrelated key. Over the 768 seen entities, a one-step map of
  the reader's form with r hidden units (q = d + Q relu(U m + c)) fitted in-sample reaches mean cosine 0.40 at r = 32
  and 0.75 at r = 128 (linear least squares 0.20 / 0.41). In replayed T2 episodes (exact hop-1 key, delta rule,
  nearest-code decoding) that gives hop-2 top-1 addressing among the 12 keys of 48% at r = 32 and 99.8% at r = 128
  (linear 30% / 99%); rotation keys normalize(Π·global_codes[e]) give 99% / 100%. So the random-key table limits
  hops ≥ 2 at r = 32 but not at r = 128. It does not explain the r = 32 failure at hop 1, which points to the other
  two random maps the LM path must learn (question state → random key; random value code → name).

Rules added:
1. **Memory-use check** (per rank, confirmation set, hop 1, full EM): B4@1, B8@4 and M1-L4@4 must each exceed
   B3-L4@4 with a 97.5% CI > 0. If any fails, that rank is reported as "the readers did not learn to use the oracle
   memory", and its G1-L, G2-L and Amendment-B6 outcomes are reported but not interpreted as evidence for or
   against multiple reads. Prediction from the logs above: r = 32 fails; r = 128 passes.
2. **r = 32**: no claim about multiple reads in the LM at r = 32, whatever the gate outcomes. The failure is a
   failure of this oracle design (three random maps learned through a rank-32 adapter), not a reader comparison.
3. **r = 128**: the value→key transition is a learned lookup over the 768 seen names, not the identity-like map of
   the symbolic oracle. A G1-L or B6 pass shows that the reader learns that transition and composes reads inside
   the LM; it does not establish equivalence with the symbolic oracle condition, and no unseen-name claim is made.
4. Descriptive additions: the memory-use check table, the training-log summary above, and the capacity table.
5. Follow-ups are registered separately and change nothing here: `docs/prereg/P1L-C.md` (causal and read-position
   analysis of the r = 128 checkpoints on test [102000, 104000), registered before any confirmation score) and,
   per `plan/followup_2026-10-27.md`, an oracle-code redesign pilot (WP-L6) and, if it passes, a new r = 32 study
   (P1L-S) on test [110000, 112000).
- Wording note (2026-10-27, before any confirmation score; raised in WP-T9): "at chance" above means at or near the
  no-memory level. In the last five logged training batches (1,280 questions) some r = 32 memory-reader seeds exceed
  B3-L4 by up to 1.25 pp (B5-L4 seed 29: 1.641% vs 0.391%). Per-run values: `reports/p1l-c1/training-logs.json`
  (`scripts/summarize_p1l_training_logs.py`; the run directories are not tracked). Rules and predictions unchanged.
