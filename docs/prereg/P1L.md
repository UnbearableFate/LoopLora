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
