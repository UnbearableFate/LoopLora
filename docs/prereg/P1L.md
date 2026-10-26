# P1-L pre-registration — single module in Qwen3-0.6B-Base (draft v2, 2026-10-27)

Status: **draft v2** (2026-10-27 03:19 JST). Written after P2-S-F1 (`reports/decisions/P2S.md` §6) and before
any P1-L T2 run; no T2 result exists. Finalised (v2 final) only if WP-L3 / WP-L4 select a writer, by filling
the ⟨writer⟩ fields from their registered rules, before Pilot-2. If neither selects a writer, P1-L stops and this
draft is not used.

## Changes from v1 (reasons in `reports/decisions/P2S.md` §6 and `docs/reports/P1L-diagnosis.md`)
- Primary reader **B8** (step-specific reads, trained at fixed L = 4); M1 trained at fixed L = 4 (M1-L4) is the
  shared variant. B8 has no depth scan, so the endpoints compare separately trained models.
- Writer from the WP-L3 / WP-L4 screens, trained on T1 and **frozen** for T2 (every reader sees the same memory).
  v1's `joint` writer is not carried: the LLM writer learns only with the screen's auxiliary losses on T1.
- Rank fixed at r = r_m = 32 (the screened writer); r = 64 is not run.
- Budget rule: jobs ≤ 8 nodes × 2 h under `scripts/submit.py --scale-up docs/prereg/P1L.md` (P2-S-F1 and the
  writer screen justify a scale-up); every job ends before 2026-10-28 09:00 JST.

## Fixed design
- Backbone `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, bf16, sdpa; module at layer 14
  `o_proj` (2048 → 1024), fp32 adapter and state, ρ = 0.25, α = 1, r = r_m = 32; fast path (`scan_final`,
  `train.prefetch_workers: 2`).
- Text: WP-E templates, single-token names, answer = one name token. Primary metric: full-vocabulary greedy EM
  at the answer position; candidate-restricted accuracy secondary.
- Protocol: **memory-only** (support truncated after the write) for every gate; context-retained only for B0,
  B1, B2 (reported, not gating).
- Writer: ⟨arm and E_L from WP-L3 / WP-L4⟩, its T1 checkpoint (seed 17) frozen for every T2 run; blake2b of the
  writer weights and of the support snapshots recorded per run.
- Task: `t2_chain_end`, n_rel 4, head sharing 0.3. If E_L = 16: E = 12, chains (8, 4) as P1-S; if E_L = 8: E = 8,
  chains (4, 4) (flagged; hops 5–8 absent). Train hops {1, 2, 3}; evaluation hops 1–8 where present.
- Readers (trained on T2 with the frozen writer, seeds 17/29/43): **B8** (L = 4), M1-L4 (L = 4), B4 (L = 1),
  B5-L4 (fixed query, L = 4), B3-L4 (no memory, L = 4). Baselines: B0 (frozen base, both protocols, no training);
  B1 LoRA and B2 single-step nonlinear adapter at the same position, trained context-retained, seeds 17/29/43.
- Secondary writer condition (run only if the primary set has finished before 2026-10-28 03:00 JST, a rule on
  time, not results): writer fine-tuned jointly on T2 with the screen's auxiliary losses, B8 and B4 only.
- Optimisation: AdamW, batch 64, warmup 20, weight decay 0.01, clip 1.0, constant lr; steps S fixed from the
  Pilot-2 smoke (the largest S whose longest run fits 2 h × 1/1.5), identical for every reader.
- Episode sets: val = split `val`, eval seed 4242, 1000 episodes; **confirmation = split `test` [100000, 102000),
  untouched until every primary run has finished training**.

## Pilot-2 (validation only, seed 17)
10% smoke for time and memory (no scores read), then lr ∈ {1e-3, 3e-3} per reader at S/3 steps; select by val EM on
hops 1–3 at the trained L (B4: L = 1; ties → lower lr); lrs and S locked in a commit before main training.

## Endpoints (memory-only, confirmation set, hops {2, 3} unless stated; 95% pooled paired episode bootstrap,
## 2000 draws; per-seed values reported)
- **G1-L (depth/capability)**: B8@4 − B4@1 ≥ 3 pp with CI > 0 and positive in every seed; hop 1: B8@4 − B4@1
  ≥ −1 pp (point estimate); replication: the same contrast on `t2_typed` (E as above, 3 same-head distractors,
  evaluated on the same checkpoints) with CI > 0.
- **G2-L (evolving query and memory)**: B8@4 − B5-L4@4 > 0 and B8@4 − B3-L4@4 > 0 (CI).
- Secondary, not gating: M1-L4 against the same comparators; M1-L4 depth scan L = 1…8 on the same x/u/M
  (Δ_reach and walk specificity as P1-S where hops 6–8 exist); B8 − M1-L4; `swap_successors` follow rates;
  B0/B1/B2 in both protocols; per-hop accuracy tables.

## Decision use
- G1-L and G2-L pass: the composition result transfers to one LLM module in the memory-only protocol; P3 may be
  planned with B8 (after the 2026-10-28 maintenance; new prereg).
- G1-L fails: negative LLM result recorded in `reports/decisions/P1L.md`; P3 is not run; the paper reports the
  symbolic results with the LLM negative result.
- G1-L passes and G2-L fails: report which comparator matches B8; no evolving-query claim for the LLM.

## Budget
Each job ≤ 8 nodes × 2 h (`--scale-up docs/prereg/P1L.md`); P1-L total ≤ 16 node·h; node·h recorded per job.
