# P1-S pre-registration — symbolic mechanism stage (draft v0, 2026-10-26)

Status: **draft** — fields marked ⟨pilot⟩ are fixed after Pilot-1/2 on validation data only and before any
confirmation-set evaluation. Changes after that point are listed under "Amendments" with reason and date.

## Question
With one module, a fixed input projection and writer, and the same memory snapshot, does the evolving-query
multi-read (M1) beat the single read (B4), the fixed-query loop (B5) and the no-memory loop (B3)? (plan §15)

## Fixed design
- Backbone: `sym.backbone`, `backbone_seed = 31415`, d_e = 64, V = 1024 (seen pool [0,768), unseen [768,1024)).
- Module: notation v1; ρ = 0.25, α = 1, r_m = r; primary r = 32, secondary r = 16.
- Writer conditions: `oracle_random`, `oracle_orth` (r = 32), `t1_frozen`, `joint` (WP-D table).
  `t1_frozen` writer: trained on `t1` with n_rel = 4, head_sharing = 0.3, E = E* ⟨pilot⟩, then frozen.
- Primary task `t2_chain_end`, E = E* facts, chain lengths ⟨pilot: (8,4,2,2) if E*=16, (8,8,4,2,2) if E*=24⟩;
  train hops {1,2,3}; evaluation hops 1–8 (one query per hop per episode).
- Replication setting: `t2_typed` (relation-0 chains plus same-head distractors in relations 1–3, same E*) for
  the learned writers; oracle keys ignore relations, so for `oracle_*` the replication is `t2_chain_end` at load
  1.5·E* (interference condition, `oracle_random`).
- Exploratory: `t2_walk` (h stated), `test_unseen`, counterfactuals, r = 16, L-extrapolation.
- Variants: M1 (`iterq`, ReLU), M1-lin (`iterq`, identity), B5 (`fixedq`), B3 (`loop`), B4 (`iterq` at L=1).
  Training depth: L ~ U{1,2,4} per batch (B4: L = 1). Same steps and batch for every variant.
- Episode sets (eval generator seed 4242 unless stated): val = split `val`, 1000 episodes; screening test = split
  `test`, 2000 episodes; **confirmation test = split `test` with eval seed 777001, 2000 episodes**, untouched until
  the gate evaluation. Every evaluation episode carries one query per hop 1–8.
- Optimisation: AdamW, batch 64 episodes, 5000 steps, warmup 20, weight decay 0.01, clip 1.0, constant lr
  after warmup; identical for all variants within a condition.

## Pilots (validation only, seed 17)
- Pilot-1: `t1` writer+L=1 reader, r ∈ {16,32}, E ∈ {8,12,16,24}, lr ∈ {3e-4,1e-3,3e-3}. Output: capacity table.
  Rule: for each (r, E) take the best-val lr; E* = largest E ∈ {16, 24} whose best-lr val single-hop accuracy is
  ≥ 0.95 at r = 32, else 12 (then chains (8,4) and a flag in the report); the writer-pretraining lr for r is the
  best-val lr at (r, E*) (clarification 2026-10-26, before any pilot result was read).
- Pilot-2: every (writer condition × variant) at r = 32, lr ∈ {3e-4,1e-3,3e-3}; select lr per pair by val
  accuracy on hops 1–3 at the variant's training-max L (L=4; B4 at L=1). Steps ⟨pilot⟩ from smoke timing.

## Endpoints (confirmation set, seeds 17/29/43, paired cluster bootstrap by episode, 2000 resamples, 95% CI)
- **G1 depth effect** (per writer condition, same checkpoint, identical x/u/M across L):
  Acc(L=4) − Acc(L=1) on hops {2,3} ≥ 3.0 pp, pooled CI lower bound > 0, positive in each seed; replicated on
  `t2_typed` (CI > 0); hop-1 accuracy at L=4 not below L=1 by more than 1.0 pp.
- **G2 evolving query and memory**: M1@L=4 − B5@L=4 > 0 and M1@L=4 − B3@L=4 > 0 on hops {2,3} (CI > 0).
- Interpretation by condition (plan §11.5): the plan's protocol is `t1_frozen`. If G1/G2 hold only for
  `oracle_*` → the reader works but T1-trained codes do not compose (writer problem). If they hold for `joint`
  with identical-snapshot depth sweeps → keep the fixed-memory read-depth claim, noting that composable codes
  need joint training. If no condition passes → the mechanism fails in this parameterization.
- Reported, not gating: L=8 vs L=4 on hops 4–8 (H6); M1 vs M1-lin; r = 16; `test_unseen`; counterfactual
  consistency (`replace_key_fact` prediction follows the new answer); `swap_memory` drops to chance.

## Stop rules
- No condition passes G1 → do not scale to P1-L; allow one follow-up F1 only if a specific observed failure
  motivates it (e.g. ReLU fails while M1-lin passes → sign-preserving activation or bias as one ablation).
- Single-hop (Pilot-1) below 0.95 at every E for r = 32 → diagnose the writer before any T2 conclusion.

## Budget
P1-S total ≤ 12 node·h; every job ≤ 8 nodes × 1 h (user cap). Actual node·h recorded per job.

## Amendments
(none yet)
