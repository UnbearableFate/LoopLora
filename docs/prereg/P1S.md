# P1-S pre-registration — symbolic mechanism stage (final v1, 2026-10-26)

Status: **final** (v1, 2026-10-26 21:15 JST, commit 3b700ee). Pilot-determined values are filled in from Pilot-1/2 on validation
data only (`docs/reports/P1S-pilot.md`); no `test` or confirmation episode has been evaluated. Changes are listed
under "Amendments" with reason and date. Draft v0 is in git history.

## Question
With one module, a fixed input projection and writer, and the same memory snapshot, does the evolving-query
multi-read (M1) beat the single read (B4), the fixed-query loop (B5) and the no-memory loop (B3)? (plan §15)

## Fixed design
- Backbone: `sym.backbone`, `backbone_seed = 31415`, d_e = 64, V = 1024 (seen pool [0,768), unseen [768,1024)).
- Module: notation v1; ρ = 0.25, α = 1, r_m = r; primary r = 32, secondary r = 16.
- Writer conditions: `oracle_random`, `oracle_orth`, `t1_frozen`, `joint` (WP-D table).
  `t1_frozen` writer: trained on `t1` with n_rel = 4, head_sharing = 0.3, E = 12, lr 3e-3, then frozen; one writer
  per training seed s (pretraining seed s), shared by every variant of that seed.
- Primary task `t2_chain_end`, **E\* = 12, chains (8,4)** (the rule's fallback; flagged: best Pilot-1 single-hop
  accuracy at r = 32 was 84.7% at E = 12, below 0.95 at every E); train hops {1,2,3}; evaluation hops 1–8 (one
  query per hop per episode).
- Replication settings: learned writers — `t2_typed`, E = 12, 3 same-head distractors in relations 1–3, chains
  (8,1); `oracle_*` — `t2_chain_end`, E = 18 (1.5·E\*), chains (8,8,2), `oracle_random` (shared by both oracle
  conditions).
- Variants: M1 (`iterq`, ReLU), M1-lin (`iterq`, identity), B5 (`fixedq`, ReLU), B5-lin (`fixedq`, identity),
  B3 (`loop`), B4 (`iterq` at L=1). M1 and M1-lin are co-primary (A1). Training depth L ~ U{1,2,4} per batch
  (B4: L = 1). Same steps and batch for every variant.
- Episode sets (eval generator seed 4242 unless stated): val = split `val`, 1000 episodes; screening test = split
  `test`, 2000 episodes; **confirmation test = split `test` with eval seed 777001, 2000 episodes**, untouched until
  every main run has finished training. Every evaluation episode carries one query per hop 1–8.
- Optimisation: AdamW, batch 64 episodes, **20000 steps** for every training stage (writer pretraining, readers,
  joint; A2), warmup 20, weight decay 0.01, clip 1.0, constant lr after warmup. Selected lr (Pilot-2, val hops 1–3
  at L = 4, B4 at L = 1):

  | Variant | oracle_random | oracle_orth | t1_frozen | joint |
  | --- | --- | --- | --- | --- |
  | M1 | 3e-3 | 3e-3 | 3e-3 | 3e-3 |
  | M1-lin | 3e-3 | 3e-3 | 3e-3 | 1e-3 |
  | B5 | 3e-3 | 1e-3 | 3e-3 | 3e-3 |
  | B5-lin | 1e-3 | 1e-3 | 3e-3 | 3e-3 |
  | B3 | 3e-4 | 1e-3 | 3e-4 | 3e-4 |
  | B4 | 3e-3 | 1e-3 | 3e-3 | 3e-3 |

  Replication and r = 16 runs reuse the lr of the same (writer, variant) pair. A run that stops on a non-finite
  loss is rerun once at the next lower grid lr; both attempts are recorded.

## Main runs (A4)
- Primary: 4 writer conditions × 6 variants × seeds {17, 29, 43}, r = 32.
- Replication: M1 and M1-lin × seeds {17, 29, 43} in each replication setting (`t1_frozen`/typed, `joint`/typed,
  `oracle_random`/E = 18).
- Secondary (reported only): r = 16, seed 17, `oracle_random`, `oracle_orth`, `joint` × M1, M1-lin, B5, B5-lin.

## Pilots (validation only, seed 17) — done
- Pilot-1: `t1` writer+L=1 reader, r ∈ {16,32}, E ∈ {8,12,16,24}, lr ∈ {3e-4,1e-3,3e-3}. Rule: for each (r, E)
  take the best-val lr; E\* = largest E ∈ {16, 24} whose best-lr val single-hop accuracy is ≥ 0.95 at r = 32,
  else 12 (then chains (8,4) and a flag); the writer-pretraining lr for r is the best-val lr at (r, E\*).
  Result: E\* = 12 (flagged), writer lr 3e-3 for r = 16 and 32.
- Pilot-2: every (writer condition × variant) at r = 32, lr ∈ {3e-4,1e-3,3e-3}, 5000 steps; lr per pair by val
  accuracy on hops 1–3 at the variant's training-max L. Result: table above.

## Endpoints (confirmation set, seeds 17/29/43, paired cluster bootstrap by episode, 2000 resamples)
- Evaluated separately for each co-primary reader P ∈ {M1, M1-lin}; its fixed-query comparator F is B5 for M1
  and B5-lin for M1-lin. Every pass requires the pooled **97.5%** CI lower bound > 0 (Bonferroni over the two
  readers); 95% CIs are also reported. Pooled = mean over seeds of per-seed paired differences, episodes
  resampled jointly across seeds.
- **G1 depth effect** (per writer condition and reader P, same checkpoint, identical x/u/M across L):
  Δ_reach = Acc(L=4) − Acc(L=1) on hops {2,3} ≥ 3.0 pp with CI > 0 and positive in each seed;
  **walk specificity (A3)**: Δ_reach − Δ_far > 0 (CI), where Δ_far = Acc(L=4) − Acc(L=1) on hops {6,7,8}, which a
  4-read walk cannot reach; hop-1 accuracy at L=4 not below L=1 by more than 1.0 pp (pooled point estimate);
  replication: Δ_reach > 0 and Δ_reach − Δ_far > 0 (CI) in the condition's replication setting.
- **G2 evolving query and memory**: P@L=4 − F@L=4 > 0 and P@L=4 − B3@L=4 > 0 on hops {2,3} (CI).
- A condition passes for reader P if P meets G1 and G2; claims name the passing reader(s).
- Interpretation by condition (plan §11.5): the plan's protocol is `t1_frozen`. If G1/G2 hold only for
  `oracle_*` → the reader works but learned codes do not compose (writer problem). If they hold for `joint`
  with identical-snapshot depth sweeps → keep the fixed-memory read-depth claim, noting that composable codes
  need joint training. If no condition passes → the mechanism fails in this parameterization.
- Reported, not gating: every criterion's components; L=8 vs L=4 on hops 4–8 (H6); M1 vs M1-lin at L=4; B4 vs
  M1@L=1; r = 16; screening-set values for seed 17; `test_unseen`; counterfactual consistency
  (`replace_key_fact` prediction follows the new answer; `swap_memory` drops to chance); training curves;
  `swap_successors` follow rate (A5): on the screening set, the share of rows whose prediction is the new
  terminal, at L = 1, 4, 8, against the share predicting it on the unswapped episode.

## Stop rules
- No condition passes G1 for either reader → do not scale to P1-L; allow one follow-up F1 only if a specific
  observed failure motivates it (e.g. both readers learn hop 1 but neither composes → bias/width ablation per
  docs/theory.md Prop. 6, or a writer-alignment diagnostic).
- Single-hop (Pilot-1) below 0.95 at every E for r = 32 → diagnose the writer before any T2 conclusion. Triggered;
  diagnosis in `docs/reports/P1S-pilot.md` (shared-head confusions, loss still falling at 5000 steps); learned-writer
  T2 results are reported together with the writer's single-hop accuracy.

## Budget
Main runs (WP-H) ≤ 8 node·h; P1-S total ≤ 13 node·h (4.04 used by pilots); every job ≤ 8 nodes × 1 h (user cap).
If measured throughput projects WP-H above 8 node·h, steps become 15000 for every stage, decided before any main
run starts and recorded.

## Amendments
- A1 (2026-10-26, before any pilot or test result was read by the orchestrator): M1-lin promoted to co-primary
  with its own fixed-query comparator B5-lin, Bonferroni 97.5% CIs for passes. Reason: docs/theory.md Prop. 6
  shows a bias-free r-wide ReLU cannot pass a positively spanning set of signed codes through Q φ(U ·), so the
  ReLU instantiation can fail for a reason unrelated to the evolving-query mechanism; testing only M1 would
  confound the two. Pilot-2 adds B5-lin to its lr selection.
- A2 (2026-10-26, after the validation pilots, before any test evaluation): steps 5000 → 20000 for every training
  stage and variant (steps were a pilot-determined field). Reason: in Pilot-2 the primary readers' L = 4 batch
  losses were still falling at 5000 steps (oracle_orth M1/M1-lin, oracle_random M1-lin, joint M1), while
  single-read baselines plateaued by ~2000 steps; equal steps keep the comparison fair. Selected lrs are kept.
- A3 (same time): walk-specificity criterion added to G1 and to its replication. Reason: Pilot-2 showed every
  learned-writer variant, including single-read B4, at 41–50% on every hop 1–8 with flat depth profiles — a
  pattern consistent with guessing among the layout's two chain terminals (a global statistic a learned writer can
  store because all chain facts share relation 0), not with following edges. A depth gain on hops {2,3} only
  counts if it exceeds the gain on hops {6,7,8}. Oracle writers store pure entity codes; their single reads were
  exactly hop-1-only (33.3%). The criterion makes passing harder, never easier.
- A4 (same time): main-run matrix fixed as above (replication limited to the two co-primary readers whose depth
  contrasts it tests; r = 16 limited to four readers and one seed), to fit the budget at 20000 steps.
- A5 (2026-10-26 21:19, commit f6e5cb9, before any test evaluation; diagnostic only, never gating): `swap_successors`
  counterfactual (`docs/tasks/WP-B2_swap_successors.md`) — rewire two chains so the terminal set is unchanged but
  the correct terminal changes; walking follows the new terminal, terminal guessing cannot. Evaluated on saved
  checkpoints after WP-B2 merges.
- A6 (2026-10-26 22:20, before any confirmation-set metric of WP-H or F1 was computed): a second follow-up, F2
  (`docs/prereg/P1S-F2.md`, input-factorized learned writer), is registered. This deviates from the stop rule's
  "one follow-up F1". Reason: the pilots' learned-writer failure (terminal guessing) has a specific mechanism
  (`docs/theory.md` Prop. 10) that F1 (rank) cannot address, and P1-L can only use learned writers. F2 is
  registered before any result it could be tuned to; it changes no P1-S gate, threshold or claim.
