# P1-S follow-up F1 — code interference and rank (pre-registered 2026-10-26 22:00, before any WP-H result)

Motivation (validation pilots, `docs/reports/P1S-pilot.md`): with interference-free codes (`oracle_orth`, r = 32)
M1-lin walks the chain (94% on hops 1–3 at L = 4), with random codes (`oracle_random`, r = 32) depth gains are
small and hop-1 accuracy falls with L (83% → 47% from L = 1 to 4), and the jointly trained learned writer settles
on terminal guessing (~45–50% on every hop for every variant). docs/theory.md Prop. 4 predicts crosstalk of order
√(E/r) amplified with depth. F1 asks whether lowering interference by raising the rank restores composition
for random codes and lets joint training find walking instead of the shortcut. This is the P1-S stop rule's
single follow-up; it is motivated by the pilot failures, not by any test result.

## Design (everything not listed is as in `docs/prereg/P1S.md` final v1)
- Task: primary layout (`t2_chain_end`, E = 12, chains (8,4)); same episode sets, seeds 17/29/43, 20000 steps,
  batch 64, the P1-S lr of the same (writer, variant) pair.
- Conditions: `oracle_random` with r = r_m ∈ {64, 128}; `joint` with r = r_m ∈ {64, 128} (d_e = 64 backbone
  unchanged; learned codes are linear images of 64-d entity embeddings, so joint interference cannot fall below
  that of random 64-d vectors — reported as a limitation). r = 32 reference values come from WP-H.
- Variants: M1, M1-lin, B5, B5-lin, B3 (B4 omitted; single reads are covered by B5/B5-lin and WP-H).

## Endpoints (confirmation set; same machinery and thresholds as P1-S, 97.5% CIs)
- Per (condition, r, reader P): G1 with walk specificity (no replication requirement in F1; stated as a weaker
  design), G2 against F and B3, hop-1 criterion.
- Trend (reported): Δ_reach and hop-1 L=4 − L=1 as functions of r for `oracle_random` and `joint`, with the
  r = 32 WP-H values and the `oracle_orth` r = 32 ceiling.
- Decision use: if `joint` passes at some r while failing at r = 32, P1-L uses the smallest passing r (and P2
  must price the larger state); if `oracle_random` passes but `joint` never does, learned codes remain the
  bottleneck and P1-L is not started before a writer redesign is specified; if neither passes, the reader's
  composition is limited to near-orthogonal codes and the claim is narrowed accordingly.

## Budget
≤ 6 node·h; every job ≤ 8 nodes × 1 h.
