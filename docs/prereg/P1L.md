# P1-L pre-registration — single module in Qwen3-0.6B-Base (draft v1, 2026-10-26)

Status: **draft v1** (2026-10-26, after the P1-S decision). P1-S passed G1/G2 for M1 with `oracle_orth` and
`joint`, so P1-L may run; it is finalised after the writer diagnosis (WP-L2; Pilot-1 found no admissible E*_L)
and Pilot-2 (validation only).

## Question
Does the P1-S mechanism survive when x comes from a real LM? Same question as P1-S, plus: does the memory add
information the frozen backbone cannot get otherwise (memory-only), and does it help when the text is still in
the backbone's attention (context-retained)?

## Fixed design
- Backbone `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, bf16, sdpa; module at layer 14
  `o_proj` (2048 → 1024), fp32 adapter/state, ρ = 0.25, α = 1 (r below).
- Text: WP-E templates; every entity a single-token name; answer = one name token; EM = greedy argmax over the
  full vocabulary at the answer position (primary); candidate-restricted accuracy secondary.
- Protocols: **memory-only** (primary for mechanism claims) and context-retained (secondary).
- Writer policy ⟨pilot: `fact_final` vs `all_tokens`, chosen by T1 val accuracy⟩.
- Writer condition carried from P1-S (`reports/decisions/P1S.md`): **`joint` only**; `t1_frozen` failed P1-S G1
  and is not carried.
- Variants: **M1** (the only reader that passed P1-S) with comparator B5, plus B4 and B3; M1-lin is not carried
  (failed G1 in all P1-S conditions at r = 32). Gate CIs at 95% (one primary reader). Baselines: B0 frozen base
  (both protocols), B1 LoRA and B2 single-step nonlinear adapter at the same position (both protocols), trained in
  the context-retained protocol with the same data and steps.
- Rank: r = r_m ∈ {32, 64}, fixed by Pilot-2 on validation before any test result (P1-S: joint M1 passes at r = 32
  with a small, seed-dependent effect; robust at r = 64 in F1). State 4 vs 16 KiB per sequence (FP32), priced in P2.
- Tasks and episode sets mirror P1-S (`t2_chain_end` primary, `t2_typed` replication, hops 1–3 train / 1–8 eval,
  confirmation set `test` [100000, 102000) untouched until gates).

## Pilots (validation only, seed 17)
- Pilot-1 (writer feasibility; runs in parallel with P1-S because it does not depend on P1-S results; task
  `docs/tasks/WP-L1_p1l_pilot1.md`): batch/procs by throughput; `write_policy` × lr ∈ {3e-4, 1e-3, 3e-3} at E = 16;
  then E ∈ {8, 12, 24} at the selected (policy, lr). Rule: E*_L = largest E ∈ {16, 24} with val single-hop EM
  ≥ 0.95, else 12 if ≥ 0.95, else 8 if ≥ 0.80 (flagged); otherwise no P1-L T2 run before a writer diagnosis.
- Pilot-2 (after P1-S): lr per (writer condition × carried reader) by val EM on hops 1–3 at L = 4 (B4: L = 1).

## Endpoints
Same G1/G2 definitions and thresholds as P1-S on the memory-only protocol (identical x/u/M across L verified by
hashes of the support snapshots and question activations). Context-retained results and B0/B1 comparisons are
reported with the protocol named; they do not gate.

## Budget
Each job ≤ 8 nodes × 1 h. P1-L total ≤ 16 node·h, re-estimated from the WP-E throughput pilot before submission.
