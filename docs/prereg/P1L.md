# P1-L pre-registration — single module in Qwen3-0.6B-Base (draft v0, 2026-10-26)

Status: **draft**; finalised after P1-S decides which writer conditions and variants carry over, and after the
P1-L pilots (validation only). Runs only if P1-S passes G1 in at least one writer condition (P1S.md stop rules).

## Question
Does the P1-S mechanism survive when x comes from a real LM? Same question as P1-S, plus: does the memory add
information the frozen backbone cannot get otherwise (memory-only), and does it help when the text is still in
the backbone's attention (context-retained)?

## Fixed design
- Backbone `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, bf16, sdpa; module at layer 14
  `o_proj` (2048 → 1024), fp32 adapter/state, r = 32, ρ = 0.25, α = 1.
- Text: WP-E templates; every entity a single-token name; answer = one name token; EM = greedy argmax over the
  full vocabulary at the answer position (primary); candidate-restricted accuracy secondary.
- Protocols: **memory-only** (primary for mechanism claims) and context-retained (secondary).
- Writer policy ⟨pilot: `fact_final` vs `all_tokens`, chosen by T1 val accuracy⟩.
- Writer conditions carried from P1-S ⟨P1-S result⟩: `t1_frozen` (writer trained on rendered `t1`, n_rel = 4) and/or
  `joint`.
- Variants: M1, B4, B5, B3 (+ M1-lin if P1-S showed it differs from M1). Baselines: B0 frozen base (both
  protocols), B1 LoRA and B2 single-step nonlinear adapter at the same position (both protocols), trained in the
  context-retained protocol with the same data and steps.
- Tasks and episode sets mirror P1-S (`t2_chain_end` primary, `t2_typed` replication, hops 1–3 train / 1–8 eval,
  confirmation set `test` [100000, 102000) untouched until gates).

## Endpoints
Same G1/G2 definitions and thresholds as P1-S on the memory-only protocol (identical x/u/M across L verified by
hashes of the support snapshots and question activations). Context-retained results and B0/B1 comparisons are
reported with the protocol named; they do not gate.

## Budget
Each job ≤ 8 nodes × 1 h. P1-L total ≤ 16 node·h, re-estimated from the WP-E throughput pilot before submission.
