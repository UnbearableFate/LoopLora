# WP-L6 — LLM oracle code designs (sn-oracle; worktree LoopLora-wt/oracle, branch wp/oracle)

Read first: `docs/tasks/_common_rules.md`, `plan/followup_2026-10-27.md` (§1, §3 F5/F6), `docs/prereg/P1L.md`
(all, especially Amendment B7), `docs/reports/P1L-state.md`, `scripts/analyze_oracle_capacity.py`.
Code: `src/looplora/llm/oracle.py`, `src/looplora/sym/writers.py` (`OracleWriter`), `src/looplora/llm/train.py`,
`scripts/run_p1l_main.py`, `scripts/launch_p1l_main.py`, `scripts/evaluate_p1l_main.py`, `configs/p1l/`,
`tests/test_llm_oracle.py`.

Why: P1-L's `oracle_random` at r = 32 never learned (all memory readers at chance). Its keys and values are
independent random tables, so the LM path must learn three random maps. Test two code designs that make the
transition (and, for O-shared, the LM-side maps) learnable.

## Step 1 (now, no compute)
1. Writers, selectable by config `writer`, r_mem ∈ {32, 128}, same fact/sentinel handling as `OracleRandomWriter`:
   - `oracle_rot`: key = normalize(Π_rel · g_e), Π_rel a fixed orthogonal matrix per relation from QR of a
     Gaussian seeded by `derive_seed(oracle_seed, "oracle", "rotation", rel)` (fix the QR sign); value = g_e
     (`global_codes`, as now).
   - `oracle_shared`: a trainable table C (init = `global_codes`); key = normalize(Π_rel · C_e),
     value = normalize(C_e); C is a writer parameter trained with the reader (state your optimizer choice for C,
     e.g. same lr, no weight decay, in the prereg).
   - `oracle_random` must stay bitwise unchanged (test against the current implementation).
2. Training, export/reload and writer signatures for both; reader parameter counts unchanged; C counted separately
   as writer parameters in accounting/records.
3. Tests (run in step 2): rotation orthogonality and unit keys; exact overwrite at β = 1; value→key map is the
   fixed linear Π_0 for `oracle_rot`; gradient reaches C for `oracle_shared` and not for the fixed writers;
   tiny-model train/export/reload for both; off-path equality.
4. Configs for the pilot: r = r_mem = 32, M1-L4, lr 3e-3, seed 17, 5,666 steps, T2 exactly as P1-L, memory-only,
   val 1000 episodes at seed 4242, evaluation at L = 1…4 on hops 1–8.
5. Pre-registration drafts: `docs/prereg/WP-L6.md` (pilot; pass = hop-1 val full EM at L = 4 ≥ 30% at the final
   checkpoint; choose O-rot if it passes, else O-shared if it passes, else stop; selection reads hop 1 only) and
   `docs/prereg/P1L-S.md` (runs iff WP-L6 passes; P1-L r = 32 design with the selected oracle; own Pilot-2;
   S = 17,000; G1-L, G2-L, Amendment B6 rule, B7 memory-use check; confirmation test [110000, 112000) seed 4242;
   95% CIs; B0–B2 not rerun). Include node·h estimates from P1-L's measured step times.
6. Step-2 commands (tests job, 10% time-only smoke, pilot job(s); pack the two arms in one 1-node job only if the
   smoke shows both fit ≤ 1 h with the 1.5 margin).
Report: `docs/reports/WP-L6.md` (state + commands). Done line in your final reply: `WP-L6 step 1 done`.

## Step 2 / 3 (only when the orchestrator says so)
Run tests, smoke and the pilot; apply the registered decision; report. Then P1L-S if it passes.
