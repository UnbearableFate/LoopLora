# WP-L7 — read-failure factorial (RFX) and regression writers (LSQ) (sn-llm; LoopLora-wt/llm, wp/llm)

Read first: `docs/tasks/_common_rules.md`, `plan/followup_2026-10-27.md` (§1, §3 F2–F4), `docs/prereg/DPFP.md`
(incl. Amendment A1), `docs/reports/P1L-read-fix.md`, `docs/review/rv-astra_2026-10-27.md` (proposal 2),
`docs/review/rv-fable_2026-10-27.md` (C3, E3, W1), your `docs/reports/WP-L5-state.md`.
Note: master (P1-L pipeline, follow-up plan) was merged into wp/llm as 77373ea; WP-L5 step-2 commands must use a
commit at or after it, and the DPFP tests must pass on it.

## Step 1 (now, no compute)
1. RFX (evaluation only) on the WP-L4 final checkpoints R/r128, R/r32 and W5/r32: cache K, V and the query q at
   the final question position on T1 train episodes (fitting set) and on a val index range never used before
   (evaluation set); replace only that position's read. Cells: storage {deployed delta, ridge
   M = V Kᵀ(K Kᵀ + ηI)⁻¹ with η chosen on the fitting set} × query {raw q, unit q with a scalar fitted on the
   fitting set, true source key}; plus clean value and zero memory. Decoders: the deployed LM path, the clean-value
   head, and a linear decoder refit on fitting-set reads. Metrics: normalized read error, read-head accuracy,
   full-vocabulary EM, NLL; write-age and write-order (permutation) sensitivity; condition numbers.
2. LSQ writer variants of the R objective: (a) ridge least-squares store with recency weights, differentiable
   solve, condition number logged; (b) K4 = the existing β = 1 delta scan repeated 4 times. Unit-normalized
   queries. Config flags; off path bitwise unchanged; tests (ridge interpolates exactly for independent keys as
   η → 0; K4 approaches the interpolant; gradients flow; export/reload).
3. Pre-registration drafts `docs/prereg/RFX.md` and `docs/prereg/LSQ.md` with the decision rules of the follow-up
   plan (RFX: "storage" if ridge + raw q halves normalized read error and adds ≥ 10 pp EM → triggers LSQ;
   "query" if only the true key rescues; "decoder" if only the refit decoder rescues. LSQ: DPFP's T1, step rule and
   gates — ≥ 0.80 select, ≥ 0.50 one E = 8 retrain; state the rank order). Node·h estimates from WP-L4 timings.
4. Update `docs/reports/WP-L5-state.md` with the new base commit and the step-2 order: DPFP tests and smokes,
   RFX, DPFP arms, LSQ if triggered.
Done line in your final reply: `WP-L7 step 1 done`.

## Step 1b (now, no compute): implement the registered rules
The orchestrator registered `docs/prereg/RFX.md` and `docs/prereg/LSQ.md` on master (077bb81, time fix 64f3e54; the
same text is on wp/llm as c3b01bf and its fix commit). The registered rules differ from the drafts. Read both files'
"Decision rule" and "Decisions at registration" sections first.
1. Merge master into wp/llm.
2. `rfx.py` / `rfx_run.py`:
   - Per variant, also compute the entity-restricted D1 EM (argmax over the 1,024 entity name tokens).
   - Rewrite `classify` to the registered verdict order: storage, normalization, query, store+query, decoder, none.
     Record every rescue flag, including ridge/unit/fit and delta/unit/fit. The decoder rule uses the
     entity-restricted baseline EMe_b.
   - Replace `lsq_ranks` with `lsq_arms` (arms in LSQ candidate order). `rfx_eta_by_rank` and `eta_by_rank` cover only
     ranks with a storage verdict.
3. `lsq.py` / `lsq_run.py`:
   - Add arms L32-U and L128-U: the delta store with its learned λ and β as R, `unit_query` on. Use the registered
     candidate order.
   - A release's `arms` equals RFX's `lsq_arms`; U and K4/RG are exclusive per rank. `release_eta` applies to RG arms
     only.
4. Tests for the new rules: verdict cases of `classify` from synthetic cell tables, `lsq_arms`, U configuration and
   validation, release checks. They stay unrun until compute is back.
5. Update `docs/reports/WP-L5-state.md` and any other doc that names `lsq_ranks` or calls the preregs drafts.
   `py_compile` and pyflakes; commit on wp/llm; no merge to master, no push.

Done line in your final reply: `WP-L7 step 1b done`.

## Step 1c (now, no compute): follow-ups from the static review
Review: `docs/review/static_2026-10-27/rfxlsq.md` (no defect in 17b27c2; O1 and O2 below).
1. O1: the entity-restricted EM glue (`rfx.py` building `entity_ids` and calling `entity_hits`) is not covered by a
   non-vacuous test: the tiny model hits entities only at chance.
   - Make the summary fail closed: if any variant of any checkpoint has `lm_em_entity < lm_em - 0.005`, raise before
     `summary.json` is written. The per-label JSON is already written first.
   - Why 0.005: the restricted argmax is over a subset that contains the full-vocabulary argmax, so only exact bf16
     ties can break the inequality.
   - Record per variant the number of rows with a full-vocabulary hit and a restricted miss, so a raise can be
     diagnosed.
   - Add a pure test. In step 2, also check the RFX smoke against this guard before the science job.
2. O2: write `scripts/prepare_lsq_release.py` now, stdlib only so it can run on the login node.
   - It reads RFX's `summary.json` and the LSQ smoke records and builds the release: `arms` = `summary['lsq_arms']`,
     with `rfx_eta_by_rank` and `eta_by_rank` copied.
   - Before writing, it asserts `arms == summary['lsq_arms']` and both eta maps equal to the summary's, and it
     passes `release_arms` and `release_eta`.
   - It refuses to overwrite an existing release.
   - Write `scripts/prepare_dpfp_release.py` the same way; `WP-L5-state.md` lists both as still to write. Pure tests
     for both.
3. Update `docs/reports/WP-L5-state.md`. Run `py_compile` and pyflakes. Commit on wp/llm; no merge to master, no push.
Done line in your final reply: `WP-L7 step 1c done`.

## Step 1d (now, no compute): second review of step 1c
Review: `docs/review/static_2026-10-27/release.md`. It found no defect in the release path. Fed WP-L4's real smoke
records, the new path reproduces the frozen A2 release exactly. Two findings:
1. **F1 (ties).** The restricted argmax breaks exact bf16 ties by entity index (a seeded shuffle). The full-vocabulary
   argmax breaks them by token id. So a correct glue can still give full-hit/restricted-miss rows. On the 64-row smoke,
   a net 2-row deficit in one of the 42 cells trips the guard with a probability of a few percent, and the smoke is
   deterministic.
   - Fix: break ties as the full-vocabulary argmax does. That is, take the argmax over the logits with every non-entity
     token masked to −inf, then map the winning token id back to its entity index. A correct glue then gives no such
     rows at all, and the guard stays as it is.
   - The registered metric ("argmax over the 1,024 entity name tokens") does not fix a tie rule, so this changes no
     registered quantity. Record it in the state doc.
   - Add a test: an exact tie between the answer and an entity whose entity index is lower but whose token id is higher.
2. **F2 (reload check).** In step 2, before a DPFP or LSQ release is committed, one short compute job per smoke arm
   reloads that arm's export (`load_inference`) and checks that its config equals the release's, as WP-L4's A2 gate did.
   Put this in the `WP-L5-state.md` step-2 commands.
3. Run `py_compile` and pyflakes. Commit on wp/llm; no merge, no push.
Done line in your final reply: `WP-L7 step 1d done`.

## Step 2 (only when the orchestrator says so)
WP-L5 step 2 (DPFP) and RFX; LSQ only for the arms RFX triggers (`lsq_arms`). Reports: `docs/reports/DPFP.md`,
`docs/reports/RFX.md`, `docs/reports/LSQ.md`.
