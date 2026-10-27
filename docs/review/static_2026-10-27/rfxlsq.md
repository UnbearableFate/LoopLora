# Review of 17b27c2 (WP-L7 step 1b: RFX and LSQ to the registered rules)

Read-only static review of branch `wp/llm` (worktree `/work/xg24i002/x10041/LoopLora-wt/llm`, head 17b27c2), by a Sonnet 5.5
subagent on 2026-10-27. Nothing was run on torch, a GPU or pytest. Authoritative text: `docs/prereg/RFX.md` ("Data", "Cells",
"Decision rule", "Decisions at registration"), `docs/prereg/LSQ.md` (arms table, "Decision rule", "Decisions at
registration"), `docs/tasks/WP-L7_rfx_lsq.md` Step 1b, `docs/reports/WP-L5-state.md`.

## Result

No defect with a concrete failure mode was found in the 1b changes or in their interaction with the existing code. The
decision logic (`classify`, `storage_ranks`, `lsq_arms`, `check_arms`, `release_eta`, `scheduled_steps`, `choose`) was
executed torch-free, with the functions extracted from the worktree sources by `ast`, and agrees with the registered text.
What remains is one test-sensitivity note, one hand-off note for the step-2 scripts, and the things only a GPU run can
settle (listed at the end).

## Observations (none is a defect)

O1 (low, test gap). `tests/test_llm_rfx.py:335-339` checks the entity-restricted EM of `run_checkpoint` against an
independent recomputation, but the model of that test is a random-initialized tiny Qwen3 (`model` fixture, lines 38-46: hidden
16, 2 layers, vocab 151,936). Its argmax over the 1,024 name tokens hits the answer at chance (about 1/1024 per row, 8 rows),
so the recorded array and the recomputed one are both all-False in nearly every run, and the assertion would also pass for a
wrong `entity_ids` slice, a wrong axis pairing or a wrong `answer_entity` in the two-line glue of `evaluate`
(`rfx.py:407` builds the ids, `rfx.py:431` calls `entity_hits`). Failure scenario: a later edit of that glue is not caught
by the tests and `lm_em_entity` (hence `baseline.lm_em_entity` and the `decoder` rescue flag, `rfx.py:353,356`) is silently
wrong in the science run. The glue is correct as written; `entity_hits` itself is covered by `tests/test_rfx.py`, and
`answer_entity` by `tests/test_llm_rfx.py:214`. Cheap mitigation on the first compute run (no code change needed): in the
smoke and science records check `cells[v]['lm_em_entity'] >= cells[v]['lm_em'] - 0.005` for every variant (the restricted
argmax is over a subset that contains the full-vocabulary argmax, so it can only be lower through exact bf16 ties, see
state doc "Choices" item 2). For the trained checkpoints (EM 12-34%) this is a non-vacuous check of the glue.

O2 (info, by design). `lsq_run.release_arms` (`lsq_run.py:35-37`) can only check that a release's `arms` form a valid RFX
output (`check_arms`); the release holds no verdicts, so it cannot detect an `arms`, `rfx_eta_by_rank` or `eta_by_rank`
that was edited away from the RFX `summary.json`. `release_eta` (`lsq_run.py:40-48`) does check `eta_by_rank` against
`lsq_eta(rfx_eta_by_rank)`, but not `rfx_eta_by_rank` against the summary. The state doc says the release copies the three
fields verbatim; the step-2 script `prepare_lsq_release.py` (not yet written) should read `summary.json` itself and assert
`arms == summary['lsq_arms']` and the two eta maps equal.

O3 (info). Runtime of the RFX job could not be estimated here (see "could not verify"): the registered estimate is 2-3 min
per checkpoint with a 30 min walltime; the 64 + 64 smoke gives the measurement before the science job.

## Verified OK (evidence)

1. `classify` (`rfx.py:342-371`) implements the registered rule exactly.
   - Verdict order storage (`ridge/raw/fit`, or `ridge/unit/fit` while not `delta/unit/fit`), normalization (`delta/unit/fit`),
     query (`delta/true/fit`), store+query (`ridge/true/fit`), decoder, none: lines 357-368, RFX.md:125-136.
   - Baselines: `e_b = err(delta/raw/fit)` (351), `EM_b` and `EMe_b` = max of as-is and gain-fitted of delta/raw (352, 353).
   - Rescue: `err <= 0.5 e_b` and `lm_em - EM_b >= 0.10 - 1e-9`, with the full-vocabulary EM for cells (354-355); decoder:
     `refit['delta/raw']['acc'] - EMe_b >= 0.10 - 1e-9` (356), i.e. against the entity-restricted baseline. The tolerance is
     only on EM rates (k/n, correctly rounded float64 means); the 0.5 ratio has none, as registered.
   - Every rescue flag (14 variants + `decoder_refit`) is recorded (369) and reaches `summary.json['rescue']`
     (`rfx_run.py:68`). `rescue['delta/raw/fit']` and `rescue['delta/raw/asis']` cannot be true by construction.
   - Edge cases. Exact 10 pp and exact 0.5-ratio ties, several rescues at once, and all 64 combinations of the six decisive
     flags are covered by the differential test below. A NaN or inf in a cell compares False (no rescue) and cannot reach
     `summary.json` silently: `_write_json` uses `allow_nan=False` (`run.py:24`) and `rfx_run.run` writes `{label}.json`
     (line 93) before the summary. All 14 cells always exist (`pieces` is built from `VARIANTS`).
2. Entity-restricted EM. `entity_hits` (`rfx.py:151-156`) is the argmax over `index_select(-1, entity_ids)` compared with the
   entity index. `load_name_pool().token_ids` is a tuple of ints (`render.py:31`) and entity e renders as `token_ids[e]`
   (`render.py:207`), so the restricted argmax is directly an entity index; `answer_entity` is `batch.arrays.answer` as
   long (`rfx.py:135`). `token_ids[:V]` has V = 1,024 entries (RFX.md:94-95). `val` answers come from the seen pool
   [0, 768) (`relations.py:332`); the 256 unseen names are distractors in D1-restricted, D2 and D3 alike, as registered.
3. `lsq_arms`, `storage_ranks`, η. `rfx.py:374-391`: per rank normalization gives U, storage gives K4 and RG, else nothing,
   in candidate order; W5-r32 is inert. `rfx_run.decisions` (`rfx_run.py:62-70`) takes
   `parameters['eta']['unit']` (the (ridge, unit) cell) of `R-r{rank}` for storage ranks only and clips with `lsq_eta`
   (`lsq.py:23-27`, bounds 1e-4..1e-1). All 216 verdict triples (R-r32, R-r128, W5-r32) give 9 distinct outputs (8
   non-empty) equal to an independent transcription of the arm table of LSQ.md:62-71; `check_arms` accepts exactly those 8
   non-empty sets and rejects every other ordered selection of the six arms (all permutations of all subsets tried).
4. LSQ arm U. `lsq.configuration` (`lsq.py:42-68`) adds only `unit_query=True` to `read_fix.configuration('R', ...)`;
   `LSQExperiment.__init__` (`lsq.py:73-86`) checks arm, rank, store, `unit_query`, `dpfp_nu == 0` and that the store fields
   equal the registered ones. The unit query is `query_code` (`core/adapters.py:189`, `F.normalize` with `cfg.eps = 1e-6`),
   which `question_terms` uses (`read_fix.py:57`), so training loss, diagnostics and export see the same query.
5. Off path. The only edit in `read_fix.py` is the extra `or adapter.cfg.unit_query` in `diagnostics` (line 153).
   `unit_query` is set by no other arm family (grep over `src` and `configs`: only `lsq.py:57`), so every WP-L4/L5 arm
   takes the old branch; for U the logged condition numbers use `gram_conditions(keys, None, None)` as K4 does.
6. Release checks. `check_arms` (`lsq.py:30-39`): each arm once, candidate order, per rank [] or U alone or K4 + RG.
   `scheduled_steps` (`lsq_run.py:51-69`): E16 release names every triggered arm, E8 only the retrained one; smoke needs an
   explicit step count and no schedule. `release_eta` applies to RG only and compares the release's clipped value.
   `choose` (`lsq_run.py:72-87`): >= 800 selects the first in candidate order; best >= 500 retries at E8; ties keep the
   candidate order; requires every triggered scientific E16 arm, complete.
7. RFX data. Fit: `train`, indices 0..4095, `FIT_SEED = derive_seed(17,'rfx','fit') = 768129440952085103` (equals
   RFX.md:61), 4 queries per episode (`rfx.py:484`; tests assert 4). Eval: `val`, [200000, 204000), seed 4242,
   `stratify_queries(..., [1], 4242)` (`rfx.py:399`). Smoke: `train` index 10,000,000, seed
   `derive_seed(17,'rfx','smoke-eval')` (`rfx.py:397`), 64 + 64 episodes (`rfx_run.py:27`). The five files read no `test`
   split and no other use of val index >= 200000 or of train index 10,000,000 exists. The three checkpoint folders on disk
   match `CHECKPOINTS` (rank, steps) and `DIGESTS` (blake2b-128 of `writer.pt` and `checkpoint.pt`) and the values printed
   in RFX.md; the folder records are `complete`, non-smoke.
8. The hook is reachable. Both injected projections call `self.adapter.process_sequence(...)` in read-only mode
   (`core/wrapper.py:78`, `llm/residual.py:55`), `IterativeQueryAdapter._read` calls `self._query(...)`
   (`core/adapters.py:244`), and `process_sequence_fast` has no caller in `src` (tests only), so the instance-level shadows
   of `FinalRead` (`rfx.py:67-78`) are what the model executes; `FinalRead._query` repeats the deployed expressions
   (`adapters.py:228-231`), so re-inserting the recorded read can be bitwise.
9. Tests, torch-free: the repo's own pure verdict tests (`test_verdicts_follow_the_registered_decision_tree`, the 0.5 /
   10 pp rescue test, the entity-baseline decoder test, `lsq_arms`, and the 64-combination test) and the LSQ logic tests
   (release arms, hand-off, eta rule, `choose`, step counts) pass against the extracted code. Mutation check, 25 mutants
   of `classify` (swapped precedence, wrong baseline, strict inequalities, missing tolerance, draft-rule variants,
   wrong cells, wrong margin): 24 are caught by the verdict tests; the survivor (`>` instead of `>=` after the 1e-9
   slack) is behaviorally equivalent, since EM differences are multiples of 1/4000.

## Could not verify (needs torch, a GPU or the cluster)

- That any `tests/test_llm_*.py` test passes on the tiny model, in particular the bf16 tolerances of the new U-arm reads
  (the state doc says tolerances come from formulas, not runs) and the bitwise identity check (`identity_max_abs_logit_diff
  == 0.`, `rfx_run.py:54`) on GPU kernels.
- Real numerics: L-BFGS convergence of the D3 decoders within 100 iterations, bf16 exact ties between EM and EMe, memory and
  runtime of the RFX job (fit set 4,096 episodes x 4 queries in batches of 32; 19 question forwards per evaluation batch)
  against the 30 min walltime.
- That the checkpoints on disk are unchanged at the time of the run (digests match now; `verify_digests` re-checks
  before anything is created).
- The step-2 scripts (`prepare_lsq_release.py` and the table scripts) do not exist yet.

## Scratch material (all under the scratchpad `review-rfxlsq/`)

`extract.py` (ast extraction of pure functions), `diff_classify.py` (300,000 random tables against an exact-arithmetic
transcription of the rule; 0 mismatches; verdict histogram decoder 59,874, none 97,551, query 20,450, storage 72,031,
normalization 33,781, store+query 16,313), `mutate_classify.py`, `run_rfx_tests_torchfree.py`, `run_lsq_tests_torchfree.py`,
`arms_enum.py`, `digests.py`, `seeds_check.py`. All run with `/usr/bin/python3 -I`, stdlib only.
