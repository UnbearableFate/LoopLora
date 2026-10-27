# Review of wp/p1l @ c5de5c2 (B7 memory-use check and P1L-C), read-only, static

Scope: `git diff master...wp/p1l` (merge-base dc659e0): 8 files, +1994/-1. Worktree
`/work/xg24i002/x10041/LoopLora-wt/p1l`. Nothing in the repo or its worktrees was edited, created, committed or
deleted; torch was never imported and pytest was never run (shared login node). Scratch files (harnesses, bytecode
cache) are only under this directory.

## Verdict

No defect was found that would make a registered run crash, mis-measure, or violate the registration
(P1L Amendment B7 rule 1, P1L-C including Clarification C1). Two low-severity findings (provenance audit gap,
registration-text drift on the branch that will be pinned) and a few notes are listed below. Everything that needs
torch or a GPU could only be traced by hand (see "Could not verify").

## Findings, ranked

### L1 (Low) The P1L-C audit does not require one source across the nine runs

- Where: `scripts/finalize_p1l_causal.py:178-182` (`shared = [...]` list compared across runs through
  `reference.setdefault`); the claim it contradicts is in `docs/reports/P1L-C-state.md` step C notes ("keep the
  `--out` root and source fixed across the three jobs, as the finalizer compares hashes across all nine runs").
- Failure scenario: three jobs (one per training seed) are submitted with `--commit "$P1LC_SOURCE"`. After an
  interruption, one seed is resubmitted from a later HEAD (a fix to `p1lc.py` or `evaluate_p1l_causal.py` was
  committed in between). The compared hashes (`spec_hash`, `episodes_hash`, `read_position_memory_hashes`,
  `worlds_hash`, `bridge_memory_hashes`) depend on episodes, worlds and the oracle memory only, not on code, so the
  finalizer accepts nine records from two source trees and reports one study. The per-run `git_commit` and
  `source_hash` are copied to `run-manifest.json` (line 190) but nothing flags a mixture.
- Fix: append `"git_commit", "source_hash"` to `shared` (line 178). `source_metadata()` always writes both keys
  (`src/looplora/sym/run.py:25-72`), and the audit fixture in `tests/test_p1lc_stats.py:239-240` already gives every
  run the same two values, so existing tests keep passing; add one case with a differing `source_hash` to
  `test_audit_loads_every_run_and_rejects_changed_inputs_or_mixed_episodes`.

### L2 (Low, process) The branch that will be pinned lacks the registered Clarification C1 and the B7 wording note

- Where: `docs/prereg/P1L-C.md` and `docs/prereg/P1L.md` on wp/p1l. `grep -c "Clarification C1"` is 0 in the
  worktree's `P1L-C.md` and `grep -c "Wording note"` is 0 in its `P1L.md`; master has both (commits f48dc00 and
  ae15f73, after the merge-base dc659e0).
- Failure scenario: `P1L_SOURCE` and `P1LC_SOURCE` are `git rev-parse HEAD` of wp/p1l (P1L-state.md step 2,
  P1L-C-state.md step C), and each run records that commit as `git_commit`. The pinned tree therefore carries the
  pre-clarification text ("b_B ... drawn with a seed derived from the episode index") while the pinned code (correctly,
  per C1 on master) uses the other chain's final edge deterministically and draws only c. An auditor reading the pinned
  commit sees a code/registration mismatch that only master resolves.
- Fix: merge master into wp/p1l before the authorization commit. The only code that master added since dc659e0 is the
  new `scripts/summarize_p1l_training_logs.py` (`git diff --stat dc659e0 master -- src scripts configs tests
  pyproject.toml uv.lock`), so the merge does not touch the confirmation path. The code itself matches master's C1
  (verified below).

## Notes (no failure mode today; not defects)

- N1. `scripts/evaluate_p1l_causal.py:54`: the equality of `make_task_spec(cfg)` and `make_task_spec(confirmation)` cannot
  fail, because `make_task_spec(..., evaluation=True)` reads only `task` and `eval.hops` (`src/looplora/llm/train.py:31-39`)
  and the two configs differ only in `eval.splits` and `allow_confirmation`. The real protection is that
  `evaluation_config` builds the config exactly as `scripts/evaluate_p1l_main.py:234-236` does (checkpoint config merged
  with `configs/p1l/main/eval-confirmation.json`), and `generate(...)` is called as in `llm/evaluate.py:144-146`. The
  state doc's wording "task spec equal to the confirmation spec, asserted" overstates the check.
- N2. `scripts/evaluate_p1l_causal.py:141` (`out.mkdir()` without `exist_ok`) is intentional (no overwrite), but after a
  walltime kill the half-written run directory blocks resubmission of that run: the operator must remove it by hand and
  submit only the missing runs. The state doc says "inspect ... existing run directories"; it does not say to delete a
  partial one.
- N3. `finalize()` wiring of B7 (`scripts/finalize_p1l_main.py:219-220, 229, 243` and the extended final print) is not driven
  by any test; each helper it calls is tested and the wiring read correct. `result["rank"]` and `result["authorization"]`
  (lines 221-222) are pre-existing keys, which is what `require_memory_use` in `evaluate_p1l_causal.py:89-93` expects.
- N4. Coverage: the B4 path (depth 1) through `read_position_samples` is not run end to end in the GPU tests (the
  parametrized `experiment` fixture is M1 and B8 only); B4 uses the same adapter class as M1 (`core_variant` "iterq"),
  `read_control(last_only=True)` does not touch `_read`, and `resolve_config` forces `train.depths == [1]` for B4, so the
  risk is low.
- N5. The GPU tests assert bitwise `torch.equal` on layer-0 outputs and logits across forwards that differ only in memory
  values or in which rows are patched. This holds if GEMM output rows do not depend on other rows' values and kernels are
  deterministic (the state doc already says to investigate before relaxing). Layer 0 was chosen so the adapter input is
  independent of memory, which makes the module-level comparisons sound.

## Verified OK

### B7 memory-use check (`scripts/finalize_p1l_main.py`)
- Contrasts: B4@1, B8@4, M1-L4@4, each minus B3-L4@4, hop 1 only, via the existing `endpoint()`
  (`condition="memory_only/original"`, `split="confirm"`, 2000 draws, paired, pooled over seeds); `passed` is
  `all(ci_low > 0)` at `confidence = 1 - .05/gated_conditions` (97.5% for k = 2), as in Amendment B7 rule 1 and as the
  gates use it. Depths match the registered readers (B4 trained at L = 1, B8 and M1-L4 at L = 4; the samples exist at those
  depths).
- `endpoints()`, `endpoint()` and every gate (G1-L, G2-L, B6 contrasts) are untouched (diff); `compare_p1l_ranks.py` imports
  only `endpoint`; no other script reads `endpoints.json` keys of P1-L. New keys `memory_use` and `memory_use_passed` are
  additive; `memory-use.csv` rows are consistent (`write_csv` takes the union of keys).
- results.md text: failure sentence worded as the amendment, Rule 2 note at r = 32, Rule 3 note at r = 128 only when passed.
  G1-L/G2-L lines still appear before the B7 section, as the amendment says outcomes are "reported but not interpreted".
- `require_memory_use` fails closed on a missing `memory_use_passed`, rank != 128 or a different authorization record.

### P1L-C library (`src/looplora/llm/p1lc.py`)
- `git diff dc659e0 --stat -- src/looplora` lists only the new file; nothing except the P1L-C scripts and tests imports it.
  Hooks are off by default: `score_logits(control=None)` is the plain path; `read_control` is a per-instance
  `unittest.mock.patch.object` of `process_sequence`, `_masked_branch` (and `_read` only for capture/donor) restored on exit
  (instance attributes are deleted again, the class methods stay).
- `read_states` reproduces `IterativeQueryAdapter._read` / `UnsharedAdapter._read` operation by operation
  (`core/adapters.py:216-221, 248-253`), and `scripts/readout_p1l.py:reader_states` is the same loop.
- Final position = `last_positions(question_mask)`, equal to the index `answer_scores` gathers (`llm/protocols.py:171-175`);
  rows are the episode-major B*Q flattening that `LLMExperiment.score` uses; the row selection in `masked_branch` equals the
  one in `AdapterBase._masked_branch`; one forward per `score`, with the `process_sequence` signature the wrapper calls.
- Patch semantics match C1: only z_1 is replaced at the final position (or at every real position for the secondary), the
  recipient's own `u` and world A's memory are used for steps 2..L; the random control is one direction per episode and
  position from `derive_seed(seed, "p1lc", "random-direction", absolute index)` with norm ‖z_1^B - z_1^A‖. Analysis A
  `last_only` zeroes memory at every position except each row's last, leaving the memory-free branch.
- The adapter at layer 14 `o_proj` (layer 0 in the tests) receives an input that does not depend on its own memory
  contribution at that layer, so z_1 of worlds A/B/C at the same position share the same `u`.
- Worlds, on the real generator (numpy-only extraction of `select_bridge_worlds`, `chain_paths`, `keep_queries`,
  `stratify_queries`): test [102000, 104000) seed 4242 yields the 256 worlds from episodes 102000..102527 (49.4% of episodes
  qualify), val [0, 64) yields 16 (episodes 0..41); 0 invariant violations in both: both continuations in every memory,
  differences only in the first edge's tail, a two-hop walk to the chain end in A (t_A) and in B (t_B, the other chain's end),
  a one-hop dead end in C, c a seen-pool entity outside the episode, t_A != t_B. Selection and c depend only on the episode.
- Shared tokens (`include_support=False`) are passed to `prepare` for worlds A/B/C, `tokenize_batch` token ids are the
  `NamePool.token_ids` used for `token_t_A/B/c`, and answer tokens are single tokens.

### P1L-C scripts
- `evaluate_p1l_causal.py`: window test [102000, 104000) seed 4242 (REGISTERED_RANGE), smoke val [0, 64), checks before any
  test episode (`require_memory_use`, `check_training(check_all=False)`), readers and depths, no duplicate keys in the
  `save(...)` call, record keys match what `audit()` expects (`git_commit`, `source_hash`, `job_id` come from
  `source_metadata()` and `runtime()`), batch size 16 as in the confirmation.
- `finalize_p1l_causal.py`: per-seed shift = rate(t_B | B) - max(rate(t_B | C), rate(t_B | R)); pooled = mean over seeds;
  95% percentile bootstrap over the 256 shared episodes with the max recomputed in every draw; claim = pooled >= 10 pp and
  ci_low > 0 and every seed > 0 on the final-position patch; Analysis A final - deployed on hops 1, 2, 3 and 1-3 pooled;
  replication M1-L4@4 and B8@4 minus B4@1 on hops {2, 3} at 95% and 97.5%. JSON/CSV/markdown writers handle the row shapes.
- The commands in `P1L-C-state.md` match the argparse of both scripts; the registered run needs `--authorization` and
  `--endpoints`; the smoke takes neither.

### Tests
- Executed by hand without torch or the pytest runner (Python 3.9, numpy 1.20, repo functions extracted by AST or loaded
  with stub heavy modules): `tests/test_p1lc_stats.py` 17 cases PASS, `tests/test_p1l_memory_use.py` 8 cases PASS, and the 7
  numpy-level cases of `tests/test_p1lc.py` (bridge worlds, unrelated entity, selection determinism, rejection cases,
  `keep_queries`, `chain_paths`, registered range and spec) PASS (26 qualifying episodes in val [0, 60) for the fixture).
- Hand-traced: the CPU adapter tests (neutrality, `last_only`, patch mechanics against `manual_branch`, `read_states`,
  error paths and restoration) and the tiny-Qwen tests (layer-0 hook placement, `question_geometry`, replica comparison,
  one-read exactness, end-to-end artifacts). No spurious failure found in tolerances, fixtures or imports; the tiny config
  follows `tests/test_llm_oracle.py` and `resolve_config` accepts it.
- Sensitivity: the B7 tests fail on wrong depth (B4@4 would raise), wrong hop, wrong confidence, `any` instead of `all`,
  or a swapped comparator; the p1lc tests compare against independently written walks and a manual branch.
- Static: `py_compile` of all 7 files OK (bytecode redirected here); `pyflakes` reports only the two unused imports
  (`LABELS`, `SEEDS`) already present in dc659e0.

## Could not verify

- Anything that needs torch or a GPU: the bitwise `torch.equal` assertions (tiny Qwen tests and the run-time audits in
  `read_position_samples` / `bridge_patch_samples`), real `read_control` behavior on a real adapter, deterministic-mode
  behavior of `torch.where`, `index_copy`, `index_select`, and the assertions that logits change under patches in bf16
  (depend on the random initialisation of the tiny model).
- The CPU adapter tests of `tests/test_p1lc.py` and all GPU tests were hand-traced, not executed.
- Python 3.13 / numpy 2.x behavior (the harness ran Python 3.9.25 / numpy 1.20.1 with shims for `numpy.typing.NDArray` and
  the PEP 604 alias in `sym/stats.py`).
- Runtime, walltime and memory (no measurement; the smoke is the intended measurement).
- The B7 `finalize()` wiring and `finalize_p1l_causal.main` on real artifacts (read, not run).

## Commands run (reproducibility)

- `git diff master...wp/p1l --stat`, `git merge-base master wp/p1l`, `git diff --stat dc659e0 master -- src scripts configs
  tests pyproject.toml uv.lock`, `git log --oneline wp/p1l..master`.
- `/usr/bin/python3 -I` scripts in this directory: `worlds_check.py` (real generator, both ranges), `harness.py` (stats
  and B7 tests), `harness2.py` (world-level tests of `test_p1lc.py`), each under `timeout 120`.
- `python3 -m py_compile` and `python3 -m pyflakes` with `PYTHONPYCACHEPREFIX` set to the `pyc/` directory here.
