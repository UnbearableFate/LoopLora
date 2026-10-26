# P1-L read fix — complete

**P1-L stops; no writer selected.** Final learned EM at rank 32: R 12.0%,
Rρ 12.2%, RK 12.0%; at rank 128: W5 29.9%, R 33.5%. The best is below both
the 80% writer gate and 50% E8 trigger, so no E8 retrain occurred. Diagnostic
rank-32 OAρ 74.7% and rank-128 OA 84.0% are ineligible. The immutable selection
is in `selection.json`.

Report: `docs/reports/P1L-read-fix.md`; task:
`docs/tasks/WP-L4_p1l_read_fix.md` including A1/A2; before-results choices:
`docs/reports/WP-L4-plan.md`. Raw artifacts persist under
`/work/xg24i002/x10041/LoopLora/runs/p1l/read-fix-20261027/`:

- `arms/{R,Rrho,RK,OArho}`: original rank32 runs, frozen source
  `0b03d606f3c752cddfcfd716946dbbc3e36bf025`, immutable `release.json`.
- `arms-r128/{W5,R,OA}`: added rank128 runs, frozen source
  `be07f7fb34c1cdd14b64f65b7f687364fcd6e36f`, `release-r128.json`.
- Each arm retains record.json, training.json, train_log.jsonl, writer.pt,
  checkpoint.pt and every val-STEP/{metrics.json,samples.npz}. The checkpoint
  retains the training head/optimizer; inference writer.pt omits both.
- `audit/{OA,W5}` retains the no-training final WP-L3 A1 audit, including all
  b/a/d coefficient vectors, keys/values/Gram/raw queries/matrices and paired
  original/zero/shuffle samples. `audit.csv`, `audit-age.csv`, and
  `audit-summary.json` preserve scalar and gradient evidence here.
- `smoke/` and `smoke-r128/` retain timing smokes; all export-check directories
  here preserve the independent final head-free inference samples.

All scientific runs used T1 E16, four relations, head sharing0.3, seed17,
layer14 o_proj, L1, B32/Q4, lr0.003, scan_final and prefetch2; val1000 episodes
at seed4242. No test/confirmation, T2, reader/joint or additional-seed science
occurred. OA means oracle addressing of learned values, not WP-C7's external
random writer. W5 uses pooled256-training-episode input normalization, normalized
address InfoNCE (weight1, temperature0.1), and clean-value CE (weight1). R adds
raw final-question read regression with the entire target/denominator stopped.
Rρ/OAρ use ρ0.5; the others0.25. RK adds squared key cosine over distinct
addresses. All extra objectives have weight1. Auxiliary labels exceed answer-only
supervision. W5/R at rank128 have132096 training-only head parameters; original
rank32 learned arms have33792. All inference exports are head-free.

A1 arrived before any scientific submission: each arm uses its largest
multiple-of-five step count satisfying
`1.5 * (S * own_smoke_mean_step + N_eval * own_max_eval100_seconds * 10) + 90 <= 7200`.
Primary validation has five equal intervals. OAρ additionally has one
predeclared6580-step diagnostic, reserved as a sixth evaluation; it does not
select a checkpoint. The original823-step smoke was10% of provisional8225;
A1's expanded schedules reused those measured timings. `release-before-A1.json`
retains the superseded proposal. New rank128 smokes used2400 steps (10% of
provisional24000) and100 validation episodes; after maximal sizing their
fractions are9.74% for W5/R and7.70% for OA, explicitly retained in the releases.
No WP-L4 accuracy determined step counts. A2 removed the study-total cap; every job
remained one node and at most2h, using submit.py --scale-up with the task.

| Rank | Arm | Steps | Primary checkpoints |
| --- | --- | ---: | --- |
|32|R|23590|4718 / 9436 / 14154 / 18872 / 23590|
|32|Rρ|24220|4844 / 9688 / 14532 / 19376 / 24220|
|32|RK|21980|4396 / 8792 / 13188 / 17584 / 21980|
|32|OAρ|27040|5408 / 10816 / 16224 / 21632 / 27040|
|128|W5|24635|4927 / 9854 / 14781 / 19708 / 24635|
|128|R|24645|4929 / 9858 / 14787 / 19716 / 24645|
|128|OA|31175|6235 / 12470 / 18705 / 24940 / 31175|

Master2844d5e (through df21654) and A2 were merged before any WP-L4 score was
read. All three rank128 scientific arms were submitted04:56:45JST and started
04:57:31, in parallel with unchanged original jobs. No WP-L4 validation score
was read before these submissions; `jobs.json` records that boundary. Rank128
retained lr0.003 without a new search, as requested, and is flagged. Rank32
and rank128 budgets differ, so cross-rank comparisons do not isolate rank from
training steps or parameter count.

Tests: initial3715417 had two failures (BF16-to-NumPy audit export, and an
untrained shuffle-logit assertion). The export casts to float32; the control
test verifies the exact deployed original/zero/donor memory and unchanged
question inputs, without requiring rounded random logits to differ. These were
fixed before science; corrected3715437 passed55. A1 tests3715659 passed57 and
independently reconstructed all audit vectors and matched prior A1 predictions.
A2 tests3716019 onmg0851 passed69 (zero failures/skips), using sourcebe07f7f:

```sh
scripts/submit.py --name t-l4-r128 --queue regular-g --nodes 1 \
  --walltime 00:05:00 --commit be07f7fb34c1cdd14b64f65b7f687364fcd6e36f -- \
  -m pytest -q tests/test_llm_read_fix.py tests/test_llm_writer_fix.py \
  tests/test_llm_prefetch.py
```

Original smokes3715461/62 and artifact audit3715504 passed. No-training A1
baseline audit3715621 onmg0837 and its independent proof3715659 onmg0844 passed.
Rank128 smokes3716039 onmg0855 and3716040 onmg0897 passed; gate3716145 onmg0842
verified all sample/config/finiteness/normalization artifacts, square memory,
and identical100 predictions after head-free reload. It exposed only timings
and artifact status, freezing rank128 release2fd95f9 before scientific training.

Final compute audit3716541 onmg0859 (sourceb5ff4ee) exited0 and reproduced every
sample metric, paired label, config/source and maximal timing schedule. It
locked all seven E16 runs, then verified all seven exports against1000 saved
predictions each without training heads. `summary.json` records all checks;
`e16-summary.json` is the preceding selection pass. The only historical config
exception is removal of the two known WP-C7 defaults writer/oracle_seed when
checking original0b03d60 records; every other field remains strict. No scientific
run was altered or repeated. The analyzer command is:

```sh
scripts/analyze_p1l_read_fix.py --root <persistent-root> --out <fresh-report-dir> \
  --release <absolute-release.json> \
  --rank128-release <absolute-release-r128.json> --verify-export
```

Run it only on compute via submit.py with an explicit source hash; HEAD in
submit.py resolves in the main checkout. An E16-only pass adds --e16-only and
omits --verify-export. The immutable candidate order is r32 R<Rρ<RK, then
r128 W5<R. A single best-arm E8 branch is implemented/tested but did not trigger;
its CLI requires the audited retry_e8 lock and the selected rank. No further
writer training or screening is authorized by WP-L4.

`runs.csv` holds all36 checkpoints, including the matched diagnostic and all
norm/read/head/key metrics. `loss_curves.csv` holds every logged objective and
gradient clipping flag. `em_curves.png` and `loss_curves.png` cover every arm.
All raw samples use the flat WP-D-compatible schema; rank is retained in run
records, release metadata, paths, CSVs and export-check names.

**Cost: 20642 allocated seconds = 5.73389 node·h, 16 jobs**, including failed tests
and every smoke/audit/scientific job. See `jobs.json` and the report table. Queue
wait is excluded. All scientific jobs exited0, every job stayed within the
per-job cap, and no pending job remains. Final compute ended06:47JST, before
the08:00 normal-operation cutoff. No torch import or pytest ran on login.
