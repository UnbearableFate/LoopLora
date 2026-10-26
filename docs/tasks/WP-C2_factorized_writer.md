# WP-C2 — implement the input-factorized writer for F2 (code and tests only)

Read first: `docs/prereg/P1S-F2.md` (Design: writer `fact`), `docs/tasks/WP-F2_sym_factorized.md`
("Implementation", binding), `docs/theory.md` Prop. 10.5, `src/looplora/sym/{writers,model,config,run,main}.py`,
`docs/reports/WP-H-main-interface.md`. Branch/worktree: `wp/core` (merge `master` first). Report:
`docs/reports/WP-C2.md`.

Implement exactly the WP-F2 "Implementation" section so that configs with `writer: t1_frozen_fact` or
`joint_fact` run end to end through the existing sym pipeline (T1 writer pretraining and freezing for
`t1_frozen_fact`, joint training, evaluation incl. counterfactuals), with the listed tests plus one GPU smoke of
each condition (≤ 200 steps) on compute via `scripts/submit.py`. No pilot or main runs (wk-symf runs F2). Keep
the change minimal; existing writer conditions must produce bit-identical results (test it).
