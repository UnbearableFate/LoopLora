# WP-C4 — P4-S extension: label-free stopping on r = 32 learned codes (evaluation only)

Read first: `docs/reports/P4S-stopping.md` (WP-C3: rules, conventions, threshold selection),
`reports/decisions/P1S.md` §7 (F2: factorized learned codes walk hop by hop but overshoot; hop-1 L4 − L1 is
−28 to −37 pp), `docs/reports/P1S-F2.md`. Branch/worktree: `wp/core` (merge `master` first). Report:
`docs/reports/P4S-stopping-r32.md`; outputs `reports/p4s-r32/` (compact CSV/MD, ≤ 1 MB in git).

## Question
Do the WP-C3 label-free stopping rules remove the overshoot of r = 32 learned walk codes: hop-1 accuracy within
1 pp of L = 1 while keeping a depth gain on hops {2, 3}?

## Design (no training; fixed checkpoints; WP-C3 code, rules, prediction convention and threshold selection)
- Checkpoints, r = 32, M1 and M1-lin, seeds 17/29/43 (18 in total; all released):
  F2 `joint_fact` and `t1_frozen_fact` (`/work/xg24i002/x10041/LoopLora-wt/symf/runs/p1s/f2/`), and WP-H `joint`
  (`/work/xg24i002/x10041/LoopLora/runs/p1s/main/20261026/`) as the unfactorized (shortcut) contrast.
- Thresholds per (family, criterion) on `val` (eval seed 4242), selected exactly as in WP-C3; report on the
  screening `test` set (eval seed 4242). No confirmation episodes.
- Report per family: accuracy by hop for fixed L ∈ {1, 4, 8, 16}, each rule, and the true-hop stop (label-informed
  reference, not an upper bound); mean steps; and the G1 quantities with the stopped reader in place of L = 4:
  hop-1 (rule − L1), reach (rule − L1 on hops {2, 3}), far (hops {6, 7, 8}), walk specificity, as screening
  point estimates with the WP-C3 bootstrap if it is already implemented.
- No latency benchmark and no contraction certificate (WP-C3 settled both).

## Report
≤ 1 page + tables: per family, whether any rule brings hop-1 within 1 pp of L = 1 with a positive reach gain
and positive walk specificity; a recommendation on whether a formal P4 test (fresh confirmation seed) is
warranted, with which rule and which checkpoints. Tests for any new code on compute. One node per job,
≤ 1 h; total ≤ 1 node·h. Final line: `WP-C4 done: docs/reports/P4S-stopping-r32.md`.
