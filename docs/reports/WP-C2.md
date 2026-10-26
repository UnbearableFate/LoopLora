# WP-C2 — input-factorized symbolic writer

Completed on `wp/core`, after merging `master` at `01fcfc5`. Implementation: `e552d74`; tests and smoke driver: `77e3cd5`.

`FactorizedWriter(d_in, r, r_mem, seed, ...)` splits the same seeded `LearnedWriter.A` into `A_k` (head/relation columns) and `A_v` (tail columns), retaining identical initial K/V/gates and the parameter budget. Keys/queries use only head/relation inputs; values use only tail inputs; memory uses the existing masked decay-first delta update. `joint_fact` and `t1_frozen_fact` work through model construction, T1 pretraining, internal/external writer checkpoints, freezing, joint export, full-checkpoint reload, evaluation/counterfactuals, and run provenance. The reader shares `A_k`, K/V/gates and declares 2*d_e input width. Only factorized checkpoint signatures add `writer_type: factorized`; existing signatures and the WP-H matrix are unchanged.

**Validation: 693 tests passed in 45.19 s**, job **3713279.opbs**, compute node **mg0119**, exit 0:

```sh
scripts/submit.py --name t-wpc2 --queue regular-g --walltime 00:10:00 \
  --worktree . --wait --tail 100 -- -m pytest -q -o addopts= \
  tests/test_sym*.py tests/test_reference_equivalence.py tests/test_causality.py \
  tests/test_state_lifecycle.py tests/test_gradient_flow.py tests/test_loop_invariants.py \
  tests/test_precision.py tests/test_contraction.py tests/test_accounting.py -rP
```

The new tests cover exact initial blocks/budget, query equivalence, output/input-gradient independence (also after updates), masks, shared parameters, T1 freeze/reload across all six F2 readers, joint updates, and end-to-end counterfactual evaluation. **All 24 legacy writer/reader combinations remained bit-identical** after training: every state tensor, prepared memory/query/value/code tensor, and L=0/1/2/4 logits match the baseline captured before production edits (job **3713238.opbs**, **mg0855**, exit 0). Fingerprints and environment metadata are committed in `tests/assets/sym_writer_legacy_v1.json`.

**Both GPU smokes passed**, job **3713316.opbs**, **mg0266 / NVIDIA GH200**, exit 0, 25 s application elapsed:

```sh
scripts/submit.py --name smoke-wpc2 --queue regular-g --walltime 00:05:00 \
  --commit 77e3cd5 --wait --tail 100 -- scripts/smoke_factorized_writer.py \
  --out /work/xg24i002/x10041/LoopLora/runs/p1s/wpc2-smoke/acceptance-v1
```

At r=r_mem=32, d_e=64, E=12, batch=64: `t1_frozen_fact` ran 100 T1 +100 reader steps (10.19 s total); `joint_fact` ran 200 steps (4.75 s total). Each produced 792 evaluation rows across val/test/test_unseen, L=0/1/2/4/8, and original/replace/remove/swap-memory/swap-successors conditions. The driver verified checkpoint roundtrips, exported writer equality and frozen/trainable status. Audit and run artifacts: `runs/p1s/wpc2-smoke/acceptance-v1/` under the main repository. PBS logs share the job-name prefixes under `/work/xg24i002/x10041/LoopLora/logs/pbs/`.

No deviations or open WP-C2 issues. No dependencies changed; no runtime/torch/pytest execution on login. Smokes are code acceptance only; no F2 pilot/main or formal confirmation set was run. Exact legacy fingerprints are scoped to the recorded PyTorch/CPU environment.
