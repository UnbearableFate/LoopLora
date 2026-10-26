# WP-H statistics interfaces (implementation contract)

No confirmation artifacts have been inspected. Runtime validation is delegated
to the root's Miyabi-G allocation.

`stats.pooled_walk_specificity(samples_by_seed, depth_a=4, depth_b=1, *,
reach_hops=(2,3), far_hops=(6,7,8), split='test', condition='original',
resamples=2000, seed=2718, confidence=0.95) -> dict` pairs query labels across
depths and seeds. It computes each episode's mean reachable-hop depth effect
minus its mean far-hop depth effect, then averages episodes and training seeds.
Bootstrap draws resample these complete episodes jointly across seeds; the two
hop subsets are never resampled independently. Every episode must cover every
requested hop. Both confidences use identical draws. Existing
`pooled_compare_runs` remains unchanged.

`gates.evaluate_gate_table(endpoints, *, evaluation_seed) -> list[dict]` accepts
exactly eight `(writer, reader)` keys: four preregistered writers crossed with
`M1`/`M1-lin`. Each value maps nine endpoint names to
`{'report': stats_at_0_95, 'gate': stats_at_0_975}`:
`reach`, `far`, `specificity`, `hop1`, `replication_reach`, `replication_far`,
`replication_specificity`, `vs_fixed`, `vs_loop`. Ordinary endpoints use
`pooled_compare_runs`; specificity endpoints use `pooled_walk_specificity`.
The caller supplies `evaluation_seed=777001` from the audited evaluation
manifest, and must wait for root's confirmation release before reading files.

The gate module validates confidence pairing, endpoint depths/hops, training
seeds `{17,29,43}`, `test/original`, 2000 episodes, 2000 resamples, and sample
counts consistent with one query per requested hop per episode. It applies
the preregistered nine predicates directly: primary reach estimate >=0.03,
reach lower CI >0, reach positive in every seed, specificity lower CI >0,
hop1 estimate >=-0.01, replication reach/specificity lower CIs >0, and both G2
lower CIs >0. `far` and `replication_far` are reported components, with no extra
pass criterion. G1/G2/overall are conjunctions; intervals are never rounded
before decisions. Output includes criterion values, thresholds, both confidence
summaries, and the proper B5/B5-lin comparator. Row keys are `writer`, `reader`,
`fixed_query_comparator`, `evaluation_seed`, `gate_confidence`,
`report_confidence`, `criteria`, `g1_passed`, `g2_passed`, `passed`, `statistics`.
Each criterion records `value`, `threshold`, `operator`, and `passed`.

Written tests cover joint covariance, per-episode specificity, missing/mismatched
labels and hops, confidence consistency, exact threshold equality, strict-zero
CI/per-seed failures, and exactly eight gate rows. Only synthetic fixtures are
used. Files are `tests/test_sym_specificity.py` and `tests/test_sym_gates.py`;
no runtime validation has been executed by this worker. Static whitespace
checks passed. The main-analysis worker owns the whole-run reporting script.
Suggested compute validation: `pytest tests/test_sym_specificity.py
tests/test_sym_gates.py tests/test_sym_multiseed.py tests/test_sym_stats.py`.
