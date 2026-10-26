# WP-D2 static pilot/runner review

This review describes the pre-A1 grid. Amendment A1 subsequently adds B5-lin:
Pilot-2/replication now contain 72/18 runs with six variants; Pilot-1 remains60.
The audit counts and behavior tests were updated accordingly. The suggested
distinct-winning-LR regression below has also been added.

Reviewed `sym/pilot.py`, `sym/run.py` cross-stage writer resolution, writer export/load/freeze,
TaskSpec construction, and focused pilot tests against WP-D2, P1S and the selected LR rule.
No source changes or runtime execution; compute validation remains with the parent.

No correctness defects found in the requested flow:

- Pilot-1: 24 joint + 24 oracle_random + 12 feasible oracle_orth = **60**.
  Orthogonal cells are r16/E8 and r32/E8,12,16; the 2E bound permits every
  head-sharing outcome without conditioning the episode distribution.
- Pilot-2: **60** for every allowed E*=12/16/24. Chain entity counts are 14/20/29,
  so all four writer conditions fit the r32 orthogonal requirement. Five variants
  each have three LRs. Replication: three conditions × five selected variants = **15**.
- `select_pilot1` first obtains the best LR separately at each E for joint/r32,
  chooses the largest qualifying E in {16,24} at the inclusive .95 threshold
  (otherwise12), then selects each condition/rank's LR at that E. Ties choose the
  smaller LR. Pilot-2 uses original val hops1–3, L4 (B4:L1).
- The selected joint/r32 row supplies the exact Pilot-1 writer.pt path at E*/LR.
  Every frozen Pilot-2 LR/variant and frozen typed replication uses that artifact.
  Structured writer_from resolves absolute paths and rejects missing files;
  train validates backbone/rank signature, loads strictly and freezes the writer.
- Replication signal chains sum to E*−E*/4 with a length8 chain; distractors
  restore total E*. Oracle_random loads18/24/36 use valid chains summing to load
  and at least two length8 chains. The typed writer pretraining task retains E*.
- Base configs, plan writing and result collection constrain training seed17,
  eval seed4242 and split[val]; no pilot config requests screening/confirmation
  evaluation. Batch64, 5000 steps and eval1000 are preserved. Threads1 is recorded
  and applied by run_one.
- Stride-three sharding covers each run exactly once. For the 60-run grids each
  shard receives one LR and the same mix of expensive dimensions/writers/variants;
  the15-run replication has five runs per shard with mixed conditions/variants.

Validation note: the existing LR selection fixture has the same winning LR at
all E. A fixture with distinct best LRs at E8 and the qualifying E would better
protect the user's bestLR-per-E → E* → LR-at-E* ordering. The implementation
currently follows that ordering. Performance at 2000×8×9 and runtime outcomes
are outside this static review; the parent must report measured results.
