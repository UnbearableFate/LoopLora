# WP-D root implementation review

Reviewed `backbone.py`, `writers.py`, `model.py`, `train.py`, and
`tests/test_sym_model.py` against WP-D, its internal contract, and binding
notation. Also inspected the current core/data implementations in their sibling
worktrees because those packages are not yet present in this worktree.

No concrete source defects established in the requested scope.

- Learned query/fact projections share writer `A`; existing memory-adapter
  writer parameters are also assigned the same parameter objects
  (`model.py:69–75`). Freezing the writer therefore freezes the shared input
  projection. Oracle queries deliberately bypass adapter `A` and use the fixed
  entity codes required by WP-D (`model.py:110–119`).
- T1 pretraining sets the reader variant/activation and initialization label
  independently of the downstream variant. Loading verifies vocabulary,
  backbone seed, and dimensions before copying/freezing writer parameters;
  checkpoint loading restores the freeze (`train.py:87–97,108–125,137–146`).
- Backbone/codebook/writer/reader/depth/data RNGs use separate `derive_seed`
  labels. Core defaults match the required initialization scales, and `B`
  starts at zero. Query preparation writes facts once and each depth reads the
  prepared snapshot without a query write.
- Evaluation sample ids are integer arrays compatible with statistics pairing.
  Stats retain split/condition identity, verify answer/hop labels, and resample
  complete episodes with query-weighted totals.

Validation limitation: this is a read-only code review, with no project imports
or runtime checks on the login node. The sharing/checkpoint regressions currently
use CPU; parent compute validation must establish CUDA behavior and the 300-step
acceptance result. Directly regenerating the same pretrained writer independently
for different downstream variants is not exercised by the existing checkpoint-
sharing test, although the inspected initialization/config path is independent
of downstream variant under otherwise identical settings.
