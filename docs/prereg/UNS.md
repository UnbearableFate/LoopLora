# UNS (F11) — unseen-pool entities: audit of the committed unseen rows and one missing evaluation, pre-registered

Registered 2026-10-27 12:44 JST, when the orchestrator committed this file to master, before any lock, smoke or run of
this study. The draft text is unchanged except where "Decisions at registration" (last section) says otherwise.

Draft by sn-sym, 2026-10-27, on the login node: no run and no evaluation; the author read no result value (difference,
interval, prediction or correctness) of any unseen-pool row or sample (what a review subagent read afterwards is stated
below). Registered at the time the orchestrator commits this file to master, before any run below.

## Known before this registration (the unseen-pool results are not blind)
- The committed `reported_contrasts` exports of P1-S, P1-S-F1 and P1-S-F2 already contain non-gating
  `dataset=unseen` rows, which no ledger has audited (WP-T9 open question (a), `docs/reports/WP-T9.md`; the paper
  ledger `docs/paper/evidence_symbolic.md` states that they support no statement until audited, and that this study
  is pending). Their values have been visible in the repository since those studies were committed; the F1 and F2
  reports state qualitative sentences about them ("preserve positive joint reach/specificity effects" in
  `docs/reports/P1S-F1.md`; "retain positive specificity and large hop-1 losses" in `docs/reports/P1S-F2.md`).
  The paper makes no claim about unseen entities.
- The author of this draft read only the metadata of those rows (counts by study, group, writer, reader, endpoint,
  confidence and pooled training seeds; no `difference`, interval or per-seed value). From the saved screening
  samples, only label arrays (split, condition, depth, hop, episode, query, answer, candidate size) of the 18 Part A
  files were read, to check the audit's structural assumptions; the author read no prediction or correctness value.
  After the draft was complete, a read-only review subagent of the author (code review, `docs/reports/WP-S9-state.md`
  §8) loaded the `prediction` and `correct` arrays of both pools of the 18 Part A files for structural checks and
  replayed with its own code 12 of the 18 reproduction comparisons below (`joint` and `joint_fact`; L4 − L1 on hops
  {1}, {2, 3}, {6, 7, 8}; both confidence levels) against the committed rows. Its report to the author says that these
  checks passed and that the reproduction was exact; it contains no value, and the author has seen none. No
  registered quantity or rule changed because of it. The inventory, from the three exports, is:

  | export | rows (screening / unseen / confirmation) | unseen rows by writer | by reader | endpoints of the unseen rows |
  | --- | --- | --- | --- | --- |
  | `reports/p1s` (r = 32 primary 224, replication 24; r = 16 secondary 96) | 344 / 344 / 200 | oracle_random 96, oracle_orth 88, t1_frozen 64, joint 96 | M1 162, M1-lin 122, B5 14, B5-lin 14, B3 8, B4 24 | reach, hop1, far, specificity, vs_fixed (28 each); vs_loop, replication_{reach, far, specificity} (16 each); L8 − L4 on hops 4–8 (84); M1 − M1-lin at L4 (20 + 20); B4@1 − M1@1 (8 + 8) |
  | `reports/p1s-f1` (r = 64, 128) | 120 / 120 / 24 | oracle_random 60, joint 60 | M1 64, M1-lin 56 | reach, hop1, far, specificity, vs_fixed, vs_loop (16 each); H6 L8 − L4 (16); M1 − M1-lin (8) |
  | `reports/p1s-f2` (r = 32) | 64 / 64 / 16 | t1_frozen_fact 32, joint_fact 32 | M1 36, M1-lin 28 | reach, hop1, far, specificity, vs_fixed, vs_loop (8 each); H6 L8 − L4 (8); M1 − M1-lin (4); B4@1 − M1@1 (4) |

  Each unseen row exists at confidence 0.95 and 0.975. The unseen rows of the P1-S primary and replication groups and
  of all F1 and F2 rows pool the three training seeds 17/29/43; the 96 P1-S secondary (r = 16) rows pool seed 17
  only. The committed `screening` rows of the same exports (the seen-pool counterparts) pool seed 17 only in every
  export, so they are not like-for-like with the unseen rows: the like-for-like seen-pool values are those that Part
  A recomputes from the `test` split of the same three seeds. The exports carry no evaluation-seed or
  episode-count column; the screening protocol of those studies is split `test_unseen`, eval
  seed 4242, 2000 episodes, hops 1–8, depths 0–16 (`docs/reports/WP-H-main-interface.md`), which Part A verifies from
  each run's own `metrics.json` before using anything. The M1 rows are the released mixed-depth M1 ({1, 2, 4}).
- Not covered by any committed row: the P2-S-F1 M1 trained at the fixed depth 4 (M1-L4) on the unseen pool, and absolute
  unseen-pool accuracies of B4.
- No gate of any study depends on the unseen pool, and none depends on UNS.

## Question
Do the depth gain of the iterative reader and its advantage over a single read, measured on the seen pool (entities
0–767, the pool of all training episodes), also appear on episodes built from the 256 entities of the unseen pool
(768–1023), for the learned writers `joint` and `joint_fact`? The answer fixes the wording of the paper's scope; it is
descriptive.

## Design
**Part A — zero-compute audit of the existing unseen-pool evaluations (no new seed).**
- Objects: the screening evaluations (split `test` = seen pool and `test_unseen` = unseen pool, eval seed 4242, 2000
  episodes, hops 1–8, depths 0–16, ordinary condition) that already exist for the r = 32 runs: writers `joint` and
  `joint_fact` (the F11 target), `oracle_random` as a control (fixed random unit codes of every entity need no
  learning, so it shows what a learned writer adds or loses); readers M1 (released, mixed-depth) and B4; seeds
  17/29/43 (18 evaluations).
  Paths: P1-S main `runs/primary-{writer}-{reader}-r32-s{seed}/screening` for `joint` and `oracle_random`; F2
  `evaluation/f2-main-joint_fact-{reader}-s{seed}/screening`.
- Audit (before any statistic; a failing run is a finding, nothing is re-evaluated and no statistic is computed for its
  writer): run identity, evaluation seed, splits, episode count, hops, depths, task and entity pools equal the
  registered values; the sample file matches its recorded blake2b hash; answers and predictions lie inside their pool;
  one query per hop per episode at every depth with identical query labels across depths; candidate size 14; saved
  correctness equals prediction == answer; no episode id shared by the two pools.
- Reproduction: for each of the three writers, the unseen-pool M1 depth contrasts (L4 − L1 on hop 1, hops {2, 3} and
  hops {6, 7, 8}; confidence 0.95 and 0.975; 2000 resamples, seed 2718) are recomputed from the saved samples and
  compared with the corresponding committed rows (one row each, 18 comparisons; reproduced if every difference and
  both interval ends agree within 1e-9). The remaining committed unseen rows are transcribed (`context.csv`: the six
  named endpoints at 95%) next to the committed screening and confirmation rows of the same export, each dataset
  with the training seeds that row pools, for context only; they are not recomputed.
- Tables: for each writer, M1 depth gain (L4 − L1) and M1@L4 − B4@L1 on hop 1, hops {2, 3} and hops {6, 7, 8}; per-hop
  accuracy of M1@L1, M1@L4 and B4@L1; each on the seen pool, the unseen pool and the unseen-minus-seen difference
  (independent bootstraps: the pools hold different episodes); chance 1/14.
**Part B — what is missing: M1 trained at the fixed depth 4 on new episodes (optional; new eval seed 999006).**
- Checkpoints: the P2-S-F1 M1-L4 runs (`joint`, `joint_fact`, `oracle_random` control; seeds 17/29/43; recorded in
  `reports/p2s-f1/checkpoint_manifest.csv` with blake2b seals) and the B4 references above: 18 frozen checkpoints.
- Evaluation: the screening protocol on new episodes: splits `test` and `test_unseen`, **eval seed 999006** (never used
  before), 2000 episodes, hops 1–8, depths 0–16, ordinary condition, batch 64; no counterfactual conditions. Nothing is
  trained; the model state hashes and the checkpoint file hashes are verified after every evaluation.
- Tables: as Part A, with M1 = M1-L4; the same audit applies with seed 999006.
- Order: `lock` (design with the 18 seals) → the orchestrator commits it → an optional timing run on two checkpoints (own
  root, partial, never analysed, deleted after its timing is read) → `run` (needs the Git-locked design) → `audit_uns.py
  --extension-root`.
- Statistics (both parts): 95% percentile intervals of a joint whole-episode bootstrap over the three training seeds,
  2000 draws, base seed 2718 (the seed of the committed analyses); per-seed values reported; no multiplicity correction
  (descriptive).

## Endpoints (all descriptive)
Hop groups {1}, {2, 3}, {6, 7, 8}. For each writer, pool and part: M1 depth gain (L4 − L1) and M1@4 − B4@1, with
intervals; unseen-minus-seen differences of both; per-hop accuracy of the three cells; the reproduction check of the
committed rows.

## Use of the results (stated before results)
- Nothing in UNS is a gate, and no decision rule is attached. It is used only for the wording of the paper's scope.
- The tables are reported as a scope row next to the seen-pool values. A (writer, reader) result is described as "the
  depth gain is also present on unseen-pool entities" only if its unseen-pool M1 depth gain on hops {2, 3} is at least
  3 pp with the interval above 0 (the project's usual reach criterion); otherwise as "not shown on unseen-pool
  entities". Part A and Part B are never pooled and are labelled by their sources.
- No sentence that composition "holds for unseen entities" or "beyond the closed vocabulary" is justified by UNS: the
  unseen pool is 256 further entities of the same 1,024-name vocabulary with fixed random features, never used in any
  training episode, and the readers were not trained to be tested on new codes.
- If Part A's audit or reproduction fails for a writer, the committed rows of that writer stay unaudited and are not cited;
  a failure is reported as it is, without re-evaluation.

## Flagged interpretations and additions (the plan does not fix these)
1. The plan's F11 is "evaluate on `test_unseen` and compare with the seen pool, descriptive". Following the coordinator's
   instruction after WP-T9 open question (a), it is split: Part A audits the existing committed rows at zero
   evaluation cost; Part B registers only what they do not cover (M1-L4). Part B is optional: Part A alone satisfies the
   zero-compute reading.
2. "M1-L4 and B4" is read as the P2-S-F1 M1 trained at L = 4 (Part B) and, for the committed rows, the released M1 read
   at L = 1 and 4 (Part A).
3. `oracle_random` is added as a control; it is not a learned writer.
4. The 3 pp wording threshold is an addition (the plan says descriptive only); it changes no gate.
5. The unseen rows were seen before this registration (above), and the review subagent's replay read some of the
   underlying samples. Part A therefore cannot be blind to the committed numbers; its purpose is to audit them, not to
   test a prediction. Only Part B, whose episodes have never been generated,
   is a new look at fresh data.

## Limits and budget
- The unseen pool shares the vocabulary, the fixed backbone and the generator with the seen pool; chain structure, entity
  pool size and read depth are unchanged. Three training seeds; intervals condition on them. Descriptive only.
- Cost: Part A at most 0.1 node·h (CPU bootstrap and file reads inside a compute allocation); Part B about 0.25 node·h
  (upper bound 0.75) from the measured screening cost of 129–154 s per checkpoint; see `docs/reports/WP-S9-state.md`.
  Each job is one node and at most 1 h.
- Code: `src/looplora/sym/uns.py`, `scripts/audit_uns.py`, `scripts/run_uns.py`; tests `tests/test_uns.py`,
  `tests/test_uns_scripts.py` (written, to be run on a compute node before any step-2 job).

## Decisions at registration (orchestrator, 2026-10-27)
- **Part B is run.** It is part of the registered study, not optional, because it is the only fresh-data look and covers
  the M1-L4 reader the paper recommends. The two-checkpoint timing run stays optional.
- Flagged items 1–5 are accepted as written.
- **Locks.** The orchestrator commits each design lock (Part B) on wp/sym, the branch whose worktree holds the lock
  files, after checking it against this text. sn-sym stops after each lock stage and reports the file path.
