# IVD (F8) — key interference versus memory load on frozen symbolic checkpoints, pre-registered

Registered 2026-10-27 12:44 JST, when the orchestrator committed this file to master, before any lock, smoke or run of
this study. The draft text is unchanged except where "Decisions at registration" (last section) says otherwise.

Draft by sn-sym, 2026-10-27, on the login node: no run, no evaluation, no result read. Registered at the time the
orchestrator commits this file to master, before any run below. Known then: IS (`docs/reports/IS.md`): the depth
gain D(E) = Acc(L4) − Acc(L1) of the same E = 12 mixed-depth M1 checkpoints shrinks as the evaluated load grows
(oracle_random D(12) = 34.75 pp, D(36) = 7.49 pp; joint_fact 24.04 pp, 4.64 pp; hops {2, 3}), but that study changed
the chain layout (and with it the number of entities in the candidate set and the hops available) together with the
load (IS report; rv-astra proposal 4, rv-fable C6). No off-path intervention on a frozen checkpoint has been run; no
IVD cell has been evaluated.

## Question
Is the shrinking depth gain of IS explained by key interference (extra facts whose keys overlap the keys of the
relevant facts) rather than by load as such (more stored facts), by the age of the relevant facts, or by the changed
task distribution?

## Design (frozen checkpoints; nothing is trained)
- Checkpoints, all r = r_mem = 32, seeds 17/29/43: writers `oracle_random` and `joint_fact` × readers M1 (mixed-depth
  training, depths {1, 2, 4}) and B4 = 12 checkpoints. `oracle_random`: P1-S main runs
  `/work/xg24i002/x10041/LoopLora/runs/p1s/main/20261026/runs/primary-oracle_random-{M1,B4}-r32-s{seed}/checkpoint.pt`;
  `joint_fact`: F2 runs `/work/xg24i002/x10041/LoopLora-wt/symf/runs/p1s/f2/evaluation/f2-main-joint_fact-{M1,B4}-s{seed}/checkpoint.pt`
  (symbolic links into `main-readers`). The design lock seals each file (blake2b) and the run re-checks the seals and
  the model state hashes after every evaluation.
- Episodes: split `test`, **eval seed 999004** (new; never evaluated before), 2000 episodes, one stratified query per
  hop 1–8 (the generator and stratification of every earlier symbolic evaluation), batch 64. In every cell the target
  chains, queries, answers and the candidate set (the 14 entities of the original twelve facts) are identical.
- Extra facts: up to 24 per episode. An extra is `(head, relation 0, tail)` with the head outside the episode's
  entities (drawn per episode from the seen pool [0, 768) minus those entities; 128 candidate heads) and the tail
  inside the episode's entities (drawn with replacement). No chain, query path or candidate set changes. The extras
  sit at fixed random slots among the twelve original facts, which keep their relative order.
- Load (two levels): L6 (the first 6 extras in slot order) and L24. The L6 sequence is the subsequence of the L24
  sequence without the later extras (nested). Total stored facts 18 and 36, as the IS loads E18/E36, with the same
  graph and candidate set as E12.
- Key overlap (three arms; low and high are the registered contrast, nat is reported):
  - `oracle_random` (keys are the global random unit codes of the head entity): per-episode code tables. `low`: the
    extras' head codes are unit vectors orthogonal to the span of the codes of the episode's entities; `nat`:
    unmodified global codes; `high`: cosine 0.5 with the code of one relevant head (slot j targets relevant fact
    j mod 12), the remaining component orthogonal to that span. The overlap of an extra is the largest absolute
    cosine of its code with the code of any relevant fact's head; the overlaps among extras are not controlled. The
    constructed codes set the write keys of the extras only: the reader decodes against the unmodified global codes
    in every cell.
  - `joint_fact`: measured overlap of an extra = largest absolute cosine of its deployed unit write key with the keys
    of the twelve relevant facts (keys of this writer depend on head and relation only, so the tail does not enter).
    From the 128 candidate heads, `low` takes the lowest-overlap heads, `high` the highest, `nat` the first in random
    order; which selected head goes to which slot is a random permutation independent of the overlap rank, and L6
    is a random subset of the L24 selection.
- Matching: within a load, low/nat/high use the same slots, tails and write order; only the extras' heads differ, so
  the age and order of the relevant facts are matched.
- Decay-only control (`joint_fact`): at the same slots and loads the extras are replaced by writes with beta = 0
  (the state decays, nothing is written). Cells: base, {L6, L24} × {low, nat, high} (+ {L6, L24}-decay for `joint_fact`).
- Readers: M1 at L = 1 and 4 (decision), B4 at L = 1 (descriptive).
- Smoke (before the main run; validation split, seed 4242, 128 episodes, seed-17 checkpoints; selects nothing): timings,
  walltime estimate, row counts, state hashes, and the achieved mean overlap must be ordered low < nat < high at every
  load for every writer, otherwise the run is refused.
- Order of operations: `lock` writes the design (settings + sealed checkpoint list) → the orchestrator commits it → `smoke`
  → `run` (requires the Git-locked design and a passed smoke report for the same design hash) → `scripts/analyze_ivd.py`,
  which audits locks, seals, structure and episode identity before computing any statistic.

## Endpoints
- **Registered (per writer):** IC = G(low) − G(high), with G(k) = Acc(L4) − Acc(L1) of M1 in the cell L24-k on hops
  {2, 3}; paired by query, pooled over the three training seeds (mean of per-seed values), joint whole-episode
  bootstrap, 2000 resamples, seed 2718. A positive IC means that high-overlap extras reduce the depth gain more.
- Confidence level 97.5% (two writers, Bonferroni). The pattern **passes** for a writer if IC ≥ 5 pp and the lower
  bound of the interval is above 0.
- Reported, not gating: IC at L6; IC by hop group {1}, {2, 3}, {4, 5}, {6, 7, 8} (every IC row at the same 97.5%
  level); per-seed IC; G for every cell with 95% intervals; absolute accuracy at L1 and L4 for every cell and B4
  accuracy at L1; the raw read residual ||M k − v|| / ||v|| at each query's start fact; the full-vocabulary exact
  match (no candidate restriction, decoded against the unmodified global codes in every cell, as a second view; the
  candidate set is also identical across cells by construction); the achieved overlap per cell (none for the
  baseline and the decay-only cells, which write no extras); the decay-only gains.

## Decision rule (as in plan F8)
At equal load, high overlap lowers the L4 − L1 gain by at least 5 pp more than low overlap with a positive paired
interval: the interference explanation of the IS results holds (for that writer). Otherwise "interference" is
withdrawn as the identified explanation of the IS results. A failure only narrows the paper's wording; no similar
study follows (plan §6).

## Use of the results (stated before results)
- Pass for a writer: the IS wording may keep "consistent with key interference, identified by IVD for <writer>".
- No pass: the paper does not offer interference as the identified explanation of the IS shrinkage, and states
  that load, graph layout and candidate distribution were changed together in IS.
- The analysis labels a non-pass `not_supported` when the achieved mean overlap gap (high − low, L24, hops 2–3) is at
  least 0.2 in absolute cosine, and `inconclusive` when it is smaller. Both labels lead to the same action (withdraw).

## Flagged interpretations and additions (the plan does not fix these)
1. The plan names no load for the decision: **L24** is registered (the largest extra load); L6 is reported.
2. The decision is per writer, with a 97.5% interval; the plan states one rule.
3. The `nat` arm and the second load level are additions to the 2 × 2 crossing and are reported only.
4. The manipulation check (the 0.2 gap and the `inconclusive` label) is an addition; it does not change the action.
5. The frozen readers were trained on twelve-fact episodes, so 18 or 36 stored facts are outside their training
   distribution, as in the same-checkpoint evaluation of IS.

## Limits
- Constructed oracle codes (cosine 0.5) can be out of distribution. For `joint_fact` the overlap is selected among
  candidates, so its result is associational unless overlap varies independently of the facts' semantics (rv-astra
  proposal 4, risk). Frozen checkpoints, three training seeds: intervals condition on those seeds.
- Cost: see `docs/reports/WP-S9-state.md` (plan estimate 0.2–0.4 node·h; upper bound 0.8 node·h from the measured
  screening-evaluation cost of 129–154 s per checkpoint; the smoke fixes the walltime). Each job is one node and at
  most 1 h; the `run` job may use `--scale-up docs/prereg/IVD.md` (at most 2 h) only if the smoke-based estimate
  requires it.
- Code: `src/looplora/sym/ivd.py`, `scripts/run_ivd.py`, `scripts/analyze_ivd.py`; tests `tests/test_ivd.py`,
  `tests/test_ivd_scripts.py` (written, to be run on a compute node before the lock).

## Decisions at registration (orchestrator, 2026-10-27)
- Flagged items 1–5 are accepted as written.
- The `--scale-up docs/prereg/IVD.md` allowance for the `run` job is accepted: at most 2 h, and only if the smoke-based
  estimate exceeds 1 h.
- **Locks.** The orchestrator commits each design lock on wp/sym, the branch whose worktree holds the lock files,
  after checking it against this text. sn-sym stops after each lock stage and reports the file path.
