# NCR (review C4) — neighbour-confusion rate of single-read errors, analysis specification

Draft by sn-sym, 2026-10-27, on the login node: nothing computed; the author read no prediction or correctness value of
any sample (only label arrays of 18 of the sample files, for the structural checks of the UNS audit). Afterwards a
read-only review subagent loaded the `prediction` and `correct` arrays of those 18 files for range and consistency
checks and for a reproduction described in `docs/prereg/UNS.md`; it computed no neighbour-confusion quantity. This is
the specification of a descriptive analysis of evaluation samples that already exist; it uses no new data and no new
seed, and no gate or decision depends on it. Registered at the time the orchestrator commits this file to master,
before the analysis is run.

## Known before this registration
- rv-fable C4 (`docs/review/rv-fable_2026-10-27.md`): hop-2 single-read accuracies of 1–2% at L = 1 (ledger row S12)
  lie below the 1/14 chance line, "i.e. the single read is systematically wrong (the 1-hop neighbour), so D partly
  measures escape from that error. Report the neighbour-confusion rate." The plan (§7 item 4) asks for the rate where
  the hop-2 single read is below the chance line, computed by sn-sym on compute nodes.
- The accuracies are known from the committed ledgers; the category rates defined below have never been computed.

## Question
When a single read (L = 1) answers a hop-2 or hop-3 query wrongly, which candidate does it return: the entity one hop
from the query (the neighbour), the entity before the answer, the query entity itself, another chain end, or another
candidate? Is the neighbour returned more often than a uniform guess would return it?

## Definitions (per query of hop h ≥ 2, in an episode with C candidates; C = 14 in the registered task)
The candidates are the entities of the episode's valid facts. The query's chain is s = p_0 → p_1 → … → p_h = answer
(recovered from the episode's facts; each row verifies that it ends at the saved answer and visits no node twice). The
C − 1 wrong candidates are partitioned into five categories, in this priority:
- `first_hop`: p_1, the successor of the query entity (the answer of a one-hop read);
- `one_short`: p_{h−1}, for h ≥ 3 only (for h = 2 it equals `first_hop`);
- `start`: the query entity p_0 itself;
- `terminal_other`: any other chain end (an entity with no outgoing fact) that is not the answer;
- `other`: every remaining candidate.
The category sizes are computed per episode and sum to C − 1.

## Design
- Data: the saved evaluation samples of the existing r = r_mem = 32 runs: the screening evaluation (split `test`, eval
  seed 4242, 2000 episodes) and the confirmation evaluation of P1-S and F2 (split `test`, eval seed 777001, 2000
  episodes; the WP-H set). Writers `t1_frozen_fact`, `joint_fact` (F2 layout); `joint`, `t1_frozen`, `oracle_random`,
  `oracle_orth` (P1-S main layout); readers M1 (the released mixed-depth reader) and B4 read at L = 1; seeds
  17/29/43; hops 2 and 3; ordinary condition. A missing directory is listed as missing and skipped; the analysis does
  not evaluate or train anything.
- Verification before any statistic: each directory's recorded evaluation config regenerates its evaluation episodes
  (same generator, stratification, seed and episode count); every used row must match the regenerated episode on the
  answer, hop, candidate count and correctness (prediction equals answer), otherwise the analysis stops.
- Statistics, per (writer, reader, dataset, hop): pooled over the three training seeds (equal weight per seed).
  - accuracy (and chance = mean of 1/C), with the flag `below_chance`;
  - rate of each category = errors in the category / all queries, against the uniform-guess value size/C;
  - share of each category = errors in the category / all errors, against the uniform-wrong-guess value size/(C − 1);
    per-seed shares are reported.
  Intervals are 95% percentile intervals of a joint whole-episode bootstrap (2000 draws, seed derived from 2718 and the
  group label); they are descriptive. A group in which some training seed has no error is reported as undefined.
- Outputs: `accuracy.csv`, `categories.csv`, `summary.md` (one table per hop, with the uniform values beside each
  rate and share), `summary.json` (counts of verified directories and rows, missing and undefined groups).

## Endpoints and use (descriptive; stated before results)
- The neighbour-confusion rate of a (writer, reader, dataset) group is its `first_hop` rate at hop 2 and the share of
  its errors that are `first_hop`, each with the uniform reference. They are reported in the paper where the hop-2
  single read is below the chance line (`below_chance` = yes), as the plan states; the other categories and hop 3
  are reported beside them.
- No threshold, test or decision is attached. A first-hop share above its uniform value says that the single read
  returns the neighbour more often than a uniform error would; it does not by itself identify the mechanism.
- The tables are used only to qualify the reading of the depth gain D in the paper (that part of D measures escape
  from the neighbour error).

## Flagged interpretations
1. "Single read" is L = 1 of M1 (mixed-depth training) and of B4 (trained at L = 1); the P2-S-F1 and later runs are not
   included because the plan names the existing C4 evidence (P1-S, F1, F2 layouts). F1 (r = 64, 128) is also not
   included.
2. The rate is computed over all queries of the hop (as accuracy), and the share over the errors; both are given
   because the rate depends on accuracy and the share does not.
3. The priority order of the categories (neighbour first) and the treatment of h = 2 (`one_short` empty) are fixed here
   before any neighbour-confusion quantity is computed from a sample.
4. The confirmation samples are already analysed by the earlier studies; NCR re-reads them only through the saved
   predictions, adds no new evaluation, and does not alter any earlier statistic.

## Limits and budget
- One answer per query: the category of a wrong answer says where the single read ended, not why. Three training seeds;
  intervals condition on them.
- CPU only, a few minutes in a compute allocation (at most 0.1 node·h); one job, one node.
- Code: `src/looplora/sym/ncr.py`, `scripts/analyze_neighbour_confusion.py`; tests `tests/test_neighbour_confusion.py`
  (written, to be run on a compute node before the analysis).
