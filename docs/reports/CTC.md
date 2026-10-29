# Study CTC (WP-L11), step 2 so far: code, tests, smokes, Stage A lock

Registered pre-registration: `docs/prereg/CTC.md` (master 9876584). Worker: sn-oracle (Claude Sonnet-5.5), branch
`wp/oracle`. Status: Stage A is done and its lock is committed. **Stage B and the evaluation have not run; they wait for
the orchestrator's OK.** No test split and no `val` [0, 2,000) smoke data were read (the smoke ranges are below).

## Code and tests
New files only: `src/looplora/llm/ctc.py`, `scripts/run_ctc.py`, `tests/test_ctc.py`. No existing module is edited, and
a test checks that nothing else imports `ctc` and that importing it leaves a reader's oracle logits and memory unchanged
(fresh processes with and without the import).
- `tests/test_ctc.py`, 17 cases: **pass** (job 3724023, exit 0, commit 4e93b9e). The first run (3724010, `-x`) stopped at
  one case: the tiny test reader ignored its memory, so a "zero memory differs" assertion could not hold. That was a test
  fault, fixed in the test (the adapter weights are randomized); no code change.
- What the cases show: head-last positions equal the character-offset positions (all 1,024 names, 24 real episodes);
  the hard lookup of correct classes is `OracleRandomWriter`'s memory **bitwise** (CPU and CUDA, with invalid slots);
  with a frozen tiny reader, perfect hard lookup reproduces its oracle logits bitwise and zero memory does not; the soft
  lookup of near-one-hot classes matches it to 1e-5 and passes gradients; unseen names are masked; the 2,098,176 /
  1,049,600 / 4,100 parameter counts; the oracle writer hash equals `db99a779…`; features of the bare backbone are
  unchanged by installed adapters; Stage B (both arms) trains only the heads and leaves the reader's hash and
  `requires_grad` unchanged; the selection and decision rules.
- Not tested on compute: the existing `test_llm_*` suites (no existing file changed).

## Stage A (registered ranges; job 3724058, 111 s)
Ranges: train [0, 8,192), seed `derive_seed(17, "ctc", "stage-a-train")` (98,304 facts); selection `val` [1,000, 2,000), seed
4242 (12,000 facts); 80 probes of 1,200 steps. Lock: `reports/ctc/stage-a-lock.json`; all cells:
`reports/ctc/stage-a-probes.csv` (blake2b-128 in the lock). Feature statistics (needed by Stage B):
`runs/ctc/stage-a/feature-stats.pt`, hash in the lock.

| target | selected (site, position, form) | selection accuracy | train acc. | parameters (1,024-way) |
| --- | --- | --- | --- | --- |
| head | h14, head_last, rank64 | 99.975 % | 100 % | 132,096 |
| relation | h14, head_last, full | 100.000 % | 100 % | 4,100 |
| tail | h14, tail, rank64 | 99.992 % | 100 % | 132,096 |

Writer parameters 268,292 (the reader's adapter has 475,138). Per-fact joint accuracy on the selection range **0.99967**
(key 0.99975); by fact index 0.999 to 1.000. Many cells are tied within 0.5 pp (head: h14, h21 and h28 at `head_last`,
rank64, all ≥ 99.96 %), so the tie-break (fewest parameters, then lowest site) chose.

Observations for the claim wording (decision 2 of the registration applies):
1. **Both entity heads read the entity at its own token** (`head_last` and `tail`), in the residual stream at block 14.
   The "classification" is a decoder of the token identity, so a candidate would be "a learned parser plus a fixed code
   table", not a learned writer. `o14` (attention output) is worse: its best head cell is 98.5 % (`rel_last`, rank64) and its best tail
   cell 98.3 % (`tail`, rank64).
2. **The relation is constant in this task.** `t2_chain_end` facts come from the fixed chain layout whose relation column
   is 0 for every fact (`data/relations.py`, `simple_chain`), so the relation head scores 100 % at every position,
   including `head_last` where the relation phrase has not been read. The key is always `pair_codes[head, 0]`. This holds
   for the oracle readers too. It is a property of the task, not a result about the writer.
3. The chain layout also makes each fact's tail the next fact's head within a chain (not used by the heads).

## Smokes (timing only; commit 4e93b9e; `reports/ctc/smoke-timings.json`)
| smoke | job | result |
| --- | --- | --- |
| Stage A: train [0, 256), val [2000, 2128), 1,200 steps | 3724033 | 74 s; 80 fits; the lock/stats/csv paths work |
| Stage B arm CE, 200 steps (smoke lock) | 3724059 | 0.0792 s/step (p95 0.0699) |
| Stage B arm CE+ans, 200 steps | 3724060 | **0.2951 s/step** (p95 0.2854); reader hash unchanged |
| evaluation: val [2000, 2032), both arms, three readers | 3724092 | 14.5 s; perfect-lookup memory equals the oracle bitwise; B3-L4 predictions identical under oracle and zero memory |

Projection at 8,000 steps: arm CE 634 s (0.18 node·h); arm CE+ans 2,361 s (0.66 node·h); evaluation of 1,000 episodes
about 460 s (0.13 node·h). Remaining total about **1.0 node·h**, plus 0.13 spent: about 1.1 against the 1.8 registered
(2.5 task budget).

**Decision needed for the CE+ans job.** Its measured step is 0.295 s, just below the 0.30 s threshold of decision 4, so
`--scale-up` is not licensed. At the registered 1.5× margin the walltime is 3,541 s plus about 90 s of load and export,
which is over 1 h (the margin is 1.49× if the job is capped at 01:00:00). Options: (a) 01:00:00 without `--scale-up`
(1.49× margin; a timeout would lose the run, since training has no resume); (b) 02:00:00 with `--scale-up docs/prereg/CTC.md`,
which decision 4 does not allow at 0.295 s. I will take (a) unless told otherwise.

## Job ledger (node·h = 1 node × elapsed time from the job log)
| job | name | node·h | outcome |
| --- | --- | --- | --- |
| 3724010 | ctc-t-unit | 0.007 | exit 1 (test fault, see above) |
| 3724023 | ctc-t-unit | 0.012 | exit 0, 17 passed |
| 3724033 | ctc-s-a | 0.026 | exit 0, Stage A smoke |
| 3724058 | ctc-stage-a | 0.031 | exit 0, **registered Stage A, lock** |
| 3724059 | ctc-s-b-ce | 0.014 | exit 0 |
| 3724060 | ctc-s-b-ans | 0.026 | exit 0 |
| 3724092 | ctc-s-eval | 0.009 | exit 0 |
| total | | 0.125 | |

## Next, after the OK (commands; `H` = the commit that holds the lock and these files)
```
C=/work/xg24i002/x10041/LoopLora; S=scripts/submit.py
python3 $S --name ctc-b-ce  --queue regular-g --walltime 00:20:00 --commit $H -- scripts/run_ctc.py train --arm ce    --out $C/runs/ctc/stage-b-ce
python3 $S --name ctc-b-ans --queue regular-g --walltime 01:00:00 --commit $H -- scripts/run_ctc.py train --arm ceans --out $C/runs/ctc/stage-b-ceans
python3 $S --name ctc-eval  --queue regular-g --walltime 00:20:00 --commit $H -- scripts/run_ctc.py evaluate --arm-dir ce=$C/runs/ctc/stage-b-ce --arm-dir ceans=$C/runs/ctc/stage-b-ceans --out $C/runs/ctc/eval
```
The training jobs may run side by side; the evaluation runs after both end. It reads val [0, 1,000), writes
`metrics.json` with the registered decision, and the report `docs/reports/CTC.md` is completed from it.
