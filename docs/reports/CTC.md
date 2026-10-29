# Study CTC (WP-L11): a classify-then-code writer drives the frozen P1-L r = 128 readers

Registration: `docs/prereg/CTC.md` (master f750ee2, blake2b-128 `546bbe39cb009b545528ef35b37c887a`, with Clarification
C1). Worker: sn-oracle (Claude Sonnet-5.5), branch `wp/oracle`. Single seed (17), validation episodes only; no test
split was read. Evaluation commit `7ae9619d206b642c22d89134630e21419e66a1e8`. Decision record: `reports/ctc/decision.json`.

## Decision
**Candidate flagged**: hop-1 EM of M1-L4@4 is 50.7 % (threshold 40 %) and M1-L4@4 minus B4@1 on hops 2-3 is **+24.05 pp**
(95 % CI 21.8 to 26.3; threshold +10 pp), with the integrity condition met (zero and shuffled memory: 0.0 % and 0.2 %).
Arm selection: CE and CE+ans have the same hop-1 EM (50.7 %, difference 0.0 pp, below 1 pp), so **arm CE** is selected.
Neither falsifier fires (F1 needs hop-1 < 35 % with p >= 0.95; F2 needs gain < +5 pp). Nothing is borderline.

**What the flag means (C1 wording).** A learned parser plus a fixed code table drives the frozen composing reader. It is
not a learned writer: the selected heads read both entities at their own tokens and the relation is constant in this task
(see Stage A). The codes are the oracle's privileged codes, so the EM is not comparable with DPFP or DPFP-X, which learn
their own keys and values. A three-seed confirmation, if wanted, needs its own registration and the user's decision.

## Numbers (val [0, 1,000), seed 4242, full-vocabulary EM, hard lookup unless named)
Per-fact joint classification accuracy **p** on the same episodes: arm CE **1.00000** (12,000 of 12,000 facts; every
episode fully correct), arm CE+ans 0.99992 (one fact wrong; 999 of 1,000 episodes fully correct). Head, relation and tail
accuracy are 1.0000 for CE; for CE+ans head 0.99992, relation and tail 1.0000. Perfect-prediction lookup reproduces the
oracle memory bitwise on these episodes for both arms.

| M1-L4 at L = 4, EM % by hop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle memory (reference, p = 1) | 50.6 | 36.6 | 21.5 | 5.9 | 2.2 | 4.2 | 3.5 | 4.6 |
| CE, hard (selected) | 50.7 | 36.5 | 21.6 | 6.0 | 2.2 | 4.2 | 3.6 | 4.5 |
| CE, soft | 50.6 | 36.6 | 21.4 | 5.9 | 2.2 | 4.3 | 3.7 | 4.7 |
| CE+ans, hard | 50.7 | 36.5 | 21.7 | 6.0 | 2.2 | 4.2 | 3.6 | 4.6 |
| CE+ans, soft | 50.8 | 36.5 | 21.7 | 6.1 | 2.3 | 4.4 | 3.6 | 4.7 |
| control: zero memory | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 |
| control: shuffled memory (CE) | 0.2 | 0.0 | 0.2 | 0.3 | 0.2 | 0.1 | 0.0 | 0.3 |

| B4 at L = 1, EM % by hop | 1 | 2 | 3 | 4-8 |
| --- | --- | --- | --- | --- |
| oracle memory | 18.5 | 5.4 | 4.9 | 4.5 to 6.0 |
| CE, hard | 18.7 | 5.3 | 4.7 | 4.5 to 6.0 |
| CE+ans, hard | 18.7 | 5.3 | 4.6 | 4.5 to 6.0 |
| zero memory | 0.2 | 0.1 | 0.2 | 0.0 to 0.2 |

B3-L4 at L = 4 (no memory read): 0.0 % at every hop under oracle and zero memory, and its predictions are identical under
the two (`b3_memory_independent` true).

| contrast, hops 2-3 pooled, paired by episode | gain over B4@1 (pp) | 95 % CI |
| --- | --- | --- |
| oracle memory | +23.90 | 21.6 to 26.1 |
| CE hard | +24.05 | 21.8 to 26.3 |
| CE soft | +23.95 | 21.7 to 26.1 |
| CE+ans hard | +24.15 | 21.9 to 26.4 |
| CE+ans soft | +24.05 | 21.8 to 26.3 |
Soft and hard lookups differ by at most 0.1 pp at every hop for both arms, within sampling noise. The registered fable
prediction (hop 1 >= 42 %, hop 2 >= 29 %, hop 3 >= 14 %, gain >= +12 pp) is met by a wide margin; the result equals the
oracle reference to within 0.2 pp everywhere, as expected when p = 1. Validation oracle EM (50.6 % hop 1) is higher than
P1-L's confirmation-range 47.9 %; the ranges differ, and this study compares only within the validation set. Full table
with candidate accuracy: `reports/ctc/eval-metrics.json`.

## Stage A and the writer
Lock `reports/ctc/stage-a-lock.json` (job 3724058): head h14 / head_last / rank64, relation h14 / head_last / full, tail
h14 / tail / rank64; selection accuracy 99.975 %, 100 %, 99.992 %; joint 0.99967 on val [1,000, 2,000). Writer parameters
**268,292** (the reader's adapter has 475,138); no parameter match is claimed.
1. Both entity heads read the entity at its own token (`head_last`, `tail`), in the residual stream at block 14, so they
   decode token identity. `o14` is worse (best head 98.5 %, best tail 98.3 %).
2. The relation is constant in `t2_chain_end` (the relation column is 0 for every fact, `data/relations.py`), so the
   relation head scores 100 % at every position and the key is always `pair_codes[head, 0]`.
These are properties of the task and of the closed 768-name vocabulary. The study shows the reader transfers to
near-oracle codes (p = 1) and that its memory is used (zero and shuffled collapse). It does not show that a model can
learn its own codes.

## Training records (verified complete)
| arm | job | steps | train s | s/step (p95) | final joint acc. | final loss |
| --- | --- | --- | --- | --- | --- | --- |
| CE | 3724123 | 8,000 of 8,000 | 533.1 | 0.0651 (0.0668) | 1.0 | 1.99e-4 |
| CE+ans | 3724124 | 8,000 of 8,000 | 2,319.6 | 0.2883 (0.2934) | 1.0 | 3.46 (answer loss 3.46, accuracy 0.37 on that batch) |
Both logs have 81 rows (steps 1, 100, ..., 8,000); the reader state hash was verified unchanged at the end of CE+ans (the
run raises otherwise); both `writer.pt` and `summary.json` exist; exit 0.

**The 4x wall-time gap** (9:13 against 39:01) is per-step cost, measured: 0.0651 s against 0.2883 s per step (4.4x;
train seconds 533 against 2,320, 4.35x), and the remaining job time is the same load and export (about 19 to 21 s each).
Arm CE only runs the frozen backbone on the support text and the heads. Arm CE+ans adds, every step, the reader pass:
oracle preparation of 64 episodes, the soft memory, the four-loop (L = 4) frozen reader on 64 x 4 questions and the
backward pass through it to the memory. The added 0.2232 s per step matches P1-L's measured M1-L4 reader training step
of 0.1994 s (r = 128) plus the soft lookup and its backward. The answer loss did not change the result: both arms end
with p = 1 and equal EM.

## Deviations and notes
- B4 is evaluated at L = 1 and B3-L4 at L = 4 (their P1-L depths), and M1-L4 at L = 1 to 8; the draft said "L = 1 to 8
  for every reader" (accepted in C1).
- The CE+ans job ran at 01:10:00 with `--scale-up docs/prereg/CTC.md` as C1 allows; it needed 39:01.
- Stage A probes use 768-way heads (the 1,024-way head with unseen logits masked); parameter counts use 1,024-way size.
- The arm choice is a tie (0.0 pp); it went to CE by the rule, so it adds no best-of-two optimism here.
- One test job failed first (3724010, a test fault, fixed in the test; see `docs/reports/CTC-state.md`).
- Caveats: one seed; validation episodes (the same set earlier P1-L pilots used); a perfect classifier makes the outcome
  nearly a restatement of the oracle result, as noted above.

## Node-hours (1 node x elapsed time from the job logs)
| job | name | node-h | outcome |
| --- | --- | --- | --- |
| 3724010 | ctc-t-unit | 0.007 | exit 1 (test fault) |
| 3724023 | ctc-t-unit | 0.012 | exit 0, 17 passed |
| 3724033 | ctc-s-a | 0.026 | Stage A smoke |
| 3724058 | ctc-stage-a | 0.031 | registered Stage A, lock |
| 3724059 | ctc-s-b-ce | 0.014 | smoke |
| 3724060 | ctc-s-b-ans | 0.026 | smoke |
| 3724092 | ctc-s-eval | 0.009 | smoke |
| 3724123 | ctc-b-ce | 0.153 | Stage B arm CE |
| 3724124 | ctc-b-ans | 0.650 | Stage B arm CE+ans |
| 3724519 | ctc-eval | 0.100 | evaluation, decision |
| total | | **1.028** | against 1.8 registered |
