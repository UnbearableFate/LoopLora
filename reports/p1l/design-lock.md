# P1-L design lock

Locked before P1-L training; source: `docs/prereg/P1L.md`, final oracle v2.
The JSON configs in `configs/p1l/main/` enumerate every run. Only `train.steps`
and `train.lr` are pilot-determined fields; their initial values are placeholders.
The main runner refuses to run without `reports/p1l/lr-S-lock.json`.

- Oracle seed 17 is fixed across reader seeds 17/29/43. Readers: B8@4,
  M1-L4@4, B4@1, B5-L4@4, B3-L4@4. B3 ignores M entirely; its unused oracle
  preparation supplies matching support hashes and does not define a writer condition.
- B1/B2 train context-retained, at the same insertion point, seeds 17/29/43.
  B0 is untrained and evaluated once in both protocols. B0/B1/B2 evaluate both
  protocols; all gates use memory-only.
- All primary configs use T2 chain-end E12, chains (8,4), Q4, four relations,
  head sharing .3, train hops 1/2/3, r32, layer14/o_proj, batch64, prefetch2.
- Validation: val [0,1000), seed4242. Pilot selection pools queries at hops
  1/2/3 at trained depth, maximizes full-vocabulary greedy EM, ties lower lr.
  Candidates are .001/.003, seed17, floor(S/3) steps. Smoke reads timing and
  memory only. No test/screening evaluation is planned before confirmation.
- Confirmation: test [100000,102000), seed4242; held until every primary and
  B0/B1/B2 training run is complete and Claude Code confirms the WP-L4 decision.
  Joint episode resampling across fixed seeds, 2000 draws, .95 confidence for
  one writer condition or .975 for two. G1/G2 follow the preregistration verbatim.
- Typed replication uses the same checkpoints: E12 total facts, 3 same-head
  distractors, chains (8,1), relations1/2/3 as distractors; this matches the
  preregistered P1-S typed layout. No replication training or lr selection.
- M1 depth scan is every integer 1 through8 on exactly the same x/u/M and
  write counts; B8 has fixed depth4. Swap-successors retains the terminal set,
  source queries and source IDs; report prediction-follow rates descriptively
  at M1 depths1/4/8 (the P1-S convention) and B8's fixed depth4.

## Per-step readout definitions

Read each z_s, s=1..4, for M1-L4/B8 on the original confirmation questions.
Each step uses the same pre-write M. The logit lens replaces the adapter's
output with alpha B z_s (at all valid question positions), then forwards the
unchanged frozen top layers. No extra reader updates or memory writes occur.
The scored position is the final real question token, whose logits predict
the one-token answer. Rank is one plus the number of strictly larger name
scores among all1024 names; report tied-top1 separately from strict argmax top1.

The hop-s entity is obtained by following s latest (head,relation) facts from
the query start. Only existing path positions are scored (s<=true hop); report
the same entity at every other reader step, with its hop and entity-step labels.
This avoids inventing entities beyond the terminal.

J-lens-inspired name-gradient probe directions are frozen-backbone name-logit gradients with respect to
the additive layer14 attention residual at the scored token (the o_proj output
addition, before layer14's MLP), averaged over512 validation questions before
unit normalization. Use original val episodes0..63, one query per hop1..8,
seed4242, question-only forward and no adapter. Cache all1024 directions once,
with model, question and direction blake2b hashes. Project alpha B z_s onto
these residual-space directions. The direction definition is shared across
models/seeds. The baseline is1024 independent unit Gaussian residual-space
directions, using derive_seed(2718,"p1l","readout-random-directions"). Report
J-lens-inspired name-gradient probe and random-direction ranks/top1 by hop, reader step and entity step.

### Amendment A1 — orchestrator instruction, before Pilot-2 scores

The primary J-lens-inspired name-gradient probe is name-contrastive: for each question subtract the uniform
mean across all1024 name-logit gradients, then average the centered gradients
across512 questions and unit-normalize each name direction. Retain the raw-logit
gradient directions above as a secondary readout. Cache and hash both versions.
For s=4 report, separately for both J-lens-inspired name-gradient probe versions and the logit lens, the
fraction whose top1 name equals the actual model's top1 name among all1024 names.
Full-vocabulary EM remains the scientific endpoint; the faithfulness reference
is the model's name-restricted argmax. The orchestrator accepted the residual
space and512-question frozen-backbone definition. Confirmation remains held
until its explicit WP-L4 decision (one condition:95% CI; two:97.5%).

## Cost and artifacts

Amendment B2 supersedes the original total cap and B1's13/16node-hour limits.
S=17000, Pilot-2=floor(S/3)=5666, fixed before any Pilot-2 score. The900-step
timing smokes remain valid by explicit orchestrator instruction. There is no
study-total cap; record node-hours for every job, including failures.
Every job<=8nodes x2hours and
must finish before2026-10-28 09:00JST. All submissions use scripts/submit.py.
Measure all7 trained variants and verify that the slowest run x1.5 fits2hours.
Packing is sequential backbone reuse, with only active-run prefetch2 and no
GPU process contention. Pack multiple trainings only when their summed
measured time x1.5 fits the job request. Raw artifacts live under
`/work/xg24i002/x10041/LoopLora/runs/p1l/main/`; compact locks, manifests,
counts, paired endpoints, readout tables and figures live in reports/p1l.
Keep adapters/checkpoints and sample arrays; no full-backbone copies or new
dependencies. All hashes are blake2b128. All runtime validation runs on compute.
