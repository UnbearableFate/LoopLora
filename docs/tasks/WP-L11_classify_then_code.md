# WP-L11: classify-then-code writer into the frozen oracle readers (G4, study CTC; sn-oracle; LoopLora-wt/oracle, wp/oracle)

Plan: `plan/followup_2026-10-29.md` G4. The user approved G1–G4 on 2026-10-29 13:00 JST.

Read first:
- `docs/tasks/_common_rules.md`
- `docs/review/rv-fable_2026-10-29.md`: §1.1 and proposal W1
- `docs/reports/WP-L3-state.md` (writer-fix probes; fable W1 quotes head 84.0 % / 95.6 %, relation 99.99 %). Read the head, relation and tail decodability.
- `docs/prereg/P1L.md` and `reports/decisions/P1L.md`. Learn how the oracle_random memory is built from its pair-key and
  value tables.
- `docs/reports/DPFP.md` and `docs/reports/DPFP-X.md`: the learned-writer stop.

## Question
The only reader that composes was trained on oracle codes. Can a writer learned from text drive that frozen reader? The
writer classifies each fact's head entity, relation and tail from the LM's hidden states, then writes the oracle's own
codes, looked up from the fixed tables. This separates write errors from the reader's composition.

## Fixed design (change only with a flagged reason)
- **Backbone and readers.** Qwen3-0.6B-Base, r = 128. The registered P1-L readers M1-L4, B4 and B3-L4 at seed 17, all
  frozen.
- **Stage A (probes).**
  - Train linear classification heads from candidate write-position hidden states to head, relation and tail, on training
    episodes.
  - Measure them on validation and pick the positions by a rule stated in the prereg.
- **Stage B (writer).**
  - Train the heads with cross-entropy, optionally plus the answer loss through the frozen M1-L4. Use the P1-L reader task
    and settings.
  - The memory is built from the oracle tables through a softmax-weighted lookup in training. Evaluate with a hard
    (argmax) lookup, and also report the soft one.
- **Writer parameters.** About 2 M at 0.6B. Report them as a cost; they are not part of any parameter match.
- **Evaluation.** Validation only, L = 1…8, with each frozen reader. Report:
  - per-fact code accuracy p;
  - hop-wise EM;
  - M1-L4@4 − B4@1 on hops 2–3;
  - zero- and shuffled-memory controls.
- **Decision rule** (fable W1; flag any change). On validation:
  - hop 1 ≥ 40 % and M1-L4@4 − B4@1 ≥ +10 pp on hops 2–3 flags a learned-writer candidate. A three-seed confirmation
    under the B6 rule on a fresh range would then need its own registration, and the orchestrator decides on it.
  - Otherwise stop.

  Record the falsifiers: hop 1 below 35 % despite p ≥ 0.95, and a gain over B4 below +5 pp.
- **Scope.**
  - The writer covers a closed vocabulary: the 768 seen names.
  - The codes are privileged information, used only in training.
  - The claim would be "a writer learned from text plus a four-read reader composes inside a frozen LM", not "the LM
    learned its own codes".
- **Budget.** About 2.5 node·h in total: Stage A 0.3, Stage B 1.4, plus tests, smokes and evaluation. Each job is 1 node,
  ≤ 1 h, or ≤ 2 h with `--scale-up docs/prereg/CTC.md` once registered.

## Step 1 (now)
1. Merge master (at or after the commit that adds this file) into wp/oracle.
2. Draft `docs/prereg/CTC.md`, based on the code:
   - which hidden states and positions are used;
   - how the lookup maps classes to the oracle's pair-key and value codes;
   - the losses, the steps and the selection rule;
   - the cost estimate.
3. STOP with the prereg path. Submit no compute job. The orchestrator registers the prereg on master.

Step 2 (code, tests on compute, smokes, Stage A, Stage B, evaluation, report `docs/reports/CTC.md` and the state file)
follows the orchestrator's OK.
