# Study CTC — a classify-then-code writer into the frozen P1-L readers, pre-registration draft

Drafted 2026-10-29 by sn-oracle (Claude Sonnet-5.5) for WP-L11 (plan G4, fable W1); **not registered**: the orchestrator
reviews it and commits it to master before any code or run. Source: `docs/tasks/WP-L11_classify_then_code.md`,
`docs/review/rv-fable_2026-10-29.md` §1.1 and W1. The design below follows the task file's fixed design; every addition or
change is listed under "Flags" for the orchestrator.

Information seen before this draft: P1-L's registered r = 128 `oracle_random` results (`reports/p1l/r128/`: M1-L4@4
hop 1/2/3 = 47.9/35.4/18.5 % and B4@1 = 19.6/5.3/5.4 % on the confirmation range, three seeds pooled); the WP-L3 pre-tail
probes (`reports/p1l-writer-fix/probes.csv`, layer-14 `o_proj` input, T1 E = 16: head 83.975 % full-linear and 95.6125 %
rank 32, relation 99.99 %); the code read for this draft. No CTC computation has run, and no test split was read (this
study uses validation episodes only).

## Question
The only reader that composes inside the LM was trained on oracle codes (P1-L, r = 128). Can a writer **learned from
text** drive the same frozen reader? The writer classifies each fact's head entity, relation and tail entity from the
backbone's hidden states over the fact's text, then writes the oracle's own codes, looked up from the fixed tables. It
separates write errors (classification) from the reader's composition.

## Fixed design
- **Backbone**: `Qwen/Qwen3-0.6B-Base` (hidden size 1,024; layer-14 `o_proj` input 2,048), frozen, no adapter during
  feature extraction.
- **Readers (all frozen, seed 17, r = r_mem = 128, `oracle_random` oracle seed 17)**, loaded with `load_checkpoint` from
  `reports/p1l/r128/training-manifest.json` (hashes re-checked): **M1-L4** (checkpoint blake2b `c90d23ff…`), **B4**
  (`acfa7848…`, evaluated at L = 1 and all L), **B3-L4** (`37020c36…`; it never reads the memory, so it is the no-memory
  reference). Oracle writer hash `db99a779…` must equal the hash of the code tables this study builds.
- **Task** (P1-L's): `t2_chain_end`, E = 12 facts (chains 8 and 4), V = 1,024 names (768 seen: entity ids 0–767), n_rel 4,
  head sharing 0.3, train hops {1, 2, 3}, memory-only protocol, evaluation hops 1–8, one question per hop. Fact text is
  `"<Head> <relation phrase> <Tail>.\n"` (`data/render.py`).
- **Evaluation data: validation only.** Final evaluation: split `val`, indices [0, 1,000), eval seed 4242 (the P1-L
  validation set). Stage A selection uses a disjoint range, `val` [1,000, 2,000), same seed. No `test` index is read.
- Seeds through `derive_seed(17, "ctc", <label>)`; hashes blake2b.

## The writer
**What the writer sees.** Only the support text (the 12 facts), never the question or the answer. Each fact is run through
the frozen backbone once per episode (causal attention over all earlier facts, as in the reader's non-oracle pipeline), and
the hidden state is read at one position per target. Candidate positions of a fact (token indices from the tokenizer's
character offsets of the rendered line; the fact's `FactSpan` and `writer_fix.fact_positions` give the last two):
`head_last` (last token of the head string; at a line start the name has no leading space and may be several tokens),
`rel_last` (= `pre_tail`, last token of the relation phrase), `tail` (the single leading-space name token, whose id is the
tail class), `final` (the fact's last token, `token_end − 1`, where P1-L's learned writers write). Candidate sites:
`o14` (layer-14 `o_proj` input, 2,048-d, the reader's own site, used by WP-L3), `h14`, `h21`, `h28` (the backbone's
`hidden_states[14|21|28]`: block-13 output = residual input of block 14, block-20 output, final output after the final
norm; 1,024-d each). Note `o14` holds only layer-14 attention output, not the residual stream, which may explain
WP-L3's 84 % head accuracy there.

**Heads.** Three linear heads on training-standardized features (statistics from Stage A's training set, frozen as
buffers): head entity (1,024 logits), relation (4), tail entity (1,024). Logits of the 256 unseen names are masked to −∞ in
training and evaluation (closed vocabulary of the 768 seen names). Each head has its own (site, position, form) chosen by
Stage A; form is `full` (d → C) or `rank64` (d → 64 → C). Writer parameters: full heads on 1,024-d states give
2 × 1,049,600 + 4,100 = **2.10 M** (the task's "about 2 M"); on `o14` the two entity heads would be 4.2 M; rank64 heads
are 0.07–0.14 M each. Reported as a cost; not part of any parameter match.

**Lookup into the oracle tables** (`llm/oracle.py::OracleRandomWriter`, oracle seed 17, r_mem = 128). The oracle writes,
for fact (h, rel, t) in list order, the decay-first delta rule (λ = β = 1) with key `pair_codes[h, rel]` (a fixed unit
vector from the [V, n_rel, 128] table; the key depends on head *and* relation) and value `global_codes[t]` (fixed unit
code of the tail). With head distribution p_h, relation p_r and tail p_t:
- **hard (evaluation, primary)**: indices argmax(p_h), argmax(p_r), argmax(p_t); the key and value are rows of the same two
  tables by indexing, with no extra arithmetic, so a correctly classified episode produces the oracle memory bitwise.
- **soft (training; also evaluated and reported)**: key = normalize(Σ_h Σ_r p_h[h] p_r[r] pair_codes[h, r]), value =
  Σ_t p_t[t] global_codes[t], temperature 1. A one-hot p gives the oracle's key and value. The key mixture is renormalized
  because the delta rule assumes unit keys (an implementation choice; see Flags).
Only valid fact slots are written, in list order; the memory goes to the frozen reader's layer-14 module exactly as
`LLMExperiment.prepare` does for the oracle (a `SupportSnapshot` from the writer's state, support text removed).

**Privileged information.** The oracle codes and the true (head, relation, tail) labels are used only as training targets
and in the control arm; evaluation uses predictions only.

## Stage A — probes (decides positions; no reader involved)
- Data: `train` indices [0, 8,192), seed `derive_seed(17, "ctc", "stage-a-train")` (98,304 facts); selection range `val`
  [1,000, 2,000) (12,000 facts). Features of all four sites at all four positions are extracted in one pass (adapters
  detached; a test shows the features equal those of the bare backbone).
- Probe: `diagnosis_probes.fit_probe` as used by WP-L3 (AdamW lr 0.01 cosine to 0.1×, weight decay 0, batch 512, 1,200
  steps, training-only standardization), per (target, site, position, form): head and tail × {full, rank64}, relation ×
  {full}; 16 (site, position) cells × 5 = 80 fits.
- **Selection rule**, per target independently: the (site, position, form) with the highest accuracy on the selection
  range; candidates within 0.5 pp of the best (about 2.5 standard errors) are tied, and a tie goes to fewer parameters,
  then the lower site index (o14 < h14 < h21 < h28), then the earlier position (head_last, rel_last, tail, final). The
  choice is written to `reports/ctc/stage-a-lock.json` and committed before any Stage B run. Stage A reports per-target
  accuracy of every cell, the per-fact joint accuracy p_A (product of the three selected accuracies is an upper-level
  estimate only), and accuracy by fact index (fact 0 starts the context without a preceding newline).
- Stage B proceeds whatever Stage A shows (its cost is small and Stage B has about 60 times more training facts than the
  probes); p_A is recorded as the expectation.

## Stage B — the writer (two registered arms, seed 17)
- Training stream: `train` split, T2 settings above, batch 64 episodes, `derive_seed(17, "ctc", "stage-b-train")`, the
  P1-L `BatchStream`. Reader depth L = 4 (the training depth of the P1-L readers).
- **Arm CE**: loss = mean over valid facts of [CE(head) + CE(relation) + CE(tail)] against the true fact.
- **Arm CE+ans**: the same plus 1.0 × the answer cross-entropy of the frozen M1-L4 (full-vocabulary logits at L = 4,
  exactly the P1-L reader loss) on the batch's questions, backpropagated through the frozen reader and the soft memory to
  the heads. The reader and backbone stay frozen (`requires_grad` false, eval mode); a test checks that no reader
  parameter or buffer changes.
- Optimization (both arms): AdamW, lr 0.01 cosine to 0.1×, warmup 20, weight decay 0, grad clip 1.0, **S = 8,000
  steps** (6.1 M fact examples). Not tuned; the recipe is Stage A's. Smoke runs only check that losses are finite and
  falling.
- **Arm selection**: the arm whose hard-lookup M1-L4@4 hop-1 EM on the final validation set is higher; a difference below
  1 pp goes to arm CE (fewer assumptions). Both arms are evaluated and reported in full; the decision below is applied
  to the selected arm, and both arms' falsifier status is recorded. The best-of-two choice adds a small optimism on the
  same validation set; any candidate needs a fresh-range confirmation anyway.

## Evaluation (validation [0, 1,000), seed 4242; hops 1–8; L = 1…8 for every reader)
Per arm and per reader (M1-L4, B4, B3-L4):
- **Code accuracy** per fact: head, relation, tail, key (head and relation), and joint **p** (all three), overall and by
  fact index, on the same validation episodes; hard and soft.
- **Hop-wise full-vocabulary EM** (and candidate accuracy) for hard and soft lookup, L = 1…8; 95 % paired episode-bootstrap
  CIs (2,000 draws, seed 2718) for the contrasts below.
- **Controls** (M1-L4, B4, hard): zero memory (all-zero state); shuffled memory (each episode reads the memory built for
  the next episode in its 16-episode evaluation batch, cyclic).
- **Oracle reference** (evaluation only): the same three readers on the same episodes with the true facts through
  `OracleRandomWriter` (p = 1), which is P1-L's `oracle_random` memory on validation.
Contrasts: **Δ = M1-L4@4 − B4@1 pooled over hops 2–3** (paired by episode and hop); hop 1 of M1-L4@4.

## Decision rule (fable W1; applied to the selected arm, hard lookup)
- **Candidate**: hop-1 EM of M1-L4@4 ≥ 40 % **and** Δ ≥ +10 pp. A candidate is flagged as "a writer learned from text drives
  the frozen composing reader"; a three-seed confirmation under the B6 rule on a fresh, unused test range then needs its
  own registration, and the orchestrator decides on it. Nothing here is a confirmation.
- **Otherwise stop**: the classify-then-code route is not carried further.
- **Falsifiers (recorded whatever the outcome)**: F1: hop-1 EM of M1-L4@4 below 35 % although per-fact joint p ≥ 0.95
  (points to soft-lookup leakage or to reader sensitivity to near-oracle codes; the soft-versus-hard comparison is the
  diagnostic); F2: Δ below +5 pp (the oracle-trained reader does not transfer to near-oracle codes).
- **Reference prediction (fable)**: with p ≥ 0.95, hop-h EM ≈ oracle × p^(h+1), i.e. hop 1 ≥ 42 %, hop 2 ≥ 29 %, hop 3 ≥ 14 %
  against the oracle's 47.9/35.4/18.5 %, and Δ ≥ +12 pp (oracle Δ = +21.6 pp). The thresholds above are the task's, not
  derived from this prediction.
- **Integrity condition (added, flagged)**: the candidate flag requires M1-L4@4 hop-1 EM under zero memory and under
  shuffled memory to be below 5 % each (P1-L's B3-L4@4 hop 1 is 0.00 %); otherwise the EM is not attributable to the
  written memory and no candidate is flagged.
- Reporting: all outcomes, including the stop, in `docs/reports/CTC.md` and `reports/ctc/`; point thresholds are applied to
  point estimates, and the CIs state how close a crossing is (at n = 1,000 the standard error of an EM near 40 % is 1.5 pp;
  a crossing within one standard error is reported as borderline for the orchestrator).

## Flags (changes or additions to the task's fixed design)
1. **Positions and sites are a Stage A choice among 16 cells**, including the residual stream; WP-L3's probe used only
   the 2,048-d `o_proj` input at `pre_tail` (84 % head accuracy), which is attention output, not residual.
2. **Stage A selection range** is a validation block disjoint from the final one (the task says "measure on validation").
3. **Heads are 1,024-way with the 256 unseen names masked**, writing codes only for the 768 seen names (task scope).
4. **Both CE and CE+ans arms are registered** (the task says "optionally plus the answer loss"), with the selection rule above.
5. **Controls and an integrity condition** (zero and shuffled memory below 5 %) are added to the decision; remove if unwanted.
6. **Oracle-code reference arm** on the same validation episodes, so Δ and hop-wise EM can be read against p = 1.
7. **Soft key renormalization and temperature 1**; hard lookup is exact table indexing (bitwise oracle memory when
   every classification is correct).
8. **Steps and optimizer** (S = 8,000, lr 0.01 cosine, answer-loss weight 1.0) are chosen here, not by the task; they are
   fixed and not tuned.
9. **Single seed (17) and validation only**, as the task states; the task's B6-style confirmation is out of scope.
10. **The reader checkpoints are re-verified by hash and loaded unchanged**; the memory enters through a new wrapper
    around `LLMExperiment.prepare` (new module `llm/ctc.py`), so `train.py`, `run.py`, `evaluate.py`, `oracle.py` and the
    P1-L scripts are not edited; a test shows the readers' oracle evaluation is bitwise unchanged with the module present.

## Cost (node·h; P1-L r = 128 smoke step times: M1-L4 0.199, B4 0.133, B3-L4 0.135 s/step; validation evaluation of M1-L4,
## 200 episodes × L = 1…8 = 12 s in P1-L's timing, so 1,000 episodes about 60 s per memory condition)
| item | estimate |
| --- | --- |
| tests (unit, GPU subset, off-path) and smokes (Stage A on 256 episodes, Stage B 200 steps per arm, evaluation on 32 episodes) | 0.3 |
| Stage A: feature extraction (9,192 episodes, four sites) and 80 probe fits | 0.2 (cap 0.3) |
| Stage B arm CE: 8,000 steps at about 0.06 s (support forward and heads, no reader backward) | 0.15 |
| Stage B arm CE+ans: 8,000 steps at about 0.25 s (reader step 0.20 s plus support forward and lookup; measured in the smoke) | 0.6 |
| evaluation, two arms: M1-L4 × {hard, soft, zero, shuffled}, B4 × 4, B3-L4, plus the oracle reference, 1,000 episodes each | 0.35 |
| model/checkpoint loading, hashing, reserve | 0.15 |
| **total** | **≈ 1.8 node·h** (task budget 2.5) |
Each job is 1 node × ≤ 1 h at 1.5 × the smoke-measured time. The CE+ans training job is expected to need 8,000 × 0.25 s ×
1.5 ≈ 0.83 h; if the smoke shows a step above 0.30 s, it is submitted with `--scale-up docs/prereg/CTC.md` (≤ 2 h;
justification: S is fixed by this registration, and the per-step time of the frozen four-loop reader is the P1-L value).
Node·h are recorded per job.

## Not claimed
No "the LM learned its own codes": the codes are privileged training information and the writer is a classifier plus a
code table over a closed vocabulary of 768 seen names (nothing about unseen names, nothing about other tasks or ranks).
No confirmation-grade statement: one seed, validation episodes that earlier P1-L pilots also used, a best-of-two arm
choice. A candidate flag only licenses a separately registered three-seed confirmation; a stop closes this route.

Executed by a Claude Sonnet-5.5 subagent after the orchestrator registers this document. Report: `docs/reports/CTC.md`;
artifacts under `reports/ctc/`.
