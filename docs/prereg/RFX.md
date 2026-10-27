# Study RFX — read-failure factorial on the frozen WP-L4 checkpoints (WP-L7, F2 of the follow-up plan), pre-registered

Registered 2026-10-27 11:05 JST by the orchestrator, before any RFX run (smoke included), any RFX or LSQ result and any
DPFP result. Drafted 09:56 JST by sn-llm (WP-L7 step 1, no compute). The orchestrator changed the decision rule at
registration (section "Decisions at registration"): the plan's rule (`plan/followup_2026-10-27.md` §3 F2) did not
cover a rescue by query normalization alone, and its "query" verdict would also have fired on the near-exact
ridge + true-key read. Every other place where the plan left a choice is marked "(operationalization)". Runs start
after the Miyabi maintenance (2026-10-28 22:00 JST), with code that implements this text (WP-L7 step 1b).

Known then: WP-L4 (`docs/reports/P1L-read-fix.md`) selected no writer. R at r = 128 ended at val EM 33.5% with
normalized read error 0.631, key top-1 88.5% and clean/read value-head accuracy 93.3% / 19.7%; R at r = 32 ended at
12.0% (read error 0.666, top-1 83.2%, head 84.5% / 8.8%); the WP-L3 A1 W5 rerun at r = 32 ended at 14.4%. Diagnostic OA
(true value fed in, memory read bypassed) reached 84.0% at r = 128. With E = 16 <= r an exactly interpolating store
exists (rv-fable C3), and the learned keys are strongly correlated (key Gram RMS about 0.3, mean max |cos| 0.85 in the
WP-L4 audit). The external reviews (`docs/review/rv-fable_2026-10-27.md` C2/C3/E3/W1, `rv-astra_2026-10-27.md` §2, §4)
ask where the deployed read goes wrong before more writers are screened.

## Question
For each frozen checkpoint, is the failure of the deployed read at the final question position a failure of the
**store** (the delta rule leaves cross-talk that an exact store would not), of the **query** (the learned q does not
address the queried key, or its scale is off), or of the **decoder** (the path after the read cannot use a read that
contains the value)? A "storage" verdict triggers the least-squares arms of the LSQ study (`docs/prereg/LSQ.md`) and a
"normalization" verdict its unit-query arm; the other verdicts trigger nothing.

## Objects (frozen, never trained or modified)
All three are final checkpoints of complete WP-L4/WP-L3 runs under `/work/xg24i002/x10041/LoopLora/runs/p1l`. The job
refuses a folder whose `record.json` is not `complete`, is a smoke, or has another rank or step count, and records the
blake2b-128 digests of `writer.pt` and `checkpoint.pt`; before creating any output it requires them to equal the values
below (`rfx_run.DIGESTS`; a mismatch voids the run).

| label | folder | rank = r_mem | steps | job | source commit | config hash |
| --- | --- | ---: | ---: | --- | --- | --- |
| R-r128 | `read-fix-20261027/arms-r128/R` | 128 | 24645 | 3716159.opbs | `be07f7fb34c1` | `65eaf986a31495bbdeb56c641911aae7` |
| R-r32 | `read-fix-20261027/arms/R` | 32 | 23590 | 3715707.opbs | `0b03d606f3c7` | `3728a5b34f1d062fa67dc2946c548bec` |
| W5-r32 | `writer-fix-20261027/rerun/W5` | 32 | 32900 | 3714914.opbs | `0713cfb66fc9` | `70b4b62607d6ce0da921e3eb46545889` |

| label | `writer.pt` blake2b-128 | `checkpoint.pt` blake2b-128 |
| --- | --- | --- |
| R-r128 | `61911b21bbfdbe48604ad3d5961ef9d3` | `0f51677aea1db4cbb5f7e9349fceffd1` |
| R-r32 | `dce6e6040122f3956444a81b9bec3bf4` | `a2e79690fafd64d540f8aef8b081efab` |
| W5-r32 | `979b387f8b04165132b91b4d174f1c80` | `687b5b46f81f845f0b6d33f2723bfe34` |

The inference export (`writer.pt`) is loaded with the checkpoint's own loader (`read_fix_run.load_inference` or
`writer_fix_run.load_inference`); the training-only clean-value head is read from `checkpoint.pt` (`value_head`) and is
used only as a decoder (below), never inside the LM path.

## Observation and replacement
One deployed forward per batch (T1, L = 1, layer-14 `o_proj`, memory-only support, `scan_final` writes) with an
instance-level hook on the installed adapter. At every question row's last real token (the token whose hidden state
predicts the single answer token) the hook records the raw learned query q = Q z and the deployed read m = M q,
where M is the deployed delta-rule state of the support. A replacement read, if given, substitutes m at exactly that
position of exactly that row; **every other position keeps its deployed read and the LM path after the read is the
unmodified one**. Re-inserting the recorded m gives bitwise-identical logits (checked on the first batch of every job;
see "Instrumentation checks"). The support (write) phase is the deployed one: keys K (unit, E = 16 per episode) and
values V are cached from it, so the ridge store below is built from exactly the pairs the delta rule saw.
Consequence for the controls: RFX "zero" zeroes the final-position read only; it is not WP-L4's zero-memory control,
which zeroes every position's read.

## Data
- **Fitting set**: T1 with the WP-L4 spec (E = 16, n_rel 4, head sharing 0.3, Q = 4 queries per episode), `train`
  split, base seed `derive_seed(17,'rfx','fit')` = 768129440952085103, episode indices 0..4095, all four queries
  (16,384 queries). The seed differs from the WP-L4 training stream (`derive_seed(17,'llm','writer-data')`), so no
  fitting episode was trained on. It selects η, the scalar gains and the refit decoders; nothing else uses it.
- **Evaluation set**: `val` split, base seed 4242, episode indices 200,000..203,999, one query per episode chosen by
  `stratify_queries(..., [1], 4242)` as in WP-L4's validation (4,000 queries; binomial SE of an EM at most 0.8 pp). The
  range was never used: earlier evaluations used val [0, 2000) and the test ranges [100000, 104000) and
  [110000, 112000). No test or confirmation data is read.
- **Smoke**: 64 + 64 episodes; the fitting set as above, the evaluation episodes drawn from the `train` split at index
  10,000,000 with seed `derive_seed(17,'rfx','smoke-eval')`, so a smoke never touches the val or test ranges.

## Cells
Storage x query, plus two controls, each evaluated by replacing the final-position read in the unmodified LM path.

| factor | levels |
| --- | --- |
| storage | **delta**: the deployed state M. **ridge**: M = V Kᵀ (K Kᵀ + η I)⁻¹ over the same E = 16 cached unit keys and values, K, V ∈ R^{r×E} (unweighted; float64, solved through the E x E system KᵀK + η I; η below). |
| query | **raw**: the deployed q. **unit**: q / ‖q‖₂ (eps 1e-6). **true**: the unit written key of the queried fact (uses the label; a diagnostic, never a deployable read; rv-fable's oracle-query control E3 is delta x true). |
| controls | **clean**: the written value of the queried fact. **zero**: a zero read. |

- **Gain** (operationalization): every one of the six cells is also evaluated with a scalar gain g fitted on the fitting
  set, g = argmin of the mean normalized error (closed form), cell `…/fit`; the cell without it is `…/asis`. For the unit
  query this is the plan's "unit q with a scalar fitted on the fitting set"; applying it to every cell removes a global
  read scale as a confound of the comparisons.
- **η** (ridge): per query type, from {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10}, the value with the smallest fitting-set
  mean normalized error after the gain fit (ties: the larger η). The EM is never used to choose η.
- **Decoders**: (D1) the deployed LM path: full-vocabulary EM (argmax over all 151,936 tokens) and NLL of the answer
  token. (D2) the checkpoint's training-only clean-value head applied to the replaced read: accuracy over the 1,024 task
  entities. (D3) a linear decoder refit on fitting-set reads of the same cell (affine multinomial logistic regression on
  standardized reads, 1,024 classes, 100 L-BFGS iterations (strong-Wolfe line search, at most 1,000 function
  evaluations), L2 weight chosen from {1e-4, 1e-3, 1e-2, 0.1} on the last
  20% of the fitting rows), applied to the evaluation-set reads of that cell: accuracy and NLL. D3 is fitted for the
  six unscaled cells, clean and zero.
- **Metrics per cell**: normalized read error ‖m − v‖² / (‖v‖² + 1e-6) against the clean value (the WP-L4 metric; mean and
  standard error), D1 EM and NLL, the entity-restricted D1 EM (argmax of the same logits over the 1,024 task-entity
  name tokens, `load_name_pool().token_ids[:V]`; the class set of D2 and D3), D2 accuracy, D3 accuracy and NLL.
- **Write age**: error and EM of five cells (delta/raw/asis, delta/raw/fit, ridge/raw/fit, delta/true/fit,
  ridge/true/fit) by the write age of the queried fact (0 = the most recent of the 16 writes).
- **Write order**: the cached writes of every evaluation episode are re-applied through the deployed decay-first delta
  rule (same learned λ, β) in four per-episode random permutations (a pure function of the episode seed and the
  permutation index); the read m = M_perm q with the deployed q goes through the LM path. Reported per permutation: error,
  EM, agreement of the predictions with the natural order, the normalized squared difference to the natural-order read,
  and error/EM by the permuted write age. The ridge store is order free; the job checks that its reads agree across a
  permutation to 1e-4. Descriptive only.
- **Conditioning**: log10 of the condition number of the key Gram KᵀK (E x E per episode) and of KᵀK + η I for each
  query's selected η (median, 90th and 99th percentile, maximum over the evaluation episodes; the smallest eigenvalue is
  floored at 1e-30, so a singular Gram reports about 30 + log10 of its largest eigenvalue). Descriptive.
- **Normalization and gain effects**: EM and error differences unit − raw (per storage, with the fitted gain) and
  fit − asis for delta/raw, reported separately; the DPFP report cites the normalization effect (DPFP Amendment A1).

## Instrumentation checks (not scientific outcomes)
Per checkpoint: (1) the logits with the recorded read re-inserted equal the deployed logits exactly; (2) M q recomputed
from the cached state and q matches the recorded deployed read to a relative 1e-4; (3) the cached unit keys and values
rebuilt through the delta rule in the natural order match the deployed state's reads to a relative 1e-4; (4) ridge reads
are permutation invariant to 1e-4. A failing check makes the job raise after it has written its files. A failed check is
a bug report, not a result: the code is fixed and the whole job rerun under the same rule; no verdict is read from a
run whose checks failed.

## Decision rule (stated before results)
Per checkpoint, on the evaluation set. Let e(c), EM(c) and EMe(c) be the mean normalized error, the D1 EM and the
entity-restricted D1 EM of cell c. The deployed baseline is e_b = e(delta/raw/fit),
EM_b = max(EM(delta/raw/asis), EM(delta/raw/fit)) and EMe_b = max(EMe(delta/raw/asis), EMe(delta/raw/fit)). A cell
**rescues** if e(c) <= 0.5 e_b and EM(c) − EM_b >= 0.10 (rates computed from counts; a 1e-9 tolerance covers
floating-point rounding of exact 10 pp differences). The verdict is the first that applies:

1. **storage** (triggers the LSQ arms K4 and RG at the checkpoint's rank): ridge/raw/fit rescues; or ridge/unit/fit
   rescues while delta/unit/fit does not. That is, the exact store rescues with the deployed query, or with the unit
   query that the LSQ arms deploy when that query alone does not rescue.
2. **normalization** (triggers the LSQ arm U at that rank): delta/unit/fit rescues, i.e. normalizing the query rescues
   the deployed store.
3. **query** (no LSQ; addressing is recorded as a follow-up candidate): delta/true/fit rescues, i.e. the true key
   rescues the deployed store.
4. **store+query** (no LSQ; reported): ridge/true/fit rescues, i.e. only the exact store and the true key together
   rescue.
5. **decoder** (no LSQ): the D3 accuracy on the delta/raw reads exceeds EMe_b by at least 0.10, i.e. only the refit
   decoder rescues (both over the same 1,024 entities).
6. **none** (no LSQ; the factors are not separated by this design).

The three checkpoints are classified separately. The LSQ arms of rank r follow the verdict of **R-r32** (r = 32) or
**R-r128** (r = 128): storage → L*r*-K4 and L*r*-RG, normalization → L*r*-U, any other verdict → none. W5-r32 is
reported with the same table and verdict but triggers nothing. The summary file lists the verdicts, every rescue flag,
the triggered arms in LSQ's candidate order (`lsq_arms`) and, for each rank with a storage verdict, the selected η of
the (ridge, unit q) cell, which the LSQ release copies.
Reading guide, descriptive and not a rule (rv-fable E3): delta/true/fit at or near the OA level (84% at r = 128) means
the query path is the bottleneck; near the checkpoint's own EM means write interference dominates.

## Limits and cost
Evaluation only: no training, no gradient step, no checkpoint is written, one seed (17), validation data only.
One job of one node: estimated 2-3 minutes per checkpoint (about 130 fitting batches and 20 question forwards per
evaluation batch, from WP-L4's 6-9 s per 1,000-episode validation), about 8-12 minutes in all, **0.15-0.2 node·h**
(plan: 0.15-0.3). A 64 + 64 episode smoke first (at most 0.05 node·h). No `--scale-up`. Node·h is recorded per job.
Commands (from `/work/xg24i002/x10041/LoopLora`; `REF` is the full hash of a commit that contains this registration
and the step-1b code implementing its rule, on `wp/llm` or on master after the merge; `cbc8389` predates `rfx_run.py`,
and step-1 commits up to `9588998` implement the draft's rule, not this one):
`scripts/submit.py --name rfx --queue regular-g --nodes 1 --walltime 00:30:00 --commit REF -- -m looplora.llm.rfx_run
--out RUNS/rfx` (smoke: add `--smoke`, separate `--out`). Report: `docs/reports/RFX.md` with every table above,
the rescue flags, the verdicts and node·h. Executed by a Claude Sonnet-5.5 subagent.

## Caveats stated in advance
- A substituted read is a distribution shift for the LM path after the read (U, B and the later layers were trained on
  deployed-delta reads). A cell that fails in D1 but succeeds in D2 or D3 is reported as such; the rule uses D1 for the
  storage and query verdicts, as the plan states.
- D3 predicts one of the 1,024 task entities. The decoder rule therefore compares it with the entity-restricted D1 EM
  of the baseline, not with the full-vocabulary EM, which would favor the decoder verdict.
- True-key cells use the label and are not deployable. The ridge store here is unweighted and uses a fitting-set η; the
  LSQ arms use recency weights (the learned decay) and are trained, not substituted.
- The checkpoints are single-seed; one evaluation set of 4,000 queries gives EM standard errors of 0.7-0.8 pp, small
  against the 10 pp threshold; error ratios are on means over the same queries.

## Decisions at registration (orchestrator, on the draft's open points)
1. **Unit query in the storage verdict.** The plan names ridge + raw q, but the LSQ arms deploy ridge + unit q. A
   ridge/unit/fit rescue therefore also counts as storage, provided delta/unit/fit does not rescue; otherwise the
   normalization alone would explain it.
2. **Normalization verdict.** A rescue by delta/unit/fit is its own verdict. It triggers LSQ's arm U (the deployed delta
   store read with the unit query), registered in `docs/prereg/LSQ.md` before any result. Without it the decision tree
   would have no outcome for this case. DPFP's report must also separate normalization from decorrelation (DPFP
   Amendment A1).
3. **Query verdict.** Only the true key on the deployed store counts. With an exact store and the true key, the read is
   the written value up to η. That is close to the clean control, so a ridge/true rescue does not isolate the query; it
   is reported as store+query.
4. **Baseline.** The better of the as-is read and the read with its fitted gain is kept (operationalization). Using
   only the as-is read would make rescues easier.
5. **Decoder comparison.** D3 is compared with the entity-restricted EM (above), not the full-vocabulary EM.
