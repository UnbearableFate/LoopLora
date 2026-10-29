# Study SLOT-LOOP-v1: explicit key-value slots against the matrix store, read by the learned low-dimensional reader (WP-L13; S0 and S1A registered here, S1B, S2, S3 by addenda), pre-registered (DRAFT)

Status: draft for registration, written on the login node before any implementation, test, smoke or run. Drafted by sn-slot (Claude Sonnet 5.5; WP-L13 step 1, wp/slot). Design source: `plan/looplora_diagnosis_and_execution_plan_2026-10-29.md`
(4e6778a, written against a255e81; "the plan"), a reference and not an authority: where it disagrees with the code the code wins, and the differences are under "Plan against code". Authorization (user, 2026-10-29): all stages S0 to S3, "按计划推进"
(`docs/tasks/WP-L13_slot_loop.md`, 8259878). The old verdicts (P1L, SQ2, LSQT, DPFP, the 0.80 writer gate) stay as they are. No old test or confirmation window is read; the old `val[0,1000)` at seed 4242 serves only the identity replays below.

## What was known at drafting (sources)
- **S-RE** (`runs/p1l/sq2/arms/S-RE/`, SQ2 job 3724486.opbs, commit `3d543b430a2533df0057647081b07d874de893db`, config hash `1bfb72b7d87bc0aff743c1e018e8680e`; D128-3 plus 6,000 steps, ridge store eta 1e-4 on DPFP nu 3 features, key dimension 768, value 128):
  on `val[0,1000)` seed 4242 723/1000, NLL 1.642127275466919, normalized read error 0.19613, clean-value head 93.6 %, read-value head 66.2 %, key top-1 87.1 % (`.../S-RE/training.json` metrics["6000"]; `docs/reports/SQ2.md`).
  The ridge swap alone on D128-3's weights gave 687/1000, read error 0.312 (`.../S-RE/record.json` `step0`). One seed, selected on that range, the best post-hoc cell of SQ2 and not a registered winner.
- **D128-3** (`runs/p1l/dpfp/arms/D128-3/`, 22,370 steps, delta store, config hash `2207a847e4967fb8e60d1663237fd979`; registered blake2b-128 `writer.pt` `31d98c01c604c05a54bc20e4b4c4f436`, `checkpoint.pt` `b6acff8b772a692311e630293add553f`, `src/looplora/llm/sq2_release.py:17-19`): 608/1000, NLL 2.1291537284851074 (`docs/reports/SQ2.md`, step-0 check).
- **RFX** (`docs/reports/RFX.md`): on the old R-r128 checkpoint the exact store or the true key alone did not rescue the read (38.8 %, 39.9 % against 33.8 %), both together read the written value exactly (84.2 % against the clean 84.4 %). It motivates selection-based reading; it is a diagnostic on another checkpoint.
- **S-RE adapter**, structure only (restricted stdlib unpickler, no torch, no hash): `A[128,2048]`, `B[1024,128]`, `K,V,Q,R,U[128,128]`, scalars `a_lambda`, `a_beta`, buffers `input_mean`, `input_std` `[2048]`; 475,138 parameters; `sigmoid(a_lambda) = 0.999999`.
- **Evaluation** has one question per episode (`read_fix_run.py:60`, `stratify_queries`, T1 hop 1, `sym/evaluate.py:30`): "episode" and "question" coincide in every evaluation below.

## Hypotheses and claim boundaries
- **H1 (plan §3.1), locked here.** With S-RE's learned representation fixed (the written unit DPFP keys `k_i` in R^768 and raw values `v_i` in R^128 of the E = 16 support facts), the same query code `q`, the same read positions (all valid question tokens, L = 1) and an equal training schedule from one parent state,
  replacing the ridge-matrix read `M q` by explicit selection `m = g * sum_i softmax_i(k_i.q / tau) v_i` lowers the actual read error and raises full-vocabulary answer EM. S0 is the frozen diagnostic; S1A holds the one confirmatory statement.
- **H2 (plan §3.2), named, not tested here:** per-transition consistency training raises the correct-address rate under free rollout and makes the gain of extra loops depend on the true path length. Tested in the S1B (does an extra read help independently of storage) and S2 addenda.
- **H3 (plan §3.3), named, not tested here:** an explicit NULL and state keeping reduce post-completion drift and help short and long hops together; any latency claim must be timed. Tested in the S2 addendum; timing in S3.
- **Boundaries.** Conditional on one parent writer, three continuation seeds (data order only), T1, E = 16, four relations, the seen-name pool, Qwen3-0.6B-Base layer 14, memory-only, L = 1. Nothing about loops or multi-hop (S1B, S2), NULL (S2), capacity, load or resources (S3), other writers or real text.
  At E = 16 a slot state is 57,368 B against 393,216 B per episode (the plan's §10.1 formula `4S(dk+dv)+S+8`; the matrix figure is `docs/reports/SQ2.md`); it grows with S and no byte claim is made here. S0 gates nothing statistically. An S1A ADVANCE would not show that interference removal is the cause: the soft mixture also changes normalization (D2).

## Stages and entry conditions (plan §0.3, §6.5, §7.4, §8.3, §10.5, §14.3)
Each stage starts only when its gate is met; a failed gate has the plan's own consequence (§14.3) and the line is not pushed past it. The orchestrator judges each gate and registers each addendum before any data of that stage is seen.

| stage | question | registered | entry condition (plan) | if the gate fails | budget node·h |
| --- | --- | --- | --- | --- | ---: |
| S0 | is S-RE's representation read better by another read operator (frozen)? | here | engineering gate of §6.4, §12 | stop comparison, fix the interface | <= 1.0 |
| S1A | does the operator beat a same-start matrix continuation (T1, L1)? | here | S0 gives a signal or a reasonable interface-mismatch reading; S0 without gain on a correct interface still allows one limited S1A (§6.5, §14.3) | interface error: no S1A | <= 4.5 |
| S1B | does a second read help independently of storage? (Memory x Depth 2x2 on a registered T2; precondition of H2) | addendum A1 | S1A valid on its dev, confirmation isolated (§0.3); this draft reads that as the S1A rule met | S1A and the §7.4 ablation null: stop expanding slots, report the negative (§7.4, §14.3) | <= 11.5 (plan) |
| S2 | do roles, per-transition supervision and NULL give intervenable multi-hop semantics? (H2, H3) | addendum A2 | S1B: independent L1 control and same-checkpoint depth sweep jointly support the loop (§0.3); no slot advantage: S2 runs on the matrix reader (§8.3) | no loop gain: claims shrink to reading or memory adaptation | in A2 |
| S3 | does the advantage survive resource limits (bytes, load, timing)? | addendum A3 | S2: transitions correct, terminal state stable, added information and supervision matched (§0.3); a real task also needs same-content, independent L1 control and reliable costs (§10.5) | no cost advantage: keep the science claim, drop the practicality claim | in A3 |

## Data and manifest (fixed before any result)
Episode identity is `derive_seed(base, "relations", "relations-v1", layout_digest, split, index)` (`src/looplora/data/relations.py:18,345`). Layout digest of S-RE's T1 task (E 16, Q 4, V 1024, n_rel 4, head sharing 0.3, hops (1,), seen [0,768), unseen [768,1024)), computed from S-RE's `record.json` task block: `b7f93c1ca56e5dd1f79a9dccfe2f6054`.
`eval_base(stage, purpose) = derive_seed(20261029, "slot-loop-v1", purpose, stage)`; `train_base(stage, run) = derive_seed(20261029, "slot-loop-v1", "train", stage, run)`; the query seed of `stratify_queries` is the set's own base seed. The manifest script re-derives the integers (a mismatch is a stop):

| stage | purpose | split, indices | episodes | base seed |
| --- | --- | --- | ---: | ---: |
| S0 | smoke (preflight, timing) | train, [0,64) | 64 | 8726904610828922277 |
| S0 | calibration (g, tau) | val, [0,2048) | 2,048 | 7872142726734868149 |
| S0 | dev (all S0 comparisons and decision) | val, [0,1000) | 1,000 | 4206094924762162946 |
| S1A | smoke (freeze, export/reload) | train, [0,64) | 64 | 5181101083092932984 |
| S1A | pilot shard (LR pilot) | val, [0,1000) | 1,000 | 2691891574861457280 |
| S1A | dev (checkpoint selection, ablation trigger) | val, [0,1000) | 1,000 | 6518772247750212169 |
| S1A | confirmation (sealed) | test, [0,2000) | 2,000 | 4685753055482933683 |
| S1A | unseen names (sealed, descriptive) | test_unseen, [0,2000) | 2,000 | 5570148062033127911 |
| S1A | train, run seed 17 / 29 / 43 | train, [0, 32 N) | stream | 5113975105655166458 / 9197923827134630300 / 5009954469258593308 |
| S1A | train, LR pilot stream | train, [0,6400) | 6,400 | 2442741374644229280 |
| S0 / S1A | bootstrap (numpy `default_rng`) | | | 2079281455337892545 / 9056755664197379988 |

Sizes as plan §5.1 (smoke 64, calibration 2,048, dev 1,000, confirmation 2,000, unseen-name diagnostic kept for S1A only); the pilot shard is added (D5). S-RE trained on `train`; `val` and `test` are the generator's disjoint fact-set partitions.
- **Namespaces.** The stage label is in the seed path, so S0 and S1A share no base seed. S1B, S2, S3 get their own labels, resolved in their addenda before their data exist. An S2 that changes after an opened S1 test uses new confirmation data (plan §5.1).
- **Old protections stay.** `CONFIRMATION_WINDOWS` (`src/looplora/llm/config.py:20`) is unchanged and a test pins its literal; the new code never calls `split_window`. The cap `eval.episodes <= 2000` (`config.py:179-180`) is not touched: sizes come from the manifest, not `resolve_config`.
- **New protection: a purpose-seed guard** (torch-free `slot_data.py`; an index window under seed 4242 cannot protect a set defined by its own seed). The registry maps `(stage, purpose)` to split, base seed and size; `(S1A, confirmation)` and `(S1A, unseen)` are sealed and raise unless the caller holds an open token: the release (this file's blake2b-128) and the orchestrator's `reports/slot-loop-v1/s1a-open.json`,
  which names the six selected-checkpoint digests and the blake2b-128 of the frozen analysis script. Test: the sealed seeds occur only in `slot_data.py` (source scan of `src/` and `scripts/`); all registered base seeds differ from each other and from the old ones
  (4242; `derive_seed(17,'llm','writer-data')` = 7003109974603132079 and `('llm','sq2-continuation')` = 292578375307266510; `derive_seed(20261027,'p1l-smoke-eval')`, `config.py:205`).
- **No overlap, checked in code** (`prepare_slot_manifest.py`, preflight): episode IDs of all registered sets and of the first `32 N` episodes of each train stream are pairwise disjoint and disjoint from these old ones: seed 4242 `val` [0,3000) and [200000,204000), `test` [100000,112000)
  (`config.py:20`, `rfx.py:37`, `lsr_rules.py:20`), and the D128-3 and S-RE training streams (22,370 x 32 and 6,000 x 32 episodes). Evaluation IDs are listed once, identical for every arm and seed; run records carry them and the analysis refuses a mismatch. The manifest holds generator version, layout digest, per-set ID digest (blake2b-128) and count.
- **Blinding.** The confirmation/unseen evaluation job and the analysis job print counts, status and digests only, no metric.

## S0: frozen read replacement on S-RE (plan §6)
Parent: S-RE `writer.pt` via `sq2.load_inference` (`sq2.py:88`); nothing is trained. L = 1, memory-only, T1/E16, same episodes and question order in every condition. `q` is the adapter's own query code (`query_code(Q z0)`, `adapters.py:189`: unit DPFP code, 768-d);
keys and values are the actually written `FixBatch.keys` and `FixBatch.values` (`writer_fix.py:132-133,167`), checked against the matrix state; `base_keys` are never used.

| condition | read `m` at a question token | role |
| --- | --- | --- |
| Native | the unmodified S-RE forward | reference, identity |
| Matrix | `g * M q`, `M` from the original ridge builder, new backend path | interface control |
| SoftSlot | `g * sum_i softmax_i(k_i.q / tau) v_i`; empty row returns zero | main new operator |
| HardSlot | `g * v_j`, `j = argmax_i k_i.q` (ties: lowest fact index) | does the query select a decodable fact |
| Clean-final | at the final valid question token `m = v_true` (g = 1, uses the label); other tokens keep the native read | oracle diagnostic, not deployable |
| SoftSlot-recency | SoftSlot with `log(lam^(E-1-i))` added to the logit of slot `i`, `lam = sigmoid(a_lambda)` | descriptive, registered now (D2) |

- **Scope.** `all_valid_question` is primary (every valid question token, as trained and deployed). `final_only` is a separate localization table: only the final valid token is replaced, the others keep the native read (a two-state diagnostic, never a deployable backend, no hidden fallback in the slot backend). A table never mixes scopes; `g` and `tau` are chosen once, under `all_valid_question`.
- **Calibration** (plan §6.3, S0 calibration split only): `tau` in {0.03, 0.1, 0.3, 1.0} (SoftSlot), `g` in {0.5, 1, 2} (Matrix, SoftSlot, HardSlot); objective mean full-vocabulary answer NLL over 2,048 episodes; one global `g` and `tau` per condition, never per example; 18 candidates (3 + 12 + 3).
  Candidates within 1e-4 nats of the minimum are tied: the `g` nearest 1, then the larger `tau`. All candidate NLLs, the time and the selected values are recorded; uncalibrated `g = 1, tau = 1` is reported too. If `tau = 0.03` is selected the report notes that HardSlot is the `tau -> 0` limit; the grid is not extended. SoftSlot-recency is evaluated at `(1,1)` and at SoftSlot's selected values (D2).
- **Metrics (plan §5.3)**, S0 dev, per condition and scope: full-vocabulary top-1 EM and NLL (primary); key top-1 (`alignment()`, `writer_fix.py:295`, final token); for the soft read the mean weight on the true slot, the mean gap between the two largest weights, the mean entropy;
  read diagnostics at the final valid token only: normalized read error, cosine of `m` and `v_true`, norm ratio `|m|/|v_true|`. Per-hop EM and UNKNOWN do not apply (T1, one hop). A descriptive value-head audit loads the training-only head from `checkpoint.pt` into a separate module, never into a prediction path.
- **Read-error definition.** The per-sample `|m - v_true|^2 / (|v_true|^2 + 1e-6)`, averaged over episodes, `m` the raw 128-d read before `U`, `v_true` the raw written value of the queried fact. This is `relative_read_error` in `src/looplora/llm/read_fix.py:176`, the "normalized read error" of `docs/reports/SQ2.md` (S-RE 0.196; the old-range replay must give 0.19612866640090942). No ratio of means is reported, so ratio and mean are never mixed.
- **Engineering gate (R0).** Native replays the old range: 723/1000, NLL 1.642127275466919, predictions identical to the saved `.../S-RE/val-6000/samples.npz` (as the SQ2 step-0 check did for D128-3); Matrix (g = 1) logits equal Native's bitwise on the same batches (no tolerance is registered: one needs an orchestrator clarification before any calibration or dev data are read);
  the D128-3 delta path replays 608/1000, NLL 2.1291537284851074 (D6). The owner/mask, isolation, frozen-set, byte and capacity checks pass before any calibration data are read.
- **Decision table.** Differences are S0 dev point estimates of the calibrated conditions against calibrated Matrix (Matrix_cal) at the same scope; "improves" = EM `>= +5.0 pp` (the plan's priority signal, §6.5) and a lower mean read error; exploratory, not confirmation. First matching row:

| row | S0 pattern | S1A backend | g, tau | S1A label and consequence |
| --- | --- | --- | --- | --- |
| R0 | Native not reproduced, Matrix differs from Native, or an owner/mask/isolation check fails | none | none | stop, fix the interface, no comparison, no S1A |
| R1 | SoftSlot improves (all_valid_question) | SoftSlot-L1 against Matrix-L1 | S0-calibrated, each arm its own | full adaptation |
| R2 | not R1; HardSlot improves | the same (HardSlot has no selection gradient) | S0-calibrated, no extra tau search | full adaptation; soft mixture or temperature suspected, HardSlot stays diagnostic |
| R3 | not R1, R2; Soft or Hard improves in `final_only` | the same, scope `all_valid_question` | S0-calibrated | limited adaptation; `final_only` reported separately; a read-position protocol study is flagged (not authorized here) |
| R4 | not R1 to R3; Clean-final `>= +5.0 pp` over Matrix_cal | the same | S0-calibrated | limited adaptation; addressing, query or role interface flagged as the main bottleneck (S2 priority) |
| R5 | none of the above | the same | S0-calibrated | limited adaptation: one pre-budgeted pass (§6.5, §14.3) |

  S1A is one identical, pre-budgeted pass in R1 to R5 and its rule below never changes. "Limited" only fixes the wording of a null: under R1, R2 a null S1A contradicts a positive S0 signal; under R3 to R5 it is the expected null and closes the single-hop slot line (§7.4, §14.3).
- **D128-3, once**, on S0 dev only: Native (delta store), Matrix and SoftSlot at S-RE's selected `(g, tau)`, HardSlot at S-RE's `g`, no recalibration. It checks that the direction of Soft minus Matrix does not depend on the ridge-trained checkpoint; descriptive.
- **Outputs:** `reports/slot-loop-v1/s0/` (metrics, per-sample predictions and slot-weight summaries, all calibration candidates), `decision_s0.md`, `preflight/`, `manifest.lock.json`, `checkpoint_inventory.json` (blake2b-128 of the real S-RE and D128-3 files, step 2).

## S1A: matched training from the same S-RE state (plan §7), locked before any S1A data are seen
- **Arms**, both from the one S-RE `writer.pt` (one digest): `Matrix-L1` (ridge store as in S-RE, Matrix backend, S0's calibrated `g`) and `SoftSlot-L1` (SoftSlot backend, S0's calibrated `g, tau`). Q, R, U, B start from S-RE. The loader keeps all buffers and never calls `WriterFixExperiment.standardize()` (`writer_fix.py:234`; it overwrites `input_mean/std`).
- **Frozen:** `A, K, V, a_lambda, a_beta` (no `w_lambda`, `w_beta`: scalar gates), buffers `input_mean, input_std`, DPFP nu 3, ridge eta 1e-4, epsilon, rho 0.25, alpha 1, `g`, `tau`, backbone. The shared `A` feeds writer and query (`adapters.py:38`), so freezing the writer freezes it.
  **Trainable:** `Q, R, U, B` only, 3 x 16,384 + 131,072 = 180,224 parameters per arm; the other 294,914 adapter parameters are frozen; no auxiliary head.
- **Freeze verification, not by names:** (i) the tensors with `requires_grad` equal `{Q,R,U,B}` and the optimizer holds exactly them; (ii) blake2b-128 of the raw bytes of every frozen tensor and buffer is equal before training, at every checkpoint and at the end, and those of Q, R, U, B changed;
  (iii) on a fixed support batch (S1A smoke set, same precision) the written keys, values and matrix state are digest-identical before and after the 20-step smoke and at the end of every run; (iv) after one backward pass frozen tensors and backbone have no gradient and Q, R, U, B have finite non-zero ones (the graph runs through the frozen upper layers, no `no_grad` on the read path). The run record stores names, counts and digests.
- **Training:** T1, E16, memory-only, L1, all valid question tokens, batch 32 episodes x 4 questions; loss = full-vocabulary answer cross-entropy only (D9); fresh AdamW (betas 0.9/0.999, eps 1e-8, weight decay 0.01), global gradient-norm clip 1.0 with `error_if_nonfinite`, linear warmup over 20 steps then constant LR (D4);
  both arms see the same batches in the same order at one run seed; run seeds 17, 29, 43 (data order only, one parent). The parent's optimizer state is unused.
- **LR pilot** (before any run, both arms alike): LR in `{3e-4, 1e-3, 3e-3}` x two arms = six runs of 200 steps from the S-RE state on the pilot train stream (the same 6,400 episodes for every candidate), each evaluated once on the 1,000-episode pilot shard after step 200, plus step 0 per arm; pilot weights are discarded.
  Per arm choose the lowest pilot-shard NLL; within 0.01 nats of the best is a tie: the lower LR. Non-finite candidates are excluded; if every candidate of an arm is worse than its step 0, STOP for the orchestrator (the grid is not extended). Unselected results are recorded.
- **Step count from the timing pilot.** Over the six pilot runs: `S` the largest steady mean step time (steps 11 to 200), `Ev` the largest 1,000-episode evaluation time, `Ld` the load time. `N = min(6000, 10 * floor(((3600 - max(90, Ld)) / 1.5 - 11 Ev) / S / 10))` (the rule of `release.largest_schedule`: 1 h, margin 1.5, 11 evaluations, multiple of 10).
  The same `N` for both arms and all seeds; walltime = (Ld + N S + 11 Ev) x 1.5 rounded up to 5 min, at most 3,600 s. If `N < 3000`: STOP; `--scale-up docs/prereg/SLOT-LOOP-v1.md` (2 h) is allowed for the six arm jobs only, with the orchestrator's OK. No other S0 or S1A job exceeds 1 h.
- **Checkpoints and selection.** Dev (1,000 episodes) is evaluated at step 0 (recorded, not selectable) and at `k N / 10`, `k = 1..10`. Selected = lowest dev NLL, a tie at 1e-6 goes to the earlier; a checkpoint is kept when it improves the best so far strictly. Every dev evaluation is logged in the run record. Selection never uses confirmation data.
- **Release** (`reports/slot-loop-v1/s1a-release.json`, committed, then STOP for `approval.json`): `N`, LR and `(g, tau)` per arm, the six science config hashes (`stable_hash`), manifest and inventory digests, this file's blake2b-128, the full 40-hex science commit, the freeze verification, and a reload check per arm
  (the export reloads, its config equals the one the release implies, a fresh 64-episode evaluation reproduces the smoke predictions; pattern `scripts/check_smoke_reload.py`).
- **Order after the runs:** six arm-seed jobs in parallel; dev selection; the §7.4 trigger on dev; the ablation if triggered; then `reports/slot-loop-v1/s1a-selection.json` (selected checkpoints and digests, dev tables, trigger outcome) and `STOP: S1A selected` (D11).
  After the orchestrator's `s1a-open.json`: the confirmation and unseen evaluation of the six selected checkpoints and of the unmodified S-RE (descriptive reference), the frozen analysis, the decision and report.

### S1A decision rule (plan §5.4), one primary contrast
SoftSlot-L1 minus Matrix-L1, full-vocabulary top-1 EM on the 2,000 confirmation episodes (one question each). With `c_asi` the 0/1 correctness of arm `a`, seed `s`, episode `i`: `d_i = (1/3) sum_s (c_Soft,s,i - c_Matrix,s,i)`, `Delta = mean_i d_i`, `Delta_s` the same per seed.
The interval is the percentile 95 % interval of 10,000 paired bootstrap resamples of the episode index (same indices for both arms and all seeds; numpy `default_rng(9056755664197379988)`); `L` is its lower end.
- **ADVANCE** (H1 supported at the registered level, conditional on this parent writer and these continuation seeds) iff `Delta >= +3.0 pp`, `L > 0`, and `Delta_s > 0` for each of seeds 17, 29, 43. It opens the S1B addendum.
- **NOT ADVANCED** otherwise. The report gives `Delta`, `L`, the three `Delta_s`, per-seed EM with paired intervals, NLL, read error, the parent S-RE row and the step-0 rows (descriptive). The §7.4 ablation, if triggered, is the only further S1A work; S1B on the slot reader needs a new addendum.
- **§7.4 ablation, one, pre-budgeted.** Trigger, on dev before the confirmation set is opened: all six runs finished, every logged loss and gradient finite, each selected checkpoint's dev NLL at most its own step-0 dev NLL (stable), and the mean over seeds of the dev EM difference (Soft minus Matrix) below +3.0 pp. If not stable: no trigger, STOP for the orchestrator.
  Ablation: both arms get their own trainable copy of `A` for the query input (initialised to the frozen `A`, +262,144 parameters, reported; the freeze verification applies with the copy added to the trainable set), seed 17, LR per arm as selected, the same `N`, dev only, no confirmation or unseen data. A dev gain of Soft over Matrix of at least +3.0 pp is flagged for a new registration with its own confirmation set and does not change the S1A verdict; no gain stops single-hop slot investment (§7.4, §14.3).
- **Descriptive, optional:** unseen-name EM of the six selected checkpoints; a common-fixed-LR comparison (seed 17, dev only): the LR `c*` minimizing the sum of the two arms' pilot NLLs, both arms at `c*`, run only if the selected LRs differ and the budget covers it after the ablation. It decides nothing and is never the primary contrast; the primary comparison is "equal tuning effort" (same candidates, steps and shard), not "same optimizer trajectory".
- **Analysis script** (`scripts/analyze_slot_study.py`, stdlib and numpy, on compute): frozen with its blake2b-128 in the release. At the `S1A selected` stop the orchestrator's independent reviewer re-runs it on the S1A dev records; its digest goes into `s1a-open.json`, and the confirmation analysis uses the same digest unchanged.

## Verification list (plan §12) and where each item is checked
Tests and preflight run on compute (`scripts/submit.py`); tiny-model tests plus preflight on the real parents.

| plan §12 item | check |
| --- | --- |
| original-path identity | preflight: Native replay of the old range (723/1000, NLL, predictions vs `val-6000/samples.npz`), Matrix bitwise equal Native, D128-3 replay; test: old matrix path bitwise unchanged with the option off (`tests/test_llm_dpfp_repro.py`, `tests/test_sq2.py` stay green) |
| owner and mask | test and preflight: two episodes with distinguishable sentinel keys/values, reordered, expanded per question, padded; only the owner's rows change |
| feature consistency | preflight: captured keys equal unit DPFP features of the recomputed `K u` (768-d), `M q` equals `sum_i d_i v_i` (`gram_relative_error`), `base_keys` unused; tiny-model test |
| slot reference read | test: hand-computed small case, empty row gives zero, padding slots inert, HardSlot ties |
| read-only isolation | test and preflight: blake2b-128 of the state before and after several questions and several L |
| writer frozen; gradient reachability | S1A smoke job on the real S-RE: the freeze verification (i) to (iv), 20 optimizer steps; tiny-model test |
| capacity boundary | test: E = S passes, E > S raises, no truncation (S = 16) |
| byte accounting | test: `sum(numel * element_size)` = 57,368 B per episode for S = 16 (57,344 + 16 valid + 8 count), 393,216 B for the matrix |
| data isolation | manifest script, preflight and guard tests above |
| export and reload | per-arm reload check (predictions identical; backend, `g`, `tau`, scope, buffers restored); an export lacking `input_mean/std` or the backend settings fails |
| NULL labels | not applicable until the S2 addendum |

## Budget (node·h; one node per job, at most 1 h, at most 6 of my jobs queued or running; walltime request = measured x 1.5)
Basis (`docs/reports/SQ2.md`, S-RE): 0.199 s per step, 6.8 to 7.2 s per 1,000-episode evaluation; load about 1 min (estimate). The slot read adds 16 slots to a 128-d read; the timing pilot decides. The totals are the caps; a job that would push the running total over its cap is not started without the orchestrator.

| job | expected | walltime request |
| --- | ---: | --- |
| S0 tests (new code, unchanged-path tests, one whole-suite run) | 0.35 | debug-g, <= 30 min each |
| S0 preflight (two identity replays, sentinels, isolation, freeze, bytes, capacity, manifest) | 0.10 | debug-g, 15 min |
| S0 run (18 x 2,048-episode calibrations about 15 s each, about 20 dev evaluations, D128-3) | 0.20 | debug-g, 25 min |
| **S0 total** | **0.65** | **cap 1.0** |
| S1A tests (trainer, release, analysis) | 0.15 | debug-g, <= 30 min |
| S1A smoke (freeze, export/reload) | 0.05 | debug-g, 15 min |
| S1A LR and timing pilot (6 x 200 steps, 8 evaluations) | 0.12 | debug-g, 25 min |
| S1A release reload checks (2 x about 25 s) | 0.03 | debug-g, 10 min |
| S1A six arm-seed jobs, each load + `N` x 0.2 s + 11 evaluations, about 0.375 | 2.25 | regular-g, about 35 min each |
| S1A confirmation and unseen evaluation (one job), analysis | 0.12 | debug-g, 20 min |
| S1A conditional ablation (two runs, about 0.375 each) | 0.75 | regular-g, about 35 min each |
| S1A optional common-LR comparison (two runs) | 0.75 | regular-g, about 35 min each |
| **S1A total** (without / with ablation / with both) | **2.8 / 3.5 / 4.3** | **cap 4.5** |

S1B (<= 11.5, the plan's figure), S2 and S3 budgets are set in their addenda from measured step times and approved by the orchestrator. Node·h per job and the cumulative total go into the state file.

## Deviations from the plan and the task (each flagged for review)
1. One file registers S0 and S1A, later stages are addenda (task scope, 8259878). The S1B gate "S1A rule met" is this draft's reading of plan §0.3, §7.4, §14.3; the plan does not say whether a matrix-only loop test (M1 against M4) may follow a null S1A.
2. SoftSlot-recency is registered but not calibrated: S-RE's decay gate is 0.999999, so the added `log(lam^(E-1-i))` is below 1.6e-5 in magnitude and the variant should equal SoftSlot up to numerical noise (the ridge store with eta 1e-4 is practically recency-free as well). It is a check at `(1,1)` and at SoftSlot's selected values.
3. The qualitative S0 table (§6.5) gets numeric thresholds (+5.0 pp on S0 dev with lower read error); ties are defined (1e-4 nats in calibration, 0.01 nats in the LR pilot, 1e-6 in checkpoint selection). The plan leaves these open.
4. Fresh AdamW gets a 20-step linear warmup, the original loop's default (`read_fix_run.py:133,154`); SQ2's continuation had none because it loaded moments. The plan is silent.
5. The LR pilot has its own 1,000-episode shard, not "the same small dev shard" (plan §7.3), so LR choice and checkpoint selection do not consume the same data.
6. The D128-3 delta path is also replayed on the old range (608/1000), an identity use like S-RE's; the task named only S-RE's replay.
7. Ten evenly spaced checkpoints plus step 0 (not selectable) instead of the original five; `N` is a multiple of 10. The plan asks only for one predetermined frequency.
8. The §7.4 ablation is fixed to the separate query projection (the plan allows that or re-initializing the reader), seed 17, dev only; the plan does not fix seeds or data.
9. The S1A loss is answer cross-entropy only (plan §7.3): both arms drop S-RE's auxiliary terms acting on Q (InfoNCE address loss, raw-read loss, training-only value head; `read_fix_run.py:159-161`). Matrix-L1 is not a pure continuation of S-RE's objective; the step-0 rows and the parent row show a drop in both arms.
10. The unseen-name diagnostic is kept for S1A (descriptive, sealed with the confirmation); S0 has none.
11. One STOP is added, `S1A selected`, between dev selection and the confirmation evaluation, so that the orchestrator's independent re-run of the analysis script and the opening record `s1a-open.json` come before any confirmation data are generated; the task lists no stop there.

## Plan against code (plan §4.3, §4.1, §4.2, each checked)
| plan claim | evidence | verdict |
| --- | --- | --- |
| §4.3(1) `rfx.FinalRead` cannot read S-RE; old L1 final-token reads only | `rfx.py:58-59` refuses DPFP, ridge, unit query, non-`iterq`; `:90-91` L1 only; reads at the last token (`:49-52,116`) | true |
| §4.3(2) `read_fix_run.validate()` fixes the old range and seed | `read_fix_run.py:57-60`: `generate(spec,'val',0,count,4242)`, stratify seed 4242 | true |
| §4.3(3) `LLMExperiment._load_writer` does not take the SQ2 export | `train.py:29` (`WRITER_NAMES` omits the standardization buffers), `:185-190` (exact `record['writer']` names and equal signature); the real `writer.pt` holds `adapter` (11 tensors) and a signature with `writer_fix`, `read_fix` fields | true |
| §4.3(4) `stage='writer'` forces T1 | `train.py:36-38`, `config.py:222-223` | true |
| §4.3(5) `writer.pt` has no value head | `read_fix_run.py:32-37`, `sq2.py:90`; checked on the real files (structure only): `checkpoint.pt` has `optimizer` and `value_head`, `writer.pt` neither | true |
| §4.3(6) no multi-node; `--commit HEAD` resolves in the main checkout | `submit.py:86-87`; `:25,43-46`; `job.pbs:31-33,54` (snapshot in a scratch directory removed on exit) | true |
| §4.3(7) `fit()` is not a general deep trainer | lr `.003` (`read_fix_run.py:139`), five checkpoints (`:142`), batch 32 (`:146`), L = 1 (`:157`), warmup 20 (`:133`) fixed in code | true |
| §4.1, §4.2 assets and loaders | directory, config hashes, commit, 6,000 steps, 723/1000, NLL (`record.json`, `training.json`); D128-3 digests, 22,370 steps (`sq2_release.py:17-19`); loaders `sq2.py:88`, `dpfp_run.py:20` | true |
| §1.2, §1.3 numbers used here (SQ2, RFX, DPFP, LSQT 46.1 %, 0.512) | agree with `docs/reports/SQ2.md`, `RFX.md`, `DPFP.md`, `LSQT.md` | true |

No claim of §4.3 is false. Differences found while checking:
- The recency premise of plan §6.2 is void for S-RE (D2); the plan's wording was hedged ("possible"), so it is not a false claim.
- `config.py:179-180` caps `eval.episodes` at 2,000, below the plan's 2,048 calibration set: sizes come from the manifest, never `resolve_config`.
- One question per episode, so the plan's "whole-episode" bootstrap equals a per-question one here.
- `WriterFixExperiment.standardize()` overwrites `input_mean/std`, and `read_fix_run.run_one` calls it (`:215`); the new loader must not follow that path (freeze verification, test).
- S-RE's own training used auxiliary losses on Q (D9): the answer-only S1A objective changes the control, not only the arm.

## Addenda for S1B, S2, S3 (rules only; their contents are not locked here)
Each is a section of this file or `docs/prereg/SLOT-LOOP-v1-S1B.md` and so on, registered by the orchestrator before any data of that stage is seen, with its own seed namespace, its tests of H2 or H3 where named, decision rule, verification list, budget table from measured step times, and the claims it makes.
Inherited unless an addendum says otherwise: the purpose-seed guard and registry, full-vocabulary EM, the paired episode bootstrap with a fixed seed, the old-study non-interference rules, blake2b only, absolute output paths under `/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/`, one-node jobs.
The `MemoryBackend` semantics fixed at the S0 code review (read-only, owner-indexed rows, capacity error, zero for empty rows, byte accounting; plan §11.2) are the interface later preparation by another worker builds on; the state file documents it.
Task cautions: S2 first checks `t2_typed` and `n_distract` in `data/relations.py` against plan §9.4; bridge, NULL and role labels reach only losses and metrics, never `forward`, `read` or `stop`; the S1B mechanism controls (plan §8.2) run only after a loop gain appears; the S3 real-task extension waits for its own addendum.

## Open questions for the orchestrator
1. S1B gate: is "S1A ADVANCE" the entry, a null S1A (and null ablation) ending the slot line (this draft), or may a matrix-only loop addendum (M1 against M4) follow a null S1A?
2. The numeric S0 thresholds (D3), the 20-step warmup (D4), the separate pilot shard (D5) and the ablation choice (D8) are mine: accept, change or strike.
3. The D128-3 identity replay on the old range (D6) goes beyond the task's single allowed use of that range.
4. The added `S1A selected` stop (D11): keep, or let the worker open the confirmation set on its own after dev selection?

## Decisions at registration (orchestrator)
(to be filled at registration; the registered digest goes into the release)
