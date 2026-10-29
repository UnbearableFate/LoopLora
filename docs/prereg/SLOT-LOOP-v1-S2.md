# SLOT-LOOP-v1 S2: role, transition and terminal-state ladder (WP-L15)

**Status: registered 2026-10-30 01:55 JST by the orchestrator (decisions at
the end; where they differ from the text above them, the decisions govern).**
Drafted by wk-slot on `wp/slot` (8948d00), after the one authorized merge of
master `0128a6e331f3492e3efa1d595c1f15e28eb0a8b7` (fast-forward from `81e8996`).
No S2 implementation, episode generation, model execution or job accompanied
this draft, and no S1B pilot, dev, confirmation or training outcome existed
when it was registered. Decision 6 sets what may start before S1B's verdict;
S2 data beyond smoke, the lock and everything after it wait for that verdict.
A ready release needs separate approval before formal training. This document
specifies future behavior, not existing runner capabilities.

Authority: [parent prereg](SLOT-LOOP-v1.md), [S1B §5](SLOT-LOOP-v1-S1B.md),
[S1B A1](SLOT-LOOP-v1-S1B-A1.md), [WP-L15](../tasks/WP-L15_slot_s2.md), and
[plan](../../plan/looplora_diagnosis_and_execution_plan_2026-10-29.md)
§§5, 9, 11–13 and its execution record. Departures and completions of the plan
are collected in §12 for explicit registration. Old verdicts and writer gates
remain closed. Written aggregate budgets do not bind; each job is ≤16 node·h.

## 1. Evidence, blindness and S1B-dependent entry

The S1A [decision](../../reports/slot-loop-v1/decision_s1a.md) is ADVANCE:
SoftSlot−Matrix +3.7333 pp, paired 95% interval [2.70, 4.8167] pp, three positive
continuation seeds, T1/E16/L1/all-valid-question. It licenses S1B's four-arm
branch, not semantic traversal on the S2 task.

Allowed S1B evidence was read with `git show` from wp/loop snapshot
`76d66e1e55535b954d3fc2ed2d7993401c2e7eb4`, without merging that branch:
`src/looplora/llm/s1b_{model,data,rules,train,analysis,artifacts,preflight,run}.py`,
`docs/reports/WP-L14.md`, and
`reports/slot-loop-v1/s1b/diagnosis-6855048/` (diagnosis, prompts and their report).
The diagnosis is job 3729676, science commit `6855048`, diagnosis-file
blake2b-128 `2ad225aa70f09940ff68b56c214ae0d3`. On calibration hop 1, actual
final-token key top-1 was 14.7949% for T2 wording, 89.4% for T1, and 59.4727%
for the same T2 graphs with T1 wording. Clean-final oracle EM was 79.49% on T2;
the frozen audit value head decoded 83.64%. These support a query/wording/position
transfer problem, with residual domain shift; they do not isolate wording from
token positions or prove perfect writer transfer. A1 removes that calibration
stop, imports S1A read settings, and increases S1B to 18,000 updates.

**Blindness:** no S1B pilot, dev, confirmation, unseen or training records under
`runs/slot-loop-v1/s1b/` were read. No S1B outcome is assumed here. Later the
orchestrator supplies a frozen verdict/open/release lineage; branch resolution
consumes that record mechanically. Timing-only facts may be supplied after this
draft, separately from S1B score payloads.

Define **qualified** exactly as S1B §5: the independently trained L4−L1 contrast
passes Δ≥3 pp, its registered adjusted lower bound >0, every seed positive,
hop-1 loss ≤2 pp; the same L4 checkpoints additionally pass the registered
pooled-positive/per-seed-nonnegative @4−@1 sweep and trace/isolation checks.
`C_storage passes` means S1B's registered S4−M4 test, not a favorable point alone.

| S1B outcome | S2 primary | S2 backends to run | interpretation |
| --- | --- | --- | --- |
| only Matrix qualified, either storage outcome | Matrix | Matrix only | only Matrix demonstrated a usable loop |
| both qualified, C_storage passes | SoftSlot | SoftSlot and Matrix, complete matched ladders | slot-led; the matrix ladder remains mandatory |
| both qualified, C_storage fails | Matrix | Matrix and SoftSlot, complete matched ladders | matrix-led; no established slot advantage |
| only SoftSlot qualified, either storage outcome | SoftSlot | SoftSlot only | a slot loop is eligible; no superiority claim if storage failed |
| neither qualified | none | none | **no S2**; claims stay at reading/memory adaptation |

Missing, invalid, unstable or provenance-inconsistent S1B results cause STOP,
not selection of a negative row. Let K=1 or 2 be the number of active backends.
No S2 score can change the primary backend, remove a comparator or change K.
S2 does not add an inferential storage-superiority claim: backend differences
are descriptive here, including in the slot-led branch.

## 2. Task, relation use and locked layout

Use `t2_typed`, **E=24, n_distract=12, n_rel=2, chain_lengths=(8,4)**, with a
registered episode-level relation-label permutation. All 24 facts remain visible
for every query and every hop: 12 chain edges and 12 distractor edges. Training
hops are sampled uniformly from 1–3; evaluation stratifies hops 1–8 on the same
graph. Chain lengths, fact count and distractor count never depend on the queried
hop. Fact order and injective entity renaming are randomized by the generator.

The complete native TaskSpec, including unused defaults, is:

```json
{"E":24,"Q":4,"V":1024,"base":null,"chain_lengths":[8,4],"head_sharing":0.0,"hops":[1,2,3],"missing_rate":0.5,"n_distract":12,"n_rel":2,"n_updates":2,"name":"t2_typed","seen":[0,768],"unseen":[768,1024],"update_query_rate":0.5}
```

Native `stable_hash(asdict(TaskSpec))` = **`2b15300ed77bd6df1751798ed2ff46ac`**.
The S2 layout is the following exact composite object; sorted compact JSON,
blake2b-128, gives **`1014521816624879d8bc3c2716fce58b`**:

```json
{"query_protocol":"stratify_raw_relation0_before_swap_v1","raw_generator":"relations-v1","schema":"slot-loop-v1-S2-typed-swap-v1","task":{"E":24,"Q":4,"V":1024,"base":null,"chain_lengths":[8,4],"head_sharing":0.0,"hops":[1,2,3],"missing_rate":0.5,"n_distract":12,"n_rel":2,"n_updates":2,"name":"t2_typed","seen":[0,768],"unseen":[768,1024],"update_query_rate":0.5},"transform":"swap_0_1_by_derive_seed_canonical_id_S2_relation_swap_mod2"}
```

Both digests were calculated on login with bounded stdlib JSON/hash arithmetic,
without importing the generator or producing episodes. Compute preflight must
re-derive them from the real TaskSpec and wrapper; no digest substitution.

**Code evidence and the necessary wrapper.** In `data/relations.py`, `kind`
maps `t2_typed` to chain-end. `_layout` subtracts `n_distract` from E; the chains
are relation 0. `generate` draws 12 distinct `(chain node, nonzero relation)`
keys, each pointing to a fresh leaf. Here there are 14 eligible chain nodes and
one nonzero relation, so the 12 distractors are legal; total visible entities
are 14+12=26. Merely setting `n_rel=2` does **not** vary the queried relation:
native `q_rel` remains 0. Also `sym/evaluate.stratify_queries` hardcodes relation
0. Passing a swapped graph into that stratifier would select the wrong paths.

The S2-only wrapper therefore follows this fixed order:

1. Generate native episodes using the exact TaskSpec, native `relations-v1`
   generator, purpose/train base seed and split/index. Keep the raw episode ID.
2. For evaluation, stratify the **raw** graph on requested hops, using that
   purpose's base seed and raw episode IDs, before relabeling. Training keeps
   the four native sampled queries. No new TaskSpec/hops digest is generated
   for evaluation; Q expansion is query metadata only.
3. Assign canonical S2 ID =
   `derive_seed(base,"relations","relations-v1",S2_layout_digest,split,index)`.
   Set b = `derive_seed(canonical_id,"S2","relation-swap") % 2`. Simultaneously
   replace every fact relation r and every question relation r by `r XOR b`.
   Entity IDs, facts' order, validity, starts, hops and answers stay unchanged.
4. Record both IDs, both layout digests and b in generator-side provenance;
   replace public `episode_seed` with the canonical S2 ID. The raw graph RNG
   still used the native ID in step 1. This two-ID convention is explicit;
   never claim that it is the unmodified generator's RNG identity. No label or
   raw ID becomes an extra model input.

Thus both `knows` and `trusts` are followed across episodes, each question names
the relation, and every graph contains other-relation edges sharing chain heads.
The Bernoulli permutation is deterministic, not forced into equal sample counts;
report counts and per-relation metrics. This task has one long-chain relation
per graph and leaf distractors, **not two interwoven multihop relation graphs**.
It does not establish variable relation-sequence composition.

Use the existing chain-end wording from `render._question` verbatim:
`Starting from {name}, keep following the relation '{relation}' until there is no further fact for that relation. Which person is reached?\nAnswer:`.
`q_hop_given=-1`; no true hop is rendered. All arms use the same names, text,
tokenization, graph, support/query boundary, span annotations and answer IDs.
Use pinned seen [0,768), unseen [768,1024), V=1024/Unknown; answer scoring is
over the complete model vocabulary. Main graphs are complete, acyclic,
functional by `(entity,relation)`, have no updates/missing facts/eviction, and
fit SoftSlot capacity **24**. Chain-end terminates successfully at an entity
with no edge of the **requested** relation, even if other-relation edges exist.
No main UNKNOWN targets occur; UNKNOWN recall is N/A, its prediction rate is
reported. Missing-fact and relation-sequence tasks need separate registration.

**Relation-use diagnostic, fixed now.** For every evaluation query, fork only
its named relation to the other relation on the unchanged graph/start. Its
visible-graph chain-end answer is that distractor leaf, or the start if no such
edge exists. Keep this h=0/1 probe outside primary hop strata. Report correct
paired changes and a byte-identical question re-render control; failure prevents
a relation-sensitive mechanism claim even if the ordinary task is solved by
detecting the graph's long-chain relation. Probe targets enter metrics only.

## 3. Position protocol and query formation

**Every S2 arm trains and evaluates with `last_valid_question`.** At layer 14's
o_proj, other valid question positions have zero adapter branch, retaining the
frozen base projection. The final nonpadding question token alone starts one
controller. This is not S0's `final_only`: `ScopedRead` keeps a native matrix
read at earlier tokens and is not an implementation of this protocol. No hidden
native/alternate-state fallback is permitted.

The S-RE parent and S1A/S1B were trained with all-token injection. Removing it
changes the upper-layer context and learned reader's operating distribution.
All S2 arms therefore start fresh from the same parent and are retrained here;
an old all-token model or inference-only final-token ablation is historical
context, never a matched S2 baseline. Reset output B to zero in **all** arms as
the plan requires. Before optimization every arm's logits must consequently
equal frozen-base logits bitwise on smoke inputs. An earlier-token branch
sentinel must show exactly zero injection, including padding/reorder cases.
The inherited unmodified all-token path still passes its old identity fixtures.

For semantic arms, c is the parent's frozen standardized input path u at the
last question token (128-d); p0 is the L2-normalized 96-d projection of the
mean frozen insertion-layer feature x over the question's **start entity span**
(2048-d). a0=0 in R32. The shared trainable `f_entity:2048→96` is also used by
the role and transition targets. It is a text-feature encoder, never an ID
embedding, token-ID lookup table or oracle codebook. Span metadata locates text;
it does not supply entity/path vectors to ordinary prediction.

This removes reliance on the final token alone to *locate* the initial entity;
it does **not** prove correct addressing or cure the S1B failure by itself.
The new address projection is random, the writer's keys remain contextual
DPFP codes, c still depends on wording, and the start mention precedes the
relation phrase in a causal model. **p0 alone cannot select the relation**;
F_addr must use c as well. No T1 rewording or oracle key is substituted.

Before any S2 optimization/calibration/pilot, smoke-only preflight checks:

- Tokenizer offsets round-trip exact head/tail/start/relation text spans;
  repeated entities, multi-token spans, padding and answer `:` are aligned.
  Current `FactSpan` covers whole facts and `RenderedQuery` exposes no mention
  spans: add explicit S2 span handling, preserving the old renderer contract.
- Captured frozen x at each mention and final position is the actual insertion
  input; averaging and normalization reproduce p0/c exactly. Verify that
  changing only the named relation changes the appropriate question text/c;
  ordinary forward accepts spans and text, never relation IDs as an extra input.
- Record untrained actual first-query key top-1/margins by relation, and ordinary
  versus true-value-final oracle diagnostics on hop-1 smoke. B=0 initially makes
  answer-logit oracle differences uninformative; audit raw reads and the already
  frozen value head separately, with no trainable audit decoder. There is **no
  untrained-EM/key-accuracy gate** for randomly initialized semantic coordinates.
- Freeze/owner/position and trace identities must pass. A failure stops before
  training; low but correctly measured random-start retrieval is reported, not
  silently repaired by another wording, pretrained query or temperature search.

## 4. State, initialization and parameter budget

The common parent is the unadapted S-RE inference export, not a selected S1B
model: `/work/xg24i002/x10041/LoopLora/runs/p1l/sq2/arms/S-RE/writer.pt`,
blake2b-128 `3994a5631021de03b2039974ac934e56`, config
`1bfb72b7d87bc0aff743c1e018e8680e`, record.json
`816570a0882ed6952b4f851b67d4ef9f`. Strict-load its original config/buffers, then
attach an explicit S2 task/reader. Qwen3-0.6B-Base/revision, layer 14, memory-only,
bf16 backbone, fp32 adapter/state, DPFP ν3 (dk=768), dv=128, ε and normalization,
ridge η=1e-4, ρ=.25 and α=1 are inherited. Read settings are **Matrix g=1;
SoftSlot g=1, τ=.1**, from S1A release `523d11d57a08a2491893ebe7cedc86a7`
and S1B A1, with no new calibration search. B=0 makes a step-0 answer-NLL read
grid degenerate. The S2 calibration set is a descriptive interface audit only.

Frozen in every arm: writer/shared A,K,V,a_lambda,a_beta, input_mean/std,
backbone and preprocessing. Never call `standardize`. Support writes remain the
original fact-final S-RE DPFP keys/raw values; the mention encoder never replaces
them. Only one backend's read-only state is visible to question forward. At E24,
fp32 slot state is `4×24×(768+128)+24+8 = 86,048 B/episode`; Matrix remains
393,216 B. These exclude transient preparation and model/controller memory.

Semantic recurrence, shared across steps (s indexes reads starting at zero):

- `q_s = query_code(F_addr([p_s,c]))`, 224→128→unit DPFP R768;
  `m_s = backend.read(state,q_s,owner)` in R128.
- `p_candidate = normalize(F_step([p_s,m_s,c]))`, 352→96;
  `a_candidate = tanh(F_answer([a_s,m_s,c]))`, 288→32.
- Without NULL, take candidates. Output at the final question position is
  `W x_q + α B[p_L;a_L]`, followed by the frozen upper model/full-vocab head.
  No extra R/U, residual u or value-to-state head is hidden in these mappings.

All new linears use Xavier-uniform weights, zero biases; normalization ε=1e-6.
B has no bias and is exactly zero initially. Base-old retains parent Q/R/U and
the original `z←u+ρ ReLU(Rz+Um)` recurrence, with B zeroed. The semantic arms
remove old unused Q/R/U from their inference export/optimizer. No inherited
optimizer state. One module-named deterministic initialization derived from
`derive_seed(20261029,"slot-loop-v1","init","S2") = 7076989969446901480`
is common across backends, rungs and seeds; derive each module's RNG seed from
this base and its fixed module name. Continuation seeds change data order only.
Every semantic rung starts from identical f_entity/F_addr/F_step/F_answer/B
bytes; no warm-start from a successful preceding rung.

| arm/module | trainable parameters | total adapter inference parameters |
| --- | ---: | ---: |
| Base-old: Q/R/U/B | 180,224 | 475,138 |
| f_entity, 2048→96 with bias | 196,704 | included below |
| F_addr, 224→128 with bias | 28,800 | included below |
| F_step, 352→96 with bias | 33,888 | included below |
| F_answer, 288→32 with bias | 9,248 | included below |
| B, 128→1024, no bias | 131,072 | included below |
| Arch / Arch+Role / +Transition | **399,712** | **694,626** |
| +Null: above + NullHead 352→1 | **400,065** | **694,979** |
| Base-budget, defined below | **399,752** | **694,666** |

The common frozen adapter part is 294,914 parameters, excluding the backbone.
Semantic trainables exceed Q/R/U/B by 219,488 (NULL: 219,841), so rank 128 does
not mean equal parameter count.

**Mandatory Base-budget control:** retain Base-old's Q/R/U/B and recurrence,
add the same f_entity(start span) and `G:96→128` with bias, setting
`u'=c+G p0`, `z0=u'`. At each read use the old query/read, then
`z_next=u'+ρ ReLU(Rz+Um)+H2(ReLU(H1(z)))`, with H1:128→40 and H2:40→128,
both biased. All additions are trainable, Xavier/zero-bias; B=0. Counts are
180,224+196,704+12,416+10,408=399,752. This is 40 above Arch and 313 below
+Null (<0.08%); equal tuning/data/updates apply. It gives the old recurrence the
same mention input and almost the same parameter budget. It is a separate arm,
not a replacement named Base-old. Arch must beat both controls before an
architecture benefit independent of these two confounds is claimed. It does
not match FLOPs or every operation. No speculative writer retraining or
S2-W, untied, attention-NULL or larger-state variant is included.

At zero B, answer loss gives zero upstream reader gradients on the first
backward; that is expected. Verify B's finite nonzero gradient at step 1 and
the other answer-only reader modules' finite nonzero gradients after B has
updated, not an impossible all-parameters-step-1 assertion. Auxiliary gradients
are checked separately. Check exact optimizer sets/counts and all frozen bytes,
plus fixed-smoke support K/V/M digests, after smoke, each checkpoint and at end.

## 5. Ladder, training and release gates

All ordinary training uses a **fixed four-read unroll**, 32 episodes ×4
questions, 18,000 updates (576,000 episodes/2,304,000 questions per arm/seed),
seeds 17/29/43, identical stream/order across backends and rungs. No mixed-depth
training. Each rung trains from the common initialization for the full schedule.

| rung | objective | launch dependency |
| --- | --- | --- |
| Base-old | final answer CE | S1B entry, registration, reviewed implementation, preflight/lock and own approved release |
| Base-budget | final answer CE | same; may run with Base-old and Arch |
| Arch | final answer CE | same; may run with both controls |
| Arch+Role | CE + .1 Role | previous wave complete and stable on dev; no requirement for positive Arch gain |
| +Transition | CE + .1 Role + 1.0 Step | Role complete/stable with valid noncollapsed codes; no requirement for a positive Role task contrast |
| +Null | CE + .1 Role + 1.0 Step + .1 NULL | Transition dev mechanism gate below, then its own smoke/pilot/release approval |

The same rung for active backends and its three seeds may run in parallel,
within scheduler limits. A later wave is never warm-started. Each release
extension lists the predetermined arm/configs and reuses the immutable prior
release lineage; do not edit an approved release file. Ideally all core code,
including NULL and analysis, exists before the first preflight so later waves
need no source changes. Do not inspect confirmation for intermediate gates.

"Stable" means complete finite training/dev, correct freeze/provenance, and
selected dev NLL no worse than that arm's step-0 NLL. Before Transition, Role's
dev visible-entity code audit must show finite normalized targets and no exact
collision between distinct names within any graph; an all-zero/collapsed target
space is not a usable semantic control. An audit failure stops for diagnosis,
not LR-grid expansion or removal of difficult entities.

**NULL entry:** on the selected Transition dev checkpoints, each seed must have
≥80% next-address top-1 and ≥80% decoded bridge accuracy on path transitions of
hops 2–3 (definitions §8). Otherwise no Null training or H3 claim; finish the
available ladder and report semantic failure. Gates cannot alter the analysis
family size or recycle confirmation. A qualified backend that fails this S2
gate remains in reporting; no backfilling a substitute backend.

Optimizer: fresh AdamW (.9/.999, eps1e-8, weight decay .01), global norm clip1
with nonfinite refusal, 20-update linear warmup then constant LR. Each arm and
backend gets exactly {3e-4,1e-3,3e-3} ×200 pilot updates, one common pilot stream,
step-0 and step-200 pilot-shard scoring on hops1–3. Choose lowest mean full-vocab
NLL; within .01 nats tie to lower LR. Exclude/record nonfinite candidates;
no finite candidate or all worse than step 0 means STOP. No weights survive
pilot. For Null, selection NLL is the **unforced raw answer distribution under
hard stopping at budget4**, saving soft/hard pilot diagnostics separately.

Dev at unselectable step0 and steps1800,3600,…,18000. Select lowest equal-weight
hop1–3 answer NLL (for Null the same raw hard-stop convention); ties within
1e-6 favor earlier step. No long-hop metric, mechanism metric, confirmation or
chosen sweep budget selects a checkpoint. No auxiliary-loss-weight, temperature,
threshold, initialization or architecture search. Every loss is averaged as
defined below before weighting; log its gradient scale, but a change requires
an amendment before main dev. Never silently shorten N to fit a job.

## 6. Role/transition losses and insufficient depth

`f_entity(x)=normalize(W_entity x+b_entity,eps=1e-6)` uses the same trainable
projection for head, tail and query-start frozen mention features. For targets
use the name's canonical text `" "+name`,
encoded by the frozen backbone at the same insertion layer, projected by the
current f_entity and **stop-gradient**. Canonical target features/codes are
training/metric temporaries, not a retained inference codebook or extra memory.
The target projection has no independently optimized copy or EMA. All rungs
have access to the same span/identity annotations for authorized losses and
audits; answer-only rungs do not receive bridge/role labels in forward.

Negatives for both losses are **all distinct other visible entity identities in
the logical 32-episode batch**, plus query-start identities if absent from
support. Deduplicate by identity; same-entity occurrences are positives, never
false negatives. Do not sample negatives from unseen/sealed pools or use gold
answer candidates at inference. If no distinct negative exists, flag invalid
batch (the registered graph normally contains 26 distinct names).

Let unit target t_e=sg(f_entity(canonical_text(e))). For occurrence code v with
identity e, `InfoNCE(v,e)=-log exp(normalize(v)·t_e/.1)/Σ_j exp(normalize(v)·t_j/.1)` over the
deduplicated batch identities. Role is its mean over head, tail and question
start occurrences, first mean within each role then mean over the three roles.
This aligns roles without weighting a repeated head as a new identity. Record
target variance, off-diagonal similarities and identity-decoding accuracy to
detect collapse; a positive-pair cosine term alone is insufficient.

For gold intent path e0→…→eh, **free rollout** computes actual p/m/a from its
own previous state/read. No teacher forcing, true key, bridge injection, answer
writeback or gold-hop update mask is allowed. For executed transition s+1≤h,
`Step_s = .5(1−cos(p_{s+1},t_{e_{s+1}})) + .5 InfoNCE(p_{s+1},e_{s+1})`.
Average over valid executed transitions within each query, then queries and
episodes. No extra "predict final answer again" step targets after h.
These are **path-intent** targets even if a rollout deviates; report them as
such, not as labels for the model's decoded current entity. Targets are detached;
gradients flow through predicted transitions and reads, not the gold branch.
Loss-only normalization uses ε=1e-6, including for soft-NULL mixed states; it
does not renormalize or otherwise alter the controller's stored state.
Role=.1, Step=1.0, NULL=.1, answer CE=1.0, temperatures=.1 are fixed.

All main training samples have h≤3 and fixed L=4, so there is sufficient depth.
For an explicitly diagnostic shortened rollout L<h, use only min(L,h) executed
bridge terms and **mask terminal answer CE/NLL from the training objective**;
do not require edge-by-edge semantics and a terminal prediction in too few
updates. No shortened rollout trains a main model. A short main unroll is an
execution failure, not an alternative schedule. Evaluation still reports final
answer EM/NLL at short budgets as descriptive limitations, with an insufficient
budget flag; never treat a bridge entity as the gold final answer. Main budget4
hops2–3 is fixed, not a best-depth selection. Tests of h4–8 use budgets1/2/4/8/9
on the same selected checkpoints, not retrained depth winners.

## 7. NULL, terminal state and read accounting

Only +Null adds `NullHead([p_s,m_s,c])`, a shared biased 352→1 linear,
weight0, bias=−log99, initial P(NULL)=.01. The Matrix head cannot inspect K/V,
max slot scores, raw support or a second backend. SoftSlot gets exactly the
same head/input contract; no attention-NULL slot is present.

**Labels:** derive `y_null(e,r)=1` iff the complete **visible graph** has no
valid `(e,r)` outgoing edge. Every training query supplies its named relation
only to the label builder. For each of its graph's 26 visible entities, form a
known anchor p_e=f_entity(canonical text), combine with that query's c, generate
the address/read through the ordinary backend, and score the head. This is a
separately labeled **anchor-state** auxiliary path, not a free-rollout event.
Detach p_e/m_e/c for this BCE so only NullHead learns from these graph labels;
free-rollout answer/step gradients still train the reader and soft gates.
Average BCE separately over NULL and non-NULL anchors, then average the two
classes and queries/episodes. Labels depend on actual `(e,r)` edges, not
`s<h`. There is **no pointwise NULL loss on ordinary rollout states**, even
when their intended gold position is known. The anchor-to-rollout transfer is
measured, never assumed. Account for these additional training reads and frozen
feature preparation; they are absent at inference.

Soft training executes all four iterations. With w0=1 and
t_s=1−P(NULL|p_s,m_s,c):
`p_{s+1}=p_s+w_s*t_s*(p_candidate−p_s)`,
`a_{s+1}=a_s+w_s*t_s*(a_candidate−a_s)`,
`w_{s+1}=w_s*t_s`.
Do not renormalize the mixed p afterward: that would change the plan's update.
Addressing still uses the fixed normalization in `query_code`. No gold
survival mask, hop-conditioned threshold or hard training stop.

Ordinary +Null inference is **hard threshold .5**, fixed without calibration:
query → read m → test NULL → if NULL hold p/a and become absorbing, otherwise
update. An absorbing row performs no further logical reads. If kernels still
process it in a batch, record that executed work. A length-h path needs **h+1
reads**, including the terminal NULL probe. Training's four reads cover h1–3;
long-hop deployment-style evaluation has maximum budget **9** for h8.

Reaching the read limit without a NULL decision sets `budget_exhausted=true`
and emits Unknown under the strict +Null task policy. Save the unforced raw
full-vocabulary distribution for diagnostics/NLL; never pick its favorable
answer to replace a strict Unknown. Main +Null EM is the emitted token:
full-vocabulary argmax when stopped, Unknown on exhaustion. Report raw argmax
EM and raw NLL separately, explicitly identifying this controller override;
do not describe the finite raw NLL as the likelihood of forced Unknown.
Non-Null arms have a fixed-budget answer and `termination_supported=false`;
they do not falsely claim a NULL certificate. The main Null comparison uses
budget4 on hops2–3, where both protocols have sufficient registered depth.

Hard-versus-soft fixed-budget inference at 4 and 9 is a mandatory paired
descriptive ablation of the **same** +Null weights, not a choice of primary
protocol. Save false/missed stop, actual reads, raw/strict EM and NLL for both.
All-empty memory still returns zero read; its head behavior is a preflight
boundary check. No overwritten/missing/evicted-memory success claim follows.

## 8. Mechanism metrics and causal intervention

Capture q_s,m_s,p_s,a_s,c, predicted NULL/w, active/absorbed/exhausted status,
owner and logical read index from the **actual final-position forward**. Old
`slot_read.final_query/final_read` recompute only read1 and cannot supply a
trajectory. Extend actual-forward S1B capture semantics, but its `path_indices`
hardcodes relation0; use the post-permutation question relation for S2 labels.
Capture must leave logits bitwise unchanged and reproduce the recurrence/read.
Base-old/Base-budget capture their actual z state; semantic p/a/bridge-decoding
and p-intervention metrics are N/A for those controls, not fabricated from an
arbitrary split of z. Task, read/address and resource metrics still apply.

Report per hop/step/seed/backend/rung:

- full-vocab answer EM and raw NLL; strict/raw +Null output distinction; UNKNOWN
  frequency; fixed/sweep budgets; paired per-relation task and relation-switch
  probe metrics;
- path next-address top-1 against the actual written DPFP keys, correct-slot
  weight, top-two gap and entropy for SoftSlot; Matrix uses the same offline
  keys only for diagnostics, never extra forward memory;
- path-target mean per-sample `||m−v_next||²/(||v_next||²+1e-6)`, cosine and norm
  ratio; labels are the next edge, N/A after gold completion. State-target read
  error is separate and only for a confidently decoded actual state;
- bridge top-1 = nearest cosine canonical entity code to p after each intended
  valid transition, among **all visible entities**, not just gold path members.
  Ties, nonfinite/zero states or undecodable states count wrong. Next-address
  accuracy means q_s top-1 on the gold edge at s<h. Completed/absent reads from
  a premature hard stop count wrong rather than disappear from denominators;
- value-to-next-query top-1 on the next gold edge after a transition (s+1<h),
  not a standalone trained value decoder; state/candidate and answer drift;
- false-stop/missed-stop conditional on actual decoded entity and requested
  relation, decoding coverage, anchor versus free-rollout head accuracy,
  end-to-end premature-stop/late-stop rates, reads and exhaustion. For confidence
  filtering require cosine≥.9 and top1−top2 cosine margin≥.1; report uncovered
  states separately, never assign their NULL labels by gold hop.

**Terminal instability D:** for each h2–3 query, compare raw argmax answers at
budgets h+1 and 9 using the same trained weights/graph and registered stop
policy. D is the fraction changed, averaged over queries/seeds. The h+1 fork is
an explicitly gold-budget diagnostic; true hop is not a normal forward input.
Also report probability/state drift, fixed-budget4→9 changes, answer correctness
and confidence coverage. Hard absorption guarantees state identity after a stop,
but useful terminal semantics additionally require correct stops/answers.

**Causal bridge test:** on every confirmation h2–3 question after its first
predicted transition, use a second question in the same fixed graph/relation
whose gold remaining length matches and terminal differs. Enumerate eligible
other-chain starts in sorted entity order and take the first; this is decided
from graph structure, not predictions. Obtain the donor bridge p by an ordinary
free rollout of that donor question after one read. Require original and donor
states to decode to their respective intended entities with the confidence
rule above; report the eligibility rate on the whole prespecified pool.
Replace **p only**, retain original a/c/read-only memory, then freely finish
under the original budget/stop policy. The predicted answer should change to
the donor branch's visible-graph terminal. This is a separately labeled oracle
mechanism intervention, not an ordinary confirmation prediction or training
sample. Gold labels select the analysis intervention only.

Two controls on the same sites: (1) apply the same donor-p substitution to a
separate, independently forked question's controller while leaving the target
question's p/a/c unchanged (unrelated-state perturbation); (2) replace the target
p by f_entity of a different textual occurrence of the **same original entity**,
keeping a/c, when that occurrence is available. The first control tests question
isolation; the second tests specificity to entity identity rather than its
particular contextual encoding. Record control coverage; no post-score donor search.
Report ordinary/intervened/donor-target correctness and answer changes for all
three operations, plus sham copy of original state as an identity fixture.

Mechanism acceptance on h2–3 requires ≥80% pooled path next-address and bridge
accuracy, ≥80% intervention eligibility, ≥80% donor-terminal correctness among
eligible swaps, and ≥90% original-answer preservation under **each** control
(each control's coverage ≥80%). Require ≥200 eligible sites in every seed;
otherwise report inadequate evidence, not success on a small selected subset.
The relation-switch probe must achieve ≥80% paired target correctness, meaning
both the original question and its changed-relation fork are answered correctly.
These are fixed descriptive acceptance screens, not extra significance claims. If
final EM improves without these checks, claim iterative computation/extra
supervision, not intervenable relation traversal. Report all h4–8 diagnostics;
they cannot rescue failed primary short-hop contrasts.
H2 acceptance uses the selected +Transition checkpoints; S3 additionally
requires these same mechanism screens on selected +Null checkpoints, so
successful stopping cannot silently replace the accepted traversal mechanism.

## 9. S2 namespace, manifests and sealed sets

Use `derive_seed(20261029,"slot-loop-v1",purpose,"S2")` for evaluation,
bootstrap and initialization; training uses
`derive_seed(20261029,"slot-loop-v1","train","S2",run)`.
Resolved integers below were computed with the source's stdlib blake2b formula;
compute preflight re-derives them. No graph or episode-ID list was generated
while drafting.

| purpose | split / indices | episodes | base seed |
| --- | --- | ---: | ---: |
| smoke / preflight / reload | train [0,64) | 64 | 5767346552835086547 |
| calibration, descriptive only | val [0,2048) | 2,048 | 4250237580548523729 |
| pilot selection | val [0,1000) | 1,000 | 1498679055494055091 |
| dev | val [0,1000) | 1,000 | 5034583424002061988 |
| confirmation, sealed | test [0,2000) | 2,000 | 2656685395072876251 |
| unseen names, sealed/descriptive | test_unseen [0,2000) | 2,000 | 6221145451142870176 |
| train seed17 | train [0,576000) | 576,000 | 8787082625600030233 |
| train seed29 | train [0,576000) | 576,000 | 5631957910583375155 |
| train seed43 | train [0,576000) | 576,000 | 8308041893740839886 |
| LR-pilot train | train [0,6400) | 6,400 | 5437067078186042258 |
| bootstrap, no episodes | numpy default_rng / 10,000 draws | — | 3685123442522494766 |

Unseen evaluation is included, descriptive, and opened with confirmation, never
substituted for it. Calibration/pilot/dev score hops1–3 only until the registered
post-selection dev diagnostic sweep. Confirmation/unseen have one query per
hop1–8 per graph. Mechanism probes/donor forks stay tied to their source episode
and never become independent bootstrap observations.

Extend the inherited stage/purpose registry and sealed-seed protections; leave
S0/S1A/S1B entries, their layouts/open tokens and old CONFIRMATION_WINDOWS
unchanged. S1B's original 6,000-step metadata is superseded by **A1's entire
576,000-episode prefix**, including when checking overlap. Check all distinct
S2 canonical and raw-ID sets/train prefixes pairwise and against the complete
registered S0/S1A/S1B and old-study streams/windows, using **each entry's own
layout**. Paired reuse across S2 arms is intentional; each unique stream is
registered once. The ID-only check may derive sealed IDs but never generate
sealed graphs/queries. Record both raw and canonical ID digests/counts and
actual disjointness results in the lock. Seed separation alone is not proof.

Registry and raw/stratified/wrapper generation all reject sealed entries
without the S2 token. Literal sealed seeds under src/scripts belong only to
the registry; public manifests omit sealed base/query seeds. An S1 open token
cannot open S2. Every science/analysis entry requires `--manifest`, checks
layout/wrapper/query protocol, IDs/order/hops/counts, full source/release/config
and checkpoint provenance, and refuses missing/extra/unstable/mismatched data.
Do not flatten questions or pair only on a filename.

Before opening: finish all triggered training waves, select every mandatory
arm/seed from dev, freeze the full analysis including gates/probes and obtain
an independent rerun of dev selection. STOP for the orchestrator's
`reports/slot-loop-v1/s2/open.json`. It binds parent/S1B/S2 prereg digests,
S1B verdict and mechanical branch, lock/inventory, every approved release file
digest and config map, selected arm/backend/seed→checkpoint digests, analysis
source/file digests, active/gated-out rungs and the **unchanged** contrast family.
If all six rungs run, the map has 18×K selected exports. References and smoke
checkpoints cannot occupy main labels. All run records carry their exact
approved release and common root lineage. Selected sweep/probe entries use the
same selected weights. Evaluator/analysis stdout prints counts/status/digests;
metric payloads are opened only at the frozen verdict stage. No confirmation
reuse for a changed S2 design: amendment plus fresh namespace/set required.

Persistent outputs: absolute paths below
`/work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s2/`; committed compact metadata
under `reports/slot-loop-v1/s2/`. Model exports contain complete preprocessing,
parameter/init, position, state, backend, task/wrapper, depth/stop and source/
parent/release metadata. Episode memory and training-only/metric codebooks/heads are
separate; no support KV/raw text/hidden history/alternate memory in prediction
exports. Preserve selected and checkpoint-selection artifacts until audited.
Caches go to job-local scratch; this task authorizes no deletion.

## 10. Confirmatory family, thresholds and S3 decision

For arm A, seed s, episode i, hop h, save correctness c[A,s,i,h] and raw NLL.
For an answer contrast A−B define
`d_i=mean_(s=17,29,43; h=2,3)(c[A,s,i,h]−c[B,s,i,h])` and Δ=mean_i d_i.
All comparisons are within one backend on the identical 2,000 episodes.
For instability use the analogous per-episode mean over seeds/h2–3.

There are **six registered contrasts per active backend**, one study-wide
family **m=6K**. Gates that leave a contrast unrun do not redistribute α.

| contrast | confirmatory interpretation and pass rule |
| --- | --- |
| C_A0 = Arch−Base-old, answer EM | ≥3 pp, adjusted lower >0, every seed positive, hop1 point loss ≤2 pp |
| C_AB = Arch−Base-budget, answer EM | same; both C_A0/C_AB required for architecture attribution |
| C_R = Role−Arch, answer EM | same; extra structural supervision claim only |
| C_T = Transition−Role, answer EM | same; H2 task benefit additionally needs §8 mechanism acceptance |
| C_Ntask = Null−Transition, answer EM at budget4 | noninferiority: adjusted lower >−2 pp and every seed point ≥−2 pp; not an EM-superiority claim |
| C_Nstable = D(Transition)−D(Null) | ≥3 pp, adjusted lower >0, every seed positive; H3 requires C_Ntask too |

Use 10,000 paired **whole-episode** resamples with numpy
`default_rng(3685123442522494766)`, one shared draw per replicate across
backends, contrasts, seeds, hops and interventions. Average the three seeds
before resampling; keep all questions/forks of an episode together. Percentile
intervals use linear interpolation at **α/(2m) and 1−α/(2m), α=.05**:
K1 m6 gives quantiles 1/240,239/240 (99⅙%); K2 m12 gives
1/480,479/480 (99 7/12%). Exact integer/rational count comparisons implement
3 pp/2 pp thresholds; never round into a pass. This is conditional on one
S-RE writer, these data-order seeds and this layout, not three writer retrains.

Report all contrasts, intervals, counts, per-seed/per-hop NLL/EM and failures.
Long hops, depth-sweep gains, raw versus strict Null answers, soft/hard
differences, backend differences, role/address/read metrics and causal screens
are preregistered descriptive outputs; no new significant subgroup replaces a
failed primary test. Mechanism thresholds do not assert additional controlled
hypothesis tests. Missing/invalid mandatory inputs stop analysis, never become
zero scores or discarded seeds. An untriggered Null rung is explicitly
"not run: Transition dev gate failed", never a confirmed Null null.

| frozen S2 pattern | conclusion / continuation |
| --- | --- |
| integrity, freeze, data or source failure | invalid comparison; STOP, resolve before opening/continuing |
| C_A0 positive but C_AB fails | composite architecture/input/budget gain only; no isolated architecture attribution |
| C_R alone passes, or C_T passes without mechanism screens | supervision/iterative-computation benefit; no semantic traversal or S3 mechanism entry |
| C_T fails, or transition/address/intervention/relation screens fail | H2 not accepted; report negative, no S3 |
| H2 passes; Null unrun or H3 tests/screens fail | transition evidence only, terminal-state gate unmet; no S3 |
| H2 passes, C_Ntask and C_Nstable pass, terminal screens below pass | eligible to draft/register S3 for this backend and verified hop scope |

Terminal screens: +Null D≤5% on its defined h2–3 population; on h1–3,
end-to-end premature-stop and missed-terminal-stop rates each ≤5%, and
exhaustion at budget4 ≤5%; actual-state
NULL label coverage ≥80%. At h4–8 report the same quantities at budget9 and
all transition/causal metrics; extrapolation is descriptive and cannot rescue
the short-hop gate. An S3 study may claim a scope only where its mechanism
screens pass. S3 may include every backend that passes; its lead remains the
S1B-selected primary if qualified here, otherwise the sole passing comparator,
by this fixed rule. No retrospective S2 storage-superiority claim follows.

For these end-to-end flags, a premature stop is a first hard stop at read index
≤h (reads indexed from 1). A missed terminal stop means no hard NULL certificate
at a state confidently decoded to the correct visible-graph terminal by read
h+1, including wrong-branch stops,
late stops and exhaustion. Denominators include every query in the stated hop
scope. These path-intent outcome flags are distinct from actual-state head
false/missed-NULL rates; they never supply labels to training or inference.

S3 still requires its own preregistration, resource accounting and release.
These S2 gains involve extra labels; no unsupervised-reasoning claim. The plan's
real-task extension additionally requires same-content/information/supervision
matching, an **independently trained L1 control on that task**, and reliable
costs. An S2 same-checkpoint @1 sweep is not that control. No real-task or
deployment claim is authorized by this addendum alone.

## 11. Preflight, source lock, cost and stop points

After registration and S1B entry only: implement/test on compute, then the
task's one core-code cross-review (rv-code plus Opus, arranged by orchestrator),
fix validation, real-parent preflight, committed S2 lock. No new review rounds
for documentation/runbooks/fixes. All compute through `scripts/submit.py
--commit <full 40-hex hash>`; no HEAD literal, extra interpreter in payload,
relative persistent output or execution on login. Keep ≤6 wk-slot jobs queued/
running; no sleep/poll loop, use WAIT for the orchestrator watcher.

Mandatory engineering evidence before calibration/pilot/dev:

| gate | required evidence |
| --- | --- |
| old compatibility / scope | inherited T1/all-token goldens untouched; Matrix/native identity on old protocol; all S2 B0 logits equal frozen base; earlier-token branch exactly zero |
| task / span | real TaskSpec and composite digests; 24 writes, key/value shapes [B,24,768]/[B,24,128]; query relation XOR and stratify-before-swap; correct named-relation path, hop/answer text and span alignment |
| state / owners | sentinel graphs, padding/reorder, multiquestion forks and depth1/2/4/8/9 use original episode owner; memory bytes unchanged; label perturbations cannot affect ordinary forward |
| frozen / parameters | exact counts/optimizer sets, delayed B0 gradient reachability, no writer/buffer/backbone change, fixed-support keys/values/Matrix unchanged, no restandardization |
| losses / NULL | role negatives exclude same entity; target stop-gradient; free rollout vs gold intent separation; short-budget CE masking; visible-graph anchors, no gold-hop NULL loss; exact soft recurrence and .5 hard action order |
| stopping | hand-built h0/h1/h8 graph cases need 1/2/9 reads; no absorption read; limit without NULL emits Unknown/sets exhaustion; kernel rows/calls recorded honestly |
| backend / interfaces | actual DPFP keys and native ridge; empty/padded read zero, E=S24 succeeds/E>S raises, storage bytes include masks/counts; no hidden second memory |
| analysis / seals | full prefixes including S1B A1, both S2 IDs and old layouts checked; unopened generators refuse; whole-episode pairing/m6K quantiles, exact thresholds, gate and missing-input handling |
| trajectory / intervention | actual q/m/p/a trace equals recurrence; capture off/on identity; fixed donor/control protocol and coverage; no label/key/codebook leak into normal forward |
| reload / release | real smoke→JSON candidate release→CLI reload reproduces IDs/logits/predictions/NLL and complete metadata; changed weights/source/analysis/release or swapped labels are refused |

The lock binds sorted src paths/file bytes via blake2b-128 (excluding bytecode),
all science scripts including analysis/data preparation, S2 config/manifest,
registered documents, parent inventory, S1B branch evidence and complete passed
preflight. **Any src or science-script change invalidates the lock** and needs
new preflight; documentation-only commits do not change source bytes. Preserve
S0/S1A/S1B locks and immutable earlier-wave releases. Score-changing fixes after
opening require a new science version and fresh confirmation, not mixed records.

Sequence per eligible wave: lock → descriptive calibration audit once per
backend (no grid) → 20-update real smoke/freeze/export → equal LR/timing pilot
→ exact candidate/CLI reload → committed ready release → **STOP for the exact
release's approval.json** → common-start main runs → frozen dev selection and
wave gate. After all triggered waves, independent dev-analysis rerun → **STOP
for open.json** → paired confirmation/unseen/sweeps/probes → frozen analysis,
artifact verification and report → **STOP for S3 decision**. This draft's STOP
precedes all these steps.

**Measured anchors, not S2 predictions.** S1A's archived
`reports/slot-loop-v1/s1a-preparation/pilot.json` reports maximum steady step
0.1589559437 s, 1,000-episode T1 evaluation 8.320275826 s and load 1.368659242 s
(job3728825). S1B A1's upcoming T2/L4 pilot is the next comparable anchor;
its outcomes were not opened for this draft. At release, obtain a separate
timing-only record with source/task/depth/backend, warmup/sample count,
steady mean/P90, load, preparation, evaluation and export overhead. No S1B
pilot-score reading is needed to import measured times.

For orientation only, 18,000×0.1589559437 = 2,861.21 s = .795 node·h of
training before evaluation/startup. A **provisional** three-query scaling of
S1A evaluation is 24.96 s/pass; 11 passes plus load yields ~3,137 s (.871 node·h),
×1.5 rounded to 10 min gives 80 min. This is **not an executable S2 request**:
E24, depth4, new losses/mention features and especially NULL's 26 anchor reads
per question change cost. Do not multiply whole step time by depth and call it
measured. With all six arms, 18×K main runs give an anchor-only ~15.7×K actual
node·h, not a cap or asserted S2 forecast.

| future job class | count if all gates pass | expected seconds before ×1.5 |
| --- | ---: | --- |
| focused tests / real-parent preflight | bounded shards | nearest measured compatible tests (S1A 37/170 s shards, old preflight53 s), then own measurement; shard before cap |
| calibration audit | K | measured load + 2048-episode hop1–3 diagnostic time, no candidate multiplier |
| smoke / reload | 6×K arms, shards | load + 20×own step time + measured smoke/trace/export/reload cost |
| LR pilot | 18×K candidates | per candidate load + 200×own step time + two 1000-episode hop1–3 evaluations + measured I/O/anchor overhead |
| main arm/seed | 18×K runs | load + 18000×own steady step + 11×1000-episode hop1–3 dev time + all checkpoint/freeze/export overhead |
| selected dev/confirmation/unseen sweeps | by arm/seed/budget | load + measured per-graph preparation + full hop/query/read counts + export; share preparation only when measured |
| causal / relation / soft-hard diagnostics | selected semantic arms/seeds | measured eligible/all-site donor and two-control forwards, graph/feature setup and serialization; time them on smoke before request |
| frozen selection/statistics | per gate / final | measured paired-array/source-audit/bootstrap time, not an unpriced login task |

For each class use its longest comparable measured finite pilot candidate and
largest observed setup/evaluation/I/O overhead, recording sampled steps11–200
and synchronization. Measure NULL anchor reads, canonical feature construction,
support preparation, question backbone and controller separately; any caching
must preserve the same frozen features and appear in bytes/time accounting.
The first S2 smoke request uses measured S1B timing plus bounded per-module
preflight measurements, not an arbitrary full 16-hour allowance.
Request `ceil_to_10min(1.5×expected_seconds)` (short checks may round to minutes),
one node, debug-g only if ≤30 min, otherwise regular-g; **≤16 h** in every case.
An over-cap indivisible main run causes STOP; shard independent candidates/
episodes/analysis instead of clamping a false estimate or shrinking science.

Log requested/actual node·h for every attempt, including failures/retries,
cumulative spend and the next wave's projected total before approval. Log
logical question reads, anchor-training reads, backend calls/row counts,
recurrent updates, GPU kernel time and complete end-to-end cost separately.
P=1 for the S2 question controller; fixed arms have L logical reads per
question, Null hard inference h+1 only when it actually stops correctly.
An apparent saving from batched calls/absorption is not a deployment latency
claim. Retain selected model exports, all selection evidence, per-step losses/
norms, paired predictions/NLL, compact traces/interventions, source inventories
and PBS ledger; no unregistered raw-support/dual-state retention for inference.

## 12. Explicit departures and choices requiring registration

1. **Typed-task completion:** native t2_typed adds relation1 distractors but only
   queries relation0. Use E24/(8,4)/12 distractors and the specified S2-only
   relation permutation, stratifying first. The composite/two-ID convention is
   a deliberate extension of the native layout identity; old generation stays
   unchanged. The same-graph relation-switch probe bounds the remaining
   long-chain-relation shortcut. No interwoven-relation/sequence claim.
2. **Parameter matching:** the plan's mention encoder alone has 196,704
   parameters; semantic trainables total 399,712 versus 180,224. Add the mandatory
   399,752-parameter mention-matched Base-budget arm and both architecture
   contrasts. Do not claim exact equality or equal FLOPs.
3. **Position/query transfer:** use last_valid_question from training, zero B
   in every arm, explicit span/feature checks, and no untrained EM threshold.
   Mention initialization is a hypothesis, not a proven repair for the S1B
   diagnosis. Native renderer spans and relation0-only path helpers need S2
   handling, not silent reuse.
4. **Read settings/schedule:** inherit S1A g/τ as S1B A1 does, rather than an
   uninformative B0 calibration search. Fix 18,000 updates, ten dev checkpoints
   and a 200-step three-LR pilot per arm/backend. The 20-step correctness smoke
   follows S1A/S1B rather than a 10%-of-main smoke; the target-path 200-step pilot
   supplies cost before release. No score-dependent N or loss-weight pilot.
5. **Loss/NULL completion:** lock the .1/1/.1 weights, symmetric role grouping,
   detached canonical targets, .1-temperature deduplicated batch negatives,
   mixed cosine/contrastive Step, short-budget CE masking, detached visible-graph
   anchor BCE, exact soft survival update and hard 0.5 policy. No rollout NULL
   label inferred from intended hop. Strict exhaustion EM and raw NLL differ
   explicitly. No new missing-fact task is smuggled into chain-end.
6. **Gates/statistics:** execute waves from identical initialization, gate NULL
   on registered dev transition checks, and reserve one six-per-backend
   Bonferroni family including Null noninferiority and stability. Plan §9 leaves
   these thresholds/family unspecified; they are new registered choices, not
   inherited S1B significant results. Unrun contrasts keep their α allocation.
7. **Mechanism/S3:** fix causal donor selection, two controls, coverage and
   numerical acceptance/terminal screens in advance. S3 entry requires both H2
   and H3 on a stated scope, not only final EM. Real-task independent L1 and
   information/supervision matching remain additional S3 obligations.
8. **Cost/process:** S1A timing is an anchor only; blind S1B timings and measured
   S2 per-arm costs must replace it before any ready release. No made-up measured
   S2 time or aggregate cap. Registration, core review, lock, release approval,
   independent dev rerun and sealed opening remain separate gates.

The orchestrator must accept or amend these items when registering. This draft
does not register itself, authorize code/jobs or resolve S1B's pending outcome.

## Decisions at registration (orchestrator, 2026-10-30 01:55 JST)

1. **Departures 1–8 of §12 are accepted**, as changed by decisions 3–6.
2. **Seeds and layout checked.** The orchestrator re-derived with stdlib blake2b:
   - the native TaskSpec digest `2b15300ed77bd6df1751798ed2ff46ac`, whose 15
     fields match `data/relations.py`;
   - the composite layout digest `1014521816624879d8bc3c2716fce58b`;
   - the initialization base seed and every base seed of §9.

   The LR-pilot training stream is
   `derive_seed(20261029,"slot-loop-v1","train","S2","pilot")`. All values match.
   The compute preflight still re-derives them with the real TaskSpec and wrapper.
3. **Two waves, not four (§5).**
   - **Wave 1:** Base-old, Base-budget, Arch, Arch+Role and +Transition launch
     together under one release, after its approval. No launch among these
     rungs depends on a score; §5 already requires no gain between them.
   - **Checks on wave 1:** the §5 stability rule and the Role target-space audit
     (finite normalized targets, no exact collision of distinct names within a
     graph) apply to every wave-1 arm's selected dev checkpoint. They run before
     the NULL gate and before opening. An audit failure stops for diagnosis.
   - **Wave 2:** +Null, only after the NULL entry gate on the selected
     +Transition dev checkpoints, with its own smoke, pilot, release and approval.
   - **Unchanged:** the ladder, common initialization, contrasts and the family
     m = 6K.

   Reason: sequential waves without score gates add release and approval
   cycles and no information.
4. **Intervention controls (§8).**
   - **Control (1)** becomes a donor **a-only swap**. At the same sites, replace
     the target's a with the donor's a after one read. Keep the target's p, c and
     memory, and finish under the original budget and stop policy. This shows
     whether the bridge identity is carried by p or by the whole controller
     state. It is reported descriptively, without a threshold.
   - **Question isolation**, the draft's control (1), is an engineering gate:
     §11 state/owners adds that perturbing one question's controller state
     leaves every other question's logits bitwise unchanged.
   - **The mechanism acceptance screen** therefore requires:
     - ≥80% pooled path next-address and bridge accuracy;
     - ≥80% intervention eligibility;
     - ≥80% donor-terminal correctness among eligible swaps;
     - ≥90% original-answer preservation under the same-entity re-encoding
       control, with coverage ≥80%;
     - ≥200 eligible sites per seed;
     - ≥80% paired correctness on the relation-switch probe.
5. **S3 entry (§10).**
   - **H2 passes, but +Null is unrun or H3 or a terminal screen fails:** the
     backend may enter an S3 registration at **fixed read budgets only**. That
     S3 covers capacity, state bytes and timing at fixed depth. It makes no
     termination, dynamic-stop, read-saving or terminal-stability claim.
   - **H2 and H3 both pass:** the full S3 scope.
   - **H2 fails:** no S3, as drafted.

   Reason: plan §10 asks whether the advantage survives resource limits, and a
   fixed-depth model can answer that.
6. **What may start before S1B's verdict.**
   - **Now:**
     - implementation of both backends' S2 paths and the analysis;
     - compute tests on synthetic fixtures and the S2 smoke namespace
       (train [0,64));
     - the one core-code cross-review;
     - fixes, checked by tests.

     To reuse the S1B loop code, wk-slot merges wp/loop at the A1 lock commit
     `258f5f5`. At that commit wp/loop holds only the S1B preflight, lock,
     stopped calibration and diagnosis. It holds no pilot, dev, confirmation or
     training outcome.
   - **After S1B's frozen verdict and the orchestrator's branch record:** the
     real-parent preflight and lock, which bind the S1B branch evidence. Then
     the calibration audit, smoke, pilot and release, and STOP for
     `approval.json`.
   - **Blindness is unchanged:** no reading of S1B pilot, dev, confirmation,
     unseen or training outputs.
   - **If neither backend qualifies:** the S2 code stays unmerged, and no S2
     job beyond tests runs.
7. **Process and resources.**
   - **Review:** one core-code cross-review (rv-code and an Opus subagent);
     fixes are checked by tests only (user, 2026-10-29: "不要搞这么多次review,核心代码没问题就行").
   - **Jobs:** each job ≤ 16 node·h, with walltime = measured time × 1.5
     (user, 2026-10-29). §11's cap of 6 wk-slot jobs is the default; the
     orchestrator may raise it in the task file once S1B has no jobs left.
