# Bibliography check: plan Appendix D (R1-R18) and section 2.2

Checked 2026-10-26 (JST) against `plan/recurrent_lora_research_plan_2026-10-26.md` (v1.1; section 2.2 and Appendix D are unchanged from v1.0, only line numbers shifted).
Method: arXiv abs pages and the pinned HTML versions fetched with curl and parsed (title, authors, submission history, abstract); PMLR, ACL Anthology, GitHub, Hugging Face and the ICLR PDF fetched directly; full text of R5 and R9 read via pdftotext; v1/v2/v3 HTML bodies diffed where the version differs. WebFetch and WebSearch were used for OpenReview and the new-work search. "unverified" means the content could not be read here.

## Problems found

1. **R1 cannot be verified from this environment.** OpenReview forum redirects to a Cloudflare "Verifying your browser" challenge, `/pdf` returns HTTP 403, the OpenReview API returns `ChallengeRequiredError`, and WebFetch is blocked as well. Two web searches list the forum page with the matching title; the snippets give authors Nanxi Yu, Kang Li, Ye Du, Guangqian Yang, Shujun Wang and the label "ICLR 2026 Conference Withdrawn Submission" (19 Sep 2025, modified 25 Sep 2025). That label is snippet-level, not opened. The plan records the uploaded draft as "ICLR 2026 under review" and does not mention withdrawal. The uploaded PDF is not in the repo (nothing matching under `/work/xg24i002/x10041`, depth <= 3), so the page-level claims (pp. 4-6, 9) are unverified by any agent.
2. **Version drift in 3 of the 7 pinned preprints (R10, R13, R14).** Details in the table; none changes the cited content (R10 license only, R13 citation fix only, R14 restructured but the equations are unchanged).
3. **R17/R18 are stale against plan section E.2.** E.2 (which "takes priority") switches to `Qwen3-0.6B-Base` @ `da87bfb...` and lists `Qwen3-1.7B-Base` or `Qwen3-4B-Base` for the confirmation scale, while Appendix D still links the post-trained `Qwen3-0.6B` and `Qwen3-4B`. All of these pages exist (HF API, see table).
4. No broken links other than R1's access block; no title/author mismatches.

## Table

| Ref | Link and identity | Version: plan -> latest (date) | Does the abstract/body support the plan's description? |
| --- | --- | --- | --- |
| R1 RecLoRA | OpenReview forum: HTTP 200 only after redirect to bot challenge; pdf: 403. Title matches the search-index entry. See Problems 1. | n/a (OpenReview) | Search-snippet abstract: "learnable router ... shared LoRA block for iterative refinement" plus a sequential knowledge-transfer scheme, consistent with "shared recursive LoRA, router, domain-incremental". Step embeddings, STE gate and all page-level claims: **unverified**. |
| R2 LoRA | arXiv 2106.09685 resolves; title and authors (Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen) match. | none pinned; latest v2 (16 Oct 2021) | Yes: "freezes the pre-trained model weights and injects trainable rank decomposition matrices"; "no additional inference latency". |
| R3 Linear attention | PMLR v119 katharopoulos20a resolves; ICML 2020; Katharopoulos, Vyas, Pappas, Fleuret. | n/a | Yes: self-attention as "a linear dot-product of kernel feature maps"; "iterative implementation ... relationship to recurrent neural networks". |
| R4 FWP / delta | PMLR v139 schlag21a resolves; ICML 2021; Schlag, Irie, Schmidhuber. | n/a | Yes: "additive outer products", "memory capacity limitation", "delta rule-like programming instruction". |
| R5 Fast Weights 2016 | arXiv 1610.06258 and the toronto.edu PDF both resolve; Ba, Hinton, Mnih, Leibo, Ionescu. | none pinned; latest v3 (5 Dec 2016) | Abstract is generic ("fast weights ... store temporary memories of the recent past"). Inner loop is in the body: section 3, Eq. 2 "inner loop ... runs for S steps" with sustained boundary conditions and fixed A(t); section 3.1 layer norm on each inner iteration. Supports the plan; pointer correct. |
| R6 Generative Adapter | arXiv 2411.05877v1 HTML resolves; title and authors (Chen, Fang, Xia, Liu, Van Durme, Zettlemoyer, Gao, Cheng) match. | v1 -> v1 (8 Nov 2024), **same** | Yes: "directly maps new contexts to low-rank LM adapters". Section 2.2 contains "a bi-linear function" and "Dynamic Streaming Update" (S_t <- S_{t-1} + A2 H_t^T H_t B1). |
| R7 delta-mem | arXiv 2605.12357v2 HTML resolves; Lei et al. (10 authors) match. | v2 -> v2 (v1 12 May 2026; v2 28 Sep 2026), **same** | Yes: "augments a frozen full-attention backbone with a compact online state of associative memory ... delta-rule learning ... low-rank corrections to the backbone's attention computation". Sections 3.1-3.5 headings match the plan's pointer. One read per token with a hidden-state query (Eq. 6). Write error uses the undecayed state (Eq. 10), consistent with the plan's section 4.5 caution. |
| R8 DEQ | arXiv 1909.01377 resolves; Bai, Kolter, Koltun. | none pinned; latest v2 (28 Oct 2019) | Yes: "directly finds these equilibrium points via root-finding"; "infinite depth (weight-tied) feedforward network". |
| R9 ELM | ICLR 2026 PDF resolves (HTTP 200, 1.26 MB); "Equilibrium Language Models", Jiang, Wang, Ding, Zhang, Wu, Zhao, Lui. | n/a (proceedings) | Partly in the abstract, fully in the body. Abstract: "compression framework that replaces groups of Transformer layers with a lightweight fixed-point network" (no LoRA). Body: section 3.1 and Fig. 3 caption "Each fixed-point layer is trained with LoRA"; section 3.4 "adaptive termination criterion" on the residual norm. Pointers correct. |
| R10 LT2 | arXiv 2605.20670v1 HTML resolves; Deng et al. match. | v1 -> **v2** (v1 20 May 2026; v2 22 May 2026) | Yes: "looping uniquely synergizes with these variants: it enables iterative memory refinement in linear attention". v1 vs v2 body diff = license line (CC BY-NC-ND -> CC BY) and version stamp only. Section 2.2 is the loop x linear-attention analysis; section 6 is a short Conclusion whose limitations say "we do not design explicit cross-loop recurrent state carry mechanisms". |
| R11 ELR | arXiv 2610.12582v1 HTML resolves; Ladhak et al. match. | v1 -> v1 (8 Oct 2026), **same** | Yes: "a gated state update with a learned step size, loop counts mixed within each optimizer update, and conditioning each pass on its loop index and budget". Section 3.2 "One-Loop Rank Overrides": rank = data-parallel rank (plan's note is correct); section 3.3 has the low-rank correction A(g(c_i) ⊙ B x̃) with loop-dependent gate. |
| R12 OperLoop | arXiv 2609.37379v1 HTML resolves; Huang et al. match. | v1 -> v1 (29 Sep 2026), **same** | Yes: "derives loop transitions in closed form from a projection, a local objective and an optimizer update rule"; "explicit weight decay, adaptive step size and a delta objective". Sections 3-4 match. Also: "We view the looped hidden state as a fast weight" (not mentioned in plan section 2.2). |
| R13 LLM Sleep | arXiv 2605.26099v2 HTML resolves; Lee, McLeish, Goldstein, Fanti match. | v2 -> **v3** (v1 25 May; v2 27 May; v3 5 Jun 2026) | Yes: "converts recent context into persistent fast weights"; "N offline recurrent passes over the accumulated context and updates the fast weights ... through a learned local rule". v2 vs v3 diff = citation edits only (Allen-Zhu and Li -> Allen-Zhu, reference format). Section 5 and Algorithm 1 exist as cited. |
| R14 Gated DeltaNet | arXiv 2412.06464v1 HTML resolves; Yang, Kautz, Hatamizadeh match. | v1 -> **v3** (v1 9 Dec 2024; v2 5 Mar 2025; v3 6 Mar 2025, "ICLR 2025 camera ready") | Yes: "gated delta rule and develop a parallel training algorithm optimized for modern hardware". v3 is restructured (new section 3.2 S-NIAH case study; chunkwise algorithm moved from section 3.2 to 3.3) but the gated delta rule S_t = S_{t-1}(alpha_t(I - beta_t k_t k_t^T)) + beta_t v_t k_t^T is identical in v1 and v3. Equivalent to decay-first delta (error on alpha S_{t-1}), matching plan section 4.5. Plan gives no section numbers, so nothing breaks. |
| R15 MuSiQue | ACL Anthology 2022.tacl-1.31 and GitHub StonyBrookNLP/musique resolve; Trivedi, Balasubramanian, Khot, Sabharwal; TACL 10, 2022. | n/a | Yes for multi-hop: "selects composable pairs of single-hop questions"; "25K 2-4 hop questions". "Real corpus" is not stated in the abstract (unverified from the page). |
| R16 HotpotQA | ACL Anthology D18-1259 and hotpotqa.github.io resolve; Yang et al.; EMNLP 2018. | n/a | Yes: "113k Wikipedia-based question-answer pairs"; "reasoning over multiple supporting documents"; "sentence-level supporting facts". |
| R17 Qwen3-0.6B | HF page resolves; Apache-2.0; HEAD `c1899de` (2025-07-26); base_model Qwen3-0.6B-Base. | n/a | Exists. Superseded by E.2: `Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` exists and equals current HEAD. |
| R18 Qwen3-4B | HF page resolves; Apache-2.0; HEAD `1cfa9a7` (2025-07-26); base_model Qwen3-4B-Base. | n/a | Exists. E.2 candidates also exist: `Qwen3-1.7B-Base` HEAD `ea980cb`, `Qwen3-4B-Base` HEAD `906bfd4`. |

## Notes bearing on plan claims

- R5's inner loop has the same read structure as the plan's V2: fixed A(t) during the loop, evolving h_s used as the query, input re-injected each step, one write after the loop. An evolving query over fixed fast weights is therefore not new; the plan already limits its claim to PEFT design, budget curves and composition evidence. R5's appendix (hyper-parameters of its fast-weights model) also says: "We experimented with more iterations for the 'inner loop' and the performance are similar", i.e. no benefit from more inner steps was reported in that setting, which is relevant to Gate 1.
- R7 reads once per token with a fixed query, so the plan's "fixed writer/state, evolving query" distinction from R7 holds as stated.
- R10 (section 2.2: each loop contributes a fresh DPLR update factor) and the MixerLoop paper below apply the full delta-rule mixer, writes included, on every loop; the plan's single-write / multi-read split is a real difference from both.

## New-work search (2025-2026)

Nothing found that combines (i) a frozen backbone with a low-rank adapter state iterated per token and (ii) a delta-rule/fast-weight memory read several times with an evolving query. Eleven searches (ten extended, one standard), then the abstracts (and, for items 1, 2 and 4, the paper text) read from arXiv. Nearest by half, closest first (links are arXiv abs pages I read):

1. [MemReasoner, 2503.07903](https://arxiv.org/abs/2503.07903) (Mar 2025). Closest on (ii): section 3.3 "Iterative Read And Query Update", z_q <- z_q + alpha*z_r, repeated until the readout converges; but the memory is an episode-level Larimar/Kanerva-style matrix written by pseudo-inverse, the model is trained end-to-end (fine-tuned from a BERT-large/GPT-2-large Larimar checkpoint), so no frozen backbone and no per-token write.
2. [Retrofitting Recurrent Depth into a Pretrained LM, 2608.11233](https://arxiv.org/abs/2608.11233) (31 Jul 2026). Closest on (i): base weights untouched, rank-16 LoRA weight-tied across loop iterations (6.01M forward-active), depth extrapolation to about 1.5x trained depth; iterated state is the full hidden state through base layers, no memory (no "memory"/"fast weight" text in the paper).
3. [Ouroboros, 2604.02051](https://arxiv.org/abs/2604.02051) (2 Apr 2026). Frozen SVD-initialised LoRA bases with a hypernetwork's per-step diagonal modulation inside a recursive block; no memory; its own abstract says held-out gains are not shown.
4. [MixerLoop, "Allocating Recurrent Compute in Looped LMs", 2608.18230](https://arxiv.org/abs/2608.18230) (18 Aug 2026). Loops only the Gated DeltaNet mixer (FFN once); each loop "recomputes its input-dependent keys, values, decay gates, and delta-rule updates" (new writes, erasures and reads per loop); trained from scratch at 15M/110M, no frozen backbone.
5. [MesaNet, 2506.05233](https://arxiv.org/abs/2506.05233) (ICLR 2026). Abstract: in-context loss "minimized to optimality at every time point using a fast conjugate gradient solver", at the cost of "additional flops spent during inference time"; a fixed CG solver rather than a learned query update; trained from scratch, no adapter.

Screened, not close (single read or no adapter): Q-Delta 2606.08804, CCQ 2606.01294, In-Place TTT 2604.06169, Trained Persistent Memory for Frozen Decoder-Only LLMs 2603.22329, ARMT 2607.11614, The Recurrent Transformer 2604.21215, RecurTrace 2609.03379. Pre-2025 context, abstract only: End-to-End Memory Networks [1503.08895](https://arxiv.org/abs/1503.08895) ("multiple computational steps (hops) are performed per output symbol"); its query-update rule was not checked (unverified).
The search index is US-only and incomplete; absence of results is not proof of absence.

## WP-T6: latent recurrence, verbalizable directions, and stability

Checked **2026-10-27 JST**, against the project at `14af341`. Method: opened primary
metadata/version histories and the pinned full-text method/analysis sections below
through web tools; distinguished author descriptions from our mathematical inferences.
Author lists are abbreviated here but their identities were checked in the linked records.
Version histories were checked, not body-diffed. No external performance comparison was
reproduced. “Full body” means the body was accessible and relevant sections read, not that
all code, image-only algorithms, or experimental artifacts were audited. The earlier
search remains a dated, incomplete search, not a novelty proof.

| Work; verified record and version | Access and inspected method | Overlap and implication for LoopLora |
| --- | --- | --- |
| Gurnee et al., **Verbalizable Representations Form a Global Workspace in Language Models**; [arXiv 2607.15495](https://arxiv.org/abs/2607.15495), v1 **2026-07-16**. [Official paper](https://transformer-circuits.pub/2026/workspace/index.html) and [Anthropic summary](https://www.anthropic.com/research/global-workspace) date publication **2026-07-06**, before arXiv submission; both dates are retained. | Full official body, Methods/J-Space/limitations opened. J-lens averages Jacobians from an earlier residual stream to later final-layer residuals and combines that transport with unembedding. J-space uses sparse nonnegative combinations of an overcomplete frame, generally a union of cones. | Prior work on verbalizable directions and intermediate representations; motivates stepwise probes and interventions. It is not the reader's transition Jacobian or a fixed rank-r linear subspace. Its feedforward-broadcast evidence is not a recurrence/stability theorem for our adapter. |
| Yang et al., **Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models (STARS)**; [2605.26733](https://arxiv.org/abs/2605.26733), v1 **2026-05-26**. | [Full HTML v1](https://arxiv.org/html/2605.26733v1), §§3–5 and Appendices A–B opened. Combines random depth with a sampled-state Jacobian-vector penalty. §5 sets practical K=1 in the K-step Algorithm 1; its squared JVP is a proxy, not a certified spectral radius. ICML metadata was not independently checked against proceedings. | Direct precedent for depth sampling plus dynamical regularization, motivating JS rather than establishing a new combination. The fixed-point motivation and the finite-state proxy must be separated. Neither reported scaling behavior nor the proxy certifies our reader, terminal correctness, or halting policy. |
| Panwar, Singh, and Bansal, **Think Deep, Speak Once: Relit, A Recursive Latent Implicit Transformer Framework**; [2608.08113](https://arxiv.org/abs/2608.08113), v1 **2026-08-08**; body styling **ReLIT**. | [Full HTML v1](https://arxiv.org/html/2608.08113v1), §§2–4/Appendix A opened; Figure 6's image-only algorithm was not separately inspected. A frozen-backbone anchor conditions full-sequence answer/scratchpad recurrence, with step embeddings, warm-up, deep supervision, and ACT halting. | Frozen-backbone latent recurrence and input reinjection have precedent. Its scratchpad is not a separately delta-written compact associative matrix. Appendix A.2's spectral-radius-to-Banach step lacks the general uniform-contraction assumptions; do not transfer that stronger conclusion. Information sufficiency in A.3 also does not guarantee training success or correctness. |
| Zhu et al., **A Survey on Latent Reasoning**; [2507.06203](https://arxiv.org/abs/2507.06203), v1 **2025-07-08**, latest v2 **2025-07-10**. | [Full HTML v2](https://arxiv.org/html/2507.06203v2), §§2–3/Table 1 opened. Distinguishes activation recurrence at fixed token/time from hidden-state recurrence across sequence history, and tabulates sharing, reinjection, depth conditioning, and stopping. | Our reader is vertical activation recurrence at fixed (u,M); the writer evolves memory horizontally across tokens. This taxonomy supplies terminology, not evidence that every step reasons or is verbalizable. The pinned survey predates the newer papers above. |
| Geiping et al., **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)**; [2502.05171](https://arxiv.org/abs/2502.05171), v1 **2025-02-07**, latest v2 **2025-02-17**. | [Full HTML v2](https://arxiv.org/html/2502.05171v2), §§3,6–7 opened. Prelude/core/coda model with input reinjection, shared Transformer recurrence, heavy-tailed depth sampling and truncated backpropagation; successive prediction-KL supports heuristic exits. | Precedent for shared vertical recurrence, sampled depth, and label-free stopping. It iterates full hidden states, not an r-dimensional adapter over our delta memory. Token-dependent trajectories/orbits are not contraction certificates. No Huginn overshoot diagnosis or speedup is transferred to our setting. |
| Zhu et al., **Scaling Latent Reasoning via Looped Language Models (Ouro)**; [2510.25741](https://arxiv.org/abs/2510.25741), v1 **2025-10-29**, latest v6 **2026-10-11**; [official project](https://ouro-llm.github.io/). This is distinct from the earlier **Ouroboros** entry. | [Full PDF v6](https://arxiv.org/pdf/2510.25741v6), §§3,5.3,7.2 opened; v6 HTML returned **404**, so PDF was used. Shared Transformer loops, per-loop supervision, entropy-regularized exits, then gate refinement with model weights frozen. §5.3 reports degradation beyond trained depth. | Learned allocation and nonmonotone depth have precedent. Its exit gate differs from random depth sampling or a residual certificate. Instrumented latent readouts and extrapolation losses do not identify our writer interference, terminal dynamics, or causal use of decoded intermediates. |
| Bae et al., **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA**; [2410.20672](https://arxiv.org/abs/2410.20672), v1 **2024-10-28**, latest v3 **2025-02-28**. | [Full HTML v3](https://arxiv.org/html/2410.20672v3), §§2.2–2.4,3.7–3.8/Appendix K opened. Full-block recursion with depth-specific LoRA relaxations; §2.3 trains all model parameters. Intermediate supervision supports exits. Continuous-depth batching throughput uses oracle-exit simulations with exclusions. | Direct boundary against novelty claims for recursion plus LoRA or relaxed sharing. Different training/frozen-weight contract and full-state loop from our reader. Oracle-exit simulations are not an implemented learned halting policy or evidence for our measured efficiency. No overshoot/verbalizability claim is inferred from these sections. |
| Bae et al., **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation**; [2507.10524](https://arxiv.org/abs/2507.10524), v1 **2025-07-14**, latest v3 **2025-10-25**. | [Full HTML v3](https://arxiv.org/html/2507.10524v3), §§2.1–2.2,4.2–4.3 opened. Shared blocks with expert-choice or token-choice routing; recursion-wise active-token KV caching, or reuse of first-recursion KV for later queries. | Shared adaptive depth and a fixed-KV/evolving-query variant already exist. Do not claim that every looped LM rewrites all context per loop. Our precise distinction is compact delta-written state and low-dimensional adapter computation; routing/caching efficiency or guarantees do not transfer automatically. |
| Hao et al., **Training Large Language Models to Reason in a Continuous Latent Space (Coconut)**; [2412.06769](https://arxiv.org/abs/2412.06769), v1 **2024-12-09**, latest v4 **2026-08-23**. | [Full HTML v4](https://arxiv.org/html/2412.06769v4), §§3–5/Appendix C.1 opened. Reuses the last hidden state as the next input embedding at additional latent sequence positions; curriculum replaces language reasoning steps. Classifier stopping is optional; default experiments use fixed latent length. | Horizontal latent positions differ from fixed-token vertical reader updates. Not emitting a word does not imply nonverbalizability. Its tree-search interpretation is not established for our chain reader; reported curriculum instability is not fixed-checkpoint inference overshoot. No stopping or transfer guarantee follows. |

All nine requested bodies were accessible by the stated route. Remaining access limits:
Ouro's pinned HTML failed (PDF opened); ReLIT's image-only algorithm and external code
were not inspected; the original RecLoRA full-text block remains unresolved. This audit
does not promote its earlier withdrawal snippet into verified status.

### Two Jacobians, and the project's proposed readouts

The published J-lens uses an averaged downstream transport
\(J_\ell=\mathbb E[\partial h_{final,t'}/\partial h_{\ell,t}]\), with later positions
included by its sampling scheme. Its lens applies final normalization and unembedding
to transported activations. The compact reader instead has, away from activation kinks,

\[
J_{read}(z)=\rho D_\phi((R+UMQ)z)(R+UMQ),
\]

holding u and M fixed. This r×r derivative governs recurrent perturbations; it is not
the residual-to-output Jacobian used to define vocabulary-associated directions.
Neither J-space's sparse cones nor the adapter's col(B) should be identified with the other.

Our inference for [P1-L's preregistered per-step probes](../prereg/P1L.md): if
\(g_j=\nabla_{h_{inj}}\mathrm{logit}_j\) is the name-logit gradient at the actual
injection point, the induced local direction in reader coordinates is
\(\alpha B^Tg_j\). Averaging name-logit gradients over validation questions is a
**J-lens-inspired sensitivity probe**; normalization and averaging mean it is not
automatically the published averaged-transport lens or its sparse J-space construction.
Decoding each z_s through hypothetical injection and frozen upper layers is another
explicit diagnostic path, not merely direct unembedding of z_s. Extra diagnostic forwards
must be costed separately from the adapter's inference loop. Ordered intermediate-name
readability would concern content; causal path-following still needs counterfactuals or
interventions. Output-direction alignment remains a proposed intervention, not a result.

### Stability qualifications for JS and plan E.2

These are our mathematical deductions, not new experiments or claims of reproduced
results from those papers. They qualify the shorthand in plan E.2 and the motivation of
[JS](../prereg/JS.md):

- For a continuously differentiable map near an **actual fixed point**, spectral radius
  below one implies local asymptotic stability. Penalizing a Jacobian at finite z_L does
  not establish that it is fixed, that the trajectory enters a suitable neighborhood, or
  that its decoded answer is correct. Residual and accuracy require separate measurements.
- A pointwise spectral radius is not a Euclidean Lipschitz bound. Nonnormal transient
  amplification and products of changing Jacobians matter. ReLU kinks require an invariant
  smooth active region or a suitable nonsmooth argument; one autodiff derivative is not
  a certificate for every neighboring direction. Do not import ReLIT Appendix A.2's stronger
  Banach conclusion without a complete invariant domain and uniform common-norm control.
- Our calculation for STARS Algorithm 1 at K=1, with normalized isotropic initial direction v, is
  \(\mathbb E\|Jv\|^2=\mathrm{tr}(J^TJ\mathbb E[vv^T])=\|J\|_F^2/d\).
  A single random JVP therefore penalizes a Frobenius proxy in expectation, not spectral
  radius squared. Even exact eigenvalues for JS's small reader matrix would not remove
  the fixed-point, neighborhood, nonsmoothness, or correctness requirements above.
- The failed P4 sufficient bounds (evidence H3) prove no nonconvergence claim; overshoot
  measures decoded-accuracy degradation, not necessarily state divergence. Prop.3 itself
  has nilpotent, radius-zero transitions with Euclidean gain above one and finite settling.
  Its fixed state is a weighted sum of visited codes, not the bare terminal code. A strict
  contraction cannot move successive equally distant bare codes to equally distant codes,
  but this special observation does not prohibit useful multistep computation in general.

The broad ideas of latent recurrence, depth sampling, relaxed sharing, adaptive halting,
reinjection, and dynamical regularization have established precedents. The defensible
project description is its specified compact adapter/delta-writer/read schedule and
the controlled results actually measured. The new literature supplies motivations and
claim boundaries, not a priority proof or predicted success of JS/P1-L.
