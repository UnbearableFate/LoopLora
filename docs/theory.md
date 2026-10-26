# WP-T: what fixed-memory reader depth can and cannot do

Binding definitions: [notation v1](notation.md). Experimental target: [P1-S](prereg/P1S.md).
This note uses column vectors, Euclidean/operator norms, fixed parameters, and a
single module unless stated otherwise. A read uses the pre-write snapshot; changing
depth never adds writes. Write count is denoted by \(E\), memory error by \(\Delta\).
Idealizations below do not replace the registered ReLU reader or learned writer.

**Status.** Propositions 1–6 and 8–9 are local derivations of elementary facts and
counterexamples, not novelty claims. Proposition 7 is standard contraction theory
with a local application. Low-rank adaptation is established by
[Hu et al. (2021)](https://arxiv.org/abs/2106.09685v2); temporary fast-weight memory
is established by [Ba et al. (2016)](https://arxiv.org/abs/1610.06258v3).
The plan's §2.3, §14.3 and Appendix C require experimental evidence for useful
depth effects, generalization, and cost advantages. None follows from these proofs.

## Proposition 1 — Output subspace, input bottleneck, and sequence derivatives

**Statement and assumptions (local derivation).** Fix \(M,L,A,B,R,U,Q\), the
activation, and masks, with no other branch input. Write \(f_{M,L}(u)=z^{(L)}\).
Then, for every input,

\[
\delta y=\alpha Bf_{M,L}(Ax)\in\operatorname{col}(B),\qquad
Ax_1=Ax_2\Longrightarrow\delta y(x_1)=\delta y(x_2).
\]

Where the reader is differentiable,
\(D_x\delta y=\alpha B D_uf_{M,L}A\), so its rank is at most
\(\min(\operatorname{rank}A,\operatorname{rank}B)\le r\).
The subspace statement holds even when memory varies; the equality-of-inputs
statement requires the same memory and depth. These claims concern the branch,
not \(Wx+b_W+\delta y\).

**Proof.** Induction on reader steps shows that \(z^{(L)}\) is a function of \(u\)
and the fixed snapshot alone. Multiplication by \(B\) gives the subspace claim.
The chain rule and rank-of-product inequality give the derivative claim. More
generally, at any point where the composed branch is differentiable, its derivative
annihilates \(\ker A\) and has image in \(\operatorname{col}(B)\), giving the same
rank bound even if a particular intermediate ReLU representation has a kink. ∎

**Counterexample: the memory path across a sequence has rank greater than \(r\).**
Use \(r=r_m=d_{in}=d_{out}=1\), \(A=B=K=V=U=Q=1\), \(R=0\),
\(\alpha=1,\rho=1/4,\lambda=\beta=1/2\), \(M_0=0\), and three positive
inputs \(x_t>\epsilon\). All tokens are valid and write once; \(L=1\).
Normalized keys equal 1, so

\[
M_t=\tfrac14M_{t-1}+\tfrac12x_t,\qquad
\delta y_t=x_t+\rho M_{t-1}x_t.
\]

Holding each output's current \(x_t\) fixed when differentiating its history path
gives the strictly lower triangular contribution

\[
J_{\rm hist}=\rho\beta
\begin{pmatrix}0&0&0\\x_2&0&0\\x_3/4&x_3&0\end{pmatrix}.
\]

Its lower-left \(2\times2\) minor has determinant
\((\rho\beta)^2x_2x_3>0\): rank 2 exceeds \(r=1\). The full sequence
Jacobian has positive diagonal and rank 3. Each *individual output's* Jacobian
with respect to all inputs still has rank at most \(r\); stacking outputs changes
the codomain to \(\operatorname{col}(B)^{\oplus T}\), whose dimension can be \(Tr\).
Different histories also give different outputs at identical current \(Ax\).

**Prediction and test variables.** At frozen \(M\), sweep \(L\) and perturb \(x\)
within \(\ker A\): the branch remains identical; its output projection outside
\(\operatorname{col}(B)\) remains zero. P1-S gains at fixed \(r\) measure computation
within a fixed bottleneck. The secondary \(r=16\) condition tests capacity versus
depth, not expansion of the output subspace.

## Proposition 2 — A linear reader is a matrix polynomial

**Statement and assumptions (local derivation).** For \(\phi=\mathrm{id}\), a
fixed snapshot, shared step matrices, \(z^{(0)}=u\), and constant reinjection of
\(u\), put \(T=\rho(R+UMQ)\). Then

\[
z^{(L)}=P_L(T)u,\qquad P_L(T)=\sum_{j=0}^{L}T^j.
\]

**Proof.** At \(L=0\) the claim is \(z^{(0)}=u\). If it holds at \(L\), then
\(z^{(L+1)}=u+T\sum_{j=0}^{L}T^ju=\sum_{j=0}^{L+1}T^ju\). ∎

Thus the degree in memory is at most \(L\). When \(R\ne0\), expansion contains
ordered, generally noncommuting words in \(R\) and \(UMQ\), not an ordinary
scalar binomial expansion. When \(R=0,U=Q=I\), it is exactly
\(\sum_{j=0}^{L}\rho^jM^ju\). At fixed \(M,L\), the branch remains linear in
\(x\), and \(\alpha BP_L(T)A\) may be precomputed. This costs matrix work and
storage per snapshot; it can be amortized over many queries, not freely across
changing online snapshots. The reference reader still uses factorized reads.

**Counterexamples.** With scalar \(u=1,T=-1\), a one-step ReLU reader returns
1 while the linear polynomial returns 0. With step-varying \(T_s\), two steps
give \(u+T_1u+T_1T_0u\), not \(u+T^{}u+T^2u\) for a shared \(T\) in general.

**Prediction and test variables.** P1-S's M1-lin, evaluated at \(L=1,2,4,8\)
with identical \(u,M\), isolates composition without ReLU. Separately vary
queries per snapshot when measuring loop versus precomputation cost. Neither
polynomial degree nor larger \(L\) guarantees a correct answer.

## Proposition 3 — Sufficient conditions for exact chain-terminal decoding

**Statement and assumptions (local derivation).** Consider disjoint directed
chains, one outgoing edge per nonterminal, no cycles, and \(N\le r=r_m\) entity
codes \(c_i\) orthonormal across the entire episode, including terminals.
Initialize memory to zero and write each edge \(i\to j\) once with
\(k=c_i,v=c_j,\lambda=\beta=1\). These gate endpoints describe the oracle writer,
not finite sigmoid logits. Take the linear reader, \(u=c_{i_0}\),

\[
R=0,\quad Q=I,\quad U=\gamma I,\quad a=\rho\gamma>1.
\]

Decode by \(\arg\max_{i\in\mathcal C}c_i^Tz^{(L)}\), with \(\mathcal C\)
all episode entities; equivalently use \(\alpha B=bI\), \(b>0\), before these
logits. For a query \(i_0\to i_1\to\cdots\to i_h\) ending at a terminal,
the unique prediction is \(i_L\) for \(L<h\) and \(i_h\) for every \(L\ge h\).
More generally it suffices that \(T c_{i_j}=a c_{i_{j+1}}\) and
\(T c_{i_h}=0\), with the same decoder and no leakage onto competing codes.

**Proof.** Before a previously unwritten orthogonal key is written,
\(M c_i=0\). The delta update therefore adds exactly \(c_jc_i^T\), independently
of fact order. Hence \(M=\sum_{i\to j}c_jc_i^T\), \(M c_i=c_j\) on edges,
and \(M c_i=0\) at terminals. Proposition 2 gives

\[
z^{(L)}=\sum_{j=0}^{\min(L,h)}a^jc_{i_j}.
\]

Visited-node logits are \(1,a,\ldots,a^{\min(L,h)}\); all others are zero.
Strict increase proves both assertions. Since the graph is acyclic, with longest
chain \(H\), \(M^{H+1}=0\); the reader stabilizes exactly after finitely many
steps despite its Euclidean operator gain exceeding 1. ∎

**Counterexamples and scope.** If \(a=1\), visited nodes tie; if \(0<a<1\), the
start wins, even after reaching the terminal. In particular the bare choice
\(U=Q=I,R=0,\rho=0.25\) does **not** solve the task with this decoder.
It needs \(\gamma>4\), or a different sufficient construction. With \(\beta<1\),
even a single isolated edge has coefficient \(a\beta\); with decay, old edges
are attenuated. Neither default gate values nor a learned \(B\) satisfy the
ideal theorem automatically. Nonorthogonal codes break exact storage and
orthogonal logit separation (Proposition 4). A cycle breaks finite termination.

**Prediction and test variables.** In `oracle_orth`, P1-S's proposed loads have
\(N=20\) or 29 (fallback 14), compatible with \(r=32\). A reader learning the
above alignment should show an accuracy transition near \(L=h\), retained
hop-1 accuracy, and successful \(L=8\) extrapolation. Test accuracy by \((h,L)\),
query/value alignment, and effective gain; this is an existence result, not a
prediction that optimization must find it. The \(r=16\) condition cannot have
20 or 29 mutually orthogonal episode codes.

## Proposition 4 — Interference and its amplification with depth

**Statement and assumptions (local derivation of the standard telescoping identity).**
For any square \(X,Y\) and integer \(j\ge1\),

\[
X^j-Y^j=\sum_{k=0}^{j-1}X^{j-1-k}(X-Y)Y^k,
\quad
\|X^j-Y^j\|\le\|X-Y\|\sum_{k=0}^{j-1}\|X\|^{j-1-k}\|Y\|^k.
\]

**Proof.** Expand each summand into
\(X^{j-k}Y^k-X^{j-1-k}Y^{k+1}\). Consecutive terms cancel, leaving the two
endpoint powers. Apply the triangle inequality and submultiplicativity. No
commutation assumption is needed. ∎

For \(M=\widetilde M+\Delta\) with shared \(u,R,U,Q\), define
\(T=\rho(R+UMQ)\), \(\widetilde T=\rho(R+U\widetilde M Q)\),
\(d=|\rho|\|U\|\|\Delta\|\|Q\|\), and
\(\kappa=\max(\|T\|,\|\widetilde T\|)\). Summing the identity gives

\[
\|z^{(L)}-\widetilde z^{(L)}\|
\le d\|u\|\sum_{j=1}^{L}\sum_{k=0}^{j-1}
\|T\|^{j-1-k}\|\widetilde T\|^k
\le d\|u\|\sum_{j=1}^{L}j\kappa^{j-1}.
\]

For \(\kappa<1\) the last sum is at most \((1-\kappa)^{-2}\); for
\(\kappa=1\) it is \(L(L+1)/2\); for \(\kappa>1\) it can grow geometrically.
This is an upper bound, not a claim that every error grows.

**What random codes change.** For additive storage
\(S=\sum_{i=1}^{E}v_i k_i^T\) with unit keys,

\[
S k_a=v_a+\sum_{i\ne a}v_i\langle k_i,k_a\rangle.
\]

If keys and values are independent isotropic unit vectors, independently drawn
across distinct facts, the cross term has zero mean and expected squared norm
\((E-1)/r\): all cross terms vanish by independence, and
\(\mathbb E\langle k_i,k_a\rangle^2=1/r\) by rotational symmetry.
This familiar \(\sqrt{E/r}\) scale is a diagnostic model, **not** a proved
formula for chain data, whose values are other facts' keys.

The implemented random oracle uses delta storage, not \(S\). Exactly, with
\(P_i=I-k_ik_i^T\) and \(\lambda=\beta=1\),

\[
M_E=\sum_{i=1}^{E}v_i k_i^T P_{i+1}\cdots P_E.
\]

Indeed \(M_i=M_{i-1}P_i+v_i k_i^T\); induction gives the expression (empty
products are identity). Nonorthogonal later writes therefore change earlier
associations and make storage order relevant. At \(E>r\), arbitrary prescribed
associations need not even be consistent with a linear map. `oracle_random`
means known codes and exact gate endpoints, not interference-free memory.

For a chain with unit codes define the actual edge residual
\(\eta_j=M c_{i_j}-c_{i_{j+1}}\). Up to the terminal,

\[
M^j c_{i_0}-c_{i_j}=\sum_{k=0}^{j-1}M^{j-1-k}\eta_k.
\]

This follows by repeatedly substituting the edge residual, and exposes the
same amplification without assuming that all nonorthogonal constraints admit
an exact \(\widetilde M\). A nonzero terminal residual \(M c_{i_h}\) creates
additional unwanted motion beyond \(h\). For the aligned reader and matched-code
decoder of Proposition 3, define the reference code sum
\(z_{\rm ideal}^{(L)}=\sum_{j=0}^{\min(L,h)}a^jc_{i_j}\) and total state error
\(\varepsilon_L=\|z^{(L)}-z_{\rm ideal}^{(L)}\|\), including terminal leakage.
Even its reference logits contain Gram terms
\(c_i^Tz_{\rm ideal}^{(L)}=\sum_{j\le\min(L,h)}a^j\langle c_i,c_{i_j}\rangle\).
If code coherence is \(\mu=\max_{i\ne j}|\langle c_i,c_j\rangle|\), these
terms perturb each orthogonal-reference logit by at most
\(\mu\sum_{j\le\min(L,h)}a^j\). Since every code has unit norm, the actual
logits differ by at most another \(\varepsilon_L\); a sufficient margin condition is

\[
a^m-a^{m-1}>2\left(\mu\sum_{j=0}^{m}a^j+\varepsilon_L\right),
\qquad m=\min(L,h)\ge1.
\]

**Counterexample.** In one dimension write \((k_1,v_1)=(1,1)\), then
\((k_2,v_2)=(1,-1)\). Delta storage ends at \(M=-1\), destroying the first
association despite \(\lambda=\beta=1\). For scalar \(X=1+\delta,Y=1\),
\(X^j-Y^j=(1+\delta)^j-1\), showing genuine growth with depth. Conversely
\(\Delta u=0\) on an invariant queried subspace can make the error zero.

**Prediction and test variables.** Compare `oracle_random` and `oracle_orth`,
then random load \(E/r\), including the preregistered \(1.5E^*\) condition.
Measure edge/terminal residuals, first-read cosine, \(\|M\|\), \(\|z^{(L)}\|\),
and accuracy by \((h,L)\). Increased load can flatten or reverse depth gains;
the bounds do not predict a numerical threshold or monotone decline for every
episode. In learned writers, code alignment and decay are additional variables.

## Proposition 5 — Single-read information and the conditional Bayes limit

**Statement and assumptions (local derivation).** With fixed \(u,L\), parameters,
and no other context access, both B4 and V2-FQ depend on memory only through
\(m=MQu\). If two episodes have identical \(u,m\) and identical decoder side
information, their predictions must coincide, whatever the depth of V2-FQ.

**Proof.** B4 returns \(u+\rho\phi(Ru+Um)\). For V2-FQ, induction in
\(z_{s+1}=u+\rho\phi(Rz_s+Um)\) shows every state is a function of \((u,m)\).
Identical functions applied to identical observations give identical predictions.
For a uniformly chosen pair with distinct answers this implies accuracy at most
\(1/2\); for \(k\) equiprobable answers at the same observation, at most \(1/k\).
Randomized prediction cannot improve this bound. ∎

**A chain pair with the same first hop.** In one two-edge component take
\(a\to b\to c\) and in another \(d\to e\to f\). Compare the graph obtained
by swapping terminal labels \(c,f\): \(a\to b\to f\), \(d\to e\to c\).
Keep all other chains, the candidate set, and the orthonormal codebook fixed.
Both queries from \(a\) have \(u=c_a,Qu=c_a\), and \(Mu=c_b\), but answers
are \(c\) and \(f\). The pair embeds in each proposed preregistration layout:
use the final two edges of two chains and retain their earlier prefixes.
A second aligned read distinguishes them.

**Distributional bound, with necessary extra assumptions.** Use the WP-B
disjoint-chain layouts and uniform injective label assignment. Condition on the
candidate set \(\mathcal C\), \(N=|\mathcal C|\), a single query, its true
hop \(h\ge2\), and its observed start/successor labels \(a,b\). Assume exact
orthogonal storage and **aligned first-hop querying** \(Qu=c_a\). The observer
may know \(h\), the chain lengths, and a fixed codebook, but gets no other query
read, graph feature, episode seed, or graph-dependent codebook information.
Then every possible prediction based on this observation obeys

\[
\Pr(\widehat{\mathrm{answer}}=\mathrm{answer}\mid h\ge2)
\le\frac{1}{N-2}.
\]

**Proof.** Conditional on the unlabeled layout, selected query position, and
labels of its first two nodes, the remaining labels are uniformly permuted over
the remaining positions. The terminal is distinct from both observed nodes.
Permuting any two remaining labels preserves the observation and probability
and exchanges the corresponding possible terminal labels. Hence each of the
\(N-2\) remaining labels has posterior probability \(1/(N-2)\). Mixing over
eligible chains or layouts preserves this result. The generator's minimum-label
split adjustment depends only on the unordered candidate set and respects node
permutations, so it does not break this symmetry. The per-hop evaluation query
sampler is uniform over eligible chains, preserving it as well. ∎

At hop 1 the answer is observed and the bound is 1. Granting the otherwise hidden
hop to the decoder gives a valid upper bound \(1/8+7/[8(N-2)]\) on a uniform
mixture of hops 1–8; hops 2–3 alone have ceiling \(1/(N-2)\).
The code's candidate-mask chance is **\(1/N\)**, not \(1/\#\text{chains}\)
or \(1/1024\). All heads and tails are candidates; terminals are not disclosed.

| P1-S layout | \(E\) | \(N=E+\#\text{chains}\) | Uniform candidate chance | Aligned-read ceiling, \(h\ge2\) |
| --- | ---: | ---: | ---: | ---: |
| (8,4,2,2) | 16 | 20 | 1/20 = 5% | 1/18 ≈ 5.56% |
| (8,8,4,2,2) | 24 | 29 | 1/29 ≈ 3.45% | 1/27 ≈ 3.70% |
| fallback (8,4) | 12 | 14 | 1/14 ≈ 7.14% | 1/12 ≈ 8.33% |

These candidate counts follow from `SymbolicModel.prepare` in
[sym/model.py](../src/looplora/sym/model.py); masking and mean reciprocal
candidate-size chance are in [sym/evaluate.py](../src/looplora/sym/evaluate.py).
The exchangeable layout is in [data/relations.py](../src/looplora/data/relations.py).

**Counterexample to an unconditional ceiling for B4/B5.** Their learned \(Q\)
is unrestricted: \(MQu\) need not equal the first-hop value. For a fixed
orthonormal codebook and finitely many distinct episode memories \(M_i\),
choose \(q\) outside \(\bigcup_{i\ne j}\ker(M_i-M_j)\). Each kernel is a proper
linear subspace, and a finite union of such subspaces has measure zero (rotate
coordinates so each lies in a coordinate hyperplane and apply finite additivity
of null sets). Thus \(M_iq\) distinguishes every memory. A matrix \(Q\) can map
a particular nonzero \(u\) to this \(q\). More generally, generic \(Q\)
simultaneously separates all memory pairs for a finite set of nonzero queries.
An unrestricted decoder can look up the graph and answer exactly after one read.
This is an information counterexample, not a claim that finite-width B4/B5 learns
that decoder or that such encoding is robust to finite precision.

Therefore the only universal numerical ceiling justified here for arbitrary
single-read algorithms is 1. Random-code interference, joint writers, and
episode-dependent `oracle_orth` codebooks also require fresh conditional laws.
In particular, the implementation seeds its per-episode QR codebook from the
episode seed; the codebook is not an independently held-fixed observation in a
literal finite-seed distribution. The identical-read pair and Bayes theorem
describe a controlled aligned-read experiment, not a proven cap on the whole
implemented training/evaluation pipeline. Fixed-seed pseudorandom sampling is
interpreted as the intended exchangeable sampling law, not secrecy from an
unbounded seed-recovering decoder.

**Prediction and test variables.** G2 compares M1 versus B4/B5 at identical
snapshots, especially hops 2–3. A matched-codebook pair with \(Qu=c_a\) must
defeat B4/B5 and can be solved with evolving queries. Unrestricted B5 exceeding
\(1/(N-2)\) is not a contradiction; inspect query alignment and the information
in its first read. Report the evaluator's candidate chance separately from the
conditional ceiling. No claim of an unconditional information separation is made.

## Proposition 6 — The bias-free ReLU code-transmission obstruction

**Statement and assumptions (local derivation).** Let \(U,Q\in\mathbb R^{r\times r}\)
and \(f(v)=Q\operatorname{ReLU}(Uv)\), with no biases. If a set of codes
\(S\) positively spans \(\mathbb R^r\), meaning its nonnegative linear
combinations fill \(\mathbb R^r\), then \(f(v)=v\) cannot hold for every \(v\in S\).

**Proof.** The image lies in \(C=Q\mathbb R_+^r\), the cone generated by the
\(r\) columns of \(Q\). If \(Q\) is singular, its image lies in a proper subspace.
If it is invertible, \(w=Q^{-T}\mathbf1\ne0\) satisfies
\(w^TQa=\mathbf1^Ta>0\) for every nonzero \(a\ge0\); thus \(C\) is a proper
pointed simplicial cone. In either case it cannot contain a set whose positive
span is all of \(\mathbb R^r\). Identity on \(S\) would require \(S\subset C\),
a contradiction. ∎

**Quantification under an explicit random-code model (local corollary).** Fix
invertible \(Q\) independently of \(n\) iid uniform spherical codes. Let
\(p(Q)\) be the spherical solid angle of \(C\). The expected fraction outside
is \(1-p(Q)\), the outside count is binomial, and the probability of containing
all codes is \(p(Q)^n\). These follow from independent membership indicators;
containment is only a necessary condition for exact transmission. For orthogonal
\(Q\), coordinate sign symmetry gives \(p(Q)=2^{-r}\). For arbitrary
invertible \(Q\), this value is not universal: in 2-D, rays separated by
\(\theta\in(0,\pi)\) give \(p=\theta/(2\pi)\). In general \(r\ge2\),
\(0<p(Q)<1/2\): the cone has interior but lies strictly within a hemisphere.
Its angle can approach either extreme. A learned \(Q\) fitted to these codes
invalidates the independence assumption; an episode's orthogonal codes are not
iid spherical draws either. No such binomial bound is asserted for those cases.

**Counterexamples and extensions.** On \(S\subset\mathbb R_+^r\), \(U=Q=I\)
transmits every code exactly. Being signed in a chosen coordinate system is not
itself the obstruction. For at most \(r\) orthonormal codes, a suitable rotation
maps them to positive coordinate axes and back; they do not positively span
\(\mathbb R^r\). A single fixed rotation need not work for many independently
rotated episode codebooks.

Adding only a hidden bias, \(Q\operatorname{ReLU}(Uv+b)\), leaves the same
output cone and does not fix the positive-spanning obstruction. An output bias
translates that cone. With both biases, any bounded set can be transmitted:
choose \(U=Q=I\), \(b\) large enough that \(v+b\ge0\) throughout the set, and
output bias \(-b\). Then \(\operatorname{ReLU}(v+b)-b=v\). A fixed translated
\(r\)-generator cone still cannot cover all of \(\mathbb R^r\).

For width \(m>r\), the range has \(m\) generators and may positively span the
whole space, but width alone does not ensure identity. Width \(2r\) suffices:

\[
U=\begin{pmatrix}I\\-I\end{pmatrix},\qquad Q=(I\; -I),\qquad
Q\operatorname{ReLU}(Uv)=v.
\]

This follows coordinatewise from \(\max(v_i,0)-\max(-v_i,0)=v_i\). Merely
having \(r+1\) positively spanning output rays does not prove such an identity.

**Prediction and test variables.** M1-lin may outperform M1 at signed-code
pointer chasing, especially as \((h,L)\) grow, with the writer and snapshot held
fixed. Measure value-to-query transmission error alongside P1-S's M1/M1-lin
accuracy. The full reader has \(u\), \(Rz\), and a learned decoder: it need not
implement this exact through-map, so the proposition does not prove universal
classification failure. A bias or width follow-up must specify which biases or
dimensions changed and account for their cost; hidden bias alone is not a proof
of restored signed identity. These are diagnostics for the registered F1 decision,
not unregistered claims of experimental improvement.

## Proposition 7 — Contraction certifies state error, not answer correctness

**Statement and assumptions (standard: Banach contraction theorem).** On
\(\mathbb R^r\), fix \(u,M\) and define \(F(z)=u+\rho\phi(Hz)\),
\(H=R+UMQ\). Suppose \(\phi\) is globally \(L_\phi\)-Lipschitz and
\(c=|\rho|L_\phi\|H\|_2<1\). Then there is a unique fixed point \(z^*\),
iteration converges to it from any starting state, and

\[
e_s=\|F(z_s)-z_s\|_2,\qquad
\|z_s-z^*\|_2\le\frac{e_s}{1-c},\qquad
\|\alpha B(z_s-z^*)\|_2\le\frac{|\alpha|\|B\|_2e_s}{1-c}.
\]

This is an application of
[Banach (1922), *Sur les opérations dans les ensembles abstraits…*](https://doi.org/10.4064/fm-3-1-133-181),
with the elementary error-bound proof included here.

**Proof.** Lipschitz continuity and the operator norm give
\(\|F(z)-F(w)\|\le c\|z-w\|\). Along iteration,
\(\|z_{n+1}-z_n\|\le c^n\|z_1-z_0\|\). For \(m>n\), summing these
increments bounds \(\|z_m-z_n\|\) by
\(c^n\|z_1-z_0\|/(1-c)\). Completeness gives a limit \(z^*\), and continuity
gives \(F(z^*)=z^*\). Two fixed points would have distance at most \(c\) times
their own distance, forcing equality. Finally,
\(\|z_s-z^*\|\le\|z_s-F(z_s)\|+\|F(z_s)-F(z^*)\|
\le e_s+c\|z_s-z^*\|\). Rearrangement and multiplication by
\(|\alpha|\|B\|\) prove the bounds. ∎

**Factorized certificate (local application).** Triangle and product norm
inequalities give

\[
c\le\bar c=|\rho|L_\phi
\left(\|R\|_2+\|U\|_2\|M\|_2\|Q\|_2\right).
\]

If \(\bar c<1\), replacing \(c\) by \(\bar c\) in the error bound is safe.
ReLU and identity have \(L_\phi=1\). Certified norm upper bounds, including
Frobenius bounds, suffice; an unconverged power-iteration estimate is not a
certificate. Failure of this sufficient test does not prove divergence.

**Counterexamples and the chain tension.** The affine map \(F(z)=z+1\) has
no fixed point at Lipschitz constant 1; \(F(z)=z\) has infinitely many. Even
with a unique fixed point, \(F(z)=(1+\varepsilon)z\) has residual
\(\varepsilon\) at \(z=1\) but distance 1 from its fixed point: a small residual
without a valid contraction constant gives no uniform error certificate.

For Proposition 3, \(\|M\|_2=1\) when there is an edge: the map sends distinct
orthonormal head codes to distinct orthonormal tail codes and kills the other
directions. Thus \(\|T\|_2=\rho\gamma=a>1\). This chain-following solution
is not a Euclidean contraction, although nilpotence gives exact finite-step
stabilization. In fact \(e_s=a^{s+1}\) for \(s<h\), and \(e_s=0\) for
\(s\ge h\): useful progress can have increasing residual. Spectral radius
zero and Euclidean one-step contraction are different properties.

Even \(F\equiv0\) is a contraction with zero residual at its fixed point and
can give the wrong target class under a fixed decoder. The theorem has no
assumption connecting the fixed point to ground truth, so none can be concluded.

**Prediction and test variables.** P4 should compare fixed-depth accuracy
against residual/output-change stopping at tolerances \(\tau\), stratified by
\(h,E/r\) and certified \(\bar c\). Enforcing \(\bar c<1\) can exclude the
simple increasing-gain chain solution and harm finite-depth performance.
Where certified, \(e_s\le\tau(1-\bar c)/(|\alpha|\|B\|)\) bounds branch
output error by \(\tau\) when the denominator is nonzero. Converting this to
label stability additionally requires a decoder margin; label stability still
does not establish correctness. Where uncertified, stopping is an empirical
policy to validate, not an application of this bound.

## Proposition 8 — Exact elimination has structural conditions

**Statement and assumptions (local derivation).** For fixed compatible
\(A\in\mathbb R^{r\times d}\), \(B\in\mathbb R^{d\times r}\), any
\(\phi:\mathbb R^r\to\mathbb R^r\), and scalar gates, suppose the *entire*
high-dimensional recurrence is

\[
h_0=x,\qquad h_{s+1}=h_s+g_s B\phi(Ah_s).
\]

Then it is implemented exactly by \(h_s=x+Bz_s\) with

\[
z_0=0,\qquad z_{s+1}=z_s+g_s\phi(Ax+(AB)z_s).
\]

**Proof.** The identity is true initially. Substitution at step \(s\) gives
\(h_{s+1}=x+B[z_s+g_s\phi(Ax+ABz_s)]\), establishing the induction. No
inverse or full column rank of \(B\) is needed; these \(z_s\) are a valid
representation, not necessarily unique. State-dependent gates are allowed if
their arguments are reproduced: a linear router score \(w_s^Th_s+b_s\) equals
\(w_s^Tx+(w_s^TB)z_s+b_s\), so its projections can be precomputed. ∎

**Counterexamples.** A high-dimensional operation whose update leaves
\(\operatorname{col}(B)\) cannot be absorbed by this argument. With
\(B=(1,0)^T,x=(0,1)^T\), the extra update \(h\mapsto2h\) produces a displacement
\((0,1)^T\) outside \(\operatorname{col}(B)\). Full attention/FFN loops require
their own proof. Moving a high-dimensional ReLU through \(A\) also fails:

\[
A=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad B=I_2,\quad
z=\begin{pmatrix}1\\-1\end{pmatrix};\qquad
A\operatorname{ReLU}(Bz)=\begin{pmatrix}1\\0\end{pmatrix}
\ne\operatorname{ReLU}(ABz)=\begin{pmatrix}0\\0\end{pmatrix}.
\]

**Prediction and test variables.** Compare original and eliminated recurrences
over depth and state-dependent router inputs for equality, then measure latency
versus \((d,r,L)\). This is an implementation equivalence, not a new function
class. It neither derives V2 from the original double-BA proposal nor licenses
replacing \(A\phi(Bz)\) by \(\phi(ABz)\) in B9. P1-S uses its registered V2
recurrence; elimination experiments must be identified separately.

## Proposition 9 — Repeated BA and the order of the zero-gradient start

**Statement and assumptions (local derivation).** Let \(A\in\mathbb R^{r\times d}\),
\(B\in\mathbb R^{d\times r}\), shared across repeats, and \(L\ge1\). Then
\((BA)^L=B(AB)^{L-1}A\). For the double-BA branch

\[
f(A,B)=\alpha BA\phi(BAx),
\]

assume \(\phi(0)=0\) and local Lipschitz continuity at zero. At every
\((A_0,0)\), its joint first derivative with respect to \((A,B)\) is zero.
Second-order behavior need not vanish and for ReLU a Hessian need not exist.

**Proof of powers and first order.** The power identity holds at \(L=1\).
Multiplying \(B(AB)^{L-1}A\) by \(BA\) yields \(B(AB)^LA\), proving induction.
For fixed \(x\) and bounded \(A\) near \(A_0\), Lipschitz continuity gives
\(\|f(A,B)\|\le C\|B\|^2\). This is
\(o(\|(A-A_0,B)\|)\), proving the zero Fréchet derivative even with inner
ReLU kinks. Composition with a differentiable loss therefore has zero gradients
for both factors. A deterministic first-order optimizer with zero initial
momentum, no perturbations, and no other loss path updating \(B\) stays at
\(B=0\); multiplicative weight decay does not free it. ∎

**Second order, including nonsmoothness.** For ReLU, homogeneity gives the
following exact identities along a direction \(D\):

\[
f(A_0,tD)=
\begin{cases}
\alpha t^2 DA_0\operatorname{ReLU}(DA_0x),&t\ge0,\\
-\alpha t^2 DA_0\operatorname{ReLU}(-DA_0x),&t\le0.
\end{cases}
\]

The right second directional derivative is
\(2\alpha DA_0\operatorname{ReLU}(DA_0x)\). With
\(A_0=x=\alpha=1\), the scalar function is \(b\operatorname{ReLU}(b)\):
its second derivative from the positive side is 2 and from the negative side 0.
There is no classical Hessian at zero. An autodiff choice of ReLU derivative at
zero cannot establish a mathematical second derivative there.

If instead \(\phi\) is \(C^2\) near zero with \(J=D\phi(0)\), Taylor expansion
gives \(f(A_0,B)=\alpha BA_0JBA_0x+O(\|B\|^3)\). Its genuine Hessian is

\[
D_B^2f(A_0,0)[D,E]
=\alpha(DA_0JEA_0x+EA_0JDA_0x).
\]

This follows by taking the coefficient of the product of two scalar perturbations
in \((tD+sE)A_0J(tD+sE)A_0x\). The \(A,A\) and mixed \(A,B\) Hessian blocks
are zero. For a loss gradient \(g=\nabla_y\ell(y_0)\), the smooth loss Hessian
contracts this branch Hessian with \(g\); there is no \((Df)^T\nabla^2\ell Df\)
term because \(Df=0\). For ReLU the one-sided loss expansion is instead
\(\ell(y_0+f(A_0,tD))=\ell(y_0)+\alpha t^2
\langle g,DA_0\operatorname{ReLU}(DA_0x)\rangle+o(t^2)\) as \(t\downarrow0\).
Neither nonzero curvature nor descent in a direction gives a first-order update
at the exact zero initialization.

**Why the skip form starts, and counterexamples to stronger claims.** For
\(s(A,B)=\alpha[BAx+\eta BA\phi(BAx)]\), the same estimate leaves

\[
Ds(A_0,0)[E,D]=\alpha DA_0x,\qquad
\nabla_B\ell=\alpha g(A_0x)^T,\quad \nabla_A\ell=0.
\]

The skip provides a nonzero first update when the batch sum of those outer
products is nonzero; \(A_0\ne0\) alone does not guarantee this (take
\(A_0x=0\), or cancelling examples). Alternatively, nonzero internal factors
with an outer zero gate allow the gate to start if its loss derivative is
nonzero. Zeroing both the gate and internal branch restores the dead start.

The assumption \(\phi(0)=0\) matters: for a nonzero constant activation vector
\(v\), \(D_Bf[D]=\alpha DA_0v\) need not vanish. Linearity matters for the
power identity's application to a reader: scalar \(BA=-1,x=1\) gives
\(BA\operatorname{ReLU}(BAx)=0\), whereas \((BA)^2x=1\).

**Prediction and test variables.** Measure initial \(\|\nabla_B\ell\|\),
\(\|\nabla_A\ell\|\), and early loss versus `double_ba_form`, initialization,
and outer gate. A pure non-skip zero-\(B\) double-BA branch does not start by
first-order learning; the registered B9 skip form permits startup. V2's single
output factor similarly permits a \(B\) gradient at zero when \(z^{(L)}\) and
the loss gradient are nondegenerate, although its internal branch gradients
are initially blocked by \(B=0\).

## Proposition 10 — Writer shortcuts, factorization, and diagnostic specificity (WP-T2)

This section is a **local derivation** for the implementation at `d243aec` and
[P1-S final v1](prereg/P1S.md), whose primary layout is now (8,4), not the
original proposed loads in Proposition 3. “Proved” refers to the stated model;
“constructed” means an existence example; “measured” refers only to the named
artifacts. The [validation pilot report](reports/P1S-pilot.md) measures joint
memory readers near 46–48% on its selection metric, including B4; the frozen
writer rows are lower, about 25–33%. Those results motivate the hypothesis and
do not identify its mechanism. No WP-H/F1 confirmation results are used here.

### 10.1 Common keys: exact delta algebra before telescoping

**Proved.** Suppose a relation's facts have one common unit key \(k\), values
\(v_i=D(e_{t_i}-e_{h_i})\), zero initial memory, constant scalar gates, and
queries with \(k^Tq=\theta>0\). Normalization must be above its clamp. Put
\(a=\lambda(1-\beta)\). The actual decay-first delta writer satisfies

\[
M_E=m_Ek^T,\quad m_E=\beta\sum_{i=1}^Ea^{E-i}v_i,\quad M_Eq=\theta m_E.
\tag{10.1}
\]

Indeed substituting \(M_{i-1}=m_{i-1}k^T\) into the writer gives
\(m_i=a m_{i-1}+\beta v_i\); induction gives (10.1). Thus shared relation
overlap *permits* a global read but does not imply an unweighted sum. With
\(\beta=1\) it keeps only the last fact. On \(a\to b\to c\), the rescaled
read is \(-a e_a+(a-1)e_b+e_c\), after applying \(D\), rather than
\(D(e_c-e_a)\), unless the coefficients agree. Here the italic node labels
are embeddings and the scalar \(a\) is the decay factor.

If \(H=\max_i\|v_i\|\), then

\[
\left\|m_E/\beta-\sum_i v_i\right\|
\le H\sum_{j=0}^{E-1}(1-a^j)
\le H(1-a)E(E-1)/2.\tag{10.2}
\]

The first inequality is the triangle inequality; the second uses
\(1-a^j=(1-a)\sum_{k<j}a^k\le j(1-a)\). Both near-unit \(\lambda\) and
small \(\beta\) make the bound small. Uniformly permuting the facts makes
the *expected* read a scalar multiple of \(\sum_i v_i\), since every value
occupies every position equally often; this does not remove realized order noise.
With distractor facts whose raw key is zero, their writes add no value but still
decay memory. The same conservative bound holds using all \(E\) write positions
and target values only, because each intervening factor is either \(a\) or
\(\lambda\), both at least \(a\).

An exact running average would use \(\lambda=1,\beta_i=1/(c+i)\): induction
gives \(m_i=\sum_{j\le i}v_j/(c+i)\). This is a **different**, position-aware
gate, not the implemented constant sigmoid gate. The reported pretrained writer
has \(\lambda=0.999968,\beta=0.565222\), hence \(a\) near 0.435; that particular
measurement is outside the small-\(\beta\) construction. Other keys, value
weights, or a different joint writer may still encode terminal statistics.

**Constructed shared-input parameters.** Let \(D\in\mathbb R^{p\times64}\)
have orthonormal rows. Choose a relation-block functional \(g\) with
\(g(e_0)=1\), \(g(e_r)=0\) for other relations, and \(g(e_{\rm hop})=0\)
for all used hop vectors, including hop 0 (unknown hop is encoded this way).
These are at most 21 independent constraints for four relations and hop 0–16;
independent Gaussian vectors in 64 dimensions permit a solution almost surely.
An ill-conditioned solution need not be numerically robust. Use shared \(A\)
to define input, anchor, and initially zero output slots:

\[
u_{\rm fact}=(D(e_t-e_h),g(e_r),0_p),\qquad
u_{\rm query}=(-De_s,1,0_p).
\]

Set \(K\) and \(Q\) to map only the anchor to a fixed unit memory vector
\(k\), and \(V\) to copy the input \(p\)-block into the first \(p\) memory
coordinates. Thus every target fact has key \(k\), and every target query has
the same \(q=k\), regardless of its start or true hop. Non-target raw keys
are zero. With identity activation, \(R=0\), let \(U\) copy those memory
coordinates into the output slots, and let \(\alpha B\) decode only those
slots by \(D^T/(\rho\beta)\). For \(\alpha\rho\ne0\),

\[
\delta y^{(L)}=D^T m_E[1:p]/\beta\quad\text{for every }L\ge1.\tag{10.3}
\]

The anchor is unchanged by every update, so \(Rz\) and \(MQz\) remain the
same; output slots start at zero and \(B\) ignores the query-dependent input
slots. This proves depth and query invariance for M1-lin, B5-lin, and B4,
using the specified \(\rho=0.25\) by rescaling \(B\).

For the **specified bias-free ReLU**, add one zero baseline slot. Let \(R\)
copy the anchor as a positive constant \(T\) into each output slot and the
baseline, while \(U\) copies the read into output slots only. Choose
\(T>\max|m_i[j]|\) over the bounded frozen table and allowed lengths. ReLU
then gives output \(\rho(T\mathbf1+m_E[1:p])\), baseline \(\rho T\).
Decode their difference with \(D^T/(\rho\beta)\) to obtain (10.3) again.
This uses a relation-dependent constant through \(Rz\), not an added bias;
it does not contradict Proposition 6. Sufficient widths are \(r\ge2p+1\)
for identity and \(r\ge2p+2\) for this ReLU construction: \(p=7,15,31,63\)
fit \(r=16,32,64,128\), respectively. These are sufficient budgets, not lower
bounds on every possible shortcut. The initialized zero \(B\) is replaced by
the displayed trained parameters; this is expressibility, not an optimization proof.

For disjoint relation-0 chains, the unweighted values telescope exactly:
\(S=\sum_iD(e_t-e_h)=\sum_{j=1}^cD(e_{T_j}-e_{R_j})\).
Under (10.2), (10.3) approximates \(D^TS\). The starts here are chain roots
\(R_j\), not necessarily the queried start. This is the precise conditional
version of the orchestrator's hypothesis. Relation overlap alone, arbitrary
gates, or an uncorrected signed ReLU do not prove it.

### 10.2 Terminal-score margins at Gaussian dimension 64

**Proved distributional calculation, not a learned-code guarantee.** Assume
\(e_i\stackrel{iid}{\sim}N(0,I_d/d)\), \(d=64\), distinct endpoints, fixed
orthonormal-row \(D\) independent of the embedding table, and exact endpoint
sum decoding \(y=D^T\sum_{j=1}^cD(e_{T_j}-e_{R_j})\). For a terminal,
interior node, and root respectively, \(\ell_i=e_i^Ty\) has moments

\[
\begin{array}{c|ccc}
 &T&\mathrm{interior}&R\\\hline
\mathbb E\ell_i&p/d&0&-p/d\\
\operatorname{Var}(\ell_i)&(2c+1)p/d^2&2cp/d^2&(2c+1)p/d^2
\end{array}\tag{10.4}
\]

For example the terminal includes its squared norm (variance \(2p/d^2\))
and \(2c-1\) uncorrelated cross terms (variance \(p/d^2\) each); conditioning
on that terminal makes the cross covariances zero. The interior is independent
of all \(2c\) endpoints. This proves the table. Terminal-minus-interior
margin has mean \(p/d\), variance \((4c+1)p/d^2\), and signal/SD
\(\sqrt{p/(4c+1)}\). At two chains it is \(\sqrt{p/9}\), not automatically
\(\sqrt{64/9}\) at a rank-32 insertion.

More exactly, this pairwise margin has the law

\[
\ell_T-\ell_i\ \overset{d}{=}
\frac{(\sqrt{4c}+1)X-(\sqrt{4c}-1)Y}{2d},\qquad
X,Y\stackrel{ind}{\sim}\chi_p^2.\tag{10.5}
\]

To see this, in each projected coordinate write the difference as
\((t-v)(t+w)\), with independent standardized Gaussians \(t,v\) and
\(w\sim N(0,2c-1)\). The quadratic form has two nonzero eigenvalues
\((1\pm\sqrt{4c})/2\); orthogonal diagonalization and summing \(p\) independent
coordinates prove (10.5). The Gaussian integral
\(\mathbb E e^{-s\chi_p^2}=(1+2s)^{-p/2}\) gives
\(\mathbb E e^{-t d(\ell_T-\ell_i)}
=[1+2t-(4c-1)t^2]^{-p/2}\) in its finite domain. Markov's inequality
at \(t=1/(4c-1)\) gives

\[
\Pr(\ell_T-\ell_i\le\eta)
\le e^{d\eta/(4c-1)}(1-1/(4c))^{p/2},\quad\eta\ge0.\tag{10.6}
\]

At zero margin, terminal-versus-root error similarly is at most
\((1-1/c)^{p/2}\) for \(c>1\): the product factors are
\((t-r)\) and \((t-r+w)\), with variances 2 and \(2c\), covariance 2;
the same Gaussian integral gives the bound. At \(c=1\) the difference is a
squared norm and positive almost surely. A union bound over
\(c(N-2c)\) terminal/interior and \(c^2\) terminal/root comparisons bounds
failure to rank **all** terminals first. It can be vacuous at small \(p\);
neither dimension 64 nor positive expected margin guarantees terminal-first
ranking in an episode. A learned \(D\) fitted to these codes also falls outside
the independence assumption.

A deterministic sufficient condition is more direct. If every endpoint's
projected norm squared is at least \(\nu\), and every distinct projected
candidate pair has absolute inner product at most \(\mu\), terminal scores
are at least \(\nu-(2c-1)\mu\), interior scores at most \(2c\mu\), and root
scores at most \(-\nu+(2c-1)\mu\). Thus the terminal/interior margin is at
least \(m_0=\nu-(4c-1)\mu\), and a positive \(m_0\) also separates roots.
For approximate delta storage subtract
\(\max_{T,i}\|D(e_T-e_i)\|\,H(1-a)E(E-1)/2\) from these margins, by
Cauchy–Schwarz and (10.2). This gives an explicit geometry/gate condition
under which the constructed single read ranks the terminal set first.

**Measured constructed example (not a trained writer).** Compute job
`3712948.opbs` used 256 validation-only (8,4) episodes, frozen backbone seed
31415, a derived generation seed, and \(D\) selecting the first \(p\)
coordinates. It computed the ideal unweighted endpoint statistic directly;
the separate shared-writer checks used the actual delta rule at small beta.
No empirical moment was required to equal its population value. Counts below
are descriptive for this fixed table and episode sample, not confidence bounds.

| \(p\) | Top prediction is a terminal | Both terminals precede all others | Minimum terminal/nonterminal margin |
| ---: | ---: | ---: | ---: |
| 7 | 150/256 | 46/256 | −0.2963 |
| 15 | 219/256 | 126/256 | −0.3607 |
| 31 | 251/256 | 205/256 | −0.3563 |
| 63 | 256/256 | 251/256 | −0.4358 |
| 64 | 256/256 | 252/256 | −0.3974 |

At \(p=15\), measured terminal/interior means were 0.23090/−0.00248, versus
theory 15/64 and 0; pair-margin variance was 0.02941 versus theory
135/4096. Negative minimum margins even at \(p=64\) demonstrate why an
expected Gaussian margin is not an all-episode guarantee. The larger projections
in this table are hypothetical endpoint statistics, not claims about the
rank-32 construction or any WP-H/F1 checkpoint. Full moments and provenance
are retained in the log linked from [WP-T2's report](reports/WP-T2.md).

### 10.3 Tail-only values: a removed statistic and remaining shortcuts

**Proved, with limited scope.** Write \(A=[A_h,A_r,A_t]\). In a factorized
writer, keys/initial queries see head and relation only and values are \(Pe_t\).
The direct linear value term \(P(e_t-e_h)\) is excluded. Nevertheless an
additive exact memory read is

\[
Mq=\sum_iPe_{t_i}\langle k(h_i,r_i),q\rangle.
\]

For delta storage the coefficient is instead
\(\beta\lambda^{E-i}k_i^T\prod_{j=i+1}^E(I-\beta k_jk_j^T)q\), with
products in chronological order, by expanding each writer update. Head
information therefore remains in coefficients even with tail-only values.
Factorization does not imply an aligned first-hop query or a universal chance
ceiling. Proposition 5's generic-probe counterexample still applies whenever
the stored memories are distinct.

An explicit finite-real example strengthens this caveat. Give entities two
independent continuous coordinates \((a_i,b_i)\), normalized keys
\((a_h,1)/\sqrt{1+a_h^2}\), values \(b_t\), and query \((1,0)\) supplied by
the relation. The additive scalar read is \(w^TFb\), where \(F\) is the
edge-count matrix and \(w_h=a_h/\sqrt{1+a_h^2}\). For \(F\ne G\),
\((F-G)^Tw\ne0\) almost surely; conditioning on \(w\), its product with the
independent continuous \(b\) is nonzero almost surely. A finite union over
graphs remains null. Thus one real read can distinguish a finite graph family
with an unrestricted lookup decoder. Small finite \(\beta=\epsilon\),
\(\lambda=1-\epsilon^2\) preserves separation for sufficiently small
\(\epsilon>0\), since its delta read is \(\epsilon w^TFb+O(\epsilon^2)\)
and finitely many nonzero leading differences remain separated. This is a
noiseless existence counterexample, not a practical FP32 decoder construction.

For an **equal-weight generic relation probe**, however, exact orthonormal
storage gives only \(b_1=\sum_{\rm edges}c_t\). Interior nodes and terminals
receive identical coefficients. For example, \(a\to b\to c\) and
\(a\to c\to b\), with the other chains unchanged, have identical candidate
and tail sets but different terminals. No function of this unweighted tail
sum alone can distinguish them. Uniform selection among its \(E_0\) target
tails has accuracy \(1/E_0\), rather than \(1/c\) or full-candidate chance
\(1/N\). This specifies an algorithm, not a bound on every decoder with
additional query information. An aligned exact first-hop read instead has
hop-1 accuracy 1 and the conditional multihop bound of Proposition 5.

**Constructed iterated generic reads.** Assume the ideal finite-episode code
geometry of Proposition 3, and that a relation-derived probe with equal overlap
on all target keys is available. This is an additional query/key condition,
not a consequence of factorization or a claim about all 1024 global codes.
Let \(S c_{j,d}=c_{j,d+1}\) along a chain with root depth zero and kill its
terminal. Induction gives

\[
b_s=S^{s-1}b_1=\sum_j\sum_{d=s}^{n_j}c_{j,d}.\tag{10.7}
\]

An evolving query can compose this operator; a cached fixed query does not
acquire these new observations. With enough separate latent channels for the
relation seed, head features and returned states, reinjection produces
\(\sum_{s=1}^La^sb_s\), \(a>0\). Each depth-\(d\) node has coefficient
\(\sum_{s=1}^{\min(d,L)}a^s\). Thus the top-scoring set is depth at least
\(L\), until the longest chain depth is reached; thereafter it remains the
deepest nodes. Nonnegative coordinate codes allow ReLU; signed codes need a
sign-preserving construction. This is a generic **root-depth filtration**, not
a walk from the queried start.

Equal-length chains of length \(H\) have \(b_H\) equal to their terminal sum.
For (8,4), no \(L\le4\) filtration isolates terminals: \(b_4\) includes five
long-chain nodes and the short-chain terminal. At \(L=8\) it isolates only the
long terminal. Differences such as \(b_2-b_3\) isolate a root-depth layer;
this is further global information even when no full terminal set is isolated.

Let \(n_L=\sum_j(n_j-L+1)_+\), \(c_h=\#\{j:n_j\ge h\}\). Uniformly
choosing among the top-depth nodes gives the **proved policy-specific** accuracy

\[
A_L(h)=\frac{\#\{j:n_j\ge\max(h,L)\}}{c_h n_L},\quad
1\le L\le\max_j n_j.\tag{10.8}
\]

There is one query-eligible chain per terminal, selected uniformly by the
per-hop evaluator; count its terminal if it belongs to the top set and divide
by the top-set size. This proves (10.8) and the following exact predictions.
Typed rows assume perfect rejection of nonzero relations.

| Layout; candidates | Generic one-read | Generic \(L=4\) | Generic \(L=8\) | Candidate chance |
| --- | --- | --- | --- | --- |
| (8,4); 14 | 1/12, all hops | 1/6, all hops | 1/2 at \(h\le4\); 1 at \(h\ge5\) | 1/14 |
| (8,1) typed; 14 | 1/9, all hops | 1/10 at \(h=1\); 1/5 otherwise | 1/2 at \(h=1\); 1 otherwise | 1/14 |
| (8,8,2); 21 | 1/18, all hops | 1/15 at \(h\le2\); 1/10 otherwise | 1/3 at \(h\le2\); 1/2 otherwise | 1/21 |

Including typed distractor tails changes the one-read support from 9 to 12,
giving 1/12 for uniform tail selection. Their relation weights and decay can
also break all ideal formulas; these must be measured in an actual model.

### 10.4 Quantitative predictions and what A3/A5 identify

**Proved for the specified policies; predictions for trained rows.** If a
query-independent endpoint statistic selects uniformly among \(c\) terminals,
its accuracy is \(1/c\) at **every** hop, even when only one chain is eligible
at that hop. In the construction (10.3), B4, M1, M1-lin, B5 and B5-lin all
have that same prediction for every \(L\ge1\) when terminal decoding succeeds.
There is no extra graph information at larger \(L\). If only a fraction
\(p_T\) of queries' global selections are terminals, exchangeability of
terminal roles gives accuracy \(p_T/c\), rather than a guaranteed \(1/c\).
This is not a theorem about an arbitrary trained classifier or a length-biased
terminal selector. B3 has no access to the stored statistic; uniform candidate
guessing is a reference, not a bound on every no-memory reader.

| Layout | Terminal-guess B4; M1/M1-lin; B5/B5-lin, all \(h\), \(L\ge1\) | Uniform candidates |
| --- | --- | --- |
| (8,4) | 1/2 = 50% | 1/14 ≈ 7.14% |
| (8,1) typed, target relation isolated | 1/2 = 50% | 1/14 ≈ 7.14% |
| (8,8,2) | 1/3 ≈ 33.33% | 1/21 ≈ 4.76% |

Thus the observed joint pilot plateau is **consistent** with a terminal statistic,
while its exact head-subtraction/common-key implementation remains unmeasured.
WP-H/F1 should check key/query start dependence, \(VA_h+VA_t\), relation
channels, gate weights, first-read/global-endpoint alignment, and memory/logit
responses to rewiring before identifying the mechanism. A higher-rank joint
row can improve terminal decoding and still remain at 50% without walking.
The fixed-projector Gaussian analysis predicts less decoding noise as its
effective dimension grows; it does not predict trained accuracy at each rank.
The tail feature's linear rank is at most 64, which is a dimension cap, not a
universal lower bound on interference after learning an anisotropic projection.

**A3 for the generic filtration policy.** Substituting (10.8) into the
registered contrasts gives:

| Layout | \(\Delta_{reach}\), hops 2–3 | \(\Delta_{far}\), hops 6–8 | Difference |
| --- | --- | --- | --- |
| (8,4) | 1/12 | 1/12 | 0 |
| (8,1) typed | 4/45 | 4/45 | 0 |
| (8,8,2) | 1/36 | 2/45 | −1/60 |

Pure terminal guessing has both deltas zero. A3 therefore rejects these
particular generic policies in expectation. In typed filtration, hop-1 accuracy
also drops by \(1/90\), slightly more than the permitted 1 percentage point.
But A3 is not an identification theorem. In (8,8,2), \(b_2-b_3\) contains
the three depth-2 nodes, including the short terminal. A policy that predicts
the start at \(L=1\) (always wrong) and uniformly selects these three nodes
at \(L=4\) has \(\Delta_{reach}=1/18>3\%\), \(\Delta_{far}=0\), and improves
hop 1, using only generic reads and subtraction. A reader with adequate
history/clock channels can construct this statistic. This is a counterexample
to universal A3 specificity, not a claim that a registered checkpoint implements
it or that it passes all G1/G2 and replication requirements.

**A5 paired prediction.** `swap_successors` preserves heads, tail multiplicities,
roots, terminal sets, and relation labels, but changes edge assignments. Any
prediction depending only on the unweighted endpoint sum or tail sum plus
unchanged query features is exactly unchanged. Thus

\[
\Pr(\widehat y_{swap}=T_{new})-
\Pr(\widehat y_{original}=T_{new})=0.\tag{10.9}
\]

This equality holds pointwise with matched tie breaking, hence also in every
old-hop and \(h_{new}\le L\) stratum. Uniform terminal guessing has both
rates 1/2 in the two-chain layouts and 1/3 in (8,8,2). A high absolute follow
rate alone is therefore insufficient. Approximate delta weighting or ties
between terminals can change predictions after a swap; (10.2) certifies
invariance only if its score perturbation is smaller than the winning-terminal
margin, not merely the terminal-versus-interior margin.

For an ideal query-local walk, let the original remaining distance be \(h\)
and the other chain length be \(b\). The generator chooses a cut uniformly
on the query's suffix and another uniformly on the entire other chain. The new
distance is \(h'=p+q\), with independent uniform
\(p\in\{1,\ldots,h\},q\in\{0,\ldots,b-1\}\). Therefore

\[
C_L(h,b)=\Pr(h'\le L)=
\frac{\sum_{p=1}^{h}\min(b,\max(0,L-p+1))}{hb}.\tag{10.10}
\]

This follows by counting the allowed suffix lengths at each first cut.
Proposition 3 then gives swapped follow \(C_L\) and unswapped new-terminal
rate zero: the other terminal was never on the original query path. Conditional
on \(h'\le L\) the ideal follow is 1; otherwise it is 0. Average (10.10) over
uniformly eligible original chains and uniformly selected other chains to get
the following **ideal-walk** predictions. Every paired unswapped entry is zero.

| (8,4), original \(h\) | \(L=1\) | \(L=4\) | \(L=8\) |
| ---: | ---: | ---: | ---: |
| 1 | 3/16 | 3/4 | 1 |
| 2 | 3/32 | 21/32 | 31/32 |
| 3 | 1/16 | 9/16 | 15/16 |
| 4 | 3/64 | 15/32 | 29/32 |
| 5 | 1/20 | 1/2 | 1 |
| 6 | 1/24 | 5/12 | 23/24 |
| 7 | 1/28 | 5/14 | 25/28 |
| 8 | 1/32 | 5/16 | 13/16 |

| (8,1) typed, original \(h\) | \(L=1\) | \(L=4\) | \(L=8\) |
| ---: | ---: | ---: | ---: |
| 1 | 9/16 | 3/4 | 1 |
| 2 | 1/2 | 1 | 1 |
| 3 | 1/3 | 1 | 1 |
| 4 | 1/4 | 1 | 1 |
| 5 | 1/5 | 4/5 | 1 |
| 6 | 1/6 | 2/3 | 1 |
| 7 | 1/7 | 4/7 | 1 |
| 8 | 1/8 | 1/2 | 1 |

| (8,8,2), original \(h\) | \(L=1\) | \(L=4\) | \(L=8\) |
| ---: | ---: | ---: | ---: |
| 1 | 1/4 | 2/3 | 1 |
| 2 | 1/8 | 5/8 | 23/24 |
| 3 | 5/48 | 11/16 | 15/16 |
| 4 | 5/64 | 19/32 | 29/32 |
| 5 | 1/16 | 19/40 | 7/8 |
| 6 | 5/96 | 19/48 | 27/32 |
| 7 | 5/112 | 19/56 | 13/16 |
| 8 | 5/128 | 19/64 | 3/4 |

In particular, old \(h\le L\) does not imply new reachability, and even
\(L=8\) can miss new paths of length up to 15.

**Remaining diagnostic counterexample.** Root-depth filtrations are not swap
invariants: swapping changes root-to-node depths and chain lengths. If the
queried chain originally has length \(a\), new lengths of it and the other
chain are \(a'=a-h+h'\), \(b'=h+b-h'\). For the generic positive-coefficient
policy put \(s=\min(L,\max_j n'_j)\). Its swapped follow is
\(\mathbf1[a'\ge s]/\sum_j(n'_j-s+1)_+\) for a fixed pair of cuts;
average this over the cuts to get its expected follow. Its unswapped rate is
\(\mathbf1[b\ge s_0]/n_{s_0}\), \(s_0=\min(L,\max_j n_j)\).
This is not the law (10.10). For typed (8,1), query the long chain's last
predecessor and swap its final edge with the short edge. At \(L=4\), the
generic policy originally scores the five deepest long-chain nodes, excluding
the new target (the old short terminal). After the swap it includes that target:
follow rises from 0 to 1/5 without a query-directed walk. At \(L=8\) it rises
from 0 to 1. A5 detects exactly invariant terminal guessing, but positive paired
follow alone does not exclude all global-statistic solutions. Stratify by new
reachability, inspect paired logit changes, and compare B5 controls; none alone
constitutes a proof of an internal algorithm.

### 10.5 Minimal symbolic writer changes and their guarantees

**Proved removal of the direct value mechanism.** A hard constraint
\(VA_h=0\) excludes the linear head subtraction in (10.1), while allowing
\(KA_h\ne0\) for addressing. Also imposing \(VA_r=0\) makes the value
exactly tail-only. For keys to be tail-blind require \(KA_t=0\). These product
constraints concern the effective maps, not merely the names of latent slots.
They are feasible with one shared \(u\): reserve head/relation slots and tail
slots in \(A\), restrict \(K,Q\) to the former initially, and \(V\) to the
latter. Query feedback from returned values still needs a specified map into
key coordinates. The allocated subspaces share the total rank budget; this is
the coherent one-\(u\) counterpart of factorizing the writer inputs. Zeroing
\(A_h\) instead would also remove the head from keys and the query, defeating
the intended associative addressing.

A smaller contract change is hard \(VA_h=0\) alone. It removes this direct
head term but allows relation-valued statistics and tail-dependent keys. It
therefore provides a narrower guarantee than full head/relation-key,
tail-only-value factorization. Neither design removes generic graph encoding,
overwrite-dependent coefficients, or (10.7). A stronger first-hop theorem also
needs aligned addressing, controlled interference, and decoder side-information
assumptions as in Proposition 5; it does not follow from a value mask alone.

**Soft penalties are predictions, not exclusions.** Penalizing
\(\|VA_h\|_F^2\) encourages the hard condition but a finite weight does not
enforce it. An empirical dot-product penalty is weaker still: a skew-symmetric
rotation \(J\) gives \(e_h^TJe_h=0\) while \(Je_h\) preserves head information
when \(J\) is invertible. In the independent isotropic Gaussian population,
complete centered cross-covariance satisfies
\(\operatorname{Cov}(v,e_h)=VA_h/d\), so making that entire matrix exactly
zero would remove the linear head term. Finite-sample scalar orthogonality,
conditional correlations, or approximate covariance penalties do not establish
this identity. Compare hard masks and penalties at matched effective rank,
writer cost, and state size; predict reduced endpoint-difference alignment and
possibly poorer single-hop storage if useful head components were removed.
No writer change is implemented or preregistration amended in this task.

### 10.6 P1-L analog: token positions do not guarantee factorization

**Observed code contract.** [The LLM interface](reports/WP-E-interface.md),
`llm/inject.py`, and `core/wrapper.py` pass the layer-14 self-attention output
*before* `o_proj` into the shared \(A,K,V\). `fact_final` chooses one position
per fact: the renderer marks `span.token_end-1` on the full span including its
trailing newline, so this need not be the tail-name token. `all_tokens` chooses
every valid position. These are write masks, not information restrictions.
At a fact-final or tail position the attention result can
depend on the head, relation, earlier facts and the current tail through causal
attention and preceding layers. A value can therefore carry head information
even at the tail-name position. `all_tokens` additionally writes
head/prefix positions and changes write count, forgetting, and order effects.
Dropping backbone KV in memory-only does not remove information already in
\(M\). No claim is made that the symbolic linear telescoping identity holds
for contextual Qwen features or its full-vocabulary decoder.

**Design options, predictions only.** A coherent strong analog buffers a key
from the head/relation prefix position immediately **before** the current tail,
and forms the value from the tail's raw single-token embedding (or a tail-only
encoder). Causality prevents that prefix activation from seeing the current
tail token; raw tail embeddings exclude contextual head information. One paired
write per fact needs explicit span alignment, a temporary key buffer, resets,
and a declared query/key representation map. Prefix features may still contain
earlier facts, so a strict head/relation-only key also needs per-fact attention
isolation or a separate prefix encoder. A lower-layer contextual tail state
alone is not a proof of value independence: even its first attention layer can
mix the head. A position before the tail also cannot by itself make a *tail*
value pure if the value is still taken from the contextual tail state.

A softer analog keeps contextual values but penalizes their conditional
predictability of the head given the tail, or projects out a measured head
subspace. This may suppress linear leakage while retaining useful semantics;
nonlinear head information can remain. Hard symbolic block masks have no exact
semantic equivalent in an already mixed feature vector. Predict that raw-tail
values reduce head reconstruction and endpoint-like global statistics, possibly
at the expense of relation/context fidelity and T1 EM. Compare paired rewiring
and hop/depth profiles in memory-only; retain context-retained as a separately
named protocol because backbone attention can solve the relation task itself.
Choose any follow-up using validation data and explicit cost accounting; this
note provides designs and falsifiable predictions, not new P1-L results.

## Validation and interpretation

[tests/test_theory_examples.py](../tests/test_theory_examples.py) checks small,
deterministic algebraic examples and the consequential counterexamples using
FP64. Execution is through `scripts/submit.py` on a Miyabi-G compute node. The
original checks are recorded in [the WP-T report](reports/WP-T.md); Proposition
10's checks and descriptive measurements are recorded in
[the WP-T2 report](reports/WP-T2.md), job `3712948.opbs` on `mg0851`
(36 tests passed, including the 16 original cases).
These checks support the derivations and implementation correspondence. They
are not P1-S training experiments or evidence that G1/G2 pass.
