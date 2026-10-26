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

## Validation and interpretation

[tests/test_theory_examples.py](../tests/test_theory_examples.py) checks small,
deterministic algebraic examples and the consequential counterexamples using
FP64. Execution is through `scripts/submit.py` on a Miyabi-G compute node; the
job, command, outcome, and scope are recorded in [the WP-T report](reports/WP-T.md).
These checks support the derivations and implementation correspondence. They
are not P1-S training experiments or evidence that G1/G2 pass.
