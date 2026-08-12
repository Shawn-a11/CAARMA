# Persistent Local Virtual Speakers with Prototype-Conditioned Adversarial Learning

## Motivation

Embedding-space class augmentation can increase speaker diversity without
recording new speakers. CAARMA establishes this direction by mixing speaker
embeddings into synthetic classes and adversarially encouraging synthetic
embeddings to resemble real ones. Two unresolved mismatches remain.

First, a one-shot mixed sample has no identity that survives across batches, so
the model cannot revisit multiple utterance-level realizations of the same
virtual speaker. Second, an unconditional discriminator estimates marginal
realism, `D(e)`, but cannot test whether an embedding is compatible with the
real or synthetic class prototype assigned to it. Random or globally nearest
mixing can also connect speakers from an incoherent region of the current
speaker manifold.

The research question is therefore not merely how to generate more synthetic
embeddings, but how to construct a locally valid virtual speaker, preserve its
identity, and train the embedding-prototype relation repeatedly.

## Method

### 1. Locally compatible parent candidates

Let `W_i` be the normalized AM-Softmax prototype of real speaker `i`. At each
epoch, deterministic spherical k-means assigns every real prototype to a
speaker group `g(i)`. The candidate parents of anchor `i` are the `k` most
similar prototypes inside that group:

$$
\mathcal{N}_i
=
\operatorname{TopK}_{j\ne i,\,g(j)=g(i)}
\cos(W_i,W_j).
$$

The cluster is a data-driven compatibility region; cosine top-k then retains
local ambiguity within that region. This is more specific than either global
top-k or uniform sampling from the whole cluster.

### 2. Persistent virtual-speaker identity

Each selected unordered parent pair `{i,j}` indexes a virtual class `c` with a
stable learnable prototype `W_syn[c]`. A CRP-inspired registry decides whether
to create an eligible new pair or revisit an existing pair. For the existing
classes accessible to anchor `i`, denoted by `C_i`, the implementation uses

$$
P(\mathrm{new}\mid i)
=
\frac{\alpha}{\alpha+\sum_{c\in\mathcal{C}_i} n_c},
\qquad
P(c\mid\mathrm{reuse},i)
=
\frac{n_c}{\sum_{r\in\mathcal{C}_i} n_r}.
$$

This is a CRP-inspired creation/reuse scheduler, not a claim of full Bayesian
posterior inference. When class `c` is first allocated, its prototype is
initialized from its parent geometry:

$$
W_{\mathrm{syn}}[c]
\leftarrow
\operatorname{SLERP}(W_i,W_j,\tfrac{1}{2}).
$$

On each revisit, different utterance embeddings from the same parents produce
a new realization of that stable identity:

$$
e_{\mathrm{syn}}^{(c)}
=
\operatorname{SLERP}(e_i,e_j,t).
$$

The joint AM-Softmax objective classifies this realization against the real
prototypes and the active persistent synthetic prototypes.

### 3. Prototype-conditioned adversarial compatibility

The condition supplied to the discriminator is the assigned class prototype:

$$
q=
\begin{cases}
W_y, & e=e_{\mathrm{real}},\\
W_{\mathrm{syn}}[c], & e=e_{\mathrm{syn}}^{(c)}.
\end{cases}
$$

Instead of only asking whether `e` looks real, the projection discriminator
scores both marginal realism and embedding-prototype compatibility:

$$
D(e,q)
=
h(\phi(e))
+
\frac{\langle\phi(e),\psi(q)\rangle}{\sqrt{d_h}}.
$$

The inner product is the load-bearing term: it exposes the relation between an
embedding and its assigned identity directly, whereas a concatenation MLP must
infer that relation implicitly.

### 4. Training objective and exposure ratio

The encoder and prototypes are optimized by real-speaker classification,
persistent synthetic-class classification, and adversarial refinement:

$$
\mathcal{L}_{M}
=
\mathcal{L}_{\mathrm{real}}
+\lambda_{\mathrm{syn}}\mathcal{L}_{\mathrm{syn}}
+\lambda_{\mathrm{adv}}\mathcal{L}_{G}.
$$

The discriminator is trained separately on correctly conditioned real and
synthetic pairs. The synthetic/real sample ratio

$$
r_{\mathrm{syn}}
=
\frac{N_{\mathrm{synthetic\ samples}}}{N_{\mathrm{real\ samples}}}
$$

controls exposure, not the number of available registry columns. It is a
sensitivity variable rather than a method contribution.

## Why It Should Work

Persistence lets one virtual identity accumulate supervision from multiple
utterance-level realizations. Parent-SLERP initialization keeps that identity
in its intended parent neighborhood. Cluster-local top-k reduces invalid
parent combinations while retaining difficult local distinctions. Finally,
Projection-D prevents the adversarial branch from reducing the task to generic
real-looking embeddings by making the assigned prototype observable.

The components are complementary: the registry supplies a stable `q`, and the
projection discriminator makes that stable identity useful during adversarial
training.

## Falsification Plan

The minimal test is a locked, multi-seed ladder comparing one-shot
unconditional augmentation, persistent unconditional augmentation,
parameter-matched Concat-D, Projection-D, parent-SLERP initialization, and
cluster-local top-k. The load-bearing observable is the matched-condition gap

$$
\Delta_q
=
D(e,q^+)-D(e,q^-),
$$

where `q+` is the assigned prototype and `q-` is a shuffled prototype from the
same real or synthetic bank. If the mechanism is correct, Projection-D should
increase `Delta_q` and reduce official-test EER relative to matched Concat-D.
Shuffling `q` is the negative-control intervention: it should remove most of
the Projection-D gain. A q-only discriminator should not recover the full
gain, and random-four sampling inside each cluster should be weaker than
cosine-ranked top-four. If these predictions fail under matched seeds and a
locked recipe, the compatibility and local-geometry claims must be rejected.

## Evidence Status

Historical single-run results are consistent with the mechanism: 3.61 to 3.57
with persistence, 3.51 with Concat-D, 3.45 with Projection-D, 3.72 when
Projection-D is used without persistence, and 3.57/3.59 for shuffled-q/q-only
controls. The best PPU result is 3.25 for cluster-local top-four, while a
matched AutoDL rerun reached 3.52. These values motivate the proposal but do
not yet constitute a publication-grade main table because environments,
selection protocols, and seeds are not fully matched.

