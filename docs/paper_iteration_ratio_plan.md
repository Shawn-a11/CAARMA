# Paper Iteration Plan: Persistent Local Virtual Speaker Learning

Updated: 2026-08-12

Research Studio artifacts:

- `docs/researchstudio/idea_card_en.md`
- `docs/researchstudio/idea_card_zh.md`
- `docs/researchstudio/reviewer_audit.md`

The Research Studio audit rates the idea at 75/100 on intrinsic quality but
finds high-to-medium prior-art overlap at the component level. The novelty must
therefore be stated as an interaction: persistent pair-indexed identity enables
assigned-prototype adversarial compatibility, and cluster-local geometry makes
the parent construction valid. CRP, SLERP, Projection-D, and clustering are not
individually novel claims.

## 1. Current Verdict

The current evidence is closest to an ICASSP paper. It is not yet sufficient
for an ICLR submission.

The paper should not be framed as a sequence of CRP, SLERP, clustering, and
discriminator tricks. The defensible problem is:

> CAARMA creates synthetic embeddings, but a temporary synthetic sample has no
> stable identity, and an unconditional discriminator only checks marginal
> realism. We instead maintain reusable virtual speaker identities and train
> them using prototype-conditioned compatibility, while restricting their
> parents to a locally coherent region of speaker space.

The three method components are therefore:

1. **Persistent identity:** CRP-style creation and reuse maintains a stable
   virtual class prototype `W_syn[c]` across batches.
2. **Prototype compatibility:** Projection-D scores whether an embedding is
   compatible with its assigned real or synthetic prototype, not only whether
   it looks marginally real.
3. **Local geometric construction:** parent-SLERP initialization and
   natural-cluster-constrained local top-k pairing reduce invalid or misplaced
   virtual speakers.

The synthetic/real sample ratio is an analysis variable, not a fourth method
contribution.

## 2. Evidence Audit

Results below are grouped by provenance. Values from different runtime
environments are not treated as paired comparisons.

### 2.1 Historical method-development evidence

| Experiment | EER (%) | Evidence supported | Limitation |
|---|---:|---|---|
| Joint-Lsyn MLP-D baseline | 3.61 | Starting point | Historical recipe |
| CRP persistent v2 | 3.57 | Persistence may help over one-shot augmentation | Small, single-run gain |
| CRP + Concat-D | 3.51 | Supplying prototype identity helps | Relation learned implicitly |
| CRP + Projection-D | 3.45 | Explicit embedding-prototype compatibility is promising | Historical single run |
| One-shot + Projection-D | 3.72 | Projection alone is insufficient without persistent identity | Historical control |
| Q-shuffled Projection-D | 3.57 | Correct assignment of `q` matters | Needs matched seeds |
| Q-only D | 3.59 | Prototype alone does not explain the 3.45 result | Prototype-type leakage still exists diagnostically |

This block supports the interaction between persistence and class-conditioned
discrimination. It does not yet prove a final publication result because the
experiments were not all replayed under one locked recipe with multiple seeds.

### 2.2 PPU geometry and candidate-policy evidence

| Experiment | EER (%) | minDCF 1e-2 / 1e-3 | Interpretation |
|---|---:|---:|---|
| E0: Xavier `W_syn` + Projection-D | 3.52 | 0.3840 / 0.5325 | Initialization control |
| E1: parent-SLERP initialized `W_syn` | 3.47 | unavailable | Small initialization benefit |
| E2: powered reuse, beta=0.75 | 3.50 | 0.3629 / 0.4189 | Flattening popularity did not improve E1 |
| Powered reuse, beta=0.5 | 3.43 | 0.3432 / 0.5014 | No stable monotonic beta story |
| Pure natural-cluster candidates | 3.38 | 0.3456 / 0.4388 | Cluster validity has some value |
| E3: cluster-constrained local top-4 | 3.25 | 0.3272 / 0.4141 | Best historical result; local ranking inside clusters appears important |
| Cluster-random-4 | 3.52 provisional record | incomplete | Supports local cosine ranking, but final artifact must be recovered |

The E3 result is currently a hypothesis-generating result, not the final main
result, because AutoDL has not reproduced it.

### 2.3 Matched AutoDL ratio evidence

| Synthetic/real sample ratio | Best EER (%) | Interpretation |
|---:|---:|---|
| 1.0 | 3.52 | Current AutoDL control |
| 0.5 | 3.57 | No observed gain from halving synthetic exposure |

The occupancy timelines verify that the implementation changes the requested
sample count correctly:

- ratio 1.0: approximately 148,644 synthetic visits per epoch;
- ratio 0.5: approximately 74,324 synthetic visits per epoch.

Both runs eventually occupy all 4,844 registry columns and have similar Gini,
effective-class ratio, and top-10% visit share. The main difference is that
occupancy counts are almost exactly halved at ratio 0.5. Therefore the result
does **not** show improved CRP balance. It only shows reduced training exposure
for each persistent pseudo-class.

The 0.05 EER difference is too small for a single-run claim. The valid statement
is:

> Under the matched AutoDL setup, reducing the synthetic/real sample ratio from
> 1.0 to 0.5 did not improve the best observed EER (3.52% versus 3.57%).

Do not compare either AutoDL number directly with PPU E3 3.25 as a ratio effect.

### 2.4 Baseline tuning integrity

The tuned MLP-D configuration reached a recorded 3.20% EER, but it was selected
through repeated official `veri_test2` evaluation. It must be labeled
**test-selected**, not an untouched final-test result. Its epoch metadata and
matching minDCF artifact are also incomplete.

This creates a central publication gate: the proposed method must be replayed
under the same locked tuned recipe and selected by a development protocol.
Otherwise 3.20 and 3.25 cannot support a fair main-table comparison.

## 3. Research Questions and Claims

### RQ1: Does a reusable virtual identity outperform temporary mixup?

Compare one-shot synthetic samples with persistent `W_syn[c]`, holding pair
generation, sample ratio, discriminator capacity, and training recipe fixed.

Claim allowed only if multi-seed improvement is positive:

> Persistent pseudo-classes turn synthetic mixup from transient perturbations
> into reusable identities that receive repeated class supervision.

### RQ2: Does the discriminator use embedding-prototype compatibility?

Compare unconditional D, parameter-matched Concat-D, Projection-D, shuffled-q,
and q-only controls.

Claim allowed only if Projection-D beats matched Concat-D and shuffled-q:

> Explicit compatibility between an embedding and its assigned prototype is
> more useful than marginal realism or prototype access alone.

### RQ3: Which parent geometry produces useful virtual speakers?

Compare global top-k, pure natural cluster, cluster-random-4, and
cluster-local cosine top-4.

Claim allowed only if cluster-local top-4 is reproducible:

> Natural clusters provide compatibility, while local cosine ranking identifies
> informative parent pairs inside each compatible region.

### RQ4: How much synthetic exposure is useful?

Treat the sample ratio as sensitivity analysis. Screen ratios 0.5, 1.0, and
2.0 on development data. The current two-point evidence does not establish an
optimum.

## 4. Required Experiment Pipeline

### Gate 0: Reproducibility lock

Before new method runs, fix and record:

- code commit and clean `git diff`;
- Python, PyTorch, Lightning, CUDA, NCCL, torchaudio/soundfile versions;
- hashes of train CSV, trial list, and representative audio files;
- feature tensor checksum for a fixed audio batch;
- seed for Python, NumPy, Torch, CUDA, DataLoader workers, and CRP sampling;
- per-rank initialization checksum for model, discriminator, `W`, and `W_syn`;
- actual `Ns_over_B`, effective batch size, and registry statistics.

Run the same fixed-seed E3 job once on PPU and AutoDL. If the early loss curves
or feature checksums differ, resolve that difference before interpreting EER.

### Gate 1: Lock a publication baseline

Select hyperparameters on a speaker-disjoint development set. Then run the
locked baseline for seeds 42, 43, and 44. Evaluate official test trials only at
the development-selected epoch.

Report mean, standard deviation, each seed, and paired seed differences.

### Gate 2: Minimal causal ablation ladder

All rows must use the same locked recipe and three seeds:

| ID | Persistent identity | D form | SLERP init | Pair pool |
|---|---|---|---|---|
| A | no | unconditional | n/a | global local rule |
| B | yes | unconditional | Xavier | global top-k |
| C | yes | Concat-D | Xavier | global top-k |
| D | yes | Projection-D | Xavier | global top-k |
| E | yes | Projection-D | parent-SLERP | global top-k |
| F | yes | Projection-D | parent-SLERP | cluster-local top-4 |

Required controls:

- Projection-D without persistence;
- shuffled q;
- q-only D;
- cluster-random-4;
- pure cluster candidate pool;
- parameter count and epoch time for all discriminator forms.

Powered CRP and boundary-utility experiments should be reported as negative
exploration or appendix analysis, not as main method components.

### Gate 3: Generalization

For ICASSP, the minimum credible package is:

- VoxCeleb1-O, VoxCeleb1-E, and VoxCeleb1-H;
- VoxCeleb2-dev training or another larger-data setting requested by the
  professor;
- at least one second encoder, preferably ECAPA-TDNN or a ResNet-based speaker
  encoder;
- EER and minDCF under the same trial protocols.

For ICLR, add at least one additional dataset/domain such as CN-Celeb and show
that the mechanism transfers beyond one speaker encoder. Ideally demonstrate
the persistent virtual-class formulation on another metric-learning or
zero-shot comparison task.

## 5. ICASSP 2027 Paper Structure

ICASSP 2027 allows four pages for technical content and an optional fifth page
for references. The submission deadline is September 16, 2026. The paper must
therefore contain one compact method and one clean evidence chain.

### Proposed title

**Persistent Local Virtual Speakers with Prototype-Conditioned Adversarial Learning**

### Abstract

Five sentences:

1. Speaker verification benefits from class diversity, motivating embedding-space virtual speakers.
2. Existing one-shot synthetic classes lack stable identity, while marginal real/fake discrimination ignores class assignment.
3. Introduce persistent virtual identities, prototype-conditioned Projection-D, and cluster-local parent construction.
4. State the locked multi-seed VoxCeleb results and key ablations.
5. State the practical conclusion about reusable, locally valid virtual classes.

### 1. Introduction (about 0.55 page)

- Problem: limited speaker-class diversity for zero-shot verification.
- CAARMA insight: synthesize classes in embedding space.
- Gap: transient synthetic samples and unconditional realism do not enforce a coherent identity.
- Insight: a useful virtual speaker needs stable identity, assigned-prototype compatibility, and locally compatible parents.
- Contributions: exactly three bullets matching the method components.

### 2. Method (about 1.35 pages)

#### 2.1 Persistent virtual speaker registry

Define real prototypes `W_y`, synthetic prototypes `W_syn[c]`, CRP-inspired
creation/reuse, and SLERP-generated samples. Avoid claiming full Bayesian
posterior inference; call it a CRP-inspired persistent reuse process.

#### 2.2 Prototype-conditioned adversarial learning

Define

`D(e,q) = h(phi(e)) + <phi(e), psi(q)>`,

where `q=W_y` for real samples and `q=W_syn[c]` for synthetic samples. Explain
that the dot product explicitly scores embedding-prototype compatibility.

#### 2.3 Locally valid parent construction

Describe deterministic spherical clusters followed by within-cluster cosine
top-k selection. Initialize a new synthetic prototype from the parent SLERP
midpoint. State the complete objective once.

### 3. Experiments (about 1.7 pages)

#### 3.1 Setup

Datasets, encoder, official trials, seed protocol, EER/minDCF, parameter count,
and development-based checkpoint selection.

#### 3.2 Main results

One table covering VoxCeleb1-O/E/H and the larger training setting. Compare the
locked CAARMA/MLP-D baseline with the complete method.

#### 3.3 Ablation and analysis

One compact ablation table for persistence, discriminator form, SLERP init,
and cluster-local selection. Include q-shuffled and cluster-random controls.
Use a small ratio/occupancy plot only if space remains.

### 4. Conclusion (about 0.2 page)

State what is supported: persistent identity and assigned-prototype
compatibility improve virtual-class learning when parents are locally valid.
Do not claim that CRP balancing or ratio 0.5 is beneficial.

### Main figures and tables

1. **Figure 1:** one end-to-end pipeline with three highlighted innovations.
2. **Table 1:** main results across O/E/H and larger-data training.
3. **Table 2:** causal ablation ladder plus shuffled-q and cluster-random controls.
4. Optional small plot: EER versus synthetic/real ratio with occupancy summary.

## 6. ICLR 2027 Extension

ICLR 2027 accepts nine-page main submissions and emphasizes representation
learning and reproducibility. The current work is too application-specific and
too dependent on single-run VoxCeleb observations for this venue.

An ICLR version would need a broader formulation:

> Persistent Virtual-Class Learning for Metric Representations

Required additions:

1. Formalize virtual classes as persistent latent identities with a
   creation/reuse prior, rather than presenting CRP as a sampling heuristic.
2. Add an explicit matched/mismatched compatibility objective and analyze how
   it differs from real/fake discrimination.
3. Provide a theoretical argument or proposition connecting stable virtual
   identities to repeated positive supervision and reduced prototype drift.
4. Demonstrate transfer across at least two datasets, two encoder families,
   and preferably a non-speaker metric-learning task.
5. Report three to five seeds, confidence intervals, robustness to ratio,
   registry capacity, alpha, and cluster size.
6. Release reproducible code/configuration and provide a reproducibility
   statement.

Suggested nine-page structure:

1. Introduction
2. Related work: virtual classes, mixup for metric learning, conditional adversarial learning, nonparametric reuse
3. Problem formulation
4. Persistent virtual-class learning
5. Prototype compatibility and local construction
6. Theoretical/optimization analysis
7. Experiments across tasks and datasets
8. Ablations, limitations, and failure cases
9. Conclusion

## 7. Immediate Iteration Order

1. **Do not add another method module.**
2. Fix seeds and runtime fingerprints; reproduce ratio 1.0 across environments.
3. Lock the tuned baseline using development selection, not official-test tuning.
4. Replay the six-row causal ablation ladder with seeds 42/43/44.
5. Run the complete method on VoxCeleb1-O/E/H and the larger training set.
6. On the locked complete E3 setup, screen sample ratios `0.5`, `1.0`, and
   `2.0` on development data. The `2.0` point tests the professor's proposed
   one-real-to-two-synthetic regime. Advance only the development winner to
   three-seed official evaluation. Treat the existing ratio 0.5 result as a
   negative preliminary observation, not a final conclusion.
7. Decide venue:
   - submit to ICASSP if the complete method gives a reproducible gain and the
     ablations validate the three components;
   - pursue ICLR only after cross-dataset/task evidence and a stronger formal
   objective are available.

The two professor-requested experiments therefore have distinct roles:

| Experiment | Fixed method | Changed variable | Question answered |
|---|---|---|---|
| Synthetic exposure | Complete E3 | `r_syn` in {0.5, 1.0, 2.0} | How much synthetic training signal is useful? |
| Bigger data | Complete E3 with the development-selected ratio | VoxCeleb1 versus VoxCeleb2-dev training | Does the mechanism remain useful when real speaker diversity increases? |

The larger-data test is especially important: if the gain disappears when real
speaker diversity is high, the method should be positioned as a low-diversity
regularizer rather than a generally superior speaker-learning framework.

## 8. Submission Gate

The ICASSP paper is ready to write when all conditions below are true:

- one locked baseline and one locked complete method;
- at least three seeds for both;
- development-selected checkpoints and one-time official test evaluation;
- reproducible improvement on VoxCeleb1-O and no material regression on E/H;
- parameter-matched Concat-D versus Projection-D result;
- shuffled-q and one-shot controls support the compatibility/persistence claim;
- cluster-random and pure-cluster controls support local top-k;
- exact commit, environment, dataset hashes, and retained logs/checkpoints.

Until then, the correct status is **promising method-development evidence, not
yet a publication-grade final result**.
