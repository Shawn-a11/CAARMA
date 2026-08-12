# Idea: Real-Gradient-Aligned CRP Reuse

## Metadata

- ID: `IDEA-20260812-01`
- Review decision: accepted for diagnostic validation
- Accepted on: 2026-08-12
- Based on branch/commit: `researchstudio-method-ratio-plan` / `c1863f6`
- Supersedes: none
- Implementation branch: not created
- Evidence level: theoretical proposal supported by completed negative controls;
  the proposed mechanism has not yet been run

## Problem

Persistent CAARMA currently uses visit frequency or proxy utility signals to
decide which synthetic speaker class should be revisited. Completed experiments
show that these proxies are insufficient:

- Boundary Utility reached EER 3.56, approximately matching CRP v2 rather than
  producing a clear gain;
- Fisher-UCB reached EER 3.57, showing that synthetic classification
  uncertainty is not sufficient;
- powered CRP variants changed occupancy concentration without establishing a
  reliable improvement mechanism;
- prototype-only virtual speakers reached EER 3.97, showing that a hard
  negative direction without a learnable synthetic identity is insufficient.

The unresolved question is whether training synthetic class `c` produces an
update that helps the real-speaker objective. Popularity, geometric ambiguity,
and synthetic classification uncertainty do not measure this quantity.

## Core Mechanism

Keep the CRP new-class probability unchanged. Change only the conditional
choice of which existing synthetic class to reuse.

Let `theta_s` denote a small parameter subspace shared by real and synthetic
speaker learning. Define

$$
g_{\mathrm{real}}=
\nabla_{\theta_s}\mathcal{L}_{\mathrm{real}},
\qquad
g_c=\nabla_{\theta_s}\mathcal{L}_{\mathrm{syn},c}.
$$

The task-alignment score is

$$
a_c=
\frac{\langle g_{\mathrm{real}},g_c\rangle}
{\lVert g_{\mathrm{real}}\rVert_2
 \lVert g_c\rVert_2+\epsilon}.
$$

Positive alignment indicates that the synthetic update is locally compatible
with reducing the real-speaker loss. Negative alignment indicates gradient
interference.

Maintain a running non-negative benefit estimate:

$$
r_c=\max(0,a_c),
\qquad
\bar a_c=\frac{1+\sum_{t=1}^{n_c}r_{c,t}}{1+n_c}.
$$

Retain the standard CRP create-versus-reuse law:

$$
P(\mathrm{new})=\frac{\alpha}{N+\alpha},
\qquad
P(\mathrm{reuse})=\frac{N}{N+\alpha}.
$$

When reuse is selected, use benefit-weighted occupancy:

$$
P(c\mid\mathrm{reuse})=
\frac{n_c\bar a_c}{\sum_r n_r\bar a_r}.
$$

If every `bar_a_c` equals one, the method exactly recovers popularity CRP.
This gives a direct control without changing class creation or synthetic data
volume.

## Derivation

Consider a small gradient step using synthetic class `c`:

$$
\theta_s'=\theta_s-\eta g_c.
$$

A first-order Taylor expansion of the real-speaker loss gives

$$
\mathcal{L}_{\mathrm{real}}(\theta_s')
\approx
\mathcal{L}_{\mathrm{real}}(\theta_s)
-\eta\langle g_{\mathrm{real}},g_c\rangle.
$$

The inner product therefore estimates the local contribution of the synthetic
update to the real objective. Cosine normalization removes gradient-magnitude
effects and makes the signal directional rather than another hardness score.

This is a local first-order argument. It does not guarantee long-horizon EER
improvement, so the alignment-to-loss-delta diagnostic is load-bearing.

## Difference from Existing Experiments

| Existing experiment | What it measures | Why this idea is different |
|---|---|---|
| Popularity CRP | past visits `n_c` | measures predicted contribution to real loss |
| Powered CRP | flattened visit counts `n_c^beta` | changes utility rather than occupancy exponent |
| Boundary Utility | parent intrusion, manifold proximity, boundary ambiguity | measures optimization effect rather than static geometry |
| Fisher-UCB | synthetic target-versus-competitor uncertainty plus exploration | distinguishes uncertainty from real-task benefit |
| Projection-D | embedding-prototype compatibility | selects which persistent identity to revisit; it does not replace Projection-D |
| Natural-cluster E3 | candidate parent locality | operates after candidate construction and keeps E3 fixed |

## Prior-Art Collision

Gradient-based data valuation is established prior art. The defensible novelty
is not the invention of gradient alignment. It is the decomposition of
persistent synthetic-class creation and reuse in CAARMA, with reuse valued by
its estimated contribution to the real-speaker objective.

Closest sources:

- GRAD-MATCH: <https://proceedings.mlr.press/v139/killamsetty21a>
- Learning to Reweight Examples for Robust Deep Learning:
  <https://proceedings.mlr.press/v80/ren18a>
- GradAlign: <https://arxiv.org/abs/2602.21492>
- CAARMA: <https://arxiv.org/abs/2503.16718>

Literature novelty assessment: medium. The speaker-specific persistent
pseudo-class formulation and controlled CRP decomposition must carry the claim.

## Minimal Decisive Experiment

- Locked baseline: runtime-matched E3 configuration
- Sole changed variable: popularity reuse versus gradient-aligned reuse
- Candidate generation: natural cluster plus within-cluster cosine top-4
- Synthetic initialization: parent SLERP
- Discriminator: Projection-D
- Synthetic/real ratio: identical across arms
- Diagnostic parameter subspace: final shared embedding projection only
- Dataset and trials: fixed development protocol for selection; official test
  only after the decision is locked
- Seeds: one matched seed for the diagnostic; three matched seeds for the final
  claim
- Metrics: alignment-to-one-step-loss-delta correlation, EER, minDCF,
  occupancy statistics, runtime, and peak memory

First run an offline diagnostic from one E3 checkpoint:

1. compute `a_c` for a fixed set of synthetic classes;
2. take a temporary small update using each class;
3. recompute the same held-out real mini-batch loss;
4. test whether alignment predicts the measured loss reduction.

Do not start a 30-epoch experiment unless this diagnostic succeeds.

## Controls

- Positive control: popularity E3 with identical runtime and candidate policy
- Negative control: preferentially reuse bottom-alignment classes
- Shuffled control: permute alignment scores across class identities
- Compute-matched control: perform the same score computation but reuse by
  popularity

The create-versus-reuse decisions and number of synthetic samples must be
identical where possible, so the experiment isolates class-specific utility.

## Decision Criteria

### Success

- alignment is positively correlated with measured one-step real-loss
  reduction;
- top-alignment classes have a better loss delta than bottom-alignment classes;
- the proposed reuse policy improves mean EER by at least 0.08 absolute over
  runtime-matched E3 across three seeds;
- minDCF does not systematically regress;
- shuffled and bottom-alignment controls lose the gain;
- compute overhead is reported and remains acceptable.

The 0.08 threshold is a project decision threshold, not a theoretical result.

### Kill

Terminate this direction if either condition holds:

1. alignment does not predict one-step real-loss change; or
2. correct class-specific alignment performs no better than shuffled alignment
   under matched compute.

Do not rescue a failed result by immediately adding temperatures, caps,
percentile windows, or additional geometric thresholds.

## Results

Not yet implemented or evaluated.

## Final Claim Boundary

If validated, the supported claim is:

> Persistent synthetic speaker classes can be revisited according to their
> predicted contribution to real-speaker optimization, rather than raw
> popularity or synthetic uncertainty alone.

Do not claim that gradient alignment is newly invented, that local alignment
guarantees verification generalization, or that the method is universally
better before testing multiple seeds and evaluation protocols.

