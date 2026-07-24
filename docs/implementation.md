# Explicit Prototype Matching Implementation

## Controlled Change

The branch starts from the reported 3.45 Projection-D experiment. For each
embedding, the existing assigned prototype is the positive condition:

$$
q^+=W_y
\quad\text{or}\quad
q^+=W_{\mathrm{syn}}[c].
$$

The hardest different prototype in the current real-plus-synthetic condition
bank is selected as the negative condition:

$$
q^-=\arg\max_{q_k:\operatorname{id}(q_k)\ne\operatorname{id}(q^+)}
\langle\phi(e),\psi(q_k)\rangle.
$$

The only new objective is the margin-free pairwise softplus loss:

$$
\mathcal L_{\mathrm{match}}
=
\log\left(1+\exp\left[s(e,q^-)-s(e,q^+)\right]\right).
$$

It is added once to the discriminator objective and once inside the existing
generator/adversarial objective. No matching threshold, margin, temperature,
or loss-weight hyperparameter is introduced.

## Experiment Isolation

- Parent: `origin/exp/innovation-projection-mlpd-crp-persistent-synth-ddp`
- Current branch:
  `exp/innovation-explicit-match-projection-crp-persistent-synth-ddp`
- Unchanged: CRP, top-k=4, alpha=1.0, SLERP, joint-L_syn, persistent table,
  random `W_syn` initialization, Projection-D parameter count, optimizers,
  schedules, and data configuration.
- Deferred: SLERP-initialized `W_syn` and natural-cluster pair generation.
