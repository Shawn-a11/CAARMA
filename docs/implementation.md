# Fisher-UCB CRP Implementation

## Controlled Change

The CRP creation decision remains:

$$
P(\mathrm{new})=\frac{\alpha}{N_i+\alpha}.
$$

Only selection among existing classes changes. The v2 visit-count weighting is
replaced by standard UCB1.

## Reward

For a synthetic sample assigned to persistent class $c$:

$$
z_c=s(\cos(e_{\mathrm{syn}},W_{\mathrm{syn}}[c])-m),
\qquad
z_-=\max_{k\ne c}z_k,
$$

$$
p_c=\sigma(z_c-z_-),
\qquad
u_c=4p_c(1-p_c).
$$

Positive learning progress rescues a difficult class that is improving:

$$
g_c=\max(0,p_c^{(t)}-p_c^{(t-1)}),
\qquad
r_c=\max(u_c,g_c).
$$

The bounded reward $r_c\in[0,1]$ is accumulated per persistent class. Reuse is:

$$
c^*=\arg\max_c\left[
\bar r_c+\sqrt{\frac{2\log T}{n_c}}
\right].
$$

## Code Flow

1. `helper/synth_table.py` deterministically reserves pair-to-column semantics,
   recycles unvisited reservations at epoch boundaries, tracks bounded rewards,
   selects reuse classes with UCB1, and all-reduces visit/reward deltas.
2. `criterion/amsoftmax_mix_gan.py` caches the D-step pair plan, reuses it in the
   M-step, and derives Fisher utility from joint-L_syn logits.
3. `train.py` commits synchronized state once after joint-L_syn on every rank.
4. `config.yaml` enables only `reuse_policy: fisher_ucb` over the CRP v2 setup.

The same code exposes `--reuse-policy popularity` and `--save-dir` so the
corrected matched control shares every DDP and pair-plan fix with the method.

## DDP Invariant

Every pair has the same `W_syn` column on every rank. Local memory banks may
differ, but successful visit and reward deltas are globally reduced before the
next pair-selection step.
