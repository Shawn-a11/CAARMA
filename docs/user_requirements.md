# User Requirements

## Fisher-UCB CRP Reuse

- Base the experiment on the reported 3.57 CRP Persistent Synth v2 code.
- Keep top-k pairing, SLERP, joint-L_syn, persistent W_syn, CRP alpha, and the
  spectral MLP discriminator unchanged.
- Replace popularity-weighted reuse with value-aware reuse only.
- Compute a bounded boundary utility from the assigned synthetic class and its
  strongest competitor; do not add percentile windows or tuned thresholds.
- Use standard UCB1 to balance useful-class reuse and under-visited exploration.
- Keep the CRP new-versus-reuse probability unchanged for causal attribution.
- Synchronize persistent class semantics and utility state across DDP ranks.
- Reuse the same synthetic pair plan in the D-step and M-step.
- Do not include experiment logs, checkpoints, or result artifacts in commits.

## Experiment Boundary

- This branch does not add gender constraints, natural clusters, Projection-D,
  boundary-window filtering, or a minimum-visit occupancy rule.
- Full four-GPU training and performance evaluation run on the SSH server.
