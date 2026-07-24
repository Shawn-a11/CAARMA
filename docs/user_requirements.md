# User Requirements

## Experiment Sequence

1. Complete the existing q-only leakage control.
2. Starting from the reported 3.45 Projection-D branch, add only an explicit
   matched-versus-mismatched prototype loss.
3. Add SLERP initialization for `W_syn` only if step 2 improves the result.
4. Consider natural-cluster pair generation only after step 3.

## Isolation Constraints

- Keep CRP alpha, top-k pairing, SLERP, persistent `W_syn`, joint-L_syn,
  discriminator architecture, optimizers, and data settings unchanged.
- Do not include Fisher-UCB, prototype-only negatives, gender filtering,
  boundary utility, or natural clusters in the explicit matching branch.
- Do not commit logs, checkpoints, or result artifacts.
