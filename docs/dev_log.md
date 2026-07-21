# Development Log

## 2026-07-21 - Fisher-UCB Reuse

- Created `exp/innovation-fisher-ucb-crp-persistent-synth-ddp` from the remote
  CRP v2 code commit, excluding local `output/` result artifacts.
- Added deterministic pair-column reservation for DDP class identity; unvisited
  reservations are recycled while created pseudo-classes remain persistent.
- Added Fisher boundary utility, positive learning progress, and UCB1 reuse.
- Kept CRP creation probability and all generation/loss settings unchanged.
- Added one pair-selection plan shared by discriminator and model updates.
- Added DDP reduction for visit, reward, probability, and activation state.
- Added checkpoint persistence for utility statistics.
- Added command-line overrides for the corrected popularity control.
- Verification status: 8 focused tests, gradient smoke test, syntax checks, and
  a two-process Gloo synchronization smoke test pass locally.

## Running

Use the existing four-GPU `train.py` entry point with this branch's
`config.yaml`. The checkpoint directory is
`caarma_mfa_ckpts_fisher_ucb_crp_persistent_synth_ddp`.

For the matched infrastructure control, run with
`--reuse-policy popularity` and a separate `--save-dir`.
