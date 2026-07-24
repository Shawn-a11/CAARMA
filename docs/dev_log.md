# Development Log

## Explicit Prototype Matching

- Added reusable pairwise compatibility scores to Projection-D.
- Added batch-hard mismatched-prototype selection with class-ID masking.
- Added a margin-free matched-versus-mismatched softplus objective to both
  discriminator and main-model updates.
- Added matching loss and compatibility-gap diagnostics.
- Kept `W_syn` initialization and all CRP/pair-generation settings unchanged.

## Verification

- Python syntax compilation passed for the modified model, training code, and
  matching test.
- Pairwise compatibility indexing and the pairwise softplus ordering test
  passed by direct execution.
- The isolated YAML configuration check passed.
- `git diff --check` passed.
- Full `train.py` import was not available locally because
  `pytorch_lightning` is not installed; end-to-end DDP validation remains a
  server-side step.

## Run Configuration

Use `config.yaml`. Checkpoints are isolated under
`caarma_mfa_ckpts_explicit_match_projection_crp_persistent_synth_ddp`.
