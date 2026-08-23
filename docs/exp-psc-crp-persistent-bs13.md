# PSC experiment: align effective global batch size to ~50

## Source arm

- Source branch: `exp/psc-crp-persistent-synth` @ `db0d3c275d45be51890a53ecdb4b589c4c3cfde8`
- Source job: `44190687` (`caarma-crp`, 4x V100-32, GPU-shared, 12h)
- Source result (test-selected checkpoint, VoxCeleb1-O):
  - **EER 3.4%**
  - **minDCF (p=1e-2) 0.3618**
  - **minDCF (p=1e-3) 0.5375**

## Delta

Only the per-rank batch size changes:

- `batch_size`: `50` -> `13`
- Effective global batch size: `13 x 4 DDP ranks = 52`, the closest
  achievable integer to the paper's stated global batch size of `50`.
- All other hyperparameters (learning rate, epochs, model, criterion,
  `lambda_adv`, CRP/persistence settings, discriminator, seed, etc.) are
  unchanged from the source arm.

## Goal

Isolate whether the effective global batch-size mismatch explains part of
the reproduction gap to the paper's reported 3.09 EER and/or shifts the
main-chain reproduction ladder.

## Reporting rule

Report in the same numerical-reproduction lane as the source arm:
VoxCeleb1-O, test-selected checkpoint.
