# exp/psc-paper-exact-bs13: Align effective global batch to ~50

## Source arm

- Branch: `exp/psc-paper-exact-repro-309`
- Base commit: `1ab649b`
- PSC job: `44192117`
- Slurm: `scripts/psc/train_vox1_paper_exact.slurm`
- Config: `config_psc_vox1_paper_exact.yaml`

## Delta

Change per-rank `batch_size` from `50` to `13`. With 4 DDP ranks this gives an
effective global batch size of `13 × 4 = 52`, the closest integer to the paper's
stated global batch size of `50`.

All other hyperparameters are kept identical to the source arm:
`init_lr 0.001`, `discriminator_lr 0.0002`, `epochs 30`, `am_margin 0.2`,
`am_scale 30`, `seed 42`, `lambda_adv_init 0.25`, etc.

## Goal

Isolate whether the mismatch in effective global batch size (source arm ran
`50` per rank = `200` global) explains the reproduction gap to the paper's
`3.09` result and/or shifts the main-chain reproduction ladder.

## Reporting rule

Same numerical-reproduction lane as the source arm: Vox1-O test-selected.
