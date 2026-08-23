# PSC Joint-Lsyn MLP-D Batch-Size 13 Experiment

## Source arm

- Branch: `exp/psc-bridges2-joint-lsyn-mlpd-reproduction@37d41df`
- Slurm job: `44072499`
- Result: 3.61% test-selected VoxCeleb1-O EER (historical AutoDL reproduction).

## Delta

Only the per-rank batch size is changed:

- Per-rank `batch_size`: `50` → `13`
- 4 DDP ranks ⇒ effective global batch size ~52 (closest practical setting to the
  paper's stated global batch size of 50).

All other hyperparameters are preserved from the source arm:

- MFA-Conformer encoder;
- plain embedding MLP discriminator (`192 -> 256 -> 128 -> 1`);
- SLERP synthetic embeddings and joint real/synthetic AM-Softmax;
- 30 epochs, seed 42, `init_lr` 0.001, AM-Softmax margin 0.2 / scale 30;
- four-GPU DDP on Bridges-2 V100-32.

## Goal

Isolate whether the effective global batch-size mismatch explains the
reproduction gap to the 3.09 target and/or shifts the main-chain ladder.

## Reporting rule

Report in the same numerical-reproduction lane as the source arm: VoxCeleb1-O
test-selected EER (and minDCF if available).
