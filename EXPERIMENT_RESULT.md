# Experiment Result

- Experiment: E0 Corrected Projection-D Control
- Branch: `exp/control-corrected-projection-mlpd-crp-persistent-synth-ddp`
- Method commit: `4adcc3b`
- Evaluation protocol: official VoxCeleb1 `veri_test2`
- Best epoch: historical record; exact epoch unavailable
- Best cosine EER: **3.61%**
- cosine minDCF at 1e-2: unavailable; original local log was removed
- cosine minDCF at 1e-3: unavailable; original local log was removed
- Status: valid historical PPU baseline
- Recorded: 2026-07-30

## Method

Corrected persistent CRP synthetic classes with Xavier-initialized synthetic
prototypes and projection-conditioned discrimination. This is the matched E0
control for testing parent-SLERP initialization in E1.

## Provenance

This file stores a manually verified experiment summary only. Checkpoints,
training logs, datasets, and generated outputs are intentionally not committed.
