# Experiment Result

- Experiment: E0 Corrected Projection-D Control
- Branch: `exp/control-corrected-projection-mlpd-crp-persistent-synth-ddp`
- Method commit: `4adcc3b`
- Evaluation protocol: official VoxCeleb1 `veri_test2`
- Best epoch: **12**
- Best cosine EER: **3.52%**
- cosine minDCF at 1e-2: **0.3840**
- cosine minDCF at 1e-3: **0.5325**
- Status: valid historical PPU baseline
- Recorded: 2026-07-30

## Method

Corrected persistent CRP synthetic classes with Xavier-initialized synthetic
prototypes and projection-conditioned discrimination. This is the matched E0
control for testing parent-SLERP initialization in E1.

## Provenance

This file stores the corrected manually verified historical summary. The
original local log is no longer available, so the values are preserved from
the previously reported result table. Checkpoints, logs, datasets, and
generated outputs are intentionally not committed.
