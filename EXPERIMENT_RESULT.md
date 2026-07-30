# Experiment Result

- Experiment: E1 SLERP-Initialized Projection-D
- Branch: `exp/innovation-slerp-initialized-projection-crp-persistent-synth-ddp`
- Method commit: `8e475dc`
- Evaluation protocol: official VoxCeleb1 `veri_test2`
- Best epoch: historical record; exact epoch unavailable
- Best cosine EER: **3.47%**
- cosine minDCF at 1e-2: unavailable; original local log was removed
- cosine minDCF at 1e-3: unavailable; original local log was removed
- Status: valid completed PPU result
- Recorded: 2026-07-30

## Method

E0 with every newly reserved persistent synthetic prototype initialized from
the spherical midpoint of its two parent speaker prototypes. Projection-D,
CRP reuse, and the remaining training configuration are retained.

## Provenance

This file stores a manually verified experiment summary only. Checkpoints,
training logs, datasets, and generated outputs are intentionally not committed.
