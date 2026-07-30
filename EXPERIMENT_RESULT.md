# Experiment Result

- Experiment: Cluster-Random-4 CRP with SLERP-Initialized Projection-D
- Branch: `exp/ablation-cluster-random4-slerp-projection-persistent-synth-ddp`
- Method commit: `bccb24e`
- Evaluation protocol: official VoxCeleb1 `veri_test2`
- Training progress at record time: epoch 19 of 30, step 330 of 744
- Best epoch so far: **18**
- Best cosine EER so far: **3.52%**
- cosine minDCF at 1e-2: pending for the epoch-18 checkpoint
- cosine minDCF at 1e-3: pending for the epoch-18 checkpoint
- Status: provisional; training in progress
- Recorded: 2026-07-30

## Method

Matched E3 ablation that retains natural clusters, popularity CRP,
parent-SLERP synthetic prototype initialization, Projection-D, and a maximum of
four candidates per anchor. It replaces within-cluster cosine top-4 ranking
with deterministic uniform random-4 sampling without replacement.

## Interpretation Boundary

This provisional result must not be used as the final ablation value until all
30 epochs finish. Comparison with E3 isolates whether nearest-neighbour ranking
inside the natural cluster is necessary.

## Provenance

Only the manually verified summary is committed. Checkpoints, logs, datasets,
and generated outputs remain external.
