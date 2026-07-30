# Experiment Result

- Experiment: E3 Natural-Cluster-Constrained Top-k CRP with SLERP-Initialized Projection-D
- Branch: `exp/innovation-natural-cluster-slerp-projection-persistent-synth-ddp`
- Method commit: `7d318cc`
- Evaluation protocol: official VoxCeleb1 `veri_test2`
- Best epoch: **20**
- Best cosine EER: **3.25%**
- cosine minDCF at 1e-2: **0.3272**
- cosine minDCF at 1e-3: **0.4141**
- Status: valid completed PPU result
- Recorded: 2026-07-30

## Method

E1 with deterministic spherical natural clusters restricting each anchor's
candidate pool. The implementation then selects the four nearest speakers by
cosine similarity within that cluster before standard popularity CRP
create-or-reuse sampling. It is therefore a cluster-constrained top-k method,
not a pure cluster-only sampler.

## Validity Note

An earlier runtime-regression run mixed incorrect DDP and loader changes and is
excluded. The values above are from the valid PPU-compatible rerun. The best
epoch was observed on the official test protocol; a paper submission must use
a development set for checkpoint selection and evaluate the test set once.

## Provenance

Only the manually verified summary is committed. Checkpoints, logs, datasets,
and generated outputs remain external.
