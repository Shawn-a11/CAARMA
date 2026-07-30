# Experiment Result

- Experiment: E2 Powered CRP with SLERP-Initialized Projection-D
- Branch: `exp/innovation-powered-crp-slerp-projection-persistent-synth-ddp`
- Method commit: `99e4426`
- Evaluation protocol: official VoxCeleb1 `veri_test2`
- Best epoch: **27**
- Best cosine EER: **3.50%**
- cosine minDCF at 1e-2: **0.3629**
- cosine minDCF at 1e-3: **0.4189**
- Status: valid completed PPU result
- Recorded: 2026-07-30

## Method

E1 with sublinear CRP reuse mass, using `reuse_power=0.75`, to reduce
rich-get-richer occupancy concentration while retaining parent-SLERP synthetic
prototype initialization, global top-k candidates, and Projection-D.

## Validity Note

An earlier runtime-regression run mixed incorrect DDP and loader changes and is
excluded. The values above are from the valid PPU-compatible rerun.

## Provenance

Only the manually verified summary is committed. Checkpoints, logs, datasets,
and generated outputs remain external.
