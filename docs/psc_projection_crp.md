# PSC Port: Persistent CRP + Projection-D

## Source

- Source branch/commit: `exp/innovation-projection-mlpd-crp-persistent-synth-ddp@48f97b3`
- Ladder row: D (persistent CRP + Projection-D)

## Method (unchanged)

- Persistent synthetic speaker bank (CRP-assigned, `persistence: true`) with
  `synth_bank_size: 10`, `synth_max_factor: 4`, `slerp_t: 0.5`.
- Pairing strategy: Chinese Restaurant Process (`pair_strategy: "crp"`,
  `crp_alpha: 1.0`, `crp_topk: 4`).
- Projection discriminator (`discriminator_type: "projection"`,
  `ProjectionDiscriminator_spectral`): explicit embedding-prototype
  compatibility via a projection inner product,

  D(e, q) = h(φ(e)) + <φ(e), ψ(q)>,

  where q is the normalized target real/pseudo-speaker prototype.

## Historical reference

- Historical reference EER: **3.45%** (test-selected, historical runtime).

## PSC differences

- Config uses `${...}` environment interpolation (`CAARMA_TRAIN_MANIFEST`,
  `CAARMA_TRIAL_PATH`, `CAARMA_VOX1_EVAL_ROOT`, `CAARMA_RUN_ROOT`), enabled by
  the `load_config` env-expansion patch in `train.py`.
- Run dir: `${CAARMA_RUN_ROOT}/projection_crp_persistent/checkpoints`;
  SLURM entrypoint: `scripts/psc/train_projection_crp.slurm`
  (job-name `caarma-proj`, 4x v100-32, 12h).

## Results

PSC result reported separately (do not mix with the historical 3.45% EER).

- EER: _TBD_
- minDCF: _TBD_
