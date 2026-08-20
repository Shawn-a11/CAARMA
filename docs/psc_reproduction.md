# PSC Bridges-2 Reproduction

This branch starts from the original CAARMA `origin/main` commit `026e145`.
It does not contain CRP, persistent synthetic classes, SLERP, Projection-D,
Natural Cluster, or any later experiment module.

## Preserved Method

- MFA-Conformer speaker encoder.
- Original one-shot Euclidean midpoint mixup (`0.5 * (x_i + x_j)`).
- Batch-local nearest-speaker pairing.
- Original AM-Softmax real and synthetic losses.
- Original HuBERT-large mixup discriminator using layers 7, 9, 11, and 12.
- Original source training state machine: five generator updates per
  discriminator update through epoch 15, then alternating one-to-one updates.
- Original adversarial weighting control law.

## Reproduction-Only Corrections

The original public main branch cannot launch the reported training directly:

1. `train.py` hard-codes an old PSC path and calls `validate` while every
   `trainer.fit` call is commented out.
2. `criterion/build_criterion.py` hard-codes 1,211 speakers regardless of the
   configured dataset.
3. `amsoftmax_mix_gan.py` allocates an unused tensor directly on `cuda:0`.
4. `model/discriminator_mix.py` imports a missing, unused module.
5. evaluation paths are concatenated as strings and require a trailing slash.
6. DDP validation merges rank-local `batch_idx` values without rank offsets.

This branch fixes only these execution and evaluation defects. The source
training algorithm is otherwise preserved. DDP optimizer stability changes are
not silently imported from later experiment branches. If source-exact DDP
hangs on Bridges-2, that failure must be recorded before a separately named
runtime-stability control is introduced.

The public source and paper disagree on several hyperparameters and update
rules. See [`source_paper_discrepancies.md`](source_paper_discrepancies.md).
The first production arm preserves the public source training state machine;
later discrepancy arms must change one item at a time.

## Current Scope

`config_psc_vox1.yaml` is the only production configuration in this branch.
The current objective is the paper's small-scale VoxCeleb1 experiment and its
reported 3.09% VoxCeleb1-O EER. VoxCeleb1+2 and all later methods are deferred
until this reproduction is resolved.

The config uses 50 samples per GPU, matching the paper's implementation
details. Four GPUs therefore produce an effective batch size of 200.

The only permitted dataset source for this reproduction is
`/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1`. The branch does not
inspect or depend on any other user's project directory. Dataset structure,
trial resolution, and train/test separation are derived from this shared
standard dataset and recorded by the audit tools.

The shared copy uses one unified `VoxCeleb1/wav` directory. The manifest job
downloads the cleaned VoxCeleb1-O trial list from OpenSLR, extracts its test
speaker IDs, and excludes them while building the 1,211-speaker training
manifest. The same unified `wav` directory is used as the evaluation root.

The `cis220031p` allocation is GPU-only. Audit, environment creation, and
manifest preparation therefore use `GPU-shared` with one `v100-16`; submitting
them to `RM-shared` fails before execution with `Invalid qos specification`.

## Required Environment

Source `scripts/psc/env.example`; it points both train discovery and evaluation
to the unified `wav` directory. Test speakers are excluded deterministically
from the official trial list before the manifest is accepted.

The HuBERT model must be available in `CAARMA_HUBERT_CACHE` before a production
job if compute nodes cannot reach Hugging Face.

## Execution Order

```bash
mkdir -p /ocean/projects/cis220031p/sge2/logs

sbatch --export=ALL scripts/psc/audit_voxceleb.slurm
sbatch --export=ALL scripts/psc/create_environment.slurm
sbatch --export=ALL scripts/psc/environment_smoke.slurm
sbatch --export=ALL scripts/psc/prepare_vox1_manifest.slurm
sbatch --export=ALL scripts/psc/train_smoke.slurm
sbatch --export=ALL scripts/psc/train_vox1.slurm
```

Do not add VoxCeleb2 or method changes until the VoxCeleb1 reproduction
protocol and runtime are frozen.

`requirements_psc.txt` replaces the public requirements file for this runtime.
The public pins combine old NumPy with a much newer SciPy release and do not
form a reliable Python 3.10 environment. The PSC file keeps the model/library
generation close to the public repository while using mutually compatible
scientific Python versions. PyTorch and torchaudio CUDA wheels are installed
separately by `create_environment.slurm`.
