# PSC Source-Faithful Reproduction of Paper EER 3.09

## Goal

Reproduce the CAARMA paper's headline result on VoxCeleb1-O:

- **Target: EER 3.09** (baseline MFA-Conformer: 3.33)
- Model: MFA-Conformer with the full CAARMA training recipe —
  synthetic-speaker loss (L_syn), adversarial training (AT), and a
  HuBERT-based mixup discriminator operating on hidden layers
  h7 / h9 / h11 / h12 of `facebook/hubert-large-ls960-ft`.
- Source: arXiv 2503.16718, Tables 2–4.

The training entry point is `train_source_faithful.py` (a state-machine
port of the public CAARMA source) driven by `config_psc_vox1.yaml`
(MFA-CONFORMER, Fbank, AMSoftmaxGAN, margin 0.2 / scale 30, 30 epochs,
batch size 50, seed 42, 4 devices, 16-mixed precision).

## Failed job 44074599 and its root cause

PSC job **44074599**, running `scripts/psc/train_vox1.slurm`, failed in
3 seconds with exit code 2. Root cause: the launcher located itself via

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
```

Under `sbatch`, `BASH_SOURCE[0]` resolves inside the Slurm spool
directory (`/var/spool/slurm/d/job<JobID>/`), not the submission
directory. `source "$SCRIPT_DIR/common.sh"` therefore failed, the conda
environment was never activated, and `python` was not found (see the
job's stderr log).

## Fix

`scripts/psc/train_vox1.slurm` was rewritten as a self-contained
launcher following the same pattern as the paper-aligned launcher
(`scripts/psc/train_vox1_paper_aligned.slurm`):

- `REPO_ROOT="${CAARMA_JOB_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"`
  instead of any `BASH_SOURCE`-based self-location.
- Explicit `module load anaconda3`, `source /jet/home/sge2/caarma_psc_env.sh`,
  and `conda activate "$CAARMA_CONDA_ENV"`, then `cd "$REPO_ROOT"`.
- `test -f` / `test -d` guards for `train_source_faithful.py`,
  `config_psc_vox1.yaml`, `$CAARMA_TRAIN_MANIFEST`, `$CAARMA_TRIAL_PATH`,
  and `$CAARMA_VOX1_EVAL_ROOT`.
- The HuBERT cache warm-up block (`HubertModel.from_pretrained` with
  `cache_dir=$CAARMA_HUBERT_CACHE`) is preserved so the four DDP ranks
  share a pre-populated cache.
- Final step unchanged:
  `srun python -u train_source_faithful.py --config config_psc_vox1.yaml --devices 4`.

`config_psc_vox1.yaml`, `train_source_faithful.py`, and
`scripts/psc/common.sh` are untouched — this fix changes only the
launcher layer. A regression guard in
`tests/test_psc_source_faithful_port.py` asserts the slurm script never
references `BASH_SOURCE` again.
