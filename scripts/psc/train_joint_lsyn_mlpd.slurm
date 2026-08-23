#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-mlpd361-devsel
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${CAARMA_JOB_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
ENV_FILE=/jet/home/sge2/caarma_psc_env.sh

module load anaconda3
source "$ENV_FILE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CAARMA_CONDA_ENV"

cd "$REPO_ROOT"
test -f "$REPO_ROOT/train.py"
test -f "$REPO_ROOT/config_psc_joint_lsyn_mlpd.yaml"

: "${CAARMA_TRAIN_MANIFEST:?CAARMA_TRAIN_MANIFEST is required}"
: "${CAARMA_TRIAL_PATH:?CAARMA_TRIAL_PATH is required}"
: "${CAARMA_VOX1_EVAL_ROOT:?CAARMA_VOX1_EVAL_ROOT is required}"
: "${CAARMA_RUN_ROOT:?CAARMA_RUN_ROOT is required}"

# Fixed dev trial for checkpoint selection; original env points to Vox1-O.
CAARMA_DEV_TRIAL_PATH=/ocean/projects/cis220031p/sge2/caarma_dev_trials/vox1_dev_trial.txt
CAARMA_TEST_TRIAL_PATH="$CAARMA_TRIAL_PATH"  # Vox1-O, saved before override
test -f "$CAARMA_DEV_TRIAL_PATH"
test -f "$CAARMA_TEST_TRIAL_PATH"
test -d "$CAARMA_VOX1_EVAL_ROOT"

RUN_TAG="joint_lsyn_mlpd_361_devsel"
CKPT_DIR="$CAARMA_RUN_ROOT/$RUN_TAG/checkpoints"
EVAL_DIR="$CAARMA_RUN_ROOT/$RUN_TAG/final_eval"
mkdir -p "$CKPT_DIR" "$EVAL_DIR"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CAARMA_CONFIG="$REPO_ROOT/config_psc_joint_lsyn_mlpd.yaml"

echo "CAARMA job repo: $REPO_ROOT"
echo "CAARMA method commit: $(git rev-parse --short HEAD)"
python -c "import train; print('PSC joint-Lsyn MLP-D import OK')"

# ── Phase 1: train with dev trial as validation / checkpoint selection ──
echo "=== Phase 1: training with dev trial ==="
export CAARMA_TRIAL_PATH="$CAARMA_DEV_TRIAL_PATH"
srun python -u train.py --config "$CAARMA_CONFIG" --mode train

# ── Phase 2: pick dev-selected best checkpoint ──
echo "=== Phase 2: selecting best dev checkpoint ==="
python3 - "$CKPT_DIR" <<'PY'
import glob, os, re, sys
ckpt_dir = sys.argv[1]
ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
best_ckpt = None
best_eer = float('inf')
for c in ckpts:
    # Lightning formats: epoch=5_cosine_eer=3.45.ckpt or epoch_5_3.45.ckpt
    m = re.search(r'(?:cosine_eer[=_])([\d.]+)\.ckpt$', c)
    if not m:
        m = re.search(r'_([\d.]+)\.ckpt$', c)
    if m:
        eer = float(m.group(1))
        if eer < best_eer:
            best_eer = eer
            best_ckpt = c
if best_ckpt is None:
    raise RuntimeError(f"No checkpoint with EER found in {ckpt_dir}")
marker = os.path.join(ckpt_dir, 'best_dev_checkpoint.txt')
with open(marker, 'w') as f:
    f.write(best_ckpt + '\n')
print(f"best_dev_eer={best_eer:.2f}")
print(f"best_dev_checkpoint={best_ckpt}")
print(f"marker={marker}")
PY

BEST_CKPT=$(cat "$CKPT_DIR/best_dev_checkpoint.txt")
echo "Best dev-selected checkpoint: $BEST_CKPT"

# ── Phase 3: one-time final evaluation on Vox1-O ──
echo "=== Phase 3: final evaluation on Vox1-O ==="
python3 - "$REPO_ROOT/config_psc_joint_lsyn_mlpd.yaml" "$BEST_CKPT" "$CAARMA_TEST_TRIAL_PATH" "$EVAL_DIR" <<'PY'
import os, sys, yaml
cfg_path, ckpt_path, test_trial, eval_dir = sys.argv[1:5]
with open(cfg_path) as f:
    cfg = yaml.safe_load(os.path.expandvars(f.read()))
cfg['trial_path'] = test_trial
cfg['checkpoint_path'] = ckpt_path
cfg['save_dir'] = eval_dir
cfg['epochs'] = 0
eval_cfg = os.path.join(eval_dir, 'eval_config.yaml')
with open(eval_cfg, 'w') as f:
    yaml.dump(cfg, f)
print(f"eval_config={eval_cfg}")
PY

export CAARMA_CONFIG="$EVAL_DIR/eval_config.yaml"
srun python -u train.py --config "$CAARMA_CONFIG" --mode eval
