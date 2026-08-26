#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-redim2-crp
#SBATCH --partition=GPU-shared
#SBATCH --exclude=v010
#SBATCH --gpus=v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${CAARMA_JOB_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
module load anaconda3
source /jet/home/sge2/caarma_psc_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CAARMA_CONDA_ENV"
cd "$REPO_ROOT"

CAARMA_REDIMNET2_CHECKPOINT="${CAARMA_REDIMNET2_CHECKPOINT:-/ocean/projects/cis220031p/sge2/caarma_runs/model_cache/redimnet2/b6-vox2-lm.pt}"
export CAARMA_REDIMNET2_CHECKPOINT

test -f train_paper_aligned.py
test -f config_psc_vox1_redimnet2_caarma.yaml
test -f "$CAARMA_TRAIN_MANIFEST"
test -f "$CAARMA_TRIAL_PATH"
test -d "$CAARMA_VOX1_EVAL_ROOT"
test -f "$CAARMA_REDIMNET2_CHECKPOINT"
mkdir -p "$CAARMA_RUN_ROOT/redimnet2_b6_vox2lm_crp_projection_slerpinit/checkpoints"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$CAARMA_RUN_ROOT/cache"
mkdir -p "$XDG_CACHE_HOME"

echo "CAARMA job repo: $REPO_ROOT"
echo "CAARMA method commit: ${CAARMA_METHOD_COMMIT:-archive-snapshot}"
echo "ReDimNet2 checkpoint: $CAARMA_REDIMNET2_CHECKPOINT"
srun --kill-on-bad-exit=1 python -u train_paper_aligned.py \
  --config config_psc_vox1_redimnet2_caarma.yaml \
  --devices 4
