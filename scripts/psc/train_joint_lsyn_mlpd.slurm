#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-mlpd361
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=/jet/home/sge2/CAARMA
ENV_FILE=/jet/home/sge2/caarma_psc_env.sh

module load anaconda3
source "$ENV_FILE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CAARMA_CONDA_ENV"

cd "$REPO_ROOT"

: "${CAARMA_TRAIN_MANIFEST:?CAARMA_TRAIN_MANIFEST is required}"
: "${CAARMA_TRIAL_PATH:?CAARMA_TRIAL_PATH is required}"
: "${CAARMA_VOX1_EVAL_ROOT:?CAARMA_VOX1_EVAL_ROOT is required}"
: "${CAARMA_RUN_ROOT:?CAARMA_RUN_ROOT is required}"

test -f "$CAARMA_TRAIN_MANIFEST"
test -f "$CAARMA_TRIAL_PATH"
test -d "$CAARMA_VOX1_EVAL_ROOT"
mkdir -p "$CAARMA_RUN_ROOT/joint_lsyn_mlpd_361/checkpoints"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAARMA_CONFIG="$REPO_ROOT/config_psc_joint_lsyn_mlpd.yaml"

python -c "import train; print('PSC joint-Lsyn MLP-D import OK')"

srun python -u train.py --config "$CAARMA_CONFIG"
