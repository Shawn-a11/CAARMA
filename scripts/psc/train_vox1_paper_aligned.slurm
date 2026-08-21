#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-paper
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
module load anaconda3
source /jet/home/sge2/caarma_psc_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CAARMA_CONDA_ENV"
cd "$REPO_ROOT"

test -f train_paper_aligned.py
test -f config_psc_vox1_paper_aligned.yaml
test -f "$CAARMA_TRAIN_MANIFEST"
test -f "$CAARMA_TRIAL_PATH"
test -d "$CAARMA_VOX1_EVAL_ROOT"
mkdir -p "$CAARMA_RUN_ROOT/caarma_paper_aligned_vox1/checkpoints"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$CAARMA_RUN_ROOT/cache"
mkdir -p "$XDG_CACHE_HOME"

python - <<'PY'
import os
from transformers import HubertModel
model = HubertModel.from_pretrained(
    "facebook/hubert-large-ls960-ft",
    cache_dir=os.environ["CAARMA_HUBERT_CACHE"],
)
print("HuBERT cache ready:", model.config.hidden_size)
PY

echo "CAARMA job repo: $REPO_ROOT"
echo "CAARMA method commit: ${CAARMA_METHOD_COMMIT:-archive-snapshot}"
srun python -u train_paper_aligned.py \
  --config config_psc_vox1_paper_aligned.yaml \
  --devices 4
