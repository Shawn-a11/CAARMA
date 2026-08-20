#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-env-create
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

CAARMA_REPO="${CAARMA_REPO:-/jet/home/sge2/CAARMA}"
ENV_PREFIX="${CAARMA_ENV_PREFIX:-/ocean/projects/cis220031p/sge2/conda_envs/caarma}"

module load anaconda3
mkdir -p "$(dirname "$ENV_PREFIX")"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create --yes --prefix "$ENV_PREFIX" python=3.10 pip
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    torch==2.1.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$CAARMA_REPO/requirements_psc.txt"

python - <<'PY'
import numpy
import pandas
import pytorch_lightning
import scipy
import torch
import torchaudio
import transformers

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("pytorch_lightning", pytorch_lightning.__version__)
print("transformers", transformers.__version__)
PY

echo "Environment ready: $ENV_PREFIX"
