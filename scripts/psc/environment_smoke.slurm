#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-env
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
prepare_runtime

nvidia-smi
python --version
python - <<'PY'
import os
import torch
import torchaudio
import pytorch_lightning as pl
from transformers import HubertModel

print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("lightning:", pl.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1

cache_dir = os.environ["CAARMA_HUBERT_CACHE"]
model = HubertModel.from_pretrained(
    "facebook/hubert-large-ls960-ft",
    cache_dir=cache_dir,
)
print("HuBERT cache ready:", cache_dir)
print("HuBERT hidden size:", model.config.hidden_size)
PY
