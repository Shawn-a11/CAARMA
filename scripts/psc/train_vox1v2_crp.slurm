#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-v12-crp
#SBATCH --partition=GPU-shared
#SBATCH --exclude=v010
#SBATCH --gpus=v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=40:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${CAARMA_JOB_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
module load anaconda3
source /jet/home/sge2/caarma_psc_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CAARMA_CONDA_ENV"
cd "$REPO_ROOT"

PYAV_VENDOR="${CAARMA_PYAV_VENDOR:-$CAARMA_RUN_ROOT/vendor/pyav-12.3.0}"
test -d "$PYAV_VENDOR"
export PYTHONPATH="$PYAV_VENDOR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -c "import av; print('PyAV ready:', av.__version__)"

export CAARMA_TRAIN_MANIFEST=/ocean/projects/cis220031p/sge2/caarma_runs/manifests/vox1v2_train.csv

test -f train_source_faithful.py
test -f config_psc_vox1v2_crp.yaml
test -f "$CAARMA_TRAIN_MANIFEST"
test -f "$CAARMA_TRIAL_PATH"
test -d "$CAARMA_VOX1_EVAL_ROOT"
mkdir -p "$CAARMA_RUN_ROOT/caarma_vox1v2_crp_persistent_hubert/checkpoints"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$CAARMA_RUN_ROOT/cache"
mkdir -p "$XDG_CACHE_HOME"

python -c "import train_source_faithful; print('Vox1+2 persistent CRP import OK')"

python - <<'PY'
import os

import pandas as pd
from functions.audio_io import load_audio_file

manifest = pd.read_csv(os.environ["CAARMA_TRAIN_MANIFEST"])
m4a_paths = manifest.loc[
    manifest["utt_paths"].str.lower().str.endswith(".m4a"), "utt_paths"
]
if m4a_paths.empty:
    raise RuntimeError("The Vox1+2 manifest contains no M4A files to validate")
waveform, sample_rate = load_audio_file(m4a_paths.iloc[0])
print(
    "VoxCeleb2 M4A decode ready:", tuple(waveform.shape), sample_rate
)
PY

python - <<'PY'
import os
from transformers import HubertModel

cache_dir = os.environ["CAARMA_HUBERT_CACHE"]
model = HubertModel.from_pretrained(
    "facebook/hubert-large-ls960-ft", cache_dir=cache_dir
)
print("HuBERT cache ready:", cache_dir, model.config.hidden_size)
PY

echo "CAARMA job repo: $REPO_ROOT"
echo "CAARMA method commit: ${CAARMA_METHOD_COMMIT:-archive-snapshot}"
echo "CAARMA train manifest: $CAARMA_TRAIN_MANIFEST"

srun python -u train_source_faithful.py \
  --config config_psc_vox1v2_crp.yaml \
  --devices 4
