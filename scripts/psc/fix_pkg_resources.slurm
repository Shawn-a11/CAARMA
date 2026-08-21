#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-envfix
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

module load anaconda3
source /jet/home/sge2/caarma_psc_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CAARMA_CONDA_ENV"

cd /jet/home/sge2/CAARMA

python -m pip install setuptools==75.8.0

python - <<'PY'
import pkg_resources
import librosa.filters

from feature.fbanks import Mel_Spectrogram

Mel_Spectrogram()
print("PSC audio frontend OK")
PY
