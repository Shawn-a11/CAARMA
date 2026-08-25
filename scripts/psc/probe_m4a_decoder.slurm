#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-m4a-probe
#SBATCH --partition=GPU-shared
#SBATCH --exclude=v010
#SBATCH --gpus=v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
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
mkdir -p "$PYAV_VENDOR"
if ! PYTHONPATH="$PYAV_VENDOR" python -c "import av" >/dev/null 2>&1; then
    python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --only-binary=:all: \
        --target "$PYAV_VENDOR" \
        "av==12.3.0"
fi
export PYTHONPATH="$PYAV_VENDOR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

M4A_PATH="${CAARMA_M4A_PROBE_PATH:-/ocean/projects/cis220031p/shared/raw/data/VoxCeleb2/audio/dev/id02295/qOEyFOkd1-w/00109.m4a}"
test -r "$M4A_PATH"

echo "M4A probe file: $M4A_PATH"
echo "CAARMA job repo: $REPO_ROOT"

python - "$M4A_PATH" <<'PY'
import sys

import av

from functions.audio_io import load_audio_file

path = sys.argv[1]
waveform, sample_rate = load_audio_file(path)
assert waveform.ndim == 2
assert waveform.shape[0] == 1
assert waveform.shape[1] > 0
print(
    f"PyAV {av.__version__}: OK shape={tuple(waveform.shape)} "
    f"sample_rate={sample_rate} min={waveform.min().item():.6f} "
    f"max={waveform.max().item():.6f}"
)
PY
