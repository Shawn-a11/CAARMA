#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-m4a-probe
#SBATCH --partition=GPU-shared
#SBATCH --exclude=v010
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

module load anaconda3
module load ffmpeg/4.3.1
source /jet/home/sge2/caarma_psc_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CAARMA_CONDA_ENV"

M4A_PATH="${CAARMA_M4A_PROBE_PATH:-/ocean/projects/cis220031p/shared/raw/data/VoxCeleb2/audio/dev/id02295/qOEyFOkd1-w/00109.m4a}"
test -r "$M4A_PATH"

echo "M4A probe file: $M4A_PATH"
ffmpeg -version | head -1

python - "$M4A_PATH" <<'PY'
import io
import subprocess
import sys

import soundfile as sf
import torch
import torchaudio

path = sys.argv[1]
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("available backends:", torchaudio.list_audio_backends())

for backend in (None, "ffmpeg"):
    label = "automatic" if backend is None else backend
    try:
        kwargs = {} if backend is None else {"backend": backend}
        waveform, sample_rate = torchaudio.load(path, **kwargs)
        print(
            f"torchaudio {label}: OK shape={tuple(waveform.shape)} "
            f"sample_rate={sample_rate}"
        )
    except Exception as exc:
        print(f"torchaudio {label}: FAIL {type(exc).__name__}: {exc}")

decoded = subprocess.run(
    [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        path,
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ],
    check=True,
    stdout=subprocess.PIPE,
).stdout
audio, sample_rate = sf.read(
    io.BytesIO(decoded), dtype="float32", always_2d=True
)
print(
    f"ffmpeg CLI fallback: OK shape={audio.shape} "
    f"sample_rate={sample_rate}"
)
PY
