#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=vox-audit
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

DATA_ROOT="${CAARMA_DATA_ROOT:-/ocean/projects/cis220031p/shared/raw/data}"
cd "$DATA_ROOT"

for dataset in VoxCeleb1; do
    echo "===== $dataset ROOT ====="
    if [[ ! -d "$dataset" ]]; then
        echo "$dataset does not exist"
        continue
    fi
    realpath "$dataset"
    du -sh "$dataset"

    echo "===== $dataset SAMPLE FILES ====="
    find "$dataset" -type f \( -iname "*.wav" -o -iname "*.flac" -o -iname "*.m4a" \) \
        | head -10 || true

    echo "===== $dataset AUDIO COUNT ====="
    find "$dataset" -type f \( -iname "*.wav" -o -iname "*.flac" -o -iname "*.m4a" \) \
        | wc -l

    echo "===== $dataset STRUCTURE ====="
    find "$dataset" -maxdepth 4 -type d | head -40
done

echo "===== TRIALS AND METADATA ====="
find VoxCeleb1 -maxdepth 7 -type f \
    \( -iname "*veri*test*" -o -iname "*trial*" -o \
       -iname "*meta*.csv" -o -iname "*iden_split*" \) \
    -print 2>/dev/null || true
