#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-assets
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

ORIGINAL_ROOT=/ocean/projects/cis220031p/mbaali/mixup
SHARED_ROOT=/ocean/projects/cis220031p/shared/raw/data

echo "===== ORIGINAL HARD-CODED ROOT ====="
if [[ -d "$ORIGINAL_ROOT" ]]; then
    realpath "$ORIGINAL_ROOT"
    find "$ORIGINAL_ROOT" -maxdepth 7 -type f \
        \( -name "config.yaml" -o -name "voxceleb_full.csv" -o \
           -iname "*veri*test*.txt" -o -iname "*trial*.txt" \) \
        -print 2>/dev/null || true
else
    echo "Original root is unavailable: $ORIGINAL_ROOT"
fi

echo "===== ORIGINAL CONFIG PATH ====="
ORIGINAL_CONFIG="$ORIGINAL_ROOT/mixup_framework/config/config.yaml"
if [[ -r "$ORIGINAL_CONFIG" ]]; then
    echo "$ORIGINAL_CONFIG"
    sed -n '1,180p' "$ORIGINAL_CONFIG"
else
    echo "Original config is not readable: $ORIGINAL_CONFIG"
fi

echo "===== SHARED VOXCELEB ASSETS ====="
if [[ -d "$SHARED_ROOT/VoxCeleb1" ]]; then
    find "$SHARED_ROOT/VoxCeleb1" -maxdepth 7 -type f \
        \( -name "voxceleb_full.csv" -o -iname "*veri*test*.txt" -o \
           -iname "*trial*.txt" -o -iname "*iden_split*" \) \
        -print 2>/dev/null || true
fi

echo "===== CSV HEADERS AND COUNTS ====="
while IFS= read -r csv_path; do
    echo "--- $csv_path"
    head -3 "$csv_path"
    wc -l "$csv_path"
done < <(
    find "$ORIGINAL_ROOT" "$SHARED_ROOT/VoxCeleb1" -maxdepth 7 \
        -type f -name "*.csv" -print 2>/dev/null || true
)
