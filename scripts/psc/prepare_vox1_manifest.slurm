#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=vox1-manifest
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
prepare_data_runtime
require_env CAARMA_VOX1_TRAIN_ROOT CAARMA_VOX1_EVAL_ROOT CAARMA_TRIAL_PATH

MANIFEST_DIR="$CAARMA_RUN_ROOT/manifests"
mkdir -p "$MANIFEST_DIR"
export CAARMA_TRAIN_MANIFEST="${CAARMA_TRAIN_MANIFEST:-$MANIFEST_DIR/vox1_train.csv}"

python tools/build_voxceleb_csv.py \
    --source "vox1=$CAARMA_VOX1_TRAIN_ROOT" \
    --output "$CAARMA_TRAIN_MANIFEST"

python tools/audit_voxceleb_protocol.py \
    --train-csv "$CAARMA_TRAIN_MANIFEST" \
    --trial-path "$CAARMA_TRIAL_PATH" \
    --eval-root "$CAARMA_VOX1_EVAL_ROOT" \
    --json-output "$MANIFEST_DIR/vox1_protocol.json"

echo "Frozen manifest: $CAARMA_TRAIN_MANIFEST"
echo "Protocol audit: $MANIFEST_DIR/vox1_protocol.json"
