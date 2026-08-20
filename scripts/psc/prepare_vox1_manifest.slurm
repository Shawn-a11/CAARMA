#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=vox1-manifest
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
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

if [[ ! -f "$CAARMA_TRIAL_PATH" ]]; then
    echo "Downloading official cleaned VoxCeleb1-O trials to $CAARMA_TRIAL_PATH"
    mkdir -p "$(dirname "$CAARMA_TRIAL_PATH")"
    temporary="$CAARMA_TRIAL_PATH.part"
    curl --fail --location --retry 3 \
        https://openslr.trmal.net/resources/49/voxceleb1_test_v2.txt \
        --output "$temporary"
    mv "$temporary" "$CAARMA_TRIAL_PATH"
fi

echo "Trial lines: $(wc -l < "$CAARMA_TRIAL_PATH")"
sha256sum "$CAARMA_TRIAL_PATH"

python tools/build_voxceleb_csv.py \
    --source "vox1=$CAARMA_VOX1_TRAIN_ROOT" \
    --exclude-speakers-from-trials "$CAARMA_TRIAL_PATH" \
    --output "$CAARMA_TRAIN_MANIFEST"

python - "$CAARMA_TRAIN_MANIFEST.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

expected = {
    "speakers": 1211,
    "utterances": 148642,
    "excluded_speaker_count": 40,
}
for key, value in expected.items():
    actual = report.get(key)
    if actual != value:
        raise SystemExit(
            f"VoxCeleb1 reproduction count mismatch: {key}={actual}, expected={value}"
        )
print("VoxCeleb1 reproduction counts verified:", expected)
PY

python tools/audit_voxceleb_protocol.py \
    --train-csv "$CAARMA_TRAIN_MANIFEST" \
    --trial-path "$CAARMA_TRIAL_PATH" \
    --eval-root "$CAARMA_VOX1_EVAL_ROOT" \
    --json-output "$MANIFEST_DIR/vox1_protocol.json"

echo "Frozen manifest: $CAARMA_TRAIN_MANIFEST"
echo "Protocol audit: $MANIFEST_DIR/vox1_protocol.json"
