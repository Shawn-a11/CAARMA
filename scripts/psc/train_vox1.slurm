#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=caarma-vox1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
prepare_runtime
require_env CAARMA_TRAIN_MANIFEST CAARMA_VOX1_EVAL_ROOT CAARMA_TRIAL_PATH
test -f "$CAARMA_TRAIN_MANIFEST"

# Lightning's SLURM launcher requires ntasks-per-node to match Trainer.devices.
srun python -u train.py --config config_psc_vox1.yaml --devices 4
