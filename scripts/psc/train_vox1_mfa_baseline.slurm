#!/bin/bash
#SBATCH --account=cis220031p
#SBATCH --job-name=mfa-lr15e4-r1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH --exclude=v003,v007,v008,v010
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis220031p/sge2/logs/%x-%j.out
#SBATCH --error=/ocean/projects/cis220031p/sge2/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${CAARMA_JOB_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
module load anaconda3
source /jet/home/sge2/caarma_psc_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CAARMA_CONDA_ENV"
cd "$REPO_ROOT"

test -f train_mfa_baseline.py
test -f config_psc_vox1_mfa_baseline.yaml
test -f tools/verify_nccl_allreduce.py
test -f "$CAARMA_TRAIN_MANIFEST"
test -f "$CAARMA_TRIAL_PATH"
test -d "$CAARMA_VOX1_EVAL_ROOT"
RUN_DIR="$CAARMA_RUN_ROOT/mfa_job44607056_lr15e4_ncclfix1"
test ! -e "$RUN_DIR" || {
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 1
}
mkdir -p "$RUN_DIR/checkpoints"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "CAARMA job repo: $REPO_ROOT"
echo "CAARMA method commit: ${CAARMA_METHOD_COMMIT:-archive-snapshot}"
echo "Paper control: Table 2 ID1, clean MFA-Conformer + real AM-Softmax only"
echo "Parent result: Job 44607056 (init_lr=2e-3, best EER 3.40%)"
echo "Tuning arm: init_lr=1.5e-3"
echo "Batch semantics: 50 samples per GPU, 4 GPUs, global batch 200"
echo "Schedule: 2000-step linear warmup, then constant learning rate"
echo "NCCL runtime: single-node shared-memory fallback with first-bucket preflight"
python tools/verify_mfa_control_isolation.py --axis init_lr --value 0.0015
srun python -u tools/verify_nccl_allreduce.py \
  --expected-world-size 4 \
  --numel 19991936 \
  --timeout-seconds 180
srun python -u train_mfa_baseline.py \
  --config config_psc_vox1_mfa_baseline.yaml \
  --devices 4
