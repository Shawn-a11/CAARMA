# PSC Bridges-2 CAARMA Policy

## Fixed Scope

- Allocation: `cis220031p`.
- Repository: `/jet/home/sge2/CAARMA`.
- Approved input dataset:
  `/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1`.
- Approved user output prefix: `/ocean/projects/cis220031p/sge2/`.
- Approved partitions: `RM-shared`, `GPU-shared`.
- Maximum GPU-shared request: four GPUs.
- Maximum job walltime: 48 hours.
- V100/L40S cost estimate: one SU per GPU-hour; H100: two SU per GPU-hour.

Do not access other users' directories. Do not scan or train VoxCeleb2 in this
reproduction skill.

## Stage Policy

| Stage | Script | Resources | Advance condition |
|---|---|---:|---|
| Storage audit | `audit_voxceleb.slurm` | RM-shared CPU | standard train/eval/trial layout identified |
| Environment create | `create_environment.slurm` | RM-shared CPU | installation exits zero |
| Environment smoke | `environment_smoke.slurm` | 1×V100 | CUDA, Lightning, torchaudio, HuBERT cache pass |
| Manifest prepare | `prepare_vox1_manifest.slurm` | RM-shared CPU | labels contiguous; no missing files or leakage |
| Code smoke | `train_smoke.slurm` | 1×V100-32 | 12 source steps complete without runtime error |
| Production | `train_vox1.slurm` | 4×V100-32 | explicit user approval after all prior summaries |

## Submission Checks

Before `submit`:

1. Ensure the Git branch is
   `exp/psc-bridges2-caarma-reproduction` at the approved commit.
2. Ensure the environment file is outside Git and has mode `600`.
3. Run the CLI `--dry-run` and report resource/SU maximum.
4. Confirm required parent stage is `COMPLETED` and clean.
5. For production, obtain explicit approval in the current conversation and
   pass a recorded completed smoke JobID through `--require-completed`.

## Monitoring and Completion

- Use `squeue` while pending/running and `sacct` after queue disappearance.
- Bound watch loops with `--max-seconds`; do not leave hidden sessions running.
- Collect both stdout and stderr.
- Parse all EER evaluations and select the minimum only for descriptive
  reproduction. Preserve its epoch and same-evaluation minDCF values.
- Report missing metrics and error signatures even when Slurm says completed.

## Cancellation

Cancellation is non-destructive: call `scancel JOB_ID`, update state, and keep
all artifacts. Require explicit user approval naming the JobID. Never use broad
process killing, wildcard cancellation, or account-wide cancellation.
