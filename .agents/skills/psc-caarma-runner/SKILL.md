---
name: psc-caarma-runner
description: "Manage CAARMA experiments on Pittsburgh Supercomputing Center Bridges-2 through a guarded Slurm lifecycle: preflight checks, safe sbatch submission, dependency chaining, queue and log monitoring, sacct collection, EER/minDCF result summarization, job listing, and explicit cancellation. Use when asked to run, queue, monitor, resume, summarize, stop, cancel, or troubleshoot CAARMA jobs on PSC/Bridges-2."
---

# PSC CAARMA Runner

Use the bundled standard-library CLI for every Slurm lifecycle operation:

```bash
RUNNER=.agents/skills/psc-caarma-runner/scripts/psc_job.py
python "$RUNNER" --help
```

Read [references/psc-policy.md](references/psc-policy.md) before submitting or
cancelling a job.

## Workflow

1. Run `preflight` on a Bridges-2 login node.
   Require the reproduction branch and a clean worktree.
2. Inspect the requested job script with `submit --dry-run`.
3. Submit only the current approved stage.
4. Record the returned JobID in the user update.
5. Use `status` for a snapshot or `watch` for bounded monitoring.
6. Use `collect` when terminal to write JSON and Markdown summaries.
7. Report state, elapsed time, exit code, best EER/epoch, minDCF, errors, and
   summary paths.
8. Use `cancel --yes` only after explicit user approval to terminate that JobID.

## Commands

```bash
python "$RUNNER" preflight

python "$RUNNER" submit \
  --script scripts/psc/audit_voxceleb.slurm \
  --label vox1-audit

python "$RUNNER" submit \
  --script scripts/psc/environment_smoke.slurm \
  --env-file /jet/home/sge2/caarma_psc_env.sh \
  --label environment-smoke

python "$RUNNER" submit \
  --script scripts/psc/train_vox1.slurm \
  --env-file /jet/home/sge2/caarma_psc_env.sh \
  --label original-caarma-vox1 \
  --require-completed SMOKE_JOB_ID \
  --confirm-production

python "$RUNNER" status --job-id JOB_ID
python "$RUNNER" watch --job-id JOB_ID --max-seconds 1800
python "$RUNNER" collect --job-id JOB_ID
python "$RUNNER" list
python "$RUNNER" cancel --job-id JOB_ID --yes
```

Use `--afterok JOB_ID` to queue a non-production stage behind a successful
dependency. Production additionally requires `--require-completed SMOKE_JOB_ID`
for a recorded terminal gate. Do not automatically chain the four-GPU job;
require a human gate after protocol and smoke summaries.

## Safety Rules

- Never perform deep scans, training, or environment installation directly on
  login nodes.
- Permit data only under
  `/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1` and user-owned outputs
  under `/ocean/projects/cis220031p/sge2`.
- Reject references to other users, VoxCeleb2, or removed recovery scripts.
- Require an environment file to have mode `600`; record its path, never its
  contents.
- Require `--confirm-production` for four-GPU or production training jobs.
- Never infer consent to cancel from a failure suspicion. Ask for the JobID and
  explicit cancellation approval, then pass `--yes`.
- Never delete logs, checkpoints, manifests, environments, or Slurm state as
  part of cancellation.
- Keep official-test-selected results labeled as numerical reproduction.

## Stage Gates

Submit stages in this order:

```text
audit_voxceleb
create_environment
environment_smoke
prepare_vox1_manifest
train_smoke
train_vox1 (explicit production approval)
```

Do not advance when a stage exits nonzero, reports missing paths, detects
speaker/utterance leakage, cannot load HuBERT, sees fewer/more GPUs than
requested, or emits DDP/NCCL/OOM errors.

## Result Handling

The CLI stores state under:

```text
$CAARMA_RUN_ROOT/job_state/JOB_ID.json
$CAARMA_RUN_ROOT/job_state/summaries/JOB_ID.json
$CAARMA_RUN_ROOT/job_state/summaries/JOB_ID.md
```

Treat state `COMPLETED` plus a clean parsed summary as success. A completed
Slurm state without expected EER/minDCF is not a successful full training run.
For `train_smoke.slurm`, additionally require `smoke_steps_complete: true`.
