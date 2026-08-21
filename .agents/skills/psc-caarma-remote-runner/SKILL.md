---
name: psc-caarma-remote-runner
description: "Operate CAARMA experiments on PSC Bridges-2 from local macOS Codex through guarded SSH-to-login-node orchestration. Use whenever the user asks Codex to connect to PSC, submit or queue sbatch jobs, prepare dependencies, monitor squeue/sacct/logs, collect EER/minDCF, diagnose failures, or cancel CAARMA jobs without running production work on login nodes."
---

# PSC CAARMA Remote Runner

Use the bundled CLI from the Mac. It connects only to the Bridges-2 login host
and permits Slurm lifecycle operations; computation remains inside `sbatch` jobs.

```bash
REMOTE_RUNNER=.agents/skills/psc-caarma-remote-runner/scripts/psc_remote.py
python3 "$REMOTE_RUNNER" --help
```

Read [references/ssh-setup.md](references/ssh-setup.md) when SSH doctor fails.
Read [references/operations.md](references/operations.md) before first submission.

## Required sequence

1. Run `doctor`. Do not submit when non-interactive SSH is unavailable.
2. Run `submit --dry-run` with the exact remote repo, script and expected commit.
   If only tracked `__pycache__/*.pyc` files are dirty, use
   `clean-generated --repo REMOTE_REPO --yes`; it refuses every other path.
3. For four-GPU jobs, require `--confirm-production`.
4. Submit one approved stage and report JobID, repo, commit, GPUs, walltime,
   maximum SU, dependency, and log paths.
5. Use `status` or bounded `watch`; never keep an unbounded SSH tail running.
6. Use `logs` for finite tails and `collect` after terminal state.
7. Use `cancel --yes` only after the user explicitly approves that JobID.

## Examples

```bash
python3 "$REMOTE_RUNNER" --host bridges2 doctor

python3 "$REMOTE_RUNNER" --host bridges2 submit \
  --repo /jet/home/sge2/CAARMA-psc-mlpd361-37d41df \
  --script scripts/psc/train_joint_lsyn_mlpd.slurm \
  --expected-commit 37d41df \
  --confirm-production \
  --dry-run

python3 "$REMOTE_RUNNER" --host bridges2 status --job-id JOB_ID
python3 "$REMOTE_RUNNER" --host bridges2 stage \
  --local-repo LOCAL_GIT_WORKTREE --ref COMMIT \
  --remote-dir /jet/home/sge2/caarma_jobs/NAME-COMMIT --yes
python3 "$REMOTE_RUNNER" --host bridges2 clean-generated \
  --repo REMOTE_REPO --yes
python3 "$REMOTE_RUNNER" --host bridges2 adopt --job-id JOB_ID \
  --repo REMOTE_REPO --commit COMMIT --job-name JOB_NAME \
  --output STDOUT_PATH --error STDERR_PATH
python3 "$REMOTE_RUNNER" --host bridges2 watch --job-id JOB_ID --max-seconds 900
python3 "$REMOTE_RUNNER" --host bridges2 logs --job-id JOB_ID --lines 100
python3 "$REMOTE_RUNNER" --host bridges2 collect --job-id JOB_ID
python3 "$REMOTE_RUNNER" --host bridges2 cancel --job-id JOB_ID --yes
```

Use `--afterok JOB_ID` to create an explicit dependency. The CLI records every
submission under `~/.codex/psc-caarma-remote/jobs/` so a new Codex session can
recover JobIDs and log paths.

## Safety

- Connect to `bridges2.psc.edu` through the local `bridges2` SSH alias.
- Never automate passwords, Duo prompts or private-key registration.
- Never run training, package installation, deep scans or evaluation on login nodes.
- Permit remote repositories only under `/jet/home/sge2/` or user-owned
  `/ocean/projects/cis220031p/sge2/`.
- Permit Slurm scripts only below the selected repo's `scripts/psc/` directory.
- Require account `cis220031p` and partition `GPU-shared`.
- Reject tracked dirty files unless `--allow-launcher-dirty` is explicit. This
  flag is for reviewed launcher-only patches, not method-code changes.
- Do not delete repositories, worktrees, environments, logs or checkpoints.
- `stage` creates a new commit snapshot and refuses to overwrite a target.
- Keep official-test-selected metrics labeled as numerical reproduction.

## Failure handling

When a job fails, collect `sacct`, stdout and stderr before proposing a change.
Classify the failure as environment, path/repo, Slurm resource, data, DDP/NCCL,
OOM, or model code. Change one layer at a time and preserve the failed JobID.
