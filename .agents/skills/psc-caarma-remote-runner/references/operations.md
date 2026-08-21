# Operations contract

## Login node

Allowed: Git metadata, small file edits, `sbatch`, `squeue`, `sacct`, `scontrol`,
bounded `tail`, and manifest existence checks.

Not allowed: training, GPU imports, dataset scans, environment installation,
feature construction, EER evaluation, or long-running agents.

## Compute nodes

All model construction, package repair, training and evaluation must be inside
Slurm scripts. Dependency jobs should gate training with `afterok`.

## Reproducibility record

Record JobID, remote repo/commit, dirty state, script, resources, maximum SU,
dependency, logs, terminal state, exit code, best EER/epoch and same-epoch
minDCF. Do not compare AutoDL and PSC without platform, manifest hash, trial
hash and environment lock.

