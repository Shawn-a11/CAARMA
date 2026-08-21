---
name: caarma-autodl-psc-gate
description: "Enforce the CAARMA development promotion workflow from local macOS Codex: power on an AutoDL Pro instance, stage an exact Git commit, run a bounded one-epoch DDP gate, require validation EER/minDCF and entry into the next epoch, stop the gate process, safely power off the AutoDL container, then promote the same passing commit to PSC through psc-caarma-remote-runner. Use whenever developing, validating, or promoting new CAARMA experiments from AutoDL to Bridges-2."
---

# CAARMA AutoDL → PSC Gate

Use this skill together with `psc-caarma-remote-runner`. AutoDL is the fast
runtime gate; PSC is the full experiment platform.

```bash
GATE=/Users/shawn/.codex/skills/caarma-autodl-psc-gate/scripts/cloud_gate.py
PSC=/Users/shawn/.codex/skills/psc-caarma-remote-runner/scripts/psc_remote.py
```

Read [references/setup.md](references/setup.md) once. Secrets stay in
`~/.codex/caarma-cloud.env` with mode 600 and are never printed or committed.

## Promotion contract

1. Resolve one explicit AutoDL Pro instance UUID; never guess a paid instance.
2. Power on and wait for `running`.
3. Stage an immutable local Git commit under `/root/autodl-tmp/caarma_gates/`.
4. Launch the supplied command in a unique tmux session.
5. A passing gate must show all of:
   - four-rank DDP initialization when four GPUs are requested;
   - `Epoch 0: 100%`;
   - cosine EER and both minDCF values;
   - entry into `Epoch 1`;
   - no traceback, NCCL error, OOM, worker failure, or stalled timeout.
6. Send Ctrl-C only to the gate tmux session, wait, then kill that session.
7. Confirm the gate process is gone and no GPU compute app remains.
8. Power off AutoDL and wait for `shutdown`/`stopped`.
9. Save a PASS receipt containing commit, metrics, log, instance and shutdown state.
10. Promote only the same commit with a PASS receipt to PSC. Run PSC staging,
    dry-run, then production submit.

On any gate failure: stop the owned session, power off when no unrelated compute
is active, save a FAIL receipt, and do not submit PSC.

## Commands

```bash
python3 "$GATE" doctor
python3 "$GATE" status

python3 "$GATE" gate \
  --local-repo /path/to/worktree \
  --ref COMMIT \
  --name experiment-name \
  --train-command 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate caarma && python -u train.py --config config_autodl_gate.yaml' \
  --max-seconds 1800 \
  --yes

python3 "$GATE" promote \
  --receipt ~/.codex/caarma-cloud-gates/receipts/RECEIPT.json \
  --local-repo /path/to/worktree \
  --ref COMMIT \
  --psc-remote-dir /jet/home/sge2/caarma_jobs/NAME-COMMIT \
  --psc-script scripts/psc/TRAIN.slurm \
  --psc-afterok ENV_JOB_ID \
  --confirm-production \
  --yes
```

## Safety

- Token comes only from `AUTODL_TOKEN` or the protected env file.
- AutoDL SSH requires an existing key; the skill never automates root passwords.
- `gate` and `promote` require `--yes` because they change paid resources.
- `release` is never called. The default lifecycle ends with power-off.
- Do not treat startup or a few training steps as PASS; complete validation and
  entry into the next epoch are required because prior failures occurred there.
- Do not promote a different commit from the receipt.
- Do not use official test EER for hyperparameter search in formal experiments;
  the runtime gate checks execution health, not model selection.

