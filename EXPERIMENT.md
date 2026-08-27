# Paper Table 2 ID 2: Lsyn Only

This branch reproduces the CAARMA Table 2 `L_syn`-only arm (reported EER
3.28%). It changes one component relative to the MFA-Conformer baseline:

`L_total = L_real + (1 / N_spk) * L_syn`

The branch uses one-shot embedding/prototype mixup to construct `L_syn`. It has
no discriminator, no adversarial optimizer, and no HuBERT dependency.

The optimizer protocol is matched to the clean Table 2 ID 1 control: automatic
Lightning optimization, AdamW, a 2,000-step linear warmup, no epoch-wise LR
decay, and the same non-SyncBN setting. The earlier audited run accidentally
combined `L_syn` with a `StepLR(gamma=0.5, step_size=4)` schedule and forced
SyncBN, so its result is not a valid ID 1 versus ID 2 comparison.

PSC entrypoint: `scripts/psc/train_vox1_table2_lsyn_only.slurm`
