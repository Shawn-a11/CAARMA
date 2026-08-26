# Paper Table 2 ID 2: Lsyn Only

This branch reproduces the CAARMA Table 2 `L_syn`-only arm (reported EER
3.28%). It changes one component relative to the MFA-Conformer baseline:

`L_total = L_real + (1 / N_spk) * L_syn`

The branch uses one-shot embedding/prototype mixup to construct `L_syn`. It has
no discriminator, no adversarial optimizer, and no HuBERT dependency.

PSC entrypoint: `scripts/psc/train_vox1_table2_lsyn_only.slurm`
