# Paper Table 2 ID 3: Adversarial Training Only

This branch reproduces the CAARMA Table 2 adversarial-training-only arm
(reported EER 3.15%). It changes one component relative to the MFA-Conformer
baseline:

`L_total = L_real + lambda_adv * L_G`

Synthetic embeddings are used by the real/fake game, but their synthetic-class
loss is excluded. The discriminator is the compact spectral MLP, not the
HuBERT Mixup Discriminator (`MD`).

PSC entrypoint: `scripts/psc/train_vox1_table2_at_only.slurm`
