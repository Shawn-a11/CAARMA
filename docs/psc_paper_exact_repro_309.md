# PSC Paper-Exact Reproduction Arm (target: VoxCeleb1-O EER 3.09)

Branch `exp/psc-paper-exact-repro-309`, based on
`exp/psc-paper-aligned-hubert-caarma` @ ae07e82 (the PSC paper-aligned arm,
job 44075531, final-epoch EER 3.44 / best 3.39).

## Goal

Reproduce the CAARMA paper's full-method result: **EER 3.09 / minDCF 0.28 on
VoxCeleb1-O** (arXiv 2503.16718, Table 2 ID 6 / Table 3 / Table 4; MFA-Conformer
+ L_syn + AT + HuBERT mixup discriminator on layers h7/h9/h11/h12; paired
baseline 3.33).

## Delta vs the paper-aligned arm (single layer: lr schedule)

Everything is identical to job 44075531 except the optimizer schedule:

- **Removed** the source repo's `StepLR(step_size=4, gamma=0.5)` on both
  optimizers. The paper's implementation details specify only AdamW
  (model lr 0.001, discriminator lr 2e-4, weight decay 1e-7) with a 2000-step
  linear warmup and no learning-rate decay. Both learning rates are now
  constant after warmup (the per-step warmup overwrite leaves the main lr at
  exactly 0.001 when the warmup ends).

Already paper-aligned in the base arm (unchanged here): Algorithm 2
every-batch D-then-M updates, unhalved `L_D` / `L_G` (paper Eq.1/Eq.2 targets),
dynamic `lambda_adv` from the `L_real / L_G` ratio (cap 0.01, floor 1e-4),
`L_syn` scaled by `1/num_spk`, trainable HuBERT-D (`freeze_hubert: false`),
AM-Softmax m=0.2 / s=30, batch 50 per rank, 30 epochs, 3 s segments, no
augmentation, seed 42.

Known remaining deviations (kept for DDP stability / comparability, recorded
here explicitly): 16-mixed precision; effective global batch 200 (50 x 4
ranks); DDP-required no_grad/toggle structure; frozen-vs-trainable HuBERT
choice inherited from the paper-aligned arm (trainable here).

## Files

- `config_psc_vox1_paper_exact.yaml` — copy of the paper-aligned config with
  isolated `save_dir` / `title` (`caarma_paper_exact_vox1`).
- `scripts/psc/train_vox1_paper_exact.slurm` — self-contained launcher
  (job-name `caarma-pexact`), same pattern as the paper-aligned launcher.
- `train_paper_aligned.py` — only the scheduler removal described above.
- `tests/test_psc_paper_exact_port.py` — port guard tests.

## Reporting rule

Numerical-reproduction lane: per-epoch validation uses the official cleaned
VoxCeleb1-O trial list and checkpoint selection monitors test EER
(test-selected). Report the PSC result separately from the paper number and
from historical-runtime numbers.
