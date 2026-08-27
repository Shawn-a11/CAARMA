# Tuned MLP-D Control: Fixed Lambda 0.01

## Purpose

Determine whether the lightweight unconditional MLP discriminator remains
weaker than HuBERT-D after applying the best current paper-aligned adversarial
weight. This is a discriminator replacement control, not a proposed-method
run.

## Branch

`exp/psc-control-mlpd-lambda001-lr1e3-dlr2e4-vs-hubert-ddp`

The branch name records the discriminator, fixed adversarial weight, main and
discriminator learning rates, comparison purpose, and DDP runtime.

## Locked Configuration

- MFA-Conformer speaker encoder
- VoxCeleb1 training set and official VoxCeleb1-O trials
- unconditional `MLPDiscriminator`, hidden dimension 256
- `init_lr = 1e-3`
- `discriminator_lr = 2e-4`
- fixed `lambda_adv = 0.01`
- paired D/G updates
- batch size 50 per GPU, four V100 GPUs
- 30 epochs, seed 42
- no CRP, persistence, synthetic prototype, Projection-D, or SLERP initialization

## Comparisons

- tuned HuBERT-D control: Job 44343185, best EER 3.38%
- default paper-aligned MLP-D control: Job 44483167, best EER 3.50%

## Decision Rule

- EER at or below 3.39% supports MLP-D as a quality-preserving HuBERT replacement.
- EER above 3.39% restricts the MLP contribution to efficiency and use as the
  foundation for class-conditioned Projection-D.
