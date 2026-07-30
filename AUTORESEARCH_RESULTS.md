# AutoResearch Development-Split Results

- Experiment family: Joint-Lsyn MLP-D baseline tuning
- Branch: `exp/autoresearch-joint-lsyn-mlpd-baseline-tuning-ddp`
- Infrastructure commit: `315abf4`
- Evaluation protocol: held-out same-speaker development trials
- Intended use: hyperparameter screening only
- Recorded: 2026-07-30

## Protocol Boundary

These development results are not official VoxCeleb1 `veri_test2` results and
must not be compared numerically with E0-E3 or other official-test experiments.
They are used only to rank configurations before a final full-budget run.

## Recorded Trials

| Stage | Only change | Budget | Best epoch | Best dev EER | minDCF 1e-2 | minDCF 1e-3 | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Main learning rate | `init_lr=1e-3` | 10 epochs | 8 | **2.09%** | 0.1908 | 0.1996 | Lock `init_lr=1e-3` |
| Main learning rate continuation | Resume the preceding `last.ckpt`, no parameter change | 20 total epochs | 18 | **1.51%** | 0.1272 | 0.1272 | Demonstrates incomplete convergence; not a new tuning point |
| Discriminator learning rate | `discriminator_lr=1e-4` | 10 epochs | 8 | **2.20%** | 0.1605 | 0.1830 | Keep default `discriminator_lr=2e-4` |
| Weight decay | `weight_decay=1e-8` | 10 epochs | 9 | **2.09%** | 0.1745 | 0.1762 | Lock `weight_decay=1e-8` |

## Current Locked Settings

- Main learning rate: `1e-3`
- Discriminator learning rate: `2e-4`
- Weight decay: `1e-8`

The continuation result shows that short-budget rankings should be confirmed at
an adequate training budget before transferring the settings to E1-E3.

## Provenance

Only the manually verified aggregate results are committed. Trial lists,
training logs, checkpoints, datasets, and generated outputs remain external.
