# exp/psc-projection-crp-bs13: Align effective global batch to ~50

## Source arm

- Branch: `exp/psc-projection-crp-persistent-synth`
- Commit: `fbd396ac861c21b9f7f598f0c12978d77451f068`
- PSC job: `44190693`
- Result (Vox1-O test-selected):
  - best EER in tail: **3.35**
  - latest minDCF@1e-2 in tail: **0.3454**
  - latest minDCF@1e-3 in tail: **0.4019**
- Config: `config_psc_projection_crp.yaml`
- Slurm: `scripts/psc/train_projection_crp.slurm`

## Delta

Only `batch_size` is changed in the per-rank config: `50 -> 13`.

With `--ntasks-per-node=4` (4 DDP ranks on a single node), the effective global batch size becomes:

```
13 per rank * 4 ranks = 52 global
```

This is the closest integer to the paper's stated global batch size of **50**. The original PSC projection arm used a per-rank batch size of 50, giving an effective global batch of 200 across 4 ranks.

## Goal

The reproduction target for the main chain is **3.09 EER on Vox1-O test-selected**. The source PSC projection arm sits at **3.35**, leaving a ~0.26 gap. The original paper reports training with a global batch size of 50; under DDP on PSC we have been using 50 per rank (200 global). This experiment isolates whether the effective global batch mismatch explains part of the reproduction gap, or whether moving toward the paper's global batch size shifts the main-chain ladder closer to 3.09.

No other hyperparameters are changed: same model (MFA-CONFORMER), criterion (AMSoftmaxGAN), learning rates (`init_lr=0.001`, `discriminator_lr` via criterion defaults), 30 epochs, seed 42, persistence, CRP pair strategy (`pair_strategy=crp`, `crp_topk=4`), and projection discriminator.

## Reporting rule

Use the same numerical-reproduction lane as the source arm: **Vox1-O test-selected**. Report final Vox1-O EER/minDCF, the selected checkpoint source, and whether the batch-size change closes any portion of the gap to 3.09.
