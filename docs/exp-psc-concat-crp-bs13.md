# PSC experiment: align effective global batch size to ~50

## Source arm

- Source branch: `exp/psc-concat-crp-persistent-synth` @ `5fdee43` (job **44191580**).
- PSC result (Vox1-O, test-selected):
  - EER: **3.56%**
  - minDCF (`1e-2`): **0.3476**
  - minDCF (`1e-3`): **0.5093**

## Delta from source arm

Only `batch_size` is changed:

- Per-rank batch size: `50` → `13`.
- With 4 DDP ranks, effective global batch size is `13 × 4 = 52`, the closest
  integer to the paper's stated global batch size of **50**.
- Save directory updated to `${CAARMA_RUN_ROOT}/concat_crp_persistent_bs13/checkpoints`.
- Job name updated to `caarma-concat13`; config title updated to
  `concat_crp_persistent_bs13_psc`.

All other hyperparameters are unchanged from the source arm (lr schedule,
`lambda_adv`, model, criterion, persistence, CRP pairing, etc.).

## Goal

Isolate whether the effective global batch-size mismatch explains part of the
reproduction gap to the paper's reported **3.09%** EER and whether it shifts the
main-chain reproduction ladder.

## Reporting rule

Report in the same numerical-reproduction lane as the source arm:
Vox1-O **test-selected** EER/minDCF from the PSC run.
