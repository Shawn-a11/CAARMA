# Public Source vs. Paper Reproduction Audit

The public `main` branch and the published CAARMA paper do not specify one
internally consistent 3.09 recipe. These differences must be resolved through
controlled runs on the shared standard VoxCeleb1 data before claiming
reproduction.

| Item | Public source | Paper | PSC first-pass treatment |
|---|---|---|---|
| Training entry | `trainer.fit` commented; only `validate` runs | 30-epoch training | enable `fit` without changing the source state machine |
| Config path | hard-coded author PSC path | not applicable | environment-driven config path |
| VoxCeleb1 batch | public config says 200 | paper says 50 on V100 | use 50 per GPU, explicitly record effective batch 200 on 4 GPUs |
| Discriminator LR | `main_lr * 0.01` = `1e-5` initially | fixed `2e-4` | preserve public source first; paper-LR arm only if separately named |
| Update schedule | 5 generator : 1 discriminator through epoch 15, then 1:1 | discriminator and model updated for every batch | preserve public source first; Algorithm-2 arm remains separate |
| Adversarial weight | `0.0005` in pretrain; later reset to `0.25` then capped by control law | dynamically adjusted, exact schedule not reported | preserve public source |
| Speaker count | config says 7,323; criterion hard-codes 1,211 | Vox1 table says 1,211; combined table says 7,205 | derive count from frozen manifest and reject mismatches |
| Vox1 utterances | manifest not published | table says 153,516 utterances but 1,211 speakers | construct and hash an official dev-only manifest from the shared standard dataset |
| Evaluation | public script evaluates configured trial every validation epoch | VoxCeleb1-O | retain for numerical reproduction but label checkpoint selection as test-selected |

## Important Dataset Inconsistency

The paper reports 1,211 VoxCeleb1 classes together with 153,516 utterances.
Those two values require explicit protocol auditing because the public
repository does not include the CSV. This branch constructs a manifest only
from `/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1` and reports the
resulting counts without reading any other user's project directory.

## Reproduction Order

1. Build and hash a dev-only manifest from the shared standard VoxCeleb1 data.
2. Run a 12-step single-GPU source smoke.
3. Run the public-source schedule on four V100-32 GPUs with the audited data.
4. If it does not reproduce 3.09, change exactly one documented source-paper
   discrepancy per arm; do not silently combine fixes.
5. Keep all official-test-selected results labeled as numerical reproduction,
   not an untouched generalization estimate.
