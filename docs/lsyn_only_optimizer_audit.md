# Lsyn-Only Optimizer Audit

## Invalid comparison

The completed Lsyn-only job `44521249` reached a best VoxCeleb1-O EER of
6.89%, but it was not optimizer-matched to the clean MFA control `44493754`
(best EER 3.84%). The method comparison was confounded by three runtime
differences:

| Setting | Clean MFA control | Lsyn-only job 44521249 |
| --- | --- | --- |
| Lightning optimization | automatic | manual |
| Learning-rate schedule | 2,000-step linear warmup only | warmup plus `StepLR(4, 0.5)` |
| SyncBatchNorm | disabled | forced enabled |

With seven epoch-wise halvings over 30 epochs, the Lsyn run ended near
`init_lr / 128`. Its real-speaker AM-Softmax accuracy remained about 45.5% at
the end of training, whereas the clean control reached about 96.3%. Therefore,
the 6.89% result does not isolate the effect of the synthetic loss.

## Matched rerun

This branch keeps the Lsyn method unchanged:

`L_total = L_real + (1 / N_spk) * L_syn`

It changes only the optimization protocol to match the clean MFA control:

- automatic Lightning optimization;
- AdamW with the same learning rate and weight decay;
- a 2,000-step linear warmup, stepped once per optimizer update;
- no epoch-wise learning-rate decay;
- SyncBatchNorm disabled.

The resulting run is the valid Table 2 ID 1 versus ID 2 comparison. The old
jobs remain diagnostic records and must not be reported as Lsyn ablations.
