# Invalid MFA Coarse Sweep: Optimizer Protocol Drift

Jobs 44593577, 44593582, 44593584, and 44593588 are not valid one-variable
ablations. Their branches replaced the Job 44493754 optimizer protocol with
manual optimization, post-step warmup, `StepLR(4, 0.5)`, and forced SyncBN.

All corrected tuning branches must preserve the locked Job 44493754 contract:

- automatic optimization;
- 2,000-step per-step `LambdaLR` warmup, then constant LR;
- `sync_batchnorm: false`;
- global batch 200 and seed 42;
- byte-identical `train_mfa_baseline.py`.

The mixed manual-optimization recipe is prohibited for this control. A
scheduler experiment must be isolated and explicitly named as such.

