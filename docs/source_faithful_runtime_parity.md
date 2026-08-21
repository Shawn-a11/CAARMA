# Source-Faithful Runtime Parity

## Reference

- Known completed branch: `exp/ddp-4gpu-v100-source_code`
- Runtime reference commit: `1c860a3`
- PSC production entrypoint: `train_source_faithful.py`

The PSC entrypoint was copied from the completed reference before applying
only cluster portability changes.

## Mechanically Identical Methods

`tests/test_source_faithful_parity.py` parses both files with Python AST and
requires exact equality for:

- `adjust_weight`
- `_d_step`
- `_g_step`
- `training_step`
- `configure_optimizers`
- `on_train_epoch_end`
- `compute_eer`
- `compute_minDCF`
- `on_validation_epoch_end`

This locks the source 5G:1D pretrain schedule, post-pretrain 1:1 schedule,
loss scaling, optimizer settings, scheduler behavior, and DDP evaluation merge
to the previously completed runtime.

## Preserved DDP Protections

- `toggle_optimizer` / `untoggle_optimizer` around both optimizers.
- encoder forward under `torch.no_grad()` for discriminator-only updates.
- fresh encoder forward inside the main-optimizer toggle for model updates.
- one concatenated real/fake discriminator forward per backward.
- frozen HuBERT backbone.
- HuBERT parameters and buffers excluded from DDP traversal.
- rank-offset correction when merging validation vectors.
- `precision="16-mixed"`.

## Deliberate PSC-Only Differences

- CLI `--config`, `--devices`, `--checkpoint`, and `--smoke-steps`.
- environment-variable expansion and manifest speaker-count validation.
- deterministic seed initialization.
- path joining for the unified PSC `VoxCeleb1/wav` layout.
- Hugging Face model ID and shared cache path instead of an AutoDL-local path.
- one-process HuBERT cache warmup before launching four DDP ranks.
- checkpoint/output paths under the user's Ocean project directory.
- 12-hour PSC walltime cap; jobs release resources immediately on completion.

No CRP, persistence, SLERP, Projection-D, Natural Cluster, gender constraint,
boundary utility, or later experiment module is present in this runtime.

## Validation Commands

```bash
python -m unittest -v tests.test_source_faithful_parity
python -c "import criterion; print('criterion import OK')"
python -m py_compile train_source_faithful.py
```

The PSC production script also runs an import preflight and warms HuBERT before
the four-rank `srun` command. Failure in either preflight stops the job before
the first batch.
