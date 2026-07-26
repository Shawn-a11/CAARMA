# CAARMA Joint-Lsyn MLP-D Autoresearch Policy

Your task is to improve the CAARMA `3.61` joint-Lsyn MLP-D baseline through
controlled hyperparameter tuning.

## Immutable method

Do not edit tracked source code, the dataset, the evaluation implementation, the
trial list, model architecture, discriminator architecture, data augmentation,
midpoint pair construction, joint-Lsyn class construction, DDP configuration,
or checkpoint-selection metric.

Do not introduce CRP, Projection-D, gender, clustering, boundary selection, or
any other proposed method. This run is baseline tuning only.

You may only launch `autoresearch/run_trial.py` with values explicitly listed in
`autoresearch/search_space.yaml`.

## Experiment loop

1. Read `autoresearch/search_space.yaml` and the untracked `results.tsv`.
2. Establish or import one baseline result before proposing changes.
3. Change one parameter group at a time:
   optimization, then schedule, then objective.
4. Use a short budget only after early-to-final ranking has been validated.
5. Record every complete, failed, and timed-out trial. Never silently rerun an
   equivalent configuration.
6. Promote only clearly better configurations to a longer budget by resuming
   from `last.ckpt`.
7. Treat EER as primary and minDCF(10-2) as a tie-breaker.
8. Do not claim an improvement from one seed. Finalists require three seeds.
9. Never use a final test trial list for autonomous selection.

Prefer simpler configurations. Do not combine near-miss settings until their
individual effects have been measured.
