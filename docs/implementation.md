# CAARMA Joint-Lsyn MLP-D Autoresearch

## Scope

Adapt the experiment loop from `karpathy/autoresearch` to CAARMA without copying
its nanochat training implementation. The target is the method baseline used by
the current paper chain, not the older HuBERT reproduction branch.

## Frozen components

- Dataset and evaluation implementation
- MFA-Conformer encoder
- MLP discriminator architecture (`192 -> 256 -> 128 -> 1`)
- Batch-local nearest-neighbor midpoint construction
- Joint real/synthetic AM-Softmax class construction
- Four-GPU DDP strategy
- No augmentation

## Editable trial parameters

Only keys declared in `autoresearch/search_space.yaml` may be overridden by the
trial runner. Default values must reproduce the parent branch exactly.

## Files

- `config.yaml`: explicit defaults for formerly hard-coded parameters.
- `train.py`: accepts `--config`, consumes explicit optimizer/loss parameters,
  and supports full Lightning resume.
- `criterion/build_criterion.py`: consumes AM-Softmax margin and scale.
- `autoresearch/run_trial.py`: validates overrides, creates an isolated trial
  directory, runs training, parses metrics, and appends an untracked TSV.
- `autoresearch/summarize.py`: ranks completed trials.
- `autoresearch/search_space.yaml`: whitelist and staged candidate values.
- `autoresearch/program.md`: operating policy for an autonomous coding agent.
- `autoresearch/README.md`: SSH/server instructions.
- `tests/test_autoresearch_tools.py`: parser and validation coverage.

## Result policy

The primary objective is development EER. minDCF(10-2) breaks near ties.
Generated artifacts are excluded from Git. Final candidates require full-budget
training and multiple seeds.

## Method boundary

This branch intentionally preserves the parent implementation's arithmetic
midpoint mixing. Converting it to true SLERP is a method change and must be
tested on a separate branch rather than attributed to hyperparameter tuning.
