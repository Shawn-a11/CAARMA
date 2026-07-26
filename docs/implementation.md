# CAARMA Autoresearch Baseline Tuning

## Scope

Adapt the experiment loop from `karpathy/autoresearch` to CAARMA without copying
its nanochat training implementation.

## Frozen components

- Dataset and evaluation implementation
- MFA-Conformer encoder
- HuBERT discriminator architecture
- One-shot LERP synthetic-speaker construction
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
