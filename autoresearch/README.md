# CAARMA Autoresearch Baseline Tuning

This branch adapts the controlled experiment loop from `karpathy/autoresearch`
to the four-GPU CAARMA baseline. It does not copy nanochat code and does not
allow the agent to edit model or evaluation code.

## Server setup

```bash
cd ~/autodl-tmp/CAARMA
git fetch origin
git checkout -B exp/autoresearch-caarma-baseline-tuning-ddp \
  origin/exp/autoresearch-caarma-baseline-tuning-ddp

conda activate caarma
export CAARMA_AUTORESEARCH_ROOT=/root/autodl-tmp/CAARMA/autoresearch_runs
mkdir -p "$CAARMA_AUTORESEARCH_ROOT"
```

Verify the fixed local assets:

```bash
test -f /root/autodl-tmp/hubert-large/config.json
test -f /root/autodl-tmp/CAARMA/voxceleb_full.csv
test -f /path/to/development_trials.txt
nvidia-smi
```

## Prepare one trial without training

```bash
python autoresearch/run_trial.py \
  --tag smoke_baseline_s42 \
  --budget 8 \
  --seed 42 \
  --trial-path /path/to/development_trials.txt \
  --dry-run
```

Inspect the generated configuration:

```bash
sed -n '1,180p' \
  "$CAARMA_AUTORESEARCH_ROOT/smoke_baseline_s42/config.yaml"
```

Delete that dry-run directory before launching the same tag, or choose a new
tag. Trial directories are immutable by design.

## Run the baseline

```bash
python autoresearch/run_trial.py \
  --tag baseline_b8_s42 \
  --budget 8 \
  --seed 42 \
  --trial-path /path/to/development_trials.txt \
  --description "unaltered reproduction baseline"
```

The command writes training output to:

```text
$CAARMA_AUTORESEARCH_ROOT/baseline_b8_s42/train.log
```

Follow progress from another SSH terminal:

```bash
tail -f "$CAARMA_AUTORESEARCH_ROOT/baseline_b8_s42/train.log"
```

## Run one approved change

```bash
python autoresearch/run_trial.py \
  --tag lr5e4_b8_s42 \
  --budget 8 \
  --seed 42 \
  --trial-path /path/to/development_trials.txt \
  --set init_lr=0.0005 \
  --description "optimization stage: lower main learning rate"
```

Overrides outside `search_space.yaml` are rejected.

## Promote a trial

```bash
python autoresearch/run_trial.py \
  --tag lr5e4_b30_s42 \
  --budget 30 \
  --seed 42 \
  --trial-path /path/to/development_trials.txt \
  --resume-from "$CAARMA_AUTORESEARCH_ROOT/lr5e4_b8_s42/checkpoints/last.ckpt" \
  --set init_lr=0.0005 \
  --description "promote lower-LR trial to full budget"
```

`budget` is the total epoch target, not the number of additional epochs.

## Register an existing experiment

Importing an existing log prevents the same configuration from being run again:

```bash
python autoresearch/run_trial.py \
  --tag existing_baseline_s42 \
  --budget 30 \
  --seed 42 \
  --trial-path /path/to/the_trials_used_for_that_run.txt \
  --import-log /path/to/existing_training.log \
  --description "existing paper-reproduction run"
```

For a controlled paper-reproduction check using a final test list, add
`--allow-final-test`. Do not use that flag for autonomous parameter selection.

## Rank trials

```bash
python autoresearch/summarize.py \
  "$CAARMA_AUTORESEARCH_ROOT/results.tsv"
```

All generated configs, logs, checkpoints, JSON, and TSV results remain
untracked. Only code and documentation belong in Git.
