# Development Log

## Progress

| Item | Status | Notes |
|---|---|---|
| Isolated baseline branch | Done | Based on `origin/exp/repro-3.09-one-shot` |
| Expose hard-coded training parameters | Done | Defaults preserve parent behaviour |
| Config-path and full-resume support | Done | `train.py --config`; Lightning `ckpt_path` |
| Restricted trial runner | Done | Whitelist, duplicate detection, import, resume |
| Server documentation | Done | SSH setup and trial lifecycle |
| Tests and review | Done | 4 unit/integration tests; compile and diff checks pass |

## Decisions

- Use fixed epoch/step budgets rather than the upstream five-minute budget.
- Keep numerical search structured; the agent proposes trials but cannot edit
  model or evaluation code.
- Require an explicit flag before using a trial path containing `test`.

## Review

- Parent defaults remain unchanged: AM margin/scale 0.2/30, discriminator head
  LR 2e-4, HuBERT LR `init_lr * 0.01`, StepLR 4/0.5, and the source controller.
- Generated artifacts are outside tracked paths and duplicate fingerprints are
  rejected.
- Timeout handling terminates the complete DDP process group.
- `--resume-from` uses Lightning's full checkpoint resume rather than loading
  model weights alone.

## Running

See `autoresearch/README.md`. The generated `autoresearch_runs/` directory and
`autoresearch/results.tsv` are ignored by Git.
