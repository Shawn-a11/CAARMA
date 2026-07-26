# Development Log

## Progress

| Item | Status | Notes |
|---|---|---|
| Isolated baseline branch | Done | Based on `origin/exp/innovation-joint-lsyn-ddp` |
| Baseline audit | Done | MLP-D; joint denominator; arithmetic midpoint mixing |
| Expose hard-coded training parameters | Done | Defaults preserve parent behaviour |
| Config-path and full-resume support | Done | `train.py --config`; Lightning `ckpt_path` |
| Restricted trial runner | Done | Whitelist, duplicate detection, import, resume |
| Server documentation | Done | SSH setup and staged trial lifecycle |
| Tests and review | Done | Compile, 4 unit tests, dry-run, and diff audit pass |

## Decisions

- Use fixed epoch/step budgets rather than the upstream five-minute budget.
- Keep numerical search structured; the agent proposes trials but cannot edit
  model or evaluation code.
- Tune the `3.61` MLP-D baseline, not the HuBERT-D reproduction branch.
- Keep midpoint construction frozen; true SLERP belongs in a separate ablation.
- Require an explicit flag before using a trial path containing `test`.

## Review

- Parent defaults remain unchanged: AM margin/scale 0.2/30, discriminator LR
  2e-4, StepLR 4/0.5, joint-Lsyn scale 1.0, and the source controller.
- Generated artifacts are outside tracked paths and duplicate fingerprints are
  rejected.
- Timeout handling terminates the complete DDP process group.
- `--resume-from` uses Lightning's full checkpoint resume rather than loading
  model weights alone.

## Running

See `autoresearch/README.md`. The generated `autoresearch_runs/` directory and
`autoresearch/results.tsv` are ignored by Git.
