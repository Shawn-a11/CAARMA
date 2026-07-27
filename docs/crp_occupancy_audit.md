# CRP Occupancy Audit

This audit reads the persistent synthetic-class registry from a Lightning
checkpoint without loading the training model or changing the checkpoint.

## Final-checkpoint distribution

```bash
python tools/audit_crp_occupancy.py \
  --checkpoint /path/to/last.ckpt
```

This reports singleton and doubleton ratios, median/p90/p99/maximum occupancy,
Gini coefficient, the entropy-based effective number of classes, registry
capacity usage, and visit concentration in the largest classes.

Some CAARMA checkpoints contain both `loss._extra_state.synth` and
`loss_syn._extra_state.synth` because the two Task attributes alias the same
criterion module. The audit verifies that their occupancy identities are
identical and collapses them into one report. It refuses to choose silently if
multiple non-identical states are found.

## Saturation timeline

```bash
python tools/audit_crp_occupancy.py \
  --checkpoint-dir /path/to/checkpoint_directory \
  --json-output /tmp/crp_occupancy_audit.json
```

The directory mode scans every saved checkpoint and reports the first checkpoint
observed with `len(pair_col) >= max_cols`. If only the best three checkpoints and
`last.ckpt` were retained, this is only the first *observed* saturation point.
For runs recorded before per-epoch logging existed, the exact saturation epoch
cannot be reconstructed without an epoch-by-epoch checkpoint.

## Per-epoch timeline log (exact saturation epoch)

Training now appends one JSON line per epoch to
`<save_dir>/crp_occupancy_timeline.jsonl` (rank 0, persistent mode only).
Each line holds the full occupancy metric set plus `display_epoch_one_based`,
`global_step`, and the per-epoch `active_synth_cols_epoch` count. Audit it with:

```bash
python tools/audit_crp_occupancy.py \
  --epoch-log /path/to/save_dir/crp_occupancy_timeline.jsonl
```

This prints a per-epoch table and the exact first saturated epoch. The flag can
be combined with `--checkpoint` / `--checkpoint-dir` in one report. Logging is
read-only with respect to training state and consumes no RNG, so runs with and
without it are numerically identical.

## Interpretation

- `allocated_pairs` is `len(pair_col)`. It measures consumed registry capacity.
- `occupied_classes` counts pairs with at least one successful synthetic visit.
- Occupancy metrics use only classes with positive visit counts.
- Effective classes are `exp(H(p))`, where `p_c = n_c / sum_c n_c`.
- Percentiles use the nearest-rank definition.
- A standard DDP checkpoint normally contains the rank-0 Python registry only;
  it is not a merged view of all ranks.

Do not claim a harmful heavy tail from one statistic alone. Inspect singleton
ratio, Gini, effective-class ratio, maximum-to-median contrast, and top-class
visit shares together, then compare them with downstream EER/minDCF.
