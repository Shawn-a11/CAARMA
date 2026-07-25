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

## Saturation timeline

```bash
python tools/audit_crp_occupancy.py \
  --checkpoint-dir /path/to/checkpoint_directory \
  --json-output /tmp/crp_occupancy_audit.json
```

The directory mode scans every saved checkpoint and reports the first checkpoint
observed with `len(pair_col) >= max_cols`. If only the best three checkpoints and
`last.ckpt` were retained, this is only the first *observed* saturation point.
The exact saturation epoch cannot be reconstructed without an epoch-by-epoch
checkpoint or an epoch-by-epoch `allocated_pairs` log.

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
