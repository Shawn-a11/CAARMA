# Corrected Popularity CRP Control

This branch is the infrastructure-matched control for persistent CRP
experiments. It keeps the original CRP popularity rule:

```text
create mass = alpha
reuse mass for class c = number of visits to c
```

It does not use Fisher utility, UCB, gender, a cosine window, boundary utility,
or natural clusters.

Compared with the historical CRP v2 implementation, this control fixes three
execution issues without changing the intended sampling rule:

1. Candidate pair columns are reserved deterministically, so every DDP rank
   maps a pair to the same `W_syn` column.
2. The discriminator and model updates reuse one pair-selection plan within a
   training step.
3. Visit and activation deltas are all-reduced before the next selection, so
   all ranks start from the same CRP occupancy state.

Use this result, not the historical 3.57 run alone, as the matched baseline for
later pair-pool experiments.
