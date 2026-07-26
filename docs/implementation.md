# Corrected Natural-Cluster CRP Implementation

## Controlled Change

The corrected popularity control builds each anchor's creation candidates from
the global top-k nearest-neighbour prototypes. This experiment replaces only
that candidate pool:

1. At every epoch start, the epoch-synced real prototypes `W` are clustered
   with deterministic spherical k-means (`num_clusters = round(C /
   cluster_size)`, cosine assignment, k-means++-style seeding, fixed seed,
   CPU float32).
2. An anchor's candidates are its top-`crp_topk` nearest neighbours *within
   its own cluster*, ranked by the same prototype cosine used by v2.
3. A singleton cluster falls back to the anchor's global top-1 neighbour, so
   every anchor keeps at least one candidate, as in v2.

`select_pair`, the create probability `alpha / (N_i + alpha)`,
popularity-weighted reuse, memory bank, joint-L_syn, and discriminator are
unchanged. Pair columns remain persistent across epochs even as clusters drift.

## Corrected Execution Invariants

1. Candidate pairs receive deterministic `W_syn` columns on every DDP rank.
2. D-step and M-step reuse the same sampled pair plan.
3. Visit and activation deltas are all-reduced before the next selection.
4. `reuse_policy: popularity` disables Fisher-UCB scoring.

## Motivation

- Top-k NN pairs can chain across the embedding manifold; within-cluster
  interpolation keeps synthetic speakers inside a natural speaker
  neighbourhood.
- The within-cluster pool generates fewer distinct pairs per epoch
  (measured ~2953 vs registry capacity 4844 at C=1211, k=4, mean cluster
  size 8), so the registry saturates later and reuse concentrates on
  coherent pairs.

## DDP Invariant

`rebuild_pairing` runs independently on every rank from identical synchronized
`W`. K-means is pinned to CPU float32 with a fixed seed, so cluster assignments,
candidate pools, and deterministic column reservations agree across ranks.

## Parameters (config.yaml)

- `candidate_pool`: `"topk"` (v2 behaviour, default) or `"cluster"`.
- `cluster_size`: target mean cluster size (default 8);
  `num_clusters = max(1, round(num_spk / cluster_size))`.
- `reuse_policy`: `"popularity"` for this experiment.
- `candidate_pool: "cluster"` requires `pair_strategy: "crp"` and
  `cluster_size >= 2`; violations raise at construction.

## Cost

One spherical k-means per epoch: ~0.07 s at C=1211, D=192 on CPU (measured),
negligible against an epoch of DDP training.

## Checkpoint State

`candidate_pool`, `cluster_size`, and the current `cluster_assign` are added
to the synth-state payload; loading tolerates their absence in older
checkpoints.
