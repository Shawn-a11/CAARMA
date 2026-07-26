# Development Log

## 2026-07-27 - Corrected Natural-Cluster CRP

- Based the experiment on the corrected popularity control rather than the
  historical CRP v2 execution path.
- Retained deterministic pair-column reservation, one shared D/M selection
  plan, and DDP all-reduction of visit and activation state.
- Kept `reuse_policy: popularity`; Fisher utility and UCB are disabled.
- Changed only the candidate pool from global top-k neighbours to top-k
  neighbours within deterministic spherical k-means clusters.
- Added isolated checkpoints and per-epoch occupancy logging.

## Verification

- Natural-cluster tests cover constructor validation, deterministic clustering,
  within-cluster candidate containment, singleton fallback, saturation,
  persistence across cluster drift, checkpoint round-trip, and the top-k
  regression path.
- Corrected-infrastructure tests cover shared selection plans and DDP state
  synchronization.
- Full Lightning/NCCL startup remains a server-side test.

## Evaluation

Run the corrected popularity control first, then this branch with the same seed
and training budget. Attribute the difference only to the candidate pool:

```text
control: candidate_pool=topk,    reuse_policy=popularity
method:  candidate_pool=cluster, reuse_policy=popularity
```

The historical 3.57 result remains context, but is not the matched control
because it predates the registry synchronization fixes.
