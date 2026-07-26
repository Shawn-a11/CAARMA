# Development Log

## 2026-07-26 - Natural-Cluster CRP

- Created `exp/innovation-natural-cluster-crp-persistent-synth-ddp` from the
  diagnostic branch tip, inheriting the CRP v2 saturation fix, checkpoint
  persistence, full Lightning resume, and the per-epoch occupancy timeline
  (training-neutral diagnostics).
- Added deterministic spherical k-means candidate pools to
  `helper/synth_table.py` behind `candidate_pool: "cluster"`; the default
  `"topk"` path is byte-for-byte the previous v2 flow.
- Threaded `candidate_pool` / `cluster_size` through the criterion builder
  and `config.yaml`; switched `save_dir` to
  `caarma_mfa_ckpts_natural_cluster_crp_persistent_synth_ddp`.
- Extended checkpoint state with the new fields (backwards tolerant).

## Verification (local)

- 8 focused unit tests pass (`python3 tests/test_natural_cluster_crp.py`):
  constructor validation, cross-state determinism, within-cluster candidate
  containment, deterministic singleton fallback to the global NN, saturation
  behaviour through `select_pair`, column persistence across cluster drift,
  state round-trip, and a top-k regression guard.
- Existing suites still pass: `tests/test_crp_occupancy_audit.py` (3),
  `tests/test_epoch_occupancy_timeline.py` (6).
- Realistic-scale check (C=1211, D=192, cluster_size=8): 151 clusters,
  sizes 3–14 (median 8), 3.95 mean candidates per anchor, 2953 unique
  epoch-pairs vs 4844 capacity, k-means 0.07 s, bitwise deterministic across
  two independent states.
- Full Lightning/DDP startup remains a server-side step (pytorch_lightning
  is not installed locally).

## Running

From the repository root on the GPU server, with this branch's
`config.yaml` (four-GPU `train.py` entry point):

```bash
mkdir -p logs
nohup python train.py > logs/natural_cluster_crp_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Startup must print `candidate_pool=cluster cluster_size=8` in the criterion
banner. Compare against CRP v2 (3.57 / .36) on cosine EER / minDCF, and
compare `crp_occupancy_timeline.jsonl` (exact saturation epoch, singleton
ratio, Gini, effective-class ratio) against the v2 audit.

For the matched control, set `candidate_pool: "topk"` with a separate
`save_dir`.
