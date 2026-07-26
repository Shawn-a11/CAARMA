# User Requirements: Natural-Cluster CRP

Execute the experiment tree's declared next step: replace the CRP v2 global
top-k nearest-neighbour candidate pool with candidates drawn from each
anchor's natural cluster, as one controlled change.

## Constraints

- Keep the CRP create-vs-reuse law, `crp_alpha`, joint-L_syn, SLERP
  generation, memory bank, persistence, and discriminator settings unchanged.
- Keep the per-anchor branching factor comparable: at most `crp_topk`
  candidates per anchor, now restricted to the anchor's own cluster.
- The clustering must be identical on every DDP rank without communication.
- `candidate_pool: "topk"` must reproduce the previous v2 behaviour exactly.
- Checkpoints must round-trip the new state fields and tolerate old
  checkpoints that lack them.
- Use an isolated `save_dir` so no earlier run is overwritten.

## Reference points for evaluation

- Top-k CRP persistent v2 (spectral D): 3.57 / .36 — the direct control.
- Prior candidate-policy attempts (hard gender 3.70, feedback alpha 3.70,
  gender p60–p90 3.55, boundary utility 3.56) — none clearly beat v2.
- The run inherits per-epoch occupancy logging, so the occupancy distribution
  under the cluster pool can be compared against the v2 audit directly.
