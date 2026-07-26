# User Requirements: Corrected Natural-Cluster CRP

Execute the experiment tree's declared next step: replace the CRP v2 global
top-k nearest-neighbour candidate pool with candidates drawn from each
anchor's natural cluster, as one controlled change.

## Constraints

- Use the corrected popularity CRP implementation as the matched control.
- Keep the CRP create-vs-reuse law, `crp_alpha`, popularity reuse, joint-L_syn,
  SLERP generation, memory bank, persistence, and discriminator unchanged.
- Keep the per-anchor branching factor comparable: at most `crp_topk`
  candidates per anchor, now restricted to the anchor's own cluster.
- The clustering must be identical on every DDP rank without communication.
- `candidate_pool: "topk"` must reproduce the corrected control behaviour.
- Pair columns and visit counts must remain synchronized across DDP ranks.
- D-step and M-step must use the same selected persistent pair.
- Checkpoints must round-trip the new state fields and tolerate old
  checkpoints that lack them.
- Use an isolated `save_dir` so no earlier run is overwritten.

## Evaluation

- Historical top-k CRP v2: 3.57 / .36 — context only.
- Corrected top-k popularity CRP — direct matched control to run first.
- Prior candidate-policy attempts (hard gender 3.70, feedback alpha 3.70,
  gender p60–p90 3.55, boundary utility 3.56) — none clearly beat v2.
- The run inherits per-epoch occupancy logging, so the occupancy distribution
  under the cluster pool can be compared against the corrected control.
