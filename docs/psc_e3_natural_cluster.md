# PSC E3 Natural-Cluster Projection-D

This branch ports `exp/innovation-natural-cluster-slerp-projection-persistent-synth-ddp@7d318cc`
to Bridges-2. The method is unchanged: SLERP-initialized persistent synthetic
prototypes, popularity CRP, natural-cluster-constrained cosine top-4 parent
selection, and Projection-D compatibility. Only config loading, PSC paths,
worker count and Slurm launch are adapted.

The historical PPU test-selected result is 3.25% EER with minDCF 0.3272 / 0.4141.
The PSC result must be reported separately until a complete curve is available.
