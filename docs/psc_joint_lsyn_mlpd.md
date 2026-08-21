# PSC Joint-Lsyn MLP-D Reproduction

This branch ports the completed `exp/innovation-joint-lsyn-ddp@3e5ef36`
experiment to Bridges-2 without changing its method.

Preserved settings:

- MFA-Conformer encoder;
- plain embedding MLP discriminator (`192 -> 256 -> 128 -> 1`);
- SLERP synthetic embeddings and joint real/synthetic AM-Softmax;
- four-GPU DDP training behavior;
- batch size 50 per rank, 30 epochs, and the original optimizer schedule.

PSC-only changes are limited to an environment-driven config path, dataset,
trial, evaluation-root and output paths, four data-loader workers, and a Slurm
launcher. No HuBERT model or Hugging Face download is used by this experiment.

The historical AutoDL result is 3.61% test-selected VoxCeleb1-O EER. The PSC
run is a platform reproduction and must be reported separately until its full
curve is available.
