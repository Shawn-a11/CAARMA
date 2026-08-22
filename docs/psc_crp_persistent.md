# PSC Port: CRP Persistent Synthetic Identity (ladder row B)

## Source

- Source branch: `exp/innovation-crp-persistent-synth-ddp` @ `ad37398`
- This PSC port changes only the launcher/config plumbing; the method code
  (`criterion/amsoftmax_mix_gan.py`, `helper/synth_table.py`, `model/discriminator_mix.py`,
  `train.py` training loop) is unchanged from the source commit.

## Method (unchanged from source)

- CRP create-vs-reuse pair selection (`pair_strategy: "crp"`, `crp_alpha: 1.0`):
  each anchor speaker chooses between reusing an already introduced synthetic pair
  and creating a new one, with stick-breaking-style mass `alpha / (visits + alpha)`.
- Persistent synthetic identity `W_syn`: each synthetic pair keeps a stable column
  across batches/epochs, accumulating training signal instead of being discarded
  after one batch (one-shot).
- Joint synthetic loss `L_syn`: synthetic embeddings and their persistent
  prototypes are trained jointly with the real AMSoftmax objective.
- Unconditional spectral discriminator (`discriminator_type: "spectral"`):
  a 1-hidden-layer MLP-D, unconditional on speaker identity.
- Global top-4 nearest-neighbour candidate pool per anchor (`crp_topk: 4`):
  new pairs are created from the anchor's top-4 neighbours in prototype space,
  recomputed each epoch from the DDP-synced real prototypes.

Note: `candidate_pool` is **not** a config key at this base commit — the top-k
neighbour pool is hardwired in `PersistentSynthState.rebuild_pairing`
(`helper/synth_table.py`), so the config carries only `crp_topk: 4`.

## PSC config

- Config: `config_psc_crp_persistent.yaml`
- Launcher: `scripts/psc/train_crp_persistent.slurm` (job name `caarma-crp`,
  4x V100-32, GPU-shared, 12h)
- Checkpoints: `${CAARMA_RUN_ROOT}/crp_persistent_synth/checkpoints`
- `train.py` `load_config` expands `${VAR}` environment variables (patch ported
  from the E3 PSC port, commit `82c98b2`) and normalises `root` with a trailing
  slash; unresolved `${...}` values raise a `ValueError`.

## Historical reference result

- Historical reference (source runtime, test-selected checkpoint):
  **EER 3.57%**, **minDCF ~0.36** on VoxCeleb1-O.
- These numbers were obtained on the original (non-PSC) runtime with
  test-selection; the PSC run must be reported **separately** and is not
  expected to reproduce them bit-for-bit.
