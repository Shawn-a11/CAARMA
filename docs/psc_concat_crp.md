# PSC port: Concat-conditioned MLP-D + persistent synthetic classes + CRP pairing

## Source

- Source branch: `exp/ablation-concat-mlpd-crp-persistent-synth-ddp` @ `29dc0c0`
- This branch ports ladder row C to PSC Bridges-2 with no method changes.

## Method

The method is unchanged from the source branch. The discriminator is
concat-conditioned: it sees `[e, q]`, where `q = W_y` for a real pair and
`q = W_syn[c]` for a synthetic pair (`c` is the sampled persistent synthetic
class). Concretely, `D([e, q]) = MLP(concat(e, normalize(q)))`, implemented by
`ConcatConditionDiscriminator_spectral` in `model/discriminator_mix.py` and
selected via `discriminator_type: "concat"` (`train.py`). Synthetic classes are
persistent (learnable `W_syn` table + per-speaker memory bank) and pairing uses
CRP create-vs-reuse sampling (`pair_strategy: "crp"`, `crp_alpha: 1.0`,
`crp_topk: 4`).

## Historical reference

- Historical reference EER: **3.51%** (test-selected, historical runtime on the
  source branch's AutoDL environment).

## PSC configuration

- Config: `config_psc_concat_crp.yaml` (paths via `${CAARMA_*}` env
  interpolation; `train.py` `load_config` expands and rejects unresolved
  `${...}` values).
- Launcher: `scripts/psc/train_concat_crp.slurm` (job-name `caarma-concat`,
  4x V100-32, run dir `$CAARMA_RUN_ROOT/concat_crp_persistent`).
- Report the PSC result separately from the historical reference; do not mix
  numbers across runtimes.
