# ReDimNet2-B6 CAARMA Integration

## Question

Does the current full CAARMA extension remain useful when the MFA-Conformer
speaker encoder is replaced by a stronger speaker-verification architecture?

## Controlled Method

The experiment keeps the synthetic-class method from PSC Job `44396204`
unchanged:

- persistent synthetic classes;
- CRP popularity reuse with `alpha=1` and cosine `top-k=4` candidates;
- parent-SLERP initialization of `W_syn`;
- joint synthetic AM-Softmax loss;
- projection-conditioned discriminator.

Only the speaker encoder is replaced by the official ReDimNet2-B6 model. The
encoder starts from the official VoxCeleb2 large-margin checkpoint and receives
a lower learning rate (`1e-5`) than the new CAARMA classifier and synthetic
prototype parameters (`1e-3`). This prevents immediate destruction of the
pretrained speaker geometry while allowing CAARMA to adapt it.

## Interpretation

This run is a practical transfer experiment, not a training-data-matched
architecture ablation: the ReDimNet2 checkpoint was pretrained on VoxCeleb2.
A causal claim that CAARMA improves ReDimNet2 requires a later matched control
using the same checkpoint, training data, optimizer groups, and evaluation
protocol without the CAARMA losses.

## Provenance

- ReDimNet2 paper: https://arxiv.org/abs/2603.11841
- Official implementation: https://github.com/PalabraAI/redimnet2
- Vendored upstream commit: `cdc875670034dd7068013ca2ab21ec083a040ff8`
- Official B6 VoxCeleb2-LM checkpoint SHA-256:
  `bc45f032099f3c9f0fb6ad756bca23daeeba456d00c515056e388824ef36f41b`
