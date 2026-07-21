# Prototype-Only Virtual Speakers: Implementation

## Base and Scope

Base branch: `exp/innovation-joint-lsyn-ddp` (reported cosine EER 3.61%).

This experiment replaces the synthetic-positive path with denominator-only
virtual classifier prototypes. It disables synthetic embeddings, `L_syn`, and
adversarial discriminator training so the meeting hypothesis is tested in
isolation.

## Data Flow

1. Normalize the real AM-Softmax speaker weights `W`.
2. For at most eight unique speakers in the current batch, sample a partner
   from the anchor's four nearest real prototypes.
3. Create `V = SLERP(W_i, W_j, 0.5)` under `torch.no_grad()`.
4. Append `V` to the AM-Softmax logits after applying the margin only to the
   correct real class.
5. Keep every target label in `[0, num_real)`, so virtual columns occur only in
   the denominator and can never receive a positive example.

## File Changes

- `criterion/build_criterion.py`: passes the virtual-negative configuration.
- `criterion/amsoftmax_mix_gan.py`: creates detached virtual prototypes and
  computes denominator-only AM-Softmax.
- `train.py`: uses a real-only optimization path without D/G or `L_syn` steps.
- `config.yaml`: selects this experiment and its checkpoint directory.

## Correctness Conditions

- The loss returns an empty synthetic-embedding tensor in this mode.
- Calling `flagSyn=True` raises an error instead of silently adding positives.
- Virtual prototypes are detached and have no optimizer state.
- No Python registry is used, avoiding CRP/DDP state mismatch in this test.
- Setting `virtual_negatives_per_batch: 0` gives a matched real-only control
  with the same optimizer and training path.
