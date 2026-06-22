# Discriminator Visualization Assets — slide 6 source materials

Generates 4 visualization assets answering Bhiksha's four questions about how
HuBERT/WavLM functions as a CAARMA discriminator.

## Run

```bash
# from repo root
pip install torchview torchinfo torchviz graphviz onnx
brew install graphviz                    # macOS — or: apt install graphviz
python viz/run_all.py
```

CPU is sufficient. First run downloads the SSL backbone into HF cache (~1.2GB
for HuBERT-Large). Subsequent runs are seconds.

## Outputs

| File | Tool | Slide it serves |
|---|---|---|
| `d_arch.png` + `.gv` | torchview | **6.2** D architecture — hierarchical forward graph |
| `d_summary.txt` | torchinfo | **6.2** Layer table caption (shape + params + trainable) |
| `d_backward.pdf/.png` | torchviz | **6.4** Gradient flow — autograd backward graph |
| `d.onnx` | torch.onnx | **6.1** Open in Netron, screenshot input node to prove "input is 192-d embedding, not audio" |

## What each tool can / cannot show

These tools draw **the structure that's in the code**. They directly answer:

- **Q1 (input contradiction)**: ONNX/torchview show the input is `(B, 192)`
  embedding, then the EnhancedAdapter projects to `(B, 1, 1024)` before the
  WavLM/HuBERT encoder. Visual proof the adapter trick resolves the
  contradiction.
- **Q2 (where is HuBERT)**: torchview at `depth=3` makes the WavLM/HuBERT
  Transformer block visible inside D between the adapter and the classifier
  head. d_summary.txt shows it as a single line item with its 315M params.
- **Q4 (gradient flow)**: torchviz draws the autograd backward graph, showing
  the gradient path from logit through every layer of D back to the input
  embedding.

What they cannot show:

- **Q3 (alternating training)** — the dual-optimizer toggle (`opt_D` step vs
  `opt_M` step, `toggle_optimizer` context, encoder forward inside `no_grad`)
  is a Python control-flow construct, not part of the forward graph. Must be
  illustrated with a manual diagram.
- **Trainable vs frozen distinction in the backward graph** — torchviz shows
  the gradient path exists, but doesn't annotate which params get updated by
  which optimizer. Manual red/green overlay required for slide 6.4.

## Branch sensitivity

The discriminator changes across branches. Pick the branch matching the slide
you want to present:

- `exp/ddp-4gpu-v100` / `exp/rewrite-algorithm2-correct-gan` → HuBERT-Large
- `exp/innovation-wavlm-ddp` → WavLM-Large
- `exp/innovation-wavlm-base-plus-ddp` → WavLM-Base+ (smaller, faster viz)
- `exp/innovation-wavlm-wganpgp-ddp` → WavLM + WGAN-GP (point 5 of your roadmap)

Re-run `python viz/run_all.py` after switching branches to regenerate.
