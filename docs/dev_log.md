# Development Log: Prototype-Only Virtual Speakers

## Progress

| Component | Status | Notes |
|---|---|---|
| Experiment branch | Done | Created from the reported 3.61 branch |
| Virtual-negative AM-Softmax | Done | Shape, gradient, and loss tests passed |
| Real-only training path | In progress | Syntax passed; server smoke test pending |
| Server training | Pending | Full DDP run is performed remotely |

## 2026-07-21: Initial Implementation

- Added detached SLERP virtual prototypes generated from real classifier
  weights.
- Added the prototypes only to the AM-Softmax denominator.
- Disabled synthetic-positive, discriminator, and generator updates in this
  experiment mode.
- Kept the method stateless across batches and ranks.
- Allowed a zero virtual-negative budget for a matched real-only control.
- Added tests for denominator-only behavior, gradients, rejection of synthetic
  positives, and the zero-budget control.

## 2026-07-21: Local Verification

- Python syntax checks passed for the criterion, criterion builder, training
  module, and tests.
- Three tensor-level tests passed under PyTorch 2.6.0.
- Full Lightning/DDP startup remains pending because the local environment does
  not include the repository's PyTorch Lightning stack.

## Run Instructions

From the repository root on the GPU server:

```bash
mkdir -p logs
nohup python train.py > logs/prototype_only_virtual_speakers_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Startup should print `prototype-only virtual-speaker mode`. Monitor `cosine
EER` and `cosine minDCF` in the generated log.
