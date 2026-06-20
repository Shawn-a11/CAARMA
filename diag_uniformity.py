#!/usr/bin/env python3
"""Wang-Isola uniformity of speaker prototypes / embeddings — money-plot axis.

Goal: test whether uniformity governs zero-shot EER. For each config, compute
uniformity here and pair it with that run's best EER -> uniformity-vs-EER scatter.

Uniformity (Wang & Isola 2020), lower = more uniform = (theory) better:
    L_unif = log E_{i != j} exp(-tau * || u_i - u_j ||^2),  u = L2-normalised.

Two axes:
  --ckpt  : prototype uniformity from loss.W (CPU, zero-cost). Reports real-only
            and real+synthetic (correct-NN, deduped midpoints). NB: this is a
            TRAINING-TIME proxy; AM-Softmax already makes real prototypes nearly
            orthogonal, so most variation is in the synthetic set.
  --emb   : uniformity on a saved (N,D) .npy of TEST embeddings (the geometry
            actually evaluated at test time -> the theory-faithful axis). To get
            it, dump self.eval_vectors at the end of on_validation_epoch_end:
                np.save('eval_emb.npy', np.asarray(eval_vectors))

Usage:
    python diag_uniformity.py --ckpt caarma_mfa_ckpts_xxx/last.ckpt
    python diag_uniformity.py --emb  eval_emb.npy
"""
import argparse
import numpy as np
import torch


def uniformity(U, tau=2.0):
    """U: (N, D) L2-normalised rows. Returns log E_{i!=j} exp(-tau ||u_i-u_j||^2)."""
    U = torch.nn.functional.normalize(U, dim=1)
    sq = torch.cdist(U, U) ** 2           # (N,N) ||u_i-u_j||^2 = 2-2cos on sphere
    N = U.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool)
    return torch.log(torch.exp(-tau * sq[mask]).mean()).item()


def load_W(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    for suff in ("loss.W", ".W"):
        for k, v in sd.items():
            if torch.is_tensor(v) and v.dim() == 2 and k.endswith(suff):
                return v.float()
    raise SystemExit("loss.W not found in checkpoint")


def reconstruct_synth(W):
    """Correct-NN, deduped LERP-midpoint synthetic prototypes from real W (D,C)."""
    Dmat = torch.cdist(W.t(), W.t())
    Dmat.fill_diagonal_(float("inf"))
    nn = Dmat.argmin(1)
    pairs = {(min(i, int(nn[i])), max(i, int(nn[i]))) for i in range(W.shape[1])}
    return torch.stack([(W[:, a] + W[:, b]) / 2 for a, b in pairs], dim=1), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--emb", help="(N,D) .npy of test embeddings")
    ap.add_argument("--tau", type=float, default=2.0)
    args = ap.parse_args()

    if args.emb:
        U = torch.from_numpy(np.load(args.emb)).float()
        print(f"test embeddings: {tuple(U.shape)}")
        print(f"uniformity (tau={args.tau}, lower=more uniform): {uniformity(U, args.tau):.4f}")
        print("  -> theory-faithful axis; pair with best EER for the money plot.")
        return

    if not args.ckpt:
        raise SystemExit("give --ckpt or --emb")
    W = load_W(args.ckpt)                       # (D, C)
    u_real = uniformity(W.t(), args.tau)
    syn, n_syn = reconstruct_synth(W)
    u_all = uniformity(torch.cat([W, syn], dim=1).t(), args.tau)
    print(f"prototypes: {W.shape[1]} real + {n_syn} synthetic (correct-NN, deduped)")
    print(f"uniformity (tau={args.tau}, lower=more uniform):")
    print(f"  real only        : {u_real:.4f}")
    print(f"  real + synthetic : {u_all:.4f}   (delta {u_all - u_real:+.4f})")
    print("Pair this number with the run's best EER to build the uniformity-vs-EER scatter.")


if __name__ == "__main__":
    main()
