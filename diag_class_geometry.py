"""Zero-cost geometric diagnostic for Hard Class Augmentation.

Loads the AM-Softmax class prototypes W from a trained checkpoint and measures
the geometry that decides whether difficulty-aware mixup pairing can work —
WITHOUT running any training.

Run on the server (where checkpoints live):
    python diag_class_geometry.py /root/autodl-tmp/CAARMA/caarma_mfa_ckpts_xxx/last.ckpt

What it answers:
  1. Distribution of pairwise prototype cosine s_ij (how crowded is class space).
  2. Where do NEAREST-NEIGHBOUR pairs (what CAARMA actually mixes) fall on s_ij.
  3. Synthetic-vs-PARENT cosine = sqrt((1+s_ij)/2)   [the 0.28 heuristic check].
  4. Synthetic-vs-SYNTHETIC cosine: nearest other synthetic prototype
     (the ACTUAL L_syn competition — the code's L_syn is syn-vs-syn, parents
     are NOT in the softmax).
  5. Whether a usable difficulty window exists between "collapse" and "trivial".

NOTE: the 0.28 collapse boundary assumes parent-in-softmax, which the code does
NOT do. So treat it as a heuristic anchor; the syn-vs-syn crowding (point 4) is
the real collapse signal. This script measures BOTH so we decide empirically.
"""
import sys
import numpy as np
import torch

M_MARGIN = 0.2   # AM-Softmax margin used in CAARMA


def load_W(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    key = next((k for k in sd if k.endswith("loss.W") or k.endswith(".W")), None)
    if key is None:
        raise KeyError(f"no AM-Softmax W found. keys sample: {list(sd)[:8]}")
    W = sd[key].float().numpy()          # (emb_dim, num_spk)
    print(f"[load] {key}  shape={W.shape}")
    return W


def main(ckpt_path):
    W = load_W(ckpt_path)                 # (D, C)
    D, C = W.shape
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)   # unit columns

    S = Wn.T @ Wn                          # (C, C) pairwise cosine
    iu = np.triu_indices(C, k=1)
    s_all = S[iu]                          # all distinct pairs
    np.fill_diagonal(S, -np.inf)
    s_nn = S.max(axis=1)                   # nearest-neighbour cosine per speaker
    nn_idx = S.argmax(axis=1)

    s_collapse = 2 * (1 - M_MARGIN) ** 2 - 1   # heuristic (parent-in-softmax)

    print("\n================ class-prototype geometry ================")
    print(f"num classes C = {C},  emb_dim D = {D}")
    print(f"all-pairs s_ij : mean={s_all.mean():.3f}  p50={np.median(s_all):.3f}  "
          f"p95={np.percentile(s_all,95):.3f}  max={s_all.max():.3f}")
    print(f"NN-pairs  s_ij : mean={s_nn.mean():.3f}  p50={np.median(s_nn):.3f}  "
          f"p05={np.percentile(s_nn,5):.3f}  min={s_nn.min():.3f}  max={s_nn.max():.3f}")
    print(f"collapse heuristic boundary (parent-in-softmax): s_ij < {s_collapse:.3f}")
    frac_nn_above = float((s_nn > s_collapse).mean())
    print(f"fraction of NN pairs ABOVE heuristic boundary  : {frac_nn_above*100:.1f}%")

    # synthetic prototypes for NN pairing: W_syn = normalize(W_i + W_nn(i))
    Wsyn = Wn + Wn[:, nn_idx]
    Wsyn = Wsyn / (np.linalg.norm(Wsyn, axis=0, keepdims=True) + 1e-12)

    cos_syn_parent = (Wsyn * Wn).sum(axis=0)
    cos_syn_parent_analytic = np.sqrt((1 + s_nn) / 2)
    print(f"\nsyn-vs-parent cos: empirical={cos_syn_parent.mean():.3f}  "
          f"analytic sqrt((1+s)/2)={cos_syn_parent_analytic.mean():.3f}  (should match)")

    Ssyn = Wsyn.T @ Wsyn
    np.fill_diagonal(Ssyn, -np.inf)
    cos_syn_syn = Ssyn.max(axis=1)
    print(f"syn-vs-syn cos (nearest other synth): mean={cos_syn_syn.mean():.3f}  "
          f"p95={np.percentile(cos_syn_syn,95):.3f}  max={cos_syn_syn.max():.3f}")
    print("  -> high = synthetic classes collide with each OTHER (the real L_syn collapse)")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].hist(s_all, bins=80, alpha=0.5, density=True, label="all pairs")
        ax[0].hist(s_nn, bins=60, alpha=0.7, density=True, label="nearest-neighbour")
        ax[0].axvline(s_collapse, color="r", ls="--", label=f"collapse heuristic {s_collapse:.2f}")
        ax[0].set_xlabel("prototype cosine  s_ij"); ax[0].set_ylabel("density")
        ax[0].set_title("Class-pair difficulty"); ax[0].legend(fontsize=8)
        ax[1].hist(cos_syn_syn, bins=60, alpha=0.7, color="purple", density=True)
        ax[1].set_xlabel("syn-vs-nearest-syn cosine")
        ax[1].set_title("Synthetic-class crowding (real L_syn competition)")
        fig.tight_layout(); fig.savefig("diag_class_geometry.png", dpi=150)
        print("\n[plot] saved diag_class_geometry.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")

    print("\n================ verdict ================")
    if frac_nn_above > 0.5:
        print("NN pairs mostly above collapse heuristic -> CAARMA may mix too-hard pairs.")
        print("=> difficulty-band pairing worth testing (run the diagnostic plot to confirm a window).")
    else:
        print("NN pairs mostly below heuristic -> NN pairs may already be safe.")
        print("=> difficulty may NOT be the lever; reconsider before spending training runs.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python diag_class_geometry.py <checkpoint.ckpt>")
        sys.exit(1)
    main(sys.argv[1])
