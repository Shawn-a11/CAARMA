#!/usr/bin/env python3
"""
No-GPU (CPU-only) diagnostic: WHY do synthetic-class prototypes collide (cos~1)?

It replays CAARMA's *original* synthetic-class construction on a trained
checkpoint's loss.W and classifies every high-cosine prototype pair as either:

  (A) SAME unordered parent pair  -> ordered-key duplicate (Bug B): the loop key
      int(str(l1)+str(l2)) makes mutual-NN pairs (i,j) & (j,i) two classes whose
      midpoint is identical (LERP/SLERP midpoint is symmetric) -> cos = 1.0.
      Fix = unordered key (already done in exp/innovation-mixup-dedup-ddp).

  (B) DIFFERENT parent pairs       -> genuine midpoint collision in W-space.
      Fix = a "new synthetic class must be far from existing ones" constraint.

NOTE on vMF: in the SLERP+vMF branch, vMF is sampled on the *embedding* x_mix
(one sample each), NOT on the class prototype w_mix (a deterministic SLERP
midpoint). So kappa CANNOT move this prototype spike; this script measures the
prototype geometry, which is what AM-Softmax classifies against.

Usage (on the server, no GPU needed):
    python diag_synth_collision.py --ckpt /path/to/last.ckpt
    python diag_synth_collision.py --ckpt ... --mode slerp   # for vMF/SLERP ckpt
"""
import argparse, math
import numpy as np
import torch


def load_W(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    cand = []
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dim() == 2 and (k.endswith("loss.W") or k.endswith(".W")):
            cand.append((k, v))
    if not cand:  # fallback: any 2D tensor that looks like (emb=192, num_spk)
        for k, v in sd.items():
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] in (192, 256, 512):
                cand.append((k, v))
    if not cand:
        raise SystemExit("Could not find loss.W in checkpoint.")
    k, W = cand[0]
    print(f"Loaded prototype matrix '{k}' shape={tuple(W.shape)} (emb, num_spk)")
    return W.float()


def buggy_dic_spk(W, set_label):
    """Verbatim replica of the upstream (buggy) nearest-neighbour pairing."""
    dic = {}
    for s in set_label:
        dists = [torch.dist(W[:, s], W[:, t]) for t in set_label if s != t]
        idx = torch.argmin(torch.tensor(dists))
        closest = set_label[idx]                       # BUG A: filtered idx on full list
        if s == int(closest):
            _, si = torch.sort(torch.tensor(dists))
            dic[s] = int(set_label[si[1]])
        else:
            dic[s] = int(closest)
    return dic


def true_dic_spk(W, set_label):
    dic = {}
    for s in set_label:
        best, bd = None, 1e9
        for t in set_label:
            if t == s:
                continue
            d = torch.dist(W[:, s], W[:, t]).item()
            if d < bd:
                bd, best = d, t
        dic[s] = best
    return dic


def midpoint(W, a, b, mode):
    if mode == "lerp":
        return (W[:, a] + W[:, b]) / 2.0
    # slerp(t=0.5)
    p0 = W[:, a] / W[:, a].norm().clamp(min=1e-7)
    p1 = W[:, b] / W[:, b].norm().clamp(min=1e-7)
    dot = (p0 * p1).sum().clamp(-1 + 1e-7, 1 - 1e-7)
    om = torch.acos(dot); so = torch.sin(om)
    return (torch.sin(0.5 * om) / so) * p0 + (torch.sin(0.5 * om) / so) * p1


def run(W, pairing, mode, batch_size, n_batches, thr, seed=0):
    rng = np.random.default_rng(seed)
    nspk = W.shape[1]
    same_pair = cross_pair = exact_dup = hi = 0
    for _ in range(n_batches):
        set_label = sorted(rng.choice(nspk, size=min(batch_size, nspk), replace=False).tolist())
        dic = (buggy_dic_spk if pairing == "buggy" else true_dic_spk)(W, set_label)
        # ordered-key loop -> list of (prototype, unordered_pair)
        newlabel, protos, upairs = {}, [], []
        for l1 in set_label:                       # one anchor per speaker (worst case for dup)
            l2 = dic[l1]
            key = int(str(int(l1)) + str(int(l2)))   # ORDERED key (Bug B)
            if key not in newlabel:
                newlabel[key] = len(protos)
                protos.append(midpoint(W, l1, l2, mode))
                upairs.append(tuple(sorted((int(l1), int(l2)))))
        if len(protos) < 2:
            continue
        P = torch.stack(protos)
        P = P / P.norm(dim=1, keepdim=True).clamp(min=1e-9)
        S = (P @ P.t()).numpy()
        n = len(protos)
        for a in range(n):
            for b in range(a + 1, n):
                if S[a, b] > thr:
                    hi += 1
                    if S[a, b] > 0.99999:
                        exact_dup += 1
                    if upairs[a] == upairs[b]:
                        same_pair += 1
                    else:
                        cross_pair += 1
    return same_pair, cross_pair, exact_dup, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["lerp", "slerp"], default="lerp",
                    help="lerp for plain/MLP-D baseline; slerp for vMF/SLERP ckpt")
    ap.add_argument("--pairing", choices=["buggy", "true"], default="buggy",
                    help="buggy = upstream code as-run; true = correct NN (post-fix)")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--n_batches", type=int, default=300)
    ap.add_argument("--thr", type=float, default=0.95)
    args = ap.parse_args()

    W = load_W(args.ckpt)
    same, cross, exact, hi = run(W, args.pairing, args.mode,
                                 args.batch_size, args.n_batches, args.thr)
    print(f"\n=== synthetic-prototype collisions  (mode={args.mode}, pairing={args.pairing},"
          f" {args.n_batches} batches of {args.batch_size}) ===")
    print(f"  high-cos pairs (cos > {args.thr}): {hi}")
    print(f"    exact duplicates (cos > 0.99999): {exact}")
    if hi:
        print(f"    SAME unordered parent pair  : {same:5d}  ({100*same/hi:.0f}%)  -> Bug B (ordered-key dup)")
        print(f"    DIFFERENT parent pairs      : {cross:5d}  ({100*cross/hi:.0f}%)  -> cross-pair collision")
        print("\nVERDICT:", end=" ")
        if same >= 4 * max(cross, 1):
            print("dominated by SAME-pair -> it's the ordered-key duplicate (Bug B).")
            print("  Fix = unordered key (done: exp/innovation-mixup-dedup-ddp). kappa is irrelevant.")
        elif cross >= same:
            print("substantial CROSS-pair collisions -> add a min-distance dedup constraint")
            print("  on newly created synthetic classes (hypothesis 2).")
        else:
            print("mostly SAME-pair with some cross-pair; unordered-key fix handles the bulk.")


if __name__ == "__main__":
    main()
