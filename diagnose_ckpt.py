import argparse
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset

from feature.build_feature import build_feature
from functions.dataset import load_audio
from model.model_build import build_model


VOX_ID_RE = re.compile(r"id\d+")


class TrialAudioDataset(Dataset):
    def __init__(self, rel_paths, root):
        self.rel_paths = list(rel_paths)
        self.root = root

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, index):
        rel = self.rel_paths[index]
        wav = load_audio(os.path.join(self.root, rel), second=-1)
        return {"waveform": wav.float(), "rel_path": rel}


def collate_one(batch):
    # Evaluation uses batch size 1 because utterances have variable full length.
    return batch[0]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_prefixed(module, state_dict, prefix, strict=False):
    plen = len(prefix)
    sub = {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sub, strict=strict)
    return len(sub), missing, unexpected


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return float(eer), float(threshold)


def compute_min_dcf(labels, scores, p_target=0.01, c_miss=1, c_fa=1):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    best = float("inf")
    best_th = thresholds[0]
    for i in range(len(fnr)):
        cost = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if cost < best:
            best = cost
            best_th = thresholds[i]
    default = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(best / default), float(best_th)


def load_gender_map(meta_csv):
    if not meta_csv or not os.path.exists(meta_csv):
        return {}
    meta = pd.read_csv(meta_csv, sep="\t")
    meta.columns = [c.strip() for c in meta.columns]
    id_col = [c for c in meta.columns if "VoxCeleb1" in c][0]
    return {
        str(row[id_col]).strip(): str(row["Gender"]).strip()
        for _, row in meta.iterrows()
        if "Gender" in meta.columns
    }


def vox_id(path):
    match = VOX_ID_RE.search(str(path))
    return match.group(0) if match else None


def describe_scores(name, values):
    if len(values) == 0:
        return "{}: n=0".format(name)
    arr = np.asarray(values, dtype=np.float64)
    return (
        "{}: n={} mean={:.4f} std={:.4f} p05={:.4f} p50={:.4f} p95={:.4f}".format(
            name, len(arr), arr.mean(), arr.std(),
            np.percentile(arr, 5), np.percentile(arr, 50), np.percentile(arr, 95)
        )
    )


def summarize_synth_state(checkpoint):
    state_dict = checkpoint.get("state_dict", {})
    synth = None
    for key, value in state_dict.items():
        if key.endswith("_extra_state") and isinstance(value, dict):
            maybe = value.get("synth")
            if isinstance(maybe, dict):
                synth = maybe
                break
    if synth is None:
        print("[synth] no persistent table state found in checkpoint")
        return

    created = synth.get("created_pairs", [])
    visits = synth.get("pair_visits", [])
    visit_counts = [int(v) for _, v in visits if int(v) > 0]
    total = sum(visit_counts)
    entropy = 0.0
    if total > 0 and len(visit_counts) > 1:
        probs = [v / total for v in visit_counts]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs) / math.log(len(probs))
    top_share = 0.0
    if total > 0:
        top_share = sum(sorted(visit_counts, reverse=True)[:100]) / total
    print(
        "[synth] strategy={} attr={} alpha={} topk={} table_size={} "
        "visited_pairs={} total_visits={} mean_visits={:.2f} entropy={:.4f} "
        "top100_visit_share={:.4f}".format(
            synth.get("pair_strategy"),
            synth.get("attr_constraint", "none"),
            synth.get("crp_alpha"),
            synth.get("crp_topk"),
            len(created),
            len(visit_counts),
            total,
            total / max(1, len(visit_counts)),
            entropy,
            top_share,
        )
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-mean-center", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("ckpt:", args.ckpt)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    summarize_synth_state(checkpoint)

    features = build_feature(config).to(device).eval()
    model = build_model(config, device).to(device).eval()
    n_feat, _, _ = load_prefixed(features, state_dict, "features.", strict=False)
    n_model, _, _ = load_prefixed(model, state_dict, "model.", strict=False)
    print("loaded params: features={} model={}".format(n_feat, n_model))

    trials = np.loadtxt(config["trial_path"], dtype=str)
    rel_paths = np.unique(np.concatenate((trials[:, 1], trials[:, 2])))
    dataset = TrialAudioDataset(rel_paths, config["root"])
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_one,
    )

    embeddings = {}
    norms = []
    for batch in loader:
        wav = batch["waveform"].unsqueeze(0).to(device)
        emb = model(features(wav))
        emb = emb.squeeze(0).detach().cpu()
        embeddings[batch["rel_path"]] = emb.numpy()
        norms.append(float(emb.norm().item()))

    mat = np.stack([embeddings[p] for p in rel_paths], axis=0)
    if not args.no_mean_center:
        mean = mat.mean(axis=0, keepdims=True)
        for path in rel_paths:
            embeddings[path] = embeddings[path] - mean.squeeze(0)

    gender = load_gender_map(config.get("meta_csv"))
    labels, scores = [], []
    pos_scores, neg_scores = [], []
    neg_same_gender, neg_cross_gender, neg_unknown_gender = [], [], []
    for label, p1, p2 in trials:
        e1 = embeddings[p1]
        e2 = embeddings[p2]
        score = float(np.dot(e1, e2) / ((np.linalg.norm(e1) * np.linalg.norm(e2)) + 1e-8))
        y = int(label)
        labels.append(y)
        scores.append(score)
        if y == 1:
            pos_scores.append(score)
        else:
            neg_scores.append(score)
            g1 = gender.get(vox_id(p1))
            g2 = gender.get(vox_id(p2))
            if g1 is None or g2 is None:
                neg_unknown_gender.append(score)
            elif g1 == g2:
                neg_same_gender.append(score)
            else:
                neg_cross_gender.append(score)

    eer, th = compute_eer(labels, scores)
    dcf2, _ = compute_min_dcf(labels, scores, p_target=0.01)
    dcf3, _ = compute_min_dcf(labels, scores, p_target=0.001)
    print("EER: {:.4f}% threshold={:.4f}".format(eer * 100, th))
    print("minDCF(1e-2): {:.4f}".format(dcf2))
    print("minDCF(1e-3): {:.4f}".format(dcf3))
    print(describe_scores("positive", pos_scores))
    print(describe_scores("negative", neg_scores))
    print(describe_scores("negative_same_gender", neg_same_gender))
    print(describe_scores("negative_cross_gender", neg_cross_gender))
    print(describe_scores("negative_unknown_gender", neg_unknown_gender))
    print(describe_scores("embedding_norm_before_score_norm", norms))

    if args.output:
        rows = []
        for label, p1, p2, score in zip(labels, trials[:, 1], trials[:, 2], scores):
            rows.append({"label": label, "path1": p1, "path2": p2, "score": score})
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print("wrote scores:", args.output)


if __name__ == "__main__":
    main()
