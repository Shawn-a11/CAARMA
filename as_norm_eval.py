#!/usr/bin/env python3
"""AS-norm vs plain-cosine EER on a checkpoint (eval-only, NO retrain).

Extracts VoxCeleb1-O test embeddings (reusing Task.validation_step), then scores
the trials two ways:
  - plain cosine (sanity: should ~match the checkpoint's reported EER)
  - adaptive score normalisation (AS-norm)
Tells you for free whether the scoring backend is leaving EER on the table.

Preprocessing matches train.py's on_validation_epoch_end: per-dim mean
subtraction over all eval vectors, then cosine.

Usage:
    python as_norm_eval.py <checkpoint.ckpt> [topk]   # topk default 300
"""
import sys
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

from functions.loader import super_dataset
from feature.build_feature import build_feature
from model.model_build import build_model
from criterion.build_criterion import build_criterion
from train import Task

CONFIG = "/root/autodl-tmp/CAARMA/config.yaml"


def eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0) * 100


def main():
    ckpt = sys.argv[1]
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    config = yaml.safe_load(open(CONFIG))
    root = config["root"]

    stash = {}

    class T(Task):
        def on_validation_epoch_end(self):           # single GPU -> full set, skip DDP gather
            stash["emb"] = list(self.eval_vectors)
            stash["map"] = dict(self.index_mapping)

    dm = super_dataset(config)
    task = T(build_feature(config), build_model(config, "cuda"), build_criterion(config), config,
             learning_rate=config["init_lr"], weight_decay=config["weight_decay"],
             batch_size=config["batch_size"], num_workers=config["num_workers"],
             max_epochs=config["epochs"], trial_path=config["trial_path"],
             warmup_step=config["warmup_step"])
    task.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    Trainer(accelerator="gpu", devices=1, logger=False, precision="16-mixed",
            num_sanity_val_steps=0).validate(task, datamodule=dm)

    E = np.asarray(stash["emb"], dtype=np.float32)
    idx = stash["map"]
    E = E - E.mean(axis=0)                            # match reported preprocessing
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    trials = np.loadtxt(config["trial_path"], str)
    lab = np.array([int(t[0]) for t in trials])
    e_rows = np.array([idx[root + t[1]] for t in trials])
    t_rows = np.array([idx[root + t[2]] for t in trials])

    s_cos = (En[e_rows] * En[t_rows]).sum(1)
    eer_cos = eer(lab, s_cos)

    # AS-norm: cohort = the eval utts themselves (self excluded), per-utt top-k mean/std.
    # NOTE: a proper cohort is held-out training impostors; using the eval set is a
    # quick directional approximation. If AS-norm helps here, re-confirm with a
    # training-derived cohort before trusting the absolute gain.
    S = En @ En.T
    np.fill_diagonal(S, -np.inf)
    part = np.partition(S, -topk, axis=1)[:, -topk:]
    mu = part.mean(1)
    sd = part.std(1) + 1e-8
    s_as = 0.5 * ((s_cos - mu[e_rows]) / sd[e_rows] + (s_cos - mu[t_rows]) / sd[t_rows])
    eer_as = eer(lab, s_as)

    print(f"\n==== {ckpt} ====")
    print(f"  EER cosine  : {eer_cos:.2f}%   (sanity: should ~match reported)")
    print(f"  EER AS-norm : {eer_as:.2f}%   (cohort=eval set, top-k={topk})")
    print(f"  delta       : {eer_as - eer_cos:+.2f}%   (negative = AS-norm helps)")


if __name__ == "__main__":
    main()
