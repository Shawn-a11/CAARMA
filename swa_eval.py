#!/usr/bin/env python3
"""SWA: average several checkpoints' weights, then eval EER (NO retrain).

Averaging weights of nearby epochs (stochastic weight averaging) often gives a
small free EER gain. Reuses Task.validation_step to extract embeddings on a
single GPU; scoring matches train.py (per-dim mean subtraction + cosine).

NOTE: BatchNorm running stats are averaged too (approximation). If SWA helps,
the proper version re-estimates BN stats with a forward pass over training data.

Usage:
    python swa_eval.py <ckpt1> <ckpt2> [<ckpt3> ...]
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
    ckpts = sys.argv[1:]
    if len(ckpts) < 2:
        sys.exit("usage: python swa_eval.py <ckpt1> <ckpt2> [<ckpt3> ...]")
    config = yaml.safe_load(open(CONFIG))
    root = config["root"]

    # ---- average the state_dicts ----
    sds = [torch.load(c, map_location="cpu")["state_dict"] for c in ckpts]
    avg = {}
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            avg[k] = sum(sd[k].float() for sd in sds) / len(sds)
        else:
            avg[k] = sds[0][k]            # int buffers (e.g. num_batches_tracked)

    stash = {}

    class T(Task):
        def on_validation_epoch_end(self):
            stash["emb"] = list(self.eval_vectors)
            stash["map"] = dict(self.index_mapping)

    dm = super_dataset(config)
    task = T(build_feature(config), build_model(config, "cuda"), build_criterion(config), config,
             learning_rate=config["init_lr"], weight_decay=config["weight_decay"],
             batch_size=config["batch_size"], num_workers=config["num_workers"],
             max_epochs=config["epochs"], trial_path=config["trial_path"],
             warmup_step=config["warmup_step"])
    task.load_state_dict(avg, strict=False)
    Trainer(accelerator="gpu", devices=1, logger=False, precision="16-mixed",
            num_sanity_val_steps=0).validate(task, datamodule=dm)

    E = np.asarray(stash["emb"], dtype=np.float32)
    idx = stash["map"]
    E = E - E.mean(axis=0)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    trials = np.loadtxt(config["trial_path"], str)
    lab = np.array([int(t[0]) for t in trials])
    er = np.array([idx[root + t[1]] for t in trials])
    tr = np.array([idx[root + t[2]] for t in trials])
    s = (En[er] * En[tr]).sum(1)

    print(f"\n==== SWA of {len(ckpts)} checkpoints ====")
    for c in ckpts:
        print(f"   {c}")
    print(f"  SWA EER : {eer(lab, s):.2f}%   (compare to each ckpt's own best EER)")


if __name__ == "__main__":
    main()
