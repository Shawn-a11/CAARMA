#!/usr/bin/env python3
"""Eval-only extraction of TEST embeddings from a checkpoint (single GPU, no DDP).

Loads a trained checkpoint, runs the encoder over the VoxCeleb1-O eval set
(reusing Task.validation_step), and saves the (N, D) embedding matrix to .npy.
Then `diag_uniformity.py --emb out.npy` gives the test-embedding uniformity --
the theory-faithful axis for the uniformity-vs-EER money plot (prototype
uniformity is pinned ~constant by AM-Softmax, so it can't discriminate configs).

Usage:
    python extract_embeddings.py <checkpoint.ckpt> <out.npy>
"""
import sys
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer

from functions.loader import super_dataset
from feature.build_feature import build_feature
from model.model_build import build_model
from criterion.build_criterion import build_criterion
from train import Task

CONFIG = "/root/autodl-tmp/CAARMA/config.yaml"


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python extract_embeddings.py <checkpoint.ckpt> <out.npy>")
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    config = yaml.safe_load(open(CONFIG))

    class ExtractTask(Task):
        # Override the eval epoch end: just dump the (single-GPU = full) eval
        # vectors and skip the parent's DDP all_gather / EER computation, which
        # needs a process group we don't have at devices=1.
        def on_validation_epoch_end(self):
            embs = np.asarray(self.eval_vectors, dtype=np.float32)
            np.save(out_path, embs)
            print(f"[extract] saved {embs.shape} test embeddings -> {out_path}")

    dm = super_dataset(config)
    features = build_feature(config)
    model = build_model(config, "cuda")
    criterion = build_criterion(config)
    task = ExtractTask(
        features, model, criterion, config,
        learning_rate=config["init_lr"], weight_decay=config["weight_decay"],
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        max_epochs=config["epochs"], trial_path=config["trial_path"],
        warmup_step=config["warmup_step"],
    )

    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = task.load_state_dict(sd, strict=False)
    print(f"[extract] loaded {ckpt_path}  (missing={len(missing)} unexpected={len(unexpected)})")

    # Single GPU, no DDP -> eval_vectors holds the full test set on one process.
    trainer = Trainer(accelerator="gpu", devices=1, logger=False,
                      precision="16-mixed", num_sanity_val_steps=0)
    trainer.validate(task, datamodule=dm)


if __name__ == "__main__":
    main()
