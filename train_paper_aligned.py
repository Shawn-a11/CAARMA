"""Paper-aligned CAARMA training arm for controlled numerical reproduction.
(massabaali7/CAARMA, commit 150001e) training behaviour, with the DDP-specific
mitigations required to run on our hardware without deadlocking.

Why this file exists
====================
Source code uses DDPStrategy + devices=-1 (multi-GPU); previous Algorithm-2-
faithful DDP run plateaued at 3.48 % vs paper 3.09 %. To isolate whether the
gap is from the practical training tricks the source repo carries beyond the
paper's Algorithm 2 pseudocode, this branch restores every source behaviour
we can verifiably enforce, while keeping the DDP fixes already proven needed.

Source-faithful restorations
============================
  1. pretrain_eps = 15  → first 15 epochs use lambda_adv = 0.0005 (weak adv.)
  2. 5:1 G:D step ratio during pretrain; 1:1 cycle post-pretrain
  3. d_loss / 2 and g_loss / 2  (source halves both; paper Eq.1/Eq.2 do not)
  4. adjust_weight: cap lambda_adv at 0.01, floor 0.0001, only after pretrain
  5. discriminator_optimizer lr = self.learning_rate * 0.01 (dynamic, halves
     with StepLR)
  6. Source's per-step state-machine counters (d_step_counter / g_step_counter)
     preserved verbatim including the reset conditions

DDP-required deviations from source (cannot be avoided without re-debugging
the deadlock root causes we already paid for)
============================================
  * HuBERT backbone frozen + `_ddp_params_and_buffers_to_ignore` set →
    necessary because DDP find_unused_parameters traversal across HuBERT's
    315M parameters deadlocks intermittently.
  * Encoder forward inside no_grad for D step / inside toggle_optimizer for
    M step → prevents stale all-reduce on encoder params when only D is
    being updated (or vice versa).
  * Discriminator forward uses a single concatenated call (real + fake in
    one tensor) → keeps DDP's bucket reducer state consistent.
  * Trainer: precision='16-mixed' (legacy precision=16 + manual_opt has known
    rank-desync issues with gradient scaler).
"""
from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from feature.build_feature import build_feature
from functions.loader import super_dataset
from criterion.build_criterion import build_criterion
from model.model_build import build_model
from model.discriminator_mix import (
    Discriminator_spectral,
    MixupDiscriminator,
    ProjectionDiscriminator_spectral,
)
from helper.config_utils import load_experiment_config

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq


class Task(LightningModule):
    def __init__(self, features, model, loss, config, learning_rate=0.2,
                 weight_decay=1.5e-6, batch_size=32, num_workers=10,
                 max_epochs=1000, trial_path="data/vox1_test.txt",
                 warmup_step=2000, **kwargs):
        super().__init__()
        self.features = features
        self.model = model
        self.loss = loss
        self.loss_syn = loss
        self.learning_rate = learning_rate
        self.encoder_learning_rate = float(
            config.get("encoder_lr", learning_rate)
        )
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.manual_accumulate_grad_batches = int(
            config.get("manual_accumulate_grad_batches", 1)
        )
        if self.manual_accumulate_grad_batches < 1:
            raise ValueError("manual_accumulate_grad_batches must be >= 1")
        self.trials = np.loadtxt(trial_path, str)
        self.config = config
        self.automatic_optimization = False

        discriminator_type = str(
            self.config.get("discriminator_type", "hubert")
        )
        if discriminator_type == "hubert":
            self.discriminator = MixupDiscriminator(
                hubert_model_name=self.config.get(
                    "hubert_model_name", "facebook/hubert-large-ls960-ft"
                ),
                cache_dir=self.config["hubert_cache_dir"],
                freeze_hubert=bool(self.config.get("freeze_hubert", False)),
            ).train()
        elif discriminator_type == "spectral":
            self.discriminator = Discriminator_spectral(
                self.config["embedding_dim"]
            ).train()
        elif discriminator_type == "projection":
            self.discriminator = ProjectionDiscriminator_spectral(
                self.config["embedding_dim"]
            ).train()
        else:
            raise ValueError(
                f"Unknown discriminator_type: {discriminator_type!r}"
            )
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        if (hasattr(self.discriminator, "hubert")
                and self.discriminator.freeze_hubert):
            hubert_ignore = []
            for name, _ in self.discriminator.hubert.named_parameters():
                hubert_ignore.append(f"discriminator.hubert.{name}")
            for name, _ in self.discriminator.hubert.named_buffers():
                hubert_ignore.append(f"discriminator.hubert.{name}")
            self._ddp_params_and_buffers_to_ignore = hubert_ignore

        # ── source hyperparameters (do not touch without re-running ablation) ──
        self.lambda_adv = float(self.config.get("lambda_adv_init", 0.25))
        self.lambda_adv_floor = float(self.config.get("lambda_adv_floor", 0.0001))
        self.lambda_adv_cap = float(self.config.get("lambda_adv_cap", 0.01))

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return torch.div(x, x_norm)

    def _disc_requires_condition(self):
        return getattr(self.discriminator, "requires_condition", False)

    def _real_conditions(self, label, device):
        label = label.to(self.loss.W.device, dtype=torch.long)
        condition = self.loss.W[:, label].detach().t()
        return F.normalize(condition, dim=1).to(device)

    def _synth_conditions(self, cols, device):
        if not getattr(self.loss, "persistence", False):
            raise RuntimeError(
                "Projection-D synthetic conditions require persistence=True"
            )
        if not cols:
            raise RuntimeError("Projection-D received no synthetic columns")
        index = torch.tensor(
            cols, device=self.loss.W_syn.device, dtype=torch.long
        )
        condition = self.loss.W_syn[:, index].detach().t()
        return F.normalize(condition, dim=1).to(device)

    def _disc_forward(self, embeddings, conditions=None):
        if self._disc_requires_condition():
            return self.discriminator(embeddings, conditions)
        return self.discriminator(embeddings)

    def forward(self, x):
        feature = self.features(x)
        return self.model(feature)

    def adjust_weight(self, amsoftmax_loss, g_loss):
        """Source-exact dynamic lambda_adv (cap 0.01, floor 0.0001).
        Note: source code only initialises loss_ratio inside the
        `if current_epoch > pretrain_eps` branch, so we mirror that by only
        calling adjust_weight from the post-pretrain branch.
        """
        loss_ratio = amsoftmax_loss / (g_loss + 1e-8)
        if loss_ratio > 1.5:
            self.lambda_adv = min(self.lambda_adv * 1.1, self.lambda_adv_cap)
        elif loss_ratio < 0.5:
            self.lambda_adv = max(self.lambda_adv * 0.9, self.lambda_adv_floor)
        return self.lambda_adv

    def on_train_epoch_start(self):
        if getattr(self.loss, "persistence", False):
            self.loss.rebuild_persistent_pairing()

    # ─────────────────────── per-step helpers ──────────────────────────
    def _d_step(self, opt_d, waveform, label, *, zero_grad=True,
                step_now=True, loss_divisor=1):
        """Discriminator update step.
        Encoder runs inside no_grad so DDP never registers encoder params
        as 'used in this forward' for this backward pass — required to
        avoid bucket-reducer races under find_unused_parameters=True.
        Single concatenated discriminator forward keeps DDP happy too.
        """
        self.toggle_optimizer(opt_d)
        with torch.no_grad():
            feature_d = self.features(waveform)
            embedding_d = self.model(feature_d)
            _, _, synth_for_d = self.loss(
                embedding_d,
                label,
                update_state=False,
                cache_selection=True,
            )
            synth_cols_d = list(
                getattr(self.loss, "last_synth_cols", [])
            )

        if zero_grad:
            opt_d.zero_grad()
        B = embedding_d.size(0)
        combined = torch.cat(
            [self.normalize(embedding_d), self.normalize(synth_for_d)], dim=0
        )
        if self._disc_requires_condition():
            if len(synth_cols_d) != synth_for_d.size(0):
                raise RuntimeError(
                    "Projection-D condition mismatch in D step: "
                    f"{len(synth_cols_d)} cols for "
                    f"{synth_for_d.size(0)} synthetic rows"
                )
            conditions = torch.cat(
                [
                    self._real_conditions(label, combined.device),
                    self._synth_conditions(synth_cols_d, combined.device),
                ],
                dim=0,
            )
        else:
            conditions = None
        preds = self._disc_forward(combined, conditions)
        real_preds, fake_preds = preds[:B], preds[B:]
        d_real_loss = self.BCE_loss(real_preds, torch.ones_like(real_preds))
        d_fake_loss = self.BCE_loss(fake_preds, torch.zeros_like(fake_preds))
        d_loss = d_real_loss + d_fake_loss

        self.manual_backward(d_loss / float(loss_divisor))
        if step_now:
            opt_d.step()
        self.untoggle_optimizer(opt_d)
        self.log('d_loss', d_loss, prog_bar=True, sync_dist=False)
        return d_loss

    def _g_step(self, opt_main, waveform, label, lambda_adv_value,
                adjust=False, *, zero_grad=True, step_now=True,
                loss_divisor=1):
        """Main encoder (M) update step.
        Encoder runs INSIDE toggle_optimizer(opt_main) so discriminator
        params are frozen via toggle and DDP only syncs encoder/loss params.
        If adjust=True, lambda_adv is dynamically adjusted by adjust_weight.
        Otherwise lambda_adv is forced to lambda_adv_value (source uses
        0.0005 during pretrain, 0.25 post-pretrain before adjust).
        """
        self.toggle_optimizer(opt_main)
        feature = self.features(waveform)
        embedding = self.model(feature)
        if zero_grad:
            opt_main.zero_grad()
        amsoftmax_loss, acc, synthetic_embeddings = self.loss(
            embedding,
            label,
            update_state=True,
            reuse_selection=True,
        )
        synth_cols_g = list(getattr(self.loss, "last_synth_cols", []))
        amsoftmax_syn_loss, _, _ = self.loss_syn(embedding, label, flagSyn=True)
        if getattr(self.loss, "persistence", False):
            self.loss.synchronize_synth_state(embedding.device)

        # Single concatenated D forward (real + synthetic in one tensor).
        Ns = synthetic_embeddings.size(0)
        combined = torch.cat(
            [self.normalize(synthetic_embeddings), self.normalize(embedding)], dim=0
        )
        if self._disc_requires_condition():
            if len(synth_cols_g) != synthetic_embeddings.size(0):
                raise RuntimeError(
                    "Projection-D condition mismatch in G step: "
                    f"{len(synth_cols_g)} cols for "
                    f"{synthetic_embeddings.size(0)} synthetic rows"
                )
            conditions = torch.cat(
                [
                    self._synth_conditions(synth_cols_g, combined.device),
                    self._real_conditions(label, combined.device),
                ],
                dim=0,
            )
        else:
            conditions = None
        preds = self._disc_forward(combined, conditions)
        fake_preds, real_preds = preds[:Ns], preds[Ns:]
        g_loss = (self.BCE_loss(fake_preds, torch.ones_like(fake_preds))
                  + self.BCE_loss(real_preds, torch.zeros_like(real_preds)))

        if adjust:
            self.lambda_adv = lambda_adv_value
            self.lambda_adv = self.adjust_weight(amsoftmax_loss, g_loss)
        else:
            self.lambda_adv = lambda_adv_value

        total_loss = (amsoftmax_loss
                      + (1.0 / self.config['num_spk']) * amsoftmax_syn_loss
                      + self.lambda_adv * g_loss)

        self.manual_backward(total_loss / float(loss_divisor))
        if step_now:
            opt_main.step()
        self.untoggle_optimizer(opt_main)

        # Source's per-step warmup overwrite.
        if step_now and self.trainer.global_step < self.config['warmup_step']:
            lr_scale = min(1., (self.trainer.global_step + 1)
                           / float(self.config['warmup_step']))
            for pg in opt_main.param_groups:
                pg['lr'] = lr_scale * pg['base_lr']

        self.log('am_loss', amsoftmax_loss, prog_bar=True, sync_dist=False)
        self.log('am_loss_syn', amsoftmax_syn_loss, prog_bar=True, sync_dist=False)
        self.log('acc', acc, prog_bar=True, sync_dist=False)
        self.log('g_loss', g_loss, prog_bar=True, sync_dist=False)
        self.log('total_loss', total_loss, prog_bar=True, sync_dist=False)
        return total_loss

    def training_step(self, batch, batch_idx):
        opt_main, opt_d = self.optimizers()
        waveform = batch['waveform']
        label = batch['mapped_id']

        accumulation = self.manual_accumulate_grad_batches
        total_batches = int(self.trainer.num_training_batches)
        group_start = (batch_idx // accumulation) * accumulation
        group_size = min(accumulation, total_batches - group_start)
        zero_grad = batch_idx == group_start
        step_now = ((batch_idx - group_start + 1) == group_size)

        self._d_step(
            opt_d,
            waveform,
            label,
            zero_grad=zero_grad,
            step_now=step_now,
            loss_divisor=group_size,
        )
        return self._g_step(
            opt_main,
            waveform,
            label,
            lambda_adv_value=self.lambda_adv,
            adjust=True,
            zero_grad=zero_grad,
            step_now=step_now,
            loss_divisor=group_size,
        )

    def configure_optimizers(self):
        embedding_optimizer = AdamW(
            [
                {
                    "params": self.model.parameters(),
                    "lr": self.encoder_learning_rate,
                    "base_lr": self.encoder_learning_rate,
                },
                {
                    "params": self.loss.parameters(),
                    "lr": self.learning_rate,
                    "base_lr": self.learning_rate,
                },
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        discriminator_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=float(self.config.get("discriminator_lr", 0.0002)),
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999),
        )
        embedding_scheduler = StepLR(embedding_optimizer, step_size=4, gamma=0.5)
        discriminator_scheduler = StepLR(discriminator_optimizer, step_size=4, gamma=0.5)
        return ([embedding_optimizer, discriminator_optimizer],
                [embedding_scheduler, discriminator_scheduler])

    def on_train_epoch_end(self):
        main_scheduler, d_scheduler = self.lr_schedulers()
        main_scheduler.step()
        d_scheduler.step()

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def on_validation_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        waveform = batch['waveform']
        path = batch['path']
        with torch.no_grad():
            x = self.features(waveform)
            self.model.eval()
            x = self.model(x)
        x = x.detach().cpu().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[path[0]] = batch_idx

    def similarity_score(self, trials, index_mapping, eval_vectors):
        labels, scores = [], []
        epsilon = 1e-8
        for item in trials:
            enroll_path = os.path.normpath(os.path.join(
                self.config['root'], str(item[1]).lstrip("/\\")
            ))
            test_path = os.path.normpath(os.path.join(
                self.config['root'], str(item[2]).lstrip("/\\")
            ))
            enroll_vector = eval_vectors[index_mapping[enroll_path]]
            test_vector = eval_vectors[index_mapping[test_path]]
            score = enroll_vector.dot(test_vector.T)
            denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
            score = score / (denom + epsilon)
            if np.isnan(score):
                print("Warning: NaN detected in score calculation. Setting score to 0.")
                score = 0.0
            labels.append(int(item[0]))
            scores.append(score)
        return labels, scores

    def compute_eer(self, labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        threshold = interp1d(fpr, thresholds)(eer)
        return eer, threshold

    def compute_minDCF(self, labels, scores, p_target=0.01, c_miss=1, c_fa=1):
        scores = np.array(scores)
        labels = np.array(labels)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr
        min_c_det = float("inf")
        min_c_det_threshold = thresholds[0]
        for i in range(len(fnr)):
            c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det
                min_c_det_threshold = thresholds[i]
        c_def = min(c_miss * p_target, c_fa * (1 - p_target))
        min_dcf = min_c_det / c_def
        return min_dcf, min_c_det_threshold

    def on_validation_epoch_end(self):
        num_gpus = torch.cuda.device_count()
        # Gather eval_vectors from all DDP ranks.
        all_eval_vectors = [None for _ in range(num_gpus)]
        dist.all_gather_object(all_eval_vectors, self.eval_vectors)

        all_index_mappings = [None for _ in range(num_gpus)]
        dist.all_gather_object(all_index_mappings, self.index_mapping)

        # Bug-fix: each rank's batch_idx is local; need offset by cumulative
        # vectors from earlier ranks before merging into the global mapping.
        index_mapping = {}
        offset = 0
        for gpu_vectors, gpu_mapping in zip(all_eval_vectors, all_index_mappings):
            for path, local_idx in gpu_mapping.items():
                index_mapping[path] = offset + local_idx
            offset += len(gpu_vectors)

        eval_vectors = np.vstack(all_eval_vectors)
        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
        labels, scores = self.similarity_score(self.trials, index_mapping, eval_vectors)
        EER, threshold = self.compute_eer(labels, scores)

        if self.trainer.is_global_zero:
            print("\ncosine EER: {:.2f}%".format(EER * 100))
            minDCF2, _ = self.compute_minDCF(labels, scores, p_target=0.01)
            print("cosine minDCF(10-2): {:.4f}".format(minDCF2))
            minDCF3, _ = self.compute_minDCF(labels, scores, p_target=0.001)
            print("cosine minDCF(10-3): {:.4f}".format(minDCF3))
        else:
            minDCF2, _ = self.compute_minDCF(labels, scores, p_target=0.01)
            minDCF3, _ = self.compute_minDCF(labels, scores, p_target=0.001)
        self.log("cosine_eer", EER * 100, sync_dist=True)
        self.log("cosine_minDCF(10-2)", minDCF2, sync_dist=True)
        self.log("cosine_minDCF(10-3)", minDCF3, sync_dist=True)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--devices", type=int, default=None)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Load model weights only and start a new training run",
    )
    checkpoint_group.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume the full Lightning training state from a checkpoint",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run N source-faithful training steps without validation/checkpoints",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    seed_everything(int(config.get("seed", 42)), workers=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    dataloader = super_dataset(config)
    features = build_feature(config)
    model = build_model(config, device)
    criterion = build_criterion(config)

    final_project = Task(
        features, model, criterion, config,
        learning_rate=config['init_lr'],
        weight_decay=config['weight_decay'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        max_epochs=config['epochs'],
        trial_path=config['trial_path'],
        warmup_step=config['warmup_step'],
    )

    checkpoint_path = args.checkpoint or config.get('checkpoint_path', 'None')
    if checkpoint_path != 'None':
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        final_project.load_state_dict(state_dict, strict=False)
        print("load weight from {}".format(checkpoint_path))

    resume_checkpoint = args.resume_checkpoint
    if resume_checkpoint is not None:
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(
                "Resume checkpoint does not exist: {}".format(
                    resume_checkpoint
                )
            )
        print("resume full training state from {}".format(resume_checkpoint))

    assert config['save_dir'] is not None
    checkpoint_callback = ModelCheckpoint(
        monitor='cosine_eer', save_top_k=3, mode='min',
        filename="{epoch}_{cosine_eer:.2f}",
        dirpath=config['save_dir'], save_last=True,
    )

    requested_devices = int(
        args.devices if args.devices is not None else config.get("devices", 4)
    )
    smoke_mode = args.smoke_steps > 0
    trainer = Trainer(
        strategy=DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
        ),
        accelerator="gpu",
        devices=requested_devices,
        max_epochs=config['epochs'],
        logger=False,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        precision="16-mixed",
        callbacks=[] if smoke_mode else [checkpoint_callback],
        enable_checkpointing=not smoke_mode,
        default_root_dir=config['save_dir'],
        reload_dataloaders_every_n_epochs=1,
        accumulate_grad_batches=1,
        log_every_n_steps=25,
        benchmark=True,
        deterministic=False,
        max_steps=args.smoke_steps if smoke_mode else -1,
        limit_val_batches=0 if smoke_mode else 1.0,
    )
    trainer.fit(
        final_project,
        datamodule=dataloader,
        ckpt_path=resume_checkpoint,
    )


if __name__ == "__main__":
    cli_main()
