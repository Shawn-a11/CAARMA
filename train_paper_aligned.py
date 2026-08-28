"""Optimizer-matched CAARMA Table 2 ID6 numerical reproduction.

The method remains the audited paper-aligned full recipe: one discriminator
and one generator update per batch, L_syn, dynamic adversarial weighting, and
the HuBERT Mixup Discriminator. The only deliberate protocol change from the
previous 3.39% full control is the optimizer schedule already validated by the
3.27% MFA baseline: 2,000 main updates of warmup at lr=2e-3 followed by one
0.5x decay at epoch index 16. The discriminator keeps the stated lr=2e-4,
receives no warmup, and receives the same one-time decay.

DDP guards retain separate optimizer toggling, a no-grad encoder pass for the
discriminator update, one concatenated discriminator forward, and mixed
precision through the current Lightning API.
"""
from argparse import ArgumentParser
import json
import os

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW

from feature.build_feature import build_feature
from functions.loader import super_dataset
from criterion.build_criterion import build_criterion
from model.model_build import build_model
from model.discriminator_mix import MixupDiscriminator
from helper.config_utils import load_experiment_config
from helper.gan_controls import (
    generator_lambda_plan,
    resolve_gan_config,
    scheduled_updates,
)

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq


def optimizer_learning_rate(
    base_lr,
    update_step,
    warmup_steps,
    current_epoch,
    decay_after_epoch,
    decay_gamma,
):
    """Return the pre-update LR with optional warmup and one epoch decay."""
    warmup = 1.0
    if warmup_steps:
        warmup = min(
            1.0,
            float(update_step + 1) / float(max(1, warmup_steps)),
        )
    decay = (
        float(decay_gamma)
        if decay_after_epoch is not None
        and int(current_epoch) >= int(decay_after_epoch)
        else 1.0
    )
    return float(base_lr) * warmup * decay


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
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.trials = np.loadtxt(trial_path, str)
        self.config = config
        self.automatic_optimization = False

        self.discriminator = MixupDiscriminator(
            hubert_model_name=self.config.get(
                "hubert_model_name", "facebook/hubert-large-ls960-ft"
            ),
            cache_dir=self.config["hubert_cache_dir"],
            freeze_hubert=bool(self.config.get("freeze_hubert", False)),
        ).train()
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        if self.discriminator.freeze_hubert:
            hubert_ignore = []
            for name, _ in self.discriminator.hubert.named_parameters():
                hubert_ignore.append(f"discriminator.hubert.{name}")
            for name, _ in self.discriminator.hubert.named_buffers():
                hubert_ignore.append(f"discriminator.hubert.{name}")
            self._ddp_params_and_buffers_to_ignore = hubert_ignore

        effective_gan_config = resolve_gan_config(
            self.config,
            main_lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.discriminator_lr = effective_gan_config["discriminator_lr"]
        self.lambda_adv_init = effective_gan_config["lambda_adv_init"]
        self.lambda_adv = self.lambda_adv_init
        self.lambda_adv_floor = effective_gan_config["lambda_adv_floor"]
        self.lambda_adv_cap = effective_gan_config["lambda_adv_cap"]
        self.lambda_adv_mode = effective_gan_config["lambda_adv_mode"]
        self.lambda_adv_fixed = effective_gan_config["lambda_adv_fixed"]
        self.lambda_adv_pretrain = effective_gan_config["lambda_adv_pretrain"]
        self.gan_schedule = effective_gan_config["gan_schedule"]
        self.pretrain_epochs = effective_gan_config["pretrain_epochs"]
        self.pretrain_g_steps = effective_gan_config["pretrain_g_steps"]
        self.register_buffer(
            "main_update_count", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "discriminator_update_count", torch.zeros((), dtype=torch.long)
        )

        # Validate controls before the first distributed forward.
        scheduled_updates(
            self.gan_schedule,
            epoch=0,
            batch_idx=0,
            pretrain_epochs=self.pretrain_epochs,
            pretrain_g_steps=self.pretrain_g_steps,
        )
        generator_lambda_plan(
            self.lambda_adv_mode,
            self.lambda_adv_fixed,
            self.lambda_adv_init,
            self.lambda_adv_pretrain,
            use_pretrain_value=False,
        )
        print(
            "EFFECTIVE_GAN_CONFIG "
            + json.dumps(effective_gan_config, sort_keys=True)
        )

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return torch.div(x, x_norm)

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

    # ─────────────────────── per-step helpers ──────────────────────────
    def _d_step(self, opt_d, waveform, label):
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
            _, _, synth_for_d = self.loss(embedding_d, label)

        opt_d.zero_grad()
        B = embedding_d.size(0)
        combined = torch.cat(
            [self.normalize(embedding_d), self.normalize(synth_for_d)], dim=0
        )
        preds = self.discriminator(combined)
        real_preds, fake_preds = preds[:B], preds[B:]
        d_real_loss = self.BCE_loss(real_preds, torch.ones_like(real_preds))
        d_fake_loss = self.BCE_loss(fake_preds, torch.zeros_like(fake_preds))
        d_loss = d_real_loss + d_fake_loss

        self.manual_backward(d_loss)
        discriminator_lr = optimizer_learning_rate(
            base_lr=self.discriminator_lr,
            update_step=int(self.discriminator_update_count.item()),
            warmup_steps=0,
            current_epoch=self.current_epoch,
            decay_after_epoch=self.config.get("lr_decay_after_epoch"),
            decay_gamma=float(self.config.get("lr_decay_gamma", 1.0)),
        )
        for group in opt_d.param_groups:
            group["lr"] = discriminator_lr
        opt_d.step()
        self.discriminator_update_count.add_(1)
        self.untoggle_optimizer(opt_d)
        self.log('d_loss', d_loss, prog_bar=True, sync_dist=False)
        self.log(
            "discriminator_learning_rate",
            discriminator_lr,
            sync_dist=False,
        )
        return d_loss

    def _g_step(self, opt_main, waveform, label, lambda_adv_value,
                adjust=False):
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
        opt_main.zero_grad()
        amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)
        amsoftmax_syn_loss, _, _ = self.loss_syn(embedding, label, flagSyn=True)

        # Single concatenated D forward (real + synthetic in one tensor).
        B = embedding.size(0)
        combined = torch.cat(
            [self.normalize(synthetic_embeddings), self.normalize(embedding)], dim=0
        )
        preds = self.discriminator(combined)
        fake_preds, real_preds = preds[:B], preds[B:]
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

        self.manual_backward(total_loss)
        main_lr = optimizer_learning_rate(
            base_lr=self.learning_rate,
            update_step=int(self.main_update_count.item()),
            warmup_steps=int(self.config.get("warmup_step", 2000)),
            current_epoch=self.current_epoch,
            decay_after_epoch=self.config.get("lr_decay_after_epoch"),
            decay_gamma=float(self.config.get("lr_decay_gamma", 1.0)),
        )
        for group in opt_main.param_groups:
            group["lr"] = main_lr
        opt_main.step()
        self.main_update_count.add_(1)
        self.untoggle_optimizer(opt_main)

        self.log('am_loss', amsoftmax_loss, prog_bar=True, sync_dist=False)
        self.log('am_loss_syn', amsoftmax_syn_loss, prog_bar=True, sync_dist=False)
        self.log('acc', acc, prog_bar=True, sync_dist=False)
        self.log('g_loss', g_loss, prog_bar=True, sync_dist=False)
        self.log('lambda_adv', float(self.lambda_adv), prog_bar=False, sync_dist=False)
        self.log("model_learning_rate", main_lr, sync_dist=False)
        self.log('total_loss', total_loss, prog_bar=True, sync_dist=False)
        return total_loss

    def training_step(self, batch, batch_idx):
        opt_main, opt_d = self.optimizers()
        waveform = batch['waveform']
        label = batch['mapped_id']

        updates = scheduled_updates(
            self.gan_schedule,
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            pretrain_epochs=self.pretrain_epochs,
            pretrain_g_steps=self.pretrain_g_steps,
        )
        result = None
        for update in updates:
            if update == "d":
                result = self._d_step(opt_d, waveform, label)
                continue

            use_pretrain_value = (
                self.gan_schedule == "source_state_machine"
                and self.current_epoch <= self.pretrain_epochs
            )
            # The paired 3.39 control carries the adjusted value across
            # batches. The public source state machine resets to its initial
            # post-pretrain value before each dynamic G update.
            dynamic_value = (
                self.lambda_adv_init
                if self.gan_schedule == "source_state_machine"
                else self.lambda_adv
            )
            lambda_value, adjust = generator_lambda_plan(
                self.lambda_adv_mode,
                self.lambda_adv_fixed,
                dynamic_value,
                self.lambda_adv_pretrain,
                use_pretrain_value=use_pretrain_value,
            )
            result = self._g_step(
                opt_main,
                waveform,
                label,
                lambda_adv_value=lambda_value,
                adjust=adjust,
            )
        return result

    def configure_optimizers(self):
        embedding_optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        discriminator_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999),
        )
        return [embedding_optimizer, discriminator_optimizer]

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
    parser.add_argument("--checkpoint", default=None)
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
        sync_batchnorm=bool(config.get("sync_batchnorm", False)),
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
    trainer.fit(final_project, datamodule=dataloader)


if __name__ == "__main__":
    cli_main()
