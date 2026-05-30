"""Single-GPU training entry that faithfully mirrors the original CAARMA repo
(massabaali7/CAARMA, commit 150001e) behaviour, while keeping the few
infrastructure tweaks needed to run on the local server.

Why this file exists
====================
Our previous Algorithm-2 rewrite removed every implementation detail that the
released paper-code carried beyond the published Algorithm 2 pseudocode, and
the resulting EER plateaued at ~3.48-3.51 % (vs. paper's 3.09 %). This file
re-introduces the source-repo behaviour in full so we can isolate whether the
gap is from those omissions or from something else.

Faithful to source
==================
  * pretrain_eps = 15  → first 15 epochs use lambda_adv = 0.0005 (weak adv.)
  * 5:1 G:D step ratio during pretrain; 1:1 after pretrain
  * d_loss and g_loss BOTH divided by 2 (source Eq.1/Eq.2 implementation)
  * adjust_weight: cap lambda_adv at 0.01, floor at 0.0001, only after pretrain
  * discriminator_optimizer lr = self.learning_rate * 0.01 (dynamic)
  * HuBERT NOT frozen (single-GPU has no DDP-deadlock constraint)
  * State-machine counters (d_step_counter / g_step_counter) untouched
  * Source's amsoftmax_loss recomputation inside each branch preserved

Diverging from source on purpose (server compat only, no math change)
====================================================================
  * Trainer: single GPU, no DDPStrategy, no wandb, no LearningRateMonitor
  * on_validation_epoch_end: rank-safe path that does NOT call
    dist.all_gather_object (source version errors out on single GPU)
  * ModelCheckpoint: keep top-3 by EER + last (we want recoverable best ckpt)
  * config.yaml path hardcoded to /root/autodl-tmp/CAARMA/config.yaml
"""
import torch
import torch.nn as nn
import numpy as np
import yaml

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from feature.build_feature import build_feature
from functions.loader import super_dataset
from criterion.build_criterion import build_criterion
from model.model_build import build_model
from model.discriminator_mix import MixupDiscriminator

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
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.trials = np.loadtxt(trial_path, str)
        self.config = config
        self.automatic_optimization = False

        self.discriminator = MixupDiscriminator(cache_dir="./cache_dir/").train()
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # ── source hyperparameters (do not touch without re-running ablation) ──
        self.lambda_adv = 0.25         # post-pretrain initial value
        self.pretrain_eps = 15         # epochs of weak-adversarial pretraining
        self.pretrain_discriminator = True
        self.discriminator_steps = 0
        # d/g_step_counter created lazily in training_step (mirrors source)

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return torch.div(x, x_norm)

    def forward(self, x):
        feature = self.features(x)
        return self.model(feature)

    def adjust_weight(self, amsoftmax_loss, g_loss):
        """Source-exact dynamic lambda_adv adjustment.

        Note: source initialises loss_ratio inside the `if current_epoch >
        pretrain_eps` branch, so calling this before pretrain ends would
        UnboundLocalError. We preserve that quirk because training_step only
        calls this in the post-pretrain branch.
        """
        if self.current_epoch > self.pretrain_eps:
            loss_ratio = amsoftmax_loss / (g_loss + 1e-8)
        if loss_ratio > 1.5:
            self.lambda_adv = min(self.lambda_adv * 1.1, 0.01)
        elif loss_ratio < 0.5:
            self.lambda_adv = max(self.lambda_adv * 0.9, 0.0001)
        return self.lambda_adv

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        optimizer_main, d_optimizer = opt

        waveform = batch['waveform']
        label = batch['mapped_id']

        # Real embeddings (one forward shared with D and G branches below).
        feature = self.features(waveform)
        embedding = self.model(feature)

        # First AM-Softmax (synthetic_embeddings reused by the discriminator).
        amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)

        # ── state-machine counters (created lazily, exactly like source) ──
        if not hasattr(self, 'd_step_counter'):
            self.d_step_counter = 0
        if not hasattr(self, 'g_step_counter'):
            self.g_step_counter = 0

        # Source's two reset conditions (5 G + 1 D during pretrain → reset;
        # 1 G + 1 D post-pretrain → reset).
        if self.d_step_counter >= 1 and self.g_step_counter >= 5:
            self.d_step_counter = 0
            self.g_step_counter = 0
        elif (self.d_step_counter >= 1 and self.g_step_counter >= 1
              and self.current_epoch > self.pretrain_eps):
            self.d_step_counter = 0
            self.g_step_counter = 0

        # ─── PRETRAIN PHASE (epoch ≤ 15): G updates 5x per D update ────────
        if self.current_epoch <= self.pretrain_eps:
            if self.g_step_counter < 5:
                # ---- Generator (M) step with weak adversarial weight ----
                self.lambda_adv = 0.0005
                optimizer_main.zero_grad()

                real_preds = self.discriminator(self.normalize(embedding.detach()))
                fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
                fake_labels = torch.zeros(real_preds.size()).to(self.device)
                real_labels = torch.ones(fake_preds.size()).to(self.device)
                # Source g_loss with /2 (paper Eq.2 doesn't have /2; source does)
                g_loss = (self.BCE_loss(fake_preds, real_labels)
                          + self.BCE_loss(real_preds, fake_labels)) / 2

                amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)
                amsoftmax_syn_loss, acc_syn, synthetic_embeddings = self.loss_syn(
                    embedding, label, flagSyn=True
                )
                total_loss = (amsoftmax_loss
                              + (1 / self.config['num_spk']) * amsoftmax_syn_loss
                              + self.lambda_adv * g_loss)

                self.manual_backward(total_loss)
                optimizer_main.step()

                if self.trainer.global_step < self.config['warmup_step']:
                    lr_scale = min(1., (self.trainer.global_step + 1)
                                   / float(self.config['warmup_step']))
                    for pg in optimizer_main.param_groups:
                        pg['lr'] = lr_scale * self.learning_rate

                self.log('am_loss', amsoftmax_loss, prog_bar=True)
                self.log('am_loss_syn', amsoftmax_syn_loss, prog_bar=True)
                self.log('acc', acc, prog_bar=True)
                self.log('g_loss', g_loss, prog_bar=True)
                self.log('total_loss', total_loss, prog_bar=True)

                self.g_step_counter += 1
                return total_loss

            elif self.d_step_counter < 1:
                # ---- Discriminator step ----
                d_optimizer.zero_grad()
                real_preds = self.discriminator(self.normalize(embedding.detach()))
                fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
                real_labels = torch.ones(real_preds.size()).to(self.device)
                fake_labels = torch.zeros(fake_preds.size()).to(self.device)
                d_real_loss = self.BCE_loss(real_preds, real_labels)
                d_fake_loss = self.BCE_loss(fake_preds, fake_labels)
                # Source: divide by 2
                d_loss = (d_real_loss + d_fake_loss) / 2

                self.manual_backward(d_loss)
                d_optimizer.step()
                self.log('d_loss', d_loss, prog_bar=True)

                self.d_step_counter += 1
                return d_loss

        # ─── POST-PRETRAIN PHASE (epoch > 15): 1 D, then 1 G per cycle ───
        else:
            if self.d_step_counter < 1:
                # ---- Discriminator step ----
                d_optimizer.zero_grad()
                real_preds = self.discriminator(self.normalize(embedding.detach()))
                fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
                real_labels = torch.ones(real_preds.size()).to(self.device)
                fake_labels = torch.zeros(fake_preds.size()).to(self.device)
                d_real_loss = self.BCE_loss(real_preds, real_labels)
                d_fake_loss = self.BCE_loss(fake_preds, fake_labels)
                d_loss = (d_real_loss + d_fake_loss) / 2

                self.manual_backward(d_loss)
                d_optimizer.step()
                self.log('d_loss', d_loss, prog_bar=True)

                self.d_step_counter += 1
                return d_loss

            elif self.d_step_counter >= 1 and self.g_step_counter < 1:
                # ---- Generator (M) step with full adversarial weight ----
                optimizer_main.zero_grad()
                self.lambda_adv = 0.25  # source resets to 0.25 every cycle

                real_preds = self.discriminator(self.normalize(embedding.detach()))
                fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
                fake_labels = torch.zeros(real_preds.size()).to(self.device)
                real_labels = torch.ones(fake_preds.size()).to(self.device)
                g_loss = (self.BCE_loss(fake_preds, real_labels)
                          + self.BCE_loss(real_preds, fake_labels)) / 2

                amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)
                amsoftmax_syn_loss, acc_syn, synthetic_embeddings = self.loss_syn(
                    embedding, label, flagSyn=True
                )
                self.lambda_adv = self.adjust_weight(amsoftmax_loss, g_loss)
                total_loss = (amsoftmax_loss
                              + (1 / self.config['num_spk']) * amsoftmax_syn_loss
                              + self.lambda_adv * g_loss)

                self.manual_backward(total_loss)
                optimizer_main.step()

                if self.trainer.global_step < self.config['warmup_step']:
                    lr_scale = min(1., (self.trainer.global_step + 1)
                                   / float(self.config['warmup_step']))
                    for pg in optimizer_main.param_groups:
                        pg['lr'] = lr_scale * self.learning_rate

                self.log('am_loss', amsoftmax_loss, prog_bar=True)
                self.log('am_loss_syn', amsoftmax_syn_loss, prog_bar=True)
                self.log('acc', acc, prog_bar=True)
                self.log('g_loss', g_loss, prog_bar=True)
                self.log('total_loss', total_loss, prog_bar=True)

                self.g_step_counter += 1
                return total_loss

    def configure_optimizers(self):
        # Source-exact optimiser setup. Note D lr is DYNAMIC (lr * 0.01),
        # so when StepLR halves the main lr, D's lr also halves implicitly
        # via the same scheduler — both share step_size=4, gamma=0.5.
        embedding_optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        discriminator_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=self.learning_rate * 0.01,
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
            enroll_vector = eval_vectors[index_mapping[self.config['root'] + item[1]]]
            test_vector = eval_vectors[index_mapping[self.config['root'] + item[2]]]
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
        # Single-GPU safe: no dist.all_gather_object. The source version
        # requires DDP to be initialised and would crash here.
        eval_vectors = np.vstack(self.eval_vectors)
        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
        labels, scores = self.similarity_score(self.trials, self.index_mapping, eval_vectors)
        EER, threshold = self.compute_eer(labels, scores)
        print("\ncosine EER: {:.2f}% with threshold {:.2f}".format(EER * 100, threshold))
        self.log("cosine_eer", EER * 100)

        minDCF2, threshold2 = self.compute_minDCF(labels, scores, p_target=0.01)
        print("cosine minDCF(10-2): {:.4f} with threshold {:.4f}".format(minDCF2, threshold2))
        self.log("cosine_minDCF(10-2)", minDCF2)

        minDCF3, threshold3 = self.compute_minDCF(labels, scores, p_target=0.001)
        print("cosine minDCF(10-3): {:.4f} with threshold {:.4f}".format(minDCF3, threshold3))
        self.log("cosine_minDCF(10-3)", minDCF3)


def cli_main():
    def load_config(p):
        with open(p) as f:
            return yaml.safe_load(f)

    config = load_config("/root/autodl-tmp/CAARMA/config.yaml")
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

    if config['checkpoint_path'] != 'None':
        state_dict = torch.load(config['checkpoint_path'], map_location="cpu")["state_dict"]
        final_project.load_state_dict(state_dict, strict=False)
        print("load weight from {}".format(config['checkpoint_path']))

    assert config['save_dir'] is not None
    checkpoint_callback = ModelCheckpoint(
        monitor='cosine_eer', save_top_k=3, mode='min',
        filename="{epoch}_{cosine_eer:.2f}",
        dirpath=config['save_dir'], save_last=True,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config['epochs'],
        logger=False,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        default_root_dir=config['save_dir'],
        reload_dataloaders_every_n_epochs=1,
        accumulate_grad_batches=1,
        log_every_n_steps=25,
        benchmark=True,
        deterministic=False,
    )
    trainer.fit(final_project, datamodule=dataloader)


if __name__ == "__main__":
    cli_main()
