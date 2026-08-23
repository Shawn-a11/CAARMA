from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union
import torch.distributed as dist
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy

import random
import torch
import torch.nn as nn
import numpy as np
import yaml

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CyclicLR

from feature.build_feature import build_feature
from functions.loader import super_dataset
from criterion.build_criterion import build_criterion
from model.model_build import build_model
from model.discriminator_mix import (
    MixupDiscriminator,
    Discriminator_spectral,
    ProjectionDiscriminator_spectral,
)
from helper.mixup_avg import mixup_data_euc_avg

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from pytorch_lightning.loggers import WandbLogger
import os

class Task(LightningModule):
    def __init__(self, features, model, loss, config, learning_rate=0.2, weight_decay=1.5e-6, 
                batch_size=32, num_workers=10, max_epochs=1000, trial_path="data/vox1_test.txt",  
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
        
        # Professor's guidance: HuBERT is wrong as the discriminator here; use a
        # simple one. "spectral" = 1-hidden-layer spectral-norm MLP (D->128->1).
        discriminator_type = config.get('discriminator_type', 'spectral')
        if discriminator_type == 'spectral':
            self.discriminator = Discriminator_spectral(config['embedding_dim']).train()
        elif discriminator_type == 'projection':
            self.discriminator = ProjectionDiscriminator_spectral(config['embedding_dim']).train()
        else:
            self.discriminator = MixupDiscriminator(cache_dir="./cache_dir/").train()
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # HuBERT/WavLM discriminators need their frozen SSL backbone excluded
        # from DDP traversal. The MLP discriminator ablation has no SSL module.
        if hasattr(self.discriminator, "hubert"):
            hubert_ignore = []
            for name, _ in self.discriminator.hubert.named_parameters():
                hubert_ignore.append(f"discriminator.hubert.{name}")
            for name, _ in self.discriminator.hubert.named_buffers():
                hubert_ignore.append(f"discriminator.hubert.{name}")
            self._ddp_params_and_buffers_to_ignore = hubert_ignore

        # Paper Algorithm 2: λ_adv dynamically adjusted based on L_real/L_G ratio
        self.lambda_adv = 0.25
        self.condition_shuffle = config.get('condition_shuffle', 'none')
        if self.condition_shuffle not in ('none', 'within_bank'):
            raise ValueError(
                "condition_shuffle must be 'none' or 'within_bank', "
                f"got {self.condition_shuffle!r}"
            )
        if self.condition_shuffle == 'within_bank':
            print(
                "[condition ablation] using within-bank shuffled q: "
                "real conditions are shuffled only among W_y rows; "
                "synthetic conditions are shuffled only among W_syn rows"
            )

    def set_discriminator_grad(self, requires_grad):
        for param in self.discriminator.parameters():
            param.requires_grad_(requires_grad)

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        return x_norm

    def _disc_requires_condition(self):
        return getattr(self.discriminator, "requires_condition", False)

    def _real_conditions(self, label, device):
        label = label.to(self.loss.W.device, dtype=torch.long)
        condition = self.loss.W[:, label].detach().t()
        return F.normalize(condition, dim=1).to(device)

    def _synth_conditions(self, cols, device):
        if not getattr(self.loss, "persistence", False):
            raise RuntimeError("Projection MLP-D synthetic conditions require persistence=True")
        if len(cols) == 0:
            raise RuntimeError("Projection MLP-D received no synthetic columns")
        idx = torch.tensor(cols, device=self.loss.W_syn.device, dtype=torch.long)
        condition = self.loss.W_syn[:, idx].detach().t()
        return F.normalize(condition, dim=1).to(device)

    def _deranged_perm(self, n, device):
        if n <= 1:
            return torch.arange(n, device=device)
        base = torch.arange(n, device=device)
        for _ in range(8):
            perm = torch.randperm(n, device=device)
            if torch.all(perm != base):
                return perm
        # Deterministic fallback: every row receives a different row's q.
        return torch.roll(base, shifts=1)

    def _maybe_shuffle_conditions(self, conditions, part_sizes):
        if self.condition_shuffle != 'within_bank' or conditions is None:
            return conditions
        parts = []
        start = 0
        for size in part_sizes:
            part = conditions[start:start + size]
            perm = self._deranged_perm(size, part.device)
            parts.append(part[perm])
            start += size
        return torch.cat(parts, dim=0)

    def _disc_forward(self, embeddings, conditions=None):
        if self._disc_requires_condition():
            return self.discriminator(embeddings, conditions)
        return self.discriminator(embeddings)

    def forward(self, x):
        feature = self.features(x)
        embedding = self.model(feature)
        return embedding
    def adjust_lambda_adv(self, am_loss, g_loss):
        # Source-faithful control law (massabaali7/CAARMA adjust_weight):
        # cap=0.01, floor=0.0001. Earlier this branch carried cap=0.5 — a
        # regression introduced by commit f18cf61, NOT the GitHub source value.
        # Combined with the per-batch reset to 0.25 (in the M step) the
        # effective trajectory is the discrete {0.01, 0.225, 0.25} set.
        loss_ratio = am_loss.detach() / (g_loss.detach() + 1e-8)
        if loss_ratio > 1.5:
            self.lambda_adv = min(self.lambda_adv * 1.1, 0.01)
        elif loss_ratio < 0.5:
            self.lambda_adv = max(self.lambda_adv * 0.9, 0.0001)
    
    def on_train_epoch_start(self):
        # Persistent synthetic classes: recompute the global-NN pair->column map
        # from the current (DDP-synced) real prototypes and reset the per-epoch
        # activated set. Identical across ranks (W is synced at the boundary).
        if getattr(self.loss, 'persistence', False):
            self.loss.synth.rebuild_pairing(self.loss.W)

    def training_step(self, batch, batch_idx):
        opt_main, opt_d = self.optimizers()

        waveform = batch['waveform']
        label = batch['mapped_id']

        # ── Algorithm 2, Step 1: Update Discriminator every batch ────────
        # Encoder is detached from the D update; only D receives gradients.
        self.toggle_optimizer(opt_d)
        self.set_discriminator_grad(True)
        model_was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                feature_d = self.features(waveform)
                embedding_d = self.model(feature_d)
                _, _, synth_for_d = self.loss(embedding_d, label)
                synth_cols_d = list(getattr(self.loss, "last_synth_cols", []))
        finally:
            if model_was_training:
                self.model.train()

        opt_d.zero_grad()
        B = embedding_d.size(0)
        combined_d = torch.cat(
            [self.normalize(embedding_d), self.normalize(synth_for_d)], dim=0
        )
        if self._disc_requires_condition():
            if len(synth_cols_d) != synth_for_d.size(0):
                raise RuntimeError(
                    "Projection MLP-D condition mismatch in D step: "
                    f"{len(synth_cols_d)} cols for {synth_for_d.size(0)} synthetic rows"
                )
            cond_d = torch.cat(
                [
                    self._real_conditions(label, combined_d.device),
                    self._synth_conditions(synth_cols_d, combined_d.device),
                ],
                dim=0,
            )
            cond_d = self._maybe_shuffle_conditions(cond_d, [B, synth_for_d.size(0)])
        else:
            cond_d = None
        preds_d_all = self._disc_forward(combined_d, cond_d)
        # real (B rows) is FIRST here, so the [:B] / [B:] split is correct even
        # when synth_for_d has Ns != B rows (persistent mode skips some samples).
        real_preds, fake_preds_d = preds_d_all[:B], preds_d_all[B:]
        # Paper Eq.(1): L_D = BCE(D(e),1) + BCE(D(e_syn),0)
        d_loss = (self.BCE_loss(real_preds, torch.ones_like(real_preds)) +
                  self.BCE_loss(fake_preds_d, torch.zeros_like(fake_preds_d)))
        with torch.no_grad():
            d_real_acc = (real_preds > 0).float().mean()
            d_fake_acc = (fake_preds_d < 0).float().mean()
            d_acc = 0.5 * (d_real_acc + d_fake_acc)
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # ── Algorithm 2, Step 2: Update M every batch ────────────────────
        # Freeze D weights for the generator/encoder update. Autograd still
        # backpropagates through D to its input embeddings, but DDP no longer
        # waits for D parameter gradients in this second backward pass.
        self.toggle_optimizer(opt_main)
        self.set_discriminator_grad(False)

        # Source-faithful: reset λ_adv to 0.25 at the start of every G step, so
        # adjust_lambda_adv becomes a one-shot per-batch decision rather than a
        # multiplicative drift that compounds and sticks at the floor.
        self.lambda_adv = 0.25

        feature = self.features(waveform)
        embedding = self.model(feature)
        opt_main.zero_grad()
        # update_state=True only here (the single L_real M-step call): the memory
        # bank and per-epoch activated-column set are advanced exactly once/step.
        amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label, update_state=True)
        synth_cols_g = list(getattr(self.loss, "last_synth_cols", []))
        amsoftmax_syn_loss, _, _ = self.loss_syn(embedding, label, flagSyn=True)

        # Paper Eq.(2): L_G = BCE(D(e_syn),1) + BCE(D(e),0)  — no pretrain phase
        # synthetic (fake) is FIRST here and in persistent mode may have Ns <= B
        # rows (samples whose partner had no batch/bank embedding are skipped).
        # Split on the actual synthetic count Ns, NOT B, otherwise (B-Ns) real
        # embeddings leak into the fake half and corrupt L_G.
        Ns = synthetic_embeddings.size(0)
        combined_g = torch.cat(
            [self.normalize(synthetic_embeddings), self.normalize(embedding)], dim=0
        )
        if self._disc_requires_condition():
            if len(synth_cols_g) != synthetic_embeddings.size(0):
                raise RuntimeError(
                    "Projection MLP-D condition mismatch in G step: "
                    f"{len(synth_cols_g)} cols for {synthetic_embeddings.size(0)} synthetic rows"
                )
            cond_g = torch.cat(
                [
                    self._synth_conditions(synth_cols_g, combined_g.device),
                    self._real_conditions(label, combined_g.device),
                ],
                dim=0,
            )
            cond_g = self._maybe_shuffle_conditions(cond_g, [Ns, embedding.size(0)])
        else:
            cond_g = None
        preds_g_all = self._disc_forward(combined_g, cond_g)
        fake_preds_g, real_preds_g = preds_g_all[:Ns], preds_g_all[Ns:]
        g_loss = (self.BCE_loss(fake_preds_g, torch.ones_like(fake_preds_g)) +
                  self.BCE_loss(real_preds_g, torch.zeros_like(real_preds_g)))

        # Paper Algorithm 2: "Adjust λ_adv based on L_real/L_G"
        self.adjust_lambda_adv(amsoftmax_loss, g_loss)

        # Paper: L_total = L_real + (1/N)*L_syn + λ_adv * L_G
        total_loss = (amsoftmax_loss
                      + (1.0 / self.config['num_spk']) * amsoftmax_syn_loss
                      + self.lambda_adv * g_loss)

        self.manual_backward(total_loss)
        opt_main.step()
        self.set_discriminator_grad(True)
        self.untoggle_optimizer(opt_main)

        # Warmup LR
        if self.trainer.global_step < self.config['warmup_step']:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.config['warmup_step']))
            for pg in opt_main.param_groups:
                pg['lr'] = lr_scale * self.learning_rate

        self.log('am_loss', amsoftmax_loss, prog_bar=True, sync_dist=False)
        self.log('am_loss_syn', amsoftmax_syn_loss, prog_bar=True, sync_dist=False)
        self.log('acc', acc, prog_bar=True, sync_dist=False)
        self.log('g_loss', g_loss, prog_bar=True, sync_dist=False)
        self.log('total_loss', total_loss, prog_bar=True, sync_dist=False)
        self.log('d_loss', d_loss, prog_bar=True, sync_dist=False)
        self.log('d_acc', d_acc, prog_bar=True, sync_dist=False)
        self.log('d_real_logit', real_preds.detach().mean(), prog_bar=False, sync_dist=False)
        self.log('d_fake_logit', fake_preds_d.detach().mean(), prog_bar=False, sync_dist=False)
        self.log('lambda_adv', self.lambda_adv, prog_bar=False, sync_dist=False)
        if getattr(self.loss, 'persistence', False):
            stats = self.loss.synth.last_stats
            self.log('synth_table_size', stats['synth_table_size'], prog_bar=False, sync_dist=False)
            self.log('active_synth_cols', stats['active_synth_cols'], prog_bar=False, sync_dist=False)
            self.log('reuse_rate', stats['reuse_rate'], prog_bar=False, sync_dist=False)
            self.log('new_pair_rate', stats['new_pair_rate'], prog_bar=False, sync_dist=False)
            self.log('mean_visits_per_pair', stats['mean_visits_per_pair'], prog_bar=False, sync_dist=False)
            self.log('pair_visit_entropy', stats['pair_visit_entropy'], prog_bar=False, sync_dist=False)
            self.log('Ns_over_B', stats['Ns_over_B'], prog_bar=False, sync_dist=False)
            self.log('bank_hit_rate', stats['bank_hit_rate'], prog_bar=False, sync_dist=False)
            self.log('batch_hit_rate', stats['batch_hit_rate'], prog_bar=False, sync_dist=False)


            
            

    def configure_optimizers(self):
        # Modified learning rates and optimizer parameters
        embedding_optimizer = AdamW(
            list(self.model.parameters())+list(self.loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)  # Standard Adam betas
        )

        # Lower learning rate for discriminator
        discriminator_optimizer = AdamW(
            self.discriminator.parameters(),
            lr= 2e-4, # Significantly reduced 2e-4
            #self.learning_rate * 0.01,  
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999)
        )
        
        embedding_scheduler = StepLR(embedding_optimizer, step_size = 4, gamma=0.5)
        discriminator_scheduler = StepLR(discriminator_optimizer, step_size = 4, gamma=0.5)

        return [embedding_optimizer, discriminator_optimizer], \
            [embedding_scheduler, discriminator_scheduler]

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
        
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
    
    def similarity_score(self, trials, index_mapping, eval_vectors):
        labels = []
        scores = []
        epsilon = 1e-8  # Small value to prevent division by zero
        for item in trials:
            enroll_vector = eval_vectors[index_mapping[self.config['root'] + item[1]]]
            test_vector = eval_vectors[index_mapping[self.config['root'] + item[2]]]
            with torch.cuda.amp.autocast():
                score = enroll_vector.dot(test_vector.T)
                denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
                score = score/ (denom + epsilon)
            if np.isnan(score):
                print("Warning: NaN detected in score calculation. Setting score to 0.")
                score = 0.0
            labels.append(int(item[0]))
            scores.append(score)
            
        scoress = torch.tensor(scores)
        meanscores = torch.mean(scoress)
        print(meanscores)
        return labels, scores
    
    def compute_eer(self, labels, scores):
        """sklearn style compute eer
        """
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        threshold = interp1d(fpr, thresholds)(eer)
        return eer, threshold 

    def compute_minDCF(self, labels, scores, p_target=0.01, c_miss=1, c_fa=1):
        """MinDCF
        Computes the minimum of the detection cost function.  The comments refer to
        equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
        """
        scores = np.array(scores)
        labels = np.array(labels)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr

        min_c_det = float("inf")
        min_c_det_threshold = thresholds[0]
        for i in range(0, len(fnr)):
            c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det
                min_c_det_threshold = thresholds[i]
        c_def = min(c_miss * p_target, c_fa * (1 - p_target))
        min_dcf = min_c_det / c_def
        return min_dcf, min_c_det_threshold
    
    # def on_validation_epoch_end(self):
    #     num_gpus = torch.cuda.device_count()
    #     eval_vectors = [None for _ in range(num_gpus)]
    #     dist.all_gather_object(eval_vectors, self.eval_vectors)
    #     eval_vectors = np.vstack(eval_vectors)

    #     table = [None for _ in range(num_gpus)]
    #     dist.all_gather_object(table, self.index_mapping)

    #     index_mapping = {}
    #     for i in table:
    #         index_mapping.update(i)

    #     eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
    #     labels, scores = self.similarity_score(self.trials, index_mapping, eval_vectors)
    #     EER, threshold = self.compute_eer(labels, scores)
    #     with open('org_inf_labels_VOX_base_3.09.txt', 'w') as f:
    #         for line in labels:
    #             f.write(f"{line}\n")
    #     with open('org_inf_scores_VOX_base_3.09.txt', 'w') as f:
    #         for line in scores:
    #             f.write(f"{line}\n")
    #     print("\ncosine EER: {:.2f}% with threshold {:.2f}".format(EER*100, threshold))
    #     self.log("cosine_eer", EER*100)
        
    #     minDCF, threshold = self.compute_minDCF(labels, scores, p_target=0.01)
    #     print("cosine minDCF(10-2): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
    #     self.log("cosine_minDCF(10-2)", minDCF)
        
    #     minDCF, threshold = self.compute_minDCF(labels, scores, p_target=0.001)
    #     print("cosine minDCF(10-3): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
    #     self.log("cosine_minDCF(10-3)", minDCF)
    def on_validation_epoch_end(self):
        num_gpus = torch.cuda.device_count()
        # Gather eval_vectors from all GPUs
        all_eval_vectors = [None for _ in range(num_gpus)]
        dist.all_gather_object(all_eval_vectors, self.eval_vectors)

        # Gather index_mapping from all GPUs
        all_index_mappings = [None for _ in range(num_gpus)]
        dist.all_gather_object(all_index_mappings, self.index_mapping)

        # Fix: local batch_idx from each GPU must be offset by the cumulative
        # number of vectors from preceding GPUs before merging into global map.
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
        # Only print and log on rank 0 to avoid duplicate output
        if self.trainer.is_global_zero:
            print("\ncosine EER: {:.2f}%".format(EER*100))
            minDCF2, _ = self.compute_minDCF(labels, scores, p_target=0.01)
            print("cosine minDCF(10-2): {:.4f}".format(minDCF2))
            minDCF3, _ = self.compute_minDCF(labels, scores, p_target=0.001)
            print("cosine minDCF(10-3): {:.4f}".format(minDCF3))
        else:
            minDCF2, _ = self.compute_minDCF(labels, scores, p_target=0.01)
            minDCF3, _ = self.compute_minDCF(labels, scores, p_target=0.001)
        self.log("cosine_eer", EER*100, sync_dist=True)
        self.log("cosine_minDCF(10-2)", minDCF2, sync_dist=True)
        self.log("cosine_minDCF(10-3)", minDCF3, sync_dist=True)


def cli_main():
    def load_config(config_file_path):
        """Load the configuration from the file."""
        with open(config_file_path) as file:
            config = yaml.safe_load(os.path.expandvars(file.read()))
        return config

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="/root/autodl-tmp/CAARMA/config.yaml",
        help="YAML experiment configuration",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    
    dataloader = super_dataset(config)

    features = build_feature(config)
        
    model = build_model(config, device)

    criterion = build_criterion(config)

    final_project = Task(features, model, criterion, config, learning_rate = config['init_lr'], weight_decay=config['weight_decay'], batch_size = config['batch_size'], num_workers = config['num_workers'], max_epochs = config['epochs'], trial_path= config['trial_path'], warmup_step = config['warmup_step'])
    
    resume_ckpt_path = config.get('resume_from_checkpoint', 'None')
    if resume_ckpt_path != 'None':
        print("resume full training state from {}".format(resume_ckpt_path))
    elif config['checkpoint_path'] != 'None':
        state_dict = torch.load(config['checkpoint_path'], map_location="cpu")["state_dict"]
        # print(state_dict.keys())
        # model_state_dict = model.state_dict()
        # model_state_dict.update(state_dict)
        final_project.load_state_dict(state_dict, strict=False)
        print("load weight from {}".format(config['checkpoint_path']))
        
    assert config['save_dir'] is not None
    checkpoint_callback = ModelCheckpoint(monitor='cosine_eer', save_top_k=3,
            mode='min', filename="{epoch}_{cosine_eer:.2f}", dirpath=config['save_dir'],
            save_last=True)


    # wandb_logger = WandbLogger(
    #     project='mixup',     # Change this to your W&B project name
    #     name='weight_decay_100BS_alternate_syn',          # Change this to your desired experiment name
    #     save_dir=config['save_dir']
    # )
    # wandb_logger.experiment.config.update(config)
    wandb_logger = None

    AVAIL_GPUS = torch.cuda.device_count()
    if AVAIL_GPUS < 1:
        raise RuntimeError("No CUDA GPU is visible to PyTorch.")
    # trainer = Trainer(
    #     strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
    #     # plugins=DDPPlugin(find_unused_parameters=False),
    #     accelerator="gpu",
    #     devices=-1,  # Use all available GPUs
    #     max_epochs=config['epochs'],
    #     # logger=wandb_logger, 
    #     logger=False,
    #     num_sanity_val_steps=0,  # Adjust for faster debugging
    #     sync_batchnorm=True,
    #     precision=16,  # Enable mixed precision training
    #     # callbacks=[checkpoint_callback, lr_monitor],
    #     callbacks=[checkpoint_callback],
    #     #     EarlyStopping(
    #     #     monitor='cosine_eer',
    #     #     patience=10,
    #     #     mode='min',
    #     #     min_delta=0.001
    #     # )
    #     #],
    #     default_root_dir=config['save_dir'],
    #     reload_dataloaders_every_n_epochs=1,
    #     accumulate_grad_batches=1,
    #     log_every_n_steps=25,
    #     benchmark=True,  # Improved speed if input sizes don't change
    #     deterministic=False,  # Better performance
    #     # Add profiler for performance monitoring
    #     profiler="simple",

    # )
    trainer = Trainer(
        strategy=DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
        ),
        accelerator="gpu",
        devices=AVAIL_GPUS,
        max_epochs=config['epochs'],
        logger=False,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        default_root_dir=config['save_dir'],
        reload_dataloaders_every_n_epochs=0,
        accumulate_grad_batches=1,
        log_every_n_steps=25,
        benchmark=True,
        deterministic=False,
    )

    #————————————————————————————————————————————————————————————————————————
    #trainer.fit(final_project, datamodule=dataloader)

    # if config.get('checkpoint_path'):
    #     trainer.fit(
    #         final_project, 
    #         datamodule=dataloader, 
    #         ckpt_path=config['checkpoint_path']
    #     )
    # else:

    #     trainer.fit(final_project, datamodule=dataloader)
    
    trainer.fit(
        final_project,
        datamodule=dataloader,
        ckpt_path=None if resume_ckpt_path == 'None' else resume_ckpt_path,
    )

    #print("\n--- Running Immediate Validation ---")
    #trainer.validate(final_project, datamodule=dataloader, ckpt_path=config['checkpoint_path'])
    
if __name__ == "__main__":
    cli_main()
    
