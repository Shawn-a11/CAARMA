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
from model.discriminator_mix import MixupDiscriminator
from helper.mixup_avg import mixup_data_euc_avg

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from pytorch_lightning.loggers import WandbLogger

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
        
        self.discriminator = MixupDiscriminator(cache_dir="./cache_dir/").train()
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # HuBERT backbone is now TRAINABLE (source-faithful), so it must NOT
        # be excluded from DDP — its gradients are produced on every D step
        # and need the all-reduce, otherwise the four replicas silently
        # diverge. The old _ddp_params_and_buffers_to_ignore workaround only
        # applied to the frozen-backbone setup; with requires_grad=True the
        # backbone params follow the same toggle_optimizer path that already
        # works for the adapter/projection/head params.

        # Paper Algorithm 2: λ_adv dynamically adjusted based on L_real/L_G ratio
        self.lambda_adv = 0.25

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        return x_norm
    def forward(self, x):
        feature = self.features(x)
        embedding = self.model(feature)
        return embedding
    def adjust_lambda_adv(self, am_loss, g_loss):
        # Source-faithful control law (massabaali7/CAARMA train.py adjust_weight):
        # cap=0.01, floor=0.0001. This caps the adversarial weight at ~1% of total
        # loss so L_G stays a weak regularizer rather than competing with L_real.
        # Combined with the per-batch reset (lambda_adv = 0.25 at start of every
        # G step) the effective trajectory is the discrete {0.01, 0.225, 0.25}
        # set from source code, not our previous compounding range [0.01, 0.5].
        loss_ratio = am_loss.detach() / (g_loss.detach() + 1e-8)
        if loss_ratio > 1.5:
            self.lambda_adv = min(self.lambda_adv * 1.1, 0.01)
        elif loss_ratio < 0.5:
            self.lambda_adv = max(self.lambda_adv * 0.9, 0.0001)
    
    def training_step(self, batch, batch_idx):
        opt_main, opt_d = self.optimizers()

        waveform = batch['waveform']
        label = batch['mapped_id']

        # ── Algorithm 2, Step 1: Update Discriminator every batch ────────
        # Encoder runs inside no_grad so DDP never registers encoder params
        # as "used in this forward" — avoids stale All-Reduce deadlock.
        # D forward uses a SINGLE concatenated call so DDP sees discriminator
        # used exactly once per backward (prevents bucket reducer confusion).
        self.toggle_optimizer(opt_d)
        with torch.no_grad():
            feature_d = self.features(waveform)
            embedding_d = self.model(feature_d)
            _, _, synth_for_d = self.loss(embedding_d, label)

        opt_d.zero_grad()
        B = embedding_d.size(0)
        combined_d = torch.cat(
            [self.normalize(embedding_d), self.normalize(synth_for_d)], dim=0
        )
        preds_d_all = self.discriminator(combined_d)
        real_preds, fake_preds_d = preds_d_all[:B], preds_d_all[B:]
        # Paper Eq.(1): L_D = BCE(D(e),1) + BCE(D(e_syn),0)
        d_loss = (self.BCE_loss(real_preds, torch.ones_like(real_preds)) +
                  self.BCE_loss(fake_preds_d, torch.zeros_like(fake_preds_d)))
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # ── Algorithm 2, Step 2: Update M every batch ────────────────────
        # Recompute embedding inside opt_main toggle so DDP only sees encoder
        # params; discriminator params are frozen by toggle_optimizer.
        # D forward uses ONE concatenated call (real + synthetic together).
        self.toggle_optimizer(opt_main)

        # Source-faithful: reset λ_adv to 0.25 at start of every G step. This
        # turns adjust_lambda_adv into a discrete one-shot decision based on
        # the current batch's am/g ratio, rather than a multiplicative drift
        # that compounds across batches.
        self.lambda_adv = 0.25

        feature = self.features(waveform)
        embedding = self.model(feature)
        opt_main.zero_grad()
        amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)
        amsoftmax_syn_loss, _, _ = self.loss_syn(embedding, label, flagSyn=True)

        # Paper Eq.(2): L_G = BCE(D(e_syn),1) + BCE(D(e),0)  — no pretrain phase
        combined_g = torch.cat(
            [self.normalize(synthetic_embeddings), self.normalize(embedding)], dim=0
        )
        preds_g_all = self.discriminator(combined_g)
        fake_preds_g, real_preds_g = preds_g_all[:B], preds_g_all[B:]
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
        self.log('lambda_adv', self.lambda_adv, prog_bar=False, sync_dist=False)
                

            
            

    def configure_optimizers(self):
        # Modified learning rates and optimizer parameters
        embedding_optimizer = AdamW(
            list(self.model.parameters())+list(self.loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)  # Standard Adam betas
        )

        # Discriminator with two learning-rate groups:
        # - HuBERT backbone at init_lr*0.01 = 1e-5, the source code's actual
        #   discriminator LR (its comment "Significantly reduced 2e-4" means
        #   2e-4 was reduced to 1e-5). A 315M pretrained backbone needs the
        #   small SSL-finetune rate or its features get destroyed.
        # - adapter/projections/head at the paper's 2e-4. Our runs where ALL
        #   D params sat at 1e-5 (01, 04) collapsed to the trivial
        #   equilibrium (d_loss pinned at ln2 — chance level) and scored
        #   worst; the head needs 2e-4 to keep the GAN game alive.
        hubert_param_ids = {id(p) for p in self.discriminator.hubert.parameters()}
        head_params = [p for p in self.discriminator.parameters()
                       if id(p) not in hubert_param_ids]
        discriminator_optimizer = AdamW(
            [
                {"params": head_params, "lr": 2e-4},
                {"params": list(self.discriminator.hubert.parameters()),
                 "lr": self.learning_rate * 0.01},
            ],
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
            config = yaml.safe_load(file)
        return config

    config = load_config("/root/autodl-tmp/CAARMA/config.yaml")

    # Reproducibility: seed everything before any randomness (encoder init,
    # AM-Softmax W init, DataLoader workers, augmentation choices).
    seed_everything(int(config.get('seed', 42)), workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    
    dataloader = super_dataset(config)

    features = build_feature(config)
        
    model = build_model(config, device)

    criterion = build_criterion(config)

    final_project = Task(features, model, criterion, config, learning_rate = config['init_lr'], weight_decay=config['weight_decay'], batch_size = config['batch_size'], num_workers = config['num_workers'], max_epochs = config['epochs'], trial_path= config['trial_path'], warmup_step = config['warmup_step'])
    
    if config['checkpoint_path'] != 'None':
        state_dict = torch.load(config['checkpoint_path'], map_location="cpu")["state_dict"]
        # print(state_dict.keys())
        # model_state_dict = model.state_dict()
        # model_state_dict.update(state_dict)
        final_project.load_state_dict(state_dict, strict=False)
        print("load weight from {}".format(config['checkpoint_path']))
        
    assert config['save_dir'] is not None
    # save_top_k=1: with the HuBERT backbone now trainable, each checkpoint
    # carries its AdamW state (~2.5 GB extra), so keeping 3+last would eat
    # >15 GB of /root/autodl-tmp.
    checkpoint_callback = ModelCheckpoint(monitor='cosine_eer', save_top_k=1,
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
            gradient_as_bucket_view=True,   # avoid bucket reducer hangs with manual opt
            static_graph=False,
        ),
        accelerator="gpu",
        devices=4,                    # 4卡 V100
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
    
    trainer.fit(final_project, datamodule=dataloader)

    #print("\n--- Running Immediate Validation ---")
    #trainer.validate(final_project, datamodule=dataloader, ckpt_path=config['checkpoint_path'])
    
if __name__ == "__main__":
    cli_main()
    