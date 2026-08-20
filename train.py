from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union
import os
import torch.distributed as dist
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy

import random
import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CyclicLR

from feature.build_feature import build_feature
from functions.loader import super_dataset
from criterion.build_criterion import build_criterion
from model.model_build import build_model
from model.discriminator_mix import MixupDiscriminator
from helper.mixup_avg import mixup_data_euc_avg
from helper.config_utils import load_experiment_config

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

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
        
        embedding_dim = self.config['embedding_dim']
        self.discriminator = MixupDiscriminator(
            cache_dir=self.config.get("hubert_cache_dir", "./cache_dir/")
        ).train()
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device) 
        
        # Add hyperparameters for GAN training
        self.lambda_adv = 0.25  # Weight for adversarial loss
        self.pretrain_eps = 15 # Number of steps to pre-train the discriminator
        
        self.pretrain_discriminator = True
        self.discriminator_steps = 0

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        return x_norm
    def forward(self, x):
        feature = self.features(x)
        embedding = self.model(feature)
        return embedding
    def adjust_weight(self,amsoftmax_loss,g_loss):
        # Dynamic adjustment after warmup
        if self.current_epoch > self.pretrain_eps:
            loss_ratio = amsoftmax_loss / (g_loss + 1e-8)
        if loss_ratio > 1.5:
            self.lambda_adv = min(self.lambda_adv * 1.1, 0.01)
        elif loss_ratio < 0.5:
            self.lambda_adv = max(self.lambda_adv * 0.9, 0.0001)
        return self.lambda_adv

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        d_sch = self.lr_schedulers()
        optimizer_main, d_optimizer = opt
        main_scheduler, d_scheduler = d_sch
        
        waveform = batch['waveform']
        label = batch['mapped_id']
        
        # Get real embeddings
        feature = self.features(waveform)
        embedding = self.model(feature)
        
        # First compute AM-Softmax loss
        amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)
        
        # Initialize counters if they don't exist
        if not hasattr(self, 'd_step_counter'):
            self.d_step_counter = 0
        if not hasattr(self, 'g_step_counter'):
            self.g_step_counter = 0
        
        if self.d_step_counter >= 1 and self.g_step_counter >= 5:
                self.d_step_counter = 0
                self.g_step_counter = 0
                print("set 0 pre-training")
        
        elif self.d_step_counter >= 1 and self.g_step_counter >= 1 and self.current_epoch > self.pretrain_eps:
                self.d_step_counter = 0
                self.g_step_counter = 0
                print("set 0 discriminator")


        if self.current_epoch <= self.pretrain_eps:
            
            if self.g_step_counter < 5:
                self.lambda_adv = 0.0005
                optimizer_main.zero_grad()
                real_preds = self.discriminator(self.normalize(embedding.detach()))
                fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
                fake_labels = torch.zeros(real_preds.size()).to(self.device) 
                
                # Simple adversarial loss for generator - try to make synthetic look real
                real_labels = torch.ones(fake_preds.size()).to(self.device) 
                g_loss = (self.BCE_loss(fake_preds, real_labels)  + self.BCE_loss(real_preds, fake_labels))/2
                # Reduced adversarial weight for better stability
                amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label) #,flagSyn=True)
                amsoftmax_syn_loss, acc_syn, synthetic_embeddings = self.loss_syn(embedding, label,flagSyn=True)
                total_loss = amsoftmax_loss + (1/self.config['num_spk']) * amsoftmax_syn_loss + self.lambda_adv * g_loss
                self.manual_backward(total_loss)
                # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Logging
                self.log('am_loss', amsoftmax_loss, prog_bar=True)
                self.log('am_loss_syn', amsoftmax_syn_loss, prog_bar=True)
                self.log('acc', acc, prog_bar=True)
                self.log('g_loss', g_loss, prog_bar=True)
                self.log('total_loss', total_loss, prog_bar=True)
                optimizer_main.step()
                if self.trainer.global_step < self.config['warmup_step']:
                    lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.config['warmup_step']))
                    for pg in optimizer_main.param_groups:
                        pg['lr'] = lr_scale * self.learning_rate
                print("gloss")
                self.g_step_counter += 1
                return total_loss
            
            elif self.d_step_counter < 1:  
                # Train Discriminator
                d_optimizer.zero_grad()
                # Real samples
                real_preds = self.discriminator(self.normalize(embedding.detach()))
                fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
                real_labels = torch.ones(real_preds.size()).to(self.device) 
                d_real_loss = self.BCE_loss(real_preds, real_labels)
                
                # Fake samples
                fake_labels = torch.zeros(fake_preds.size()).to(self.device)
                d_fake_loss = self.BCE_loss(fake_preds, fake_labels)
                
                # Simple discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                self.manual_backward(d_loss)
                self.log('d_loss', d_loss, prog_bar=True)
                d_optimizer.step()
                self.d_step_counter += 1
                print("d_loss")
                return d_loss
        
        else:
            if self.d_step_counter < 1:  

                # Train Discriminator
                d_optimizer.zero_grad()                
                # Real samples
                real_preds = self.discriminator(self.normalize(embedding.detach()))
                fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
                real_labels = torch.ones(real_preds.size()).to(self.device) 
                d_real_loss = self.BCE_loss(real_preds, real_labels)
                
                # Fake samples
                fake_labels = torch.zeros(fake_preds.size()).to(self.device) 
                d_fake_loss = self.BCE_loss(fake_preds, fake_labels)
                
                # Simple discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                    
                self.manual_backward(d_loss)
                self.log('d_loss', d_loss, prog_bar=True)
                d_optimizer.step()
                self.d_step_counter += 1
                print("d_loss")
                return d_loss

            elif self.d_step_counter >= 1 and self.g_step_counter < 1:  # Train the generator for 1 step
                # Train Generator (Main Model) first
                optimizer_main.zero_grad()
                self.lambda_adv = 0.25  # Weight for adversarial loss
                real_preds = self.discriminator(self.normalize(embedding.detach()))
                fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
                fake_labels = torch.zeros(real_preds.size()).to(self.device) 
                # Simple adversarial loss for generator - try to make synthetic look real
                real_labels = torch.ones(fake_preds.size()).to(self.device)
                g_loss = (self.BCE_loss(fake_preds, real_labels)  + self.BCE_loss(real_preds, fake_labels))/2

                amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label) 
                amsoftmax_syn_loss, acc_syn, synthetic_embeddings = self.loss_syn(embedding, label,flagSyn=True)
                self.lambda_adv = self.adjust_weight(amsoftmax_loss,g_loss)
                # Reduced adversarial weight for better stability
                total_loss = amsoftmax_loss + (1/self.config['num_spk']) * amsoftmax_syn_loss + self.lambda_adv * g_loss
                self.manual_backward(total_loss)
                # Logging
                self.log('am_loss', amsoftmax_loss, prog_bar=True)
                self.log('am_loss_syn', amsoftmax_syn_loss, prog_bar=True)
                self.log('acc', acc, prog_bar=True)
                self.log('g_loss', g_loss, prog_bar=True)
                self.log('total_loss', total_loss, prog_bar=True)
                optimizer_main.step()
                if self.trainer.global_step < self.config['warmup_step']:
                    lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.config['warmup_step']))
                    for pg in optimizer_main.param_groups:
                        pg['lr'] = lr_scale * self.learning_rate
                print("gloss")
                self.g_step_counter += 1

                return total_loss
                

            
            

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
            lr= self.learning_rate * 0.01,  # Significantly reduced 2e-4
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999)
        )
        
        embedding_scheduler = StepLR(embedding_optimizer, step_size = 4, gamma=0.5)
        discriminator_scheduler = StepLR(discriminator_optimizer, step_size = 4, gamma=0.5)

        return [embedding_optimizer, discriminator_optimizer], \
            [embedding_scheduler, discriminator_scheduler]

    def on_train_epoch_end(self):
        d_sch = self.lr_schedulers()
        main_scheduler, d_scheduler = d_sch         
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
            enroll_path = os.path.normpath(os.path.join(
                self.config['root'], str(item[1]).lstrip("/\\")
            ))
            test_path = os.path.normpath(os.path.join(
                self.config['root'], str(item[2]).lstrip("/\\")
            ))
            enroll_vector = eval_vectors[index_mapping[enroll_path]]
            test_vector = eval_vectors[index_mapping[test_path]]
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
    
    def on_validation_epoch_end(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        vectors_by_rank = [None for _ in range(world_size)]
        mappings_by_rank = [None for _ in range(world_size)]
        if world_size > 1:
            dist.all_gather_object(vectors_by_rank, self.eval_vectors)
            dist.all_gather_object(mappings_by_rank, self.index_mapping)
        else:
            vectors_by_rank[0] = self.eval_vectors
            mappings_by_rank[0] = self.index_mapping

        # batch_idx is local to each rank. Offset every rank-local mapping by
        # the number of vectors gathered from preceding ranks.
        index_mapping = {}
        offset = 0
        for rank_vectors, rank_mapping in zip(vectors_by_rank, mappings_by_rank):
            for path, local_index in rank_mapping.items():
                index_mapping[path] = offset + int(local_index)
            offset += len(rank_vectors)

        eval_vectors = np.vstack(vectors_by_rank)
        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
        labels, scores = self.similarity_score(self.trials, index_mapping, eval_vectors)
        EER, threshold = self.compute_eer(labels, scores)
        minDCF2, threshold2 = self.compute_minDCF(labels, scores, p_target=0.01)
        minDCF3, threshold3 = self.compute_minDCF(labels, scores, p_target=0.001)

        if self.trainer.is_global_zero:
            score_dir = os.path.join(self.config['save_dir'], "scores")
            os.makedirs(score_dir, exist_ok=True)
            with open(os.path.join(score_dir, 'labels.txt'), 'w') as handle:
                handle.writelines(f"{line}\n" for line in labels)
            with open(os.path.join(score_dir, 'scores.txt'), 'w') as handle:
                handle.writelines(f"{line}\n" for line in scores)
            print("\ncosine EER: {:.2f}% with threshold {:.4f}".format(
                EER * 100, threshold
            ))
            print("cosine minDCF(10-2): {:.4f} with threshold {:.4f}".format(
                minDCF2, threshold2
            ))
            print("cosine minDCF(10-3): {:.4f} with threshold {:.4f}".format(
                minDCF3, threshold3
            ))

        self.log("cosine_eer", EER * 100, sync_dist=True)
        self.log("cosine_minDCF(10-2)", minDCF2, sync_dist=True)
        self.log("cosine_minDCF(10-3)", minDCF3, sync_dist=True)

def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("train", "validate"), default="train")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run only N training steps and skip validation/checkpointing",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    seed_everything(int(config.get("seed", 42)), workers=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    
    dataloader = super_dataset(config)

    features = build_feature(config)
        
    model = build_model(config, device)

    criterion = build_criterion(config)

    final_project = Task(features, model, criterion, config, learning_rate = config['init_lr'], weight_decay=config['weight_decay'], batch_size = config['batch_size'], num_workers = config['num_workers'], max_epochs = config['epochs'], trial_path= config['trial_path'], warmup_step = config['warmup_step'])
    
    checkpoint_path = args.checkpoint or config.get('checkpoint_path', 'None')
    if args.mode == "train" and checkpoint_path != 'None':
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        # print(state_dict.keys())
        # model_state_dict = model.state_dict()
        # model_state_dict.update(state_dict)
        final_project.load_state_dict(state_dict, strict=False)
        print("load weight from {}".format(checkpoint_path))
        
    assert config['save_dir'] is not None
    checkpoint_callback = ModelCheckpoint(
        monitor='cosine_eer',
        save_top_k=int(config.get('save_top_k', 3)),
        mode='min',
        filename="{epoch}_{cosine_eer:.2f}",
        dirpath=config['save_dir'],
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    requested_devices = int(
        args.devices if args.devices is not None
        else config.get('devices', torch.cuda.device_count())
    )
    if requested_devices < 1:
        raise ValueError(f"devices must be positive, got {requested_devices}")
    strategy = (
        DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        if requested_devices > 1 else "auto"
    )
    smoke_mode = args.smoke_steps > 0
    trainer = Trainer(
        strategy=strategy,
        accelerator="gpu",
        devices=requested_devices,
        num_nodes=int(config.get('num_nodes', 1)),
        max_epochs=config['epochs'],
        logger=False,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        precision=config.get('precision', '16-mixed'),
        callbacks=[] if smoke_mode else [checkpoint_callback, lr_monitor],
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

    if args.mode == "train":
        trainer.fit(final_project, datamodule=dataloader)
    else:
        if checkpoint_path == 'None':
            raise ValueError("--mode validate requires --checkpoint or checkpoint_path")
        trainer.validate(final_project, datamodule=dataloader, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    cli_main()
