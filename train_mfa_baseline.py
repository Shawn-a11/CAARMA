"""Paper Table 2 ID 1: MFA-Conformer with real-speaker AM-Softmax only.

This control intentionally contains no adversarial training, mixup
discriminator, synthetic embeddings, or synthetic-class loss. It isolates the
paper's reported 3.33% MFA-Conformer baseline under the current PSC protocol.
"""

from argparse import ArgumentParser
import os

import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from criterion.build_criterion import build_criterion
from feature.build_feature import build_feature
from functions.loader import super_dataset
from helper.config_utils import load_experiment_config
from model.model_build import build_model


def warmup_then_single_decay_multiplier(
    step,
    warmup_steps,
    current_epoch,
    decay_after_epoch,
    decay_gamma,
):
    """Warm up per step, then apply one decay after the named display epoch."""
    warmup = min(1.0, float(step + 1) / float(max(1, warmup_steps)))
    if decay_after_epoch is None:
        return warmup
    completed_epochs = int(current_epoch)
    return warmup * (float(decay_gamma) if completed_epochs >= decay_after_epoch else 1.0)


class Task(LightningModule):
    def __init__(
        self,
        features,
        model,
        loss,
        config,
        learning_rate,
        weight_decay,
        trial_path,
        **kwargs,
    ):
        super().__init__()
        self.features = features
        self.model = model
        self.loss = loss
        self.config = config
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.trials = np.loadtxt(trial_path, str)

    def forward(self, waveform):
        return self.model(self.features(waveform))

    def training_step(self, batch, batch_idx):
        embedding = self(batch['waveform'])
        am_loss, acc, _ = self.loss(embedding, batch['mapped_id'])

        self.log(
            'am_loss', am_loss, prog_bar=True, on_step=True, on_epoch=True,
            sync_dist=True,
        )
        self.log(
            'acc', acc, prog_bar=True, on_step=True, on_epoch=True,
            sync_dist=True,
        )
        self.log(
            'learning_rate',
            self.trainer.optimizers[0].param_groups[0]['lr'],
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        return am_loss

    def on_train_epoch_start(self):
        decay_after_epoch = self.config.get('lr_decay_after_epoch')
        if decay_after_epoch is None or self.current_epoch != int(decay_after_epoch):
            return
        decay_gamma = float(self.config.get('lr_decay_gamma', 1.0))
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group['lr'] = self.learning_rate * decay_gamma

    def configure_optimizers(self):
        optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        warmup_steps = max(1, int(self.config.get('warmup_step', 2000)))
        decay_after_epoch = self.config.get('lr_decay_after_epoch')
        if decay_after_epoch is not None:
            decay_after_epoch = int(decay_after_epoch)
        decay_gamma = float(self.config.get('lr_decay_gamma', 1.0))
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_then_single_decay_multiplier(
                step=step,
                warmup_steps=warmup_steps,
                current_epoch=self.current_epoch,
                decay_after_epoch=decay_after_epoch,
                decay_gamma=decay_gamma,
            ),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def on_validation_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            embedding = self(batch['waveform'])
        self.eval_vectors.append(embedding.detach().cpu().numpy()[0])
        self.index_mapping[batch['path'][0]] = batch_idx

    def similarity_score(self, trials, index_mapping, eval_vectors):
        labels, scores = [], []
        for item in trials:
            enroll_path = os.path.normpath(
                os.path.join(self.config['root'], str(item[1]).lstrip('/\\'))
            )
            test_path = os.path.normpath(
                os.path.join(self.config['root'], str(item[2]).lstrip('/\\'))
            )
            enroll = eval_vectors[index_mapping[enroll_path]]
            test = eval_vectors[index_mapping[test_path]]
            denominator = np.linalg.norm(enroll) * np.linalg.norm(test)
            score = enroll.dot(test.T) / (denominator + 1e-8)
            labels.append(int(item[0]))
            scores.append(0.0 if np.isnan(score) else float(score))
        return labels, scores

    @staticmethod
    def compute_eer(labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        eer = brentq(
            lambda value: 1.0 - value - interp1d(fpr, tpr)(value),
            0.0,
            1.0,
        )
        return eer, interp1d(fpr, thresholds)(eer)

    @staticmethod
    def compute_min_dcf(labels, scores, p_target=0.01, c_miss=1, c_fa=1):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr
        costs = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        index = int(np.argmin(costs))
        default_cost = min(c_miss * p_target, c_fa * (1 - p_target))
        return float(costs[index] / default_cost), thresholds[index]

    def on_validation_epoch_end(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_vectors = [None for _ in range(world_size)]
        all_mappings = [None for _ in range(world_size)]
        if dist.is_initialized():
            dist.all_gather_object(all_vectors, self.eval_vectors)
            dist.all_gather_object(all_mappings, self.index_mapping)
        else:
            all_vectors[0] = self.eval_vectors
            all_mappings[0] = self.index_mapping

        index_mapping = {}
        offset = 0
        for rank_vectors, rank_mapping in zip(all_vectors, all_mappings):
            for path, local_index in rank_mapping.items():
                index_mapping[path] = offset + local_index
            offset += len(rank_vectors)

        eval_vectors = np.vstack(all_vectors)
        eval_vectors -= np.mean(eval_vectors, axis=0)
        labels, scores = self.similarity_score(
            self.trials, index_mapping, eval_vectors
        )
        eer, _ = self.compute_eer(labels, scores)
        min_dcf_1e2, _ = self.compute_min_dcf(
            labels, scores, p_target=0.01
        )
        min_dcf_1e3, _ = self.compute_min_dcf(
            labels, scores, p_target=0.001
        )

        if self.trainer.is_global_zero:
            print('\ncosine EER: {:.2f}%'.format(eer * 100))
            print('cosine minDCF(10-2): {:.4f}'.format(min_dcf_1e2))
            print('cosine minDCF(10-3): {:.4f}'.format(min_dcf_1e3))
        self.log('cosine_eer', eer * 100, sync_dist=True)
        self.log('cosine_minDCF(10-2)', min_dcf_1e2, sync_dist=True)
        self.log('cosine_minDCF(10-3)', min_dcf_1e3, sync_dist=True)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--smoke-steps', type=int, default=0)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config['criterion'] != 'AMSoftmax' or bool(config.get('mixup', False)):
        raise ValueError(
            'Table 2 ID1 requires criterion=AMSoftmax and mixup=false'
        )

    seed_everything(int(config.get('seed', 42)), workers=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    datamodule = super_dataset(config)
    project = Task(
        build_feature(config),
        build_model(config, device),
        build_criterion(config),
        config,
        learning_rate=config['init_lr'],
        weight_decay=config['weight_decay'],
        trial_path=config['trial_path'],
    )

    checkpoint_path = args.checkpoint or config.get('checkpoint_path', 'None')
    if checkpoint_path != 'None':
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        project.load_state_dict(state_dict, strict=True)
        print('load weight from {}'.format(checkpoint_path))

    smoke_mode = args.smoke_steps > 0
    checkpoint_callback = ModelCheckpoint(
        monitor='cosine_eer',
        save_top_k=int(config.get('save_top_k', 3)),
        mode='min',
        filename='{epoch}_{cosine_eer:.2f}',
        dirpath=config['save_dir'],
        save_last=True,
    )
    devices = int(args.devices or config.get('devices', 4))
    trainer = Trainer(
        strategy=DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        ),
        accelerator='gpu',
        devices=devices,
        max_epochs=int(config['epochs']),
        logger=False,
        num_sanity_val_steps=0,
        sync_batchnorm=bool(config.get('sync_batchnorm', False)),
        precision=str(config.get('precision', '16-mixed')),
        callbacks=[] if smoke_mode else [checkpoint_callback],
        enable_checkpointing=not smoke_mode,
        default_root_dir=config['save_dir'],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=25,
        benchmark=True,
        deterministic=False,
        max_steps=args.smoke_steps if smoke_mode else -1,
        limit_val_batches=0 if smoke_mode else 1.0,
    )
    trainer.fit(project, datamodule=datamodule)


if __name__ == '__main__':
    cli_main()
