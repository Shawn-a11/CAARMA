import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import Evaluation_Dataset, Train_Dataset
from .augmentation import Augmentation


class super_dataset(LightningDataModule):
    def __init__(
        self,
        config,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.config = config


    def train_dataloader(self) -> DataLoader:
        # Bug fix: previously augmentation=None was hardcoded → do_augmentation
        # flag had no effect. Now we actually construct Augmentation when
        # do_augmentation=True, and pass the noise/reverb CSV paths from config
        # so add_noise / add_reverb can read MUSAN / RIRS_NOISES wav lists.
        augmentation = None
        if self.config.get('do_augmentation', False):
            aug_cfg = self.config.get('augmentations', {})
            augmentation = Augmentation(
                add_noise=bool(aug_cfg.get('add_noise', False)),
                add_reverb=bool(aug_cfg.get('add_reverb', False)),
                drop_freq=bool(aug_cfg.get('drop_freq', False)),
                drop_chunk=bool(aug_cfg.get('drop_chunk', False)),
                noise_csv=self.config.get('noise_csv'),
                reverb_csv=self.config.get('reverb_csv'),
            )
        train_dataset = Train_Dataset(self.config['dataset'], self.config['second'],
                                      do_augmentation=self.config.get('do_augmentation', False),
                                      augmentation=augmentation)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=self.config['num_workers'],
                batch_size=self.config['batch_size'],
                pin_memory=True,
                drop_last=False,
                )
        return loader

    def val_dataloader(self) -> DataLoader:
        trials = np.loadtxt(self.config['trial_path'], str)
        self.trials = trials
        eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
        print("number of enroll: {}".format(len(set(trials.T[1]))))
        print("number of test: {}".format(len(set(trials.T[2]))))
        print("number of evaluation: {}".format(len(eval_path)))
        # eval_dataset = Evaluation_Dataset(eval_path, second=-1)
        eval_dataset = Evaluation_Dataset(eval_path, root=self.config['root'])
        loader = torch.utils.data.DataLoader(eval_dataset,
                                             num_workers=10,
                                             shuffle=False, 
                                             batch_size=1)
        return loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


