import torch.nn as nn

from .fbanks import Mel_Spectrogram


def build_feature(config):
    if config['features'] == 'Fbank':
        features = Mel_Spectrogram()
    elif config['features'] == 'Passthrough':
        features = nn.Identity()
    else:
        raise NotImplementedError

    return features
