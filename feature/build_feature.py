import torch.nn as nn

from .fbanks import Mel_Spectrogram


class WaveformPassthrough(nn.Module):
    """Identity feature extractor — returns the raw waveform unchanged.

    Used by encoders that compute their own front-end (e.g. ReDimNet).
    """

    def forward(self, x):
        return x


def build_feature(config):
    if config['features'] == 'Fbank':
        # SpecAugment is optional, controlled by a top-level config flag.
        # Default off for backward-compatibility with branches that predate
        # this flag.
        spec_aug = bool(config.get('spec_aug', False))
        return Mel_Spectrogram(
            spec_aug=spec_aug,
            freq_mask_param=int(config.get('freq_mask_param', 15)),
            time_mask_param=int(config.get('time_mask_param', 20)),
            num_freq_masks=int(config.get('num_freq_masks', 2)),
            num_time_masks=int(config.get('num_time_masks', 2)),
        )
    if config['features'] == 'Passthrough':
        return WaveformPassthrough()
    raise NotImplementedError(
        f"features={config['features']!r} not supported "
        "(known: 'Fbank', 'Passthrough')"
    )
