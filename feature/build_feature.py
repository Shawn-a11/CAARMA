import torch.nn as nn

from .fbanks import Mel_Spectrogram


class WaveformPassthrough(nn.Module):
    """Identity feature extractor — returns the raw waveform unchanged.

    Used by encoders that compute their own front-end (e.g. ReDimNet has an
    internal Mel-spectrogram + 2D stem). The CAARMA pipeline expects
    `self.features(waveform)` to return a tensor that is then fed to
    `self.model`; for such encoders the right thing is to let the waveform
    pass through verbatim and have the encoder do its own feature work.
    """

    def forward(self, x):
        return x


def build_feature(config):
    if config['features'] == 'Fbank':
        return Mel_Spectrogram()
    if config['features'] == 'Passthrough':
        return WaveformPassthrough()
    raise NotImplementedError(
        f"features={config['features']!r} not supported "
        "(known: 'Fbank', 'Passthrough')"
    )
