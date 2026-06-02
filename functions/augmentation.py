"""Wave-form augmentation for speaker-verification training.

This module covers the augmentations that operate on the RAW WAVEFORM:
  * add_noise    — overlay MUSAN noise at a random SNR
  * add_reverb   — convolve with a Room Impulse Response (RIRS_NOISES)
  * drop_chunk   — zero-out random time chunks (waveform-level dropout)

SpecAugment-style frequency/time masking on the MEL-SPECTROGRAM is done
separately inside ``feature/fbanks.py:Mel_Spectrogram`` — much more
correct than trying to apply ``transforms.FrequencyMasking`` on raw audio.
The ``drop_freq`` flag in this class is therefore disabled by design;
keep the kwarg only for backward-compat with old configs.

Bug-fix history vs the original code shipped with CAARMA repo:
  1. CSV loading was unconditional → would crash on import if noise.csv /
     rir.csv didn't exist on disk. Now we only load the CSVs whose
     corresponding flag is True.
  2. ``dataset.py`` called ``self.augmentation(x)`` with one positional
     argument, but ``__call__(self, x, sr)`` expects two → guaranteed
     TypeError at the first batch. Fix is on the dataset side now; here
     we keep the two-argument signature.
  3. The ``add_real_noise`` / ``add_reverberate`` helpers dtype-cast back
     to ``waveform.dtype`` via ``.type_as(waveform)`` so downstream
     STFT / Mel layers keep their expected dtype.
"""

import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy import signal


class Augmentation:
    def __init__(self,
                 add_noise: bool = True,
                 add_reverb: bool = True,
                 drop_freq: bool = False,
                 drop_chunk: bool = False,
                 noise_csv: Optional[str] = None,
                 reverb_csv: Optional[str] = None):
        self.add_noise = add_noise
        self.add_reverb = add_reverb
        self.drop_freq = drop_freq        # kept for compat, not recommended
        self.drop_chunk = drop_chunk

        # Bug-fix #1: only load CSVs whose flag is on. Avoids crash when
        # only SpecAugment / drop_chunk is enabled and no MUSAN/RIR data
        # is present on disk.
        self.noise_paths = None
        self.reverb_paths = None
        if add_noise:
            if noise_csv is None:
                raise ValueError("add_noise=True requires noise_csv path")
            self.noise_paths = pd.read_csv(noise_csv)["wav"].tolist()
            if len(self.noise_paths) == 0:
                raise ValueError(f"noise_csv {noise_csv} is empty")
        if add_reverb:
            if reverb_csv is None:
                raise ValueError("add_reverb=True requires reverb_csv path")
            self.reverb_paths = pd.read_csv(reverb_csv)["wav"].tolist()
            if len(self.reverb_paths) == 0:
                raise ValueError(f"reverb_csv {reverb_csv} is empty")

    # ──────────────────────────────────────────────────────────────────
    def compute_dB(self, waveform: torch.Tensor) -> torch.Tensor:
        val = torch.clamp(torch.mean(torch.pow(waveform, 2)), min=0.0)
        return 10.0 * torch.log10(val + 1e-4)

    def add_real_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        clean_dB = self.compute_dB(waveform)

        idx = np.random.randint(0, len(self.noise_paths))
        noise, _ = torchaudio.load(self.noise_paths[idx])
        # mono: average down if stereo
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        noise = noise.to(dtype=torch.float32).squeeze(0)  # (T,)

        snr = float(np.random.uniform(15, 25))

        noise_length = noise.shape[-1]
        audio_length = waveform.shape[-1]

        if audio_length >= noise_length:
            # Tile noise (wrap-around) so the padding isn't silent. .repeat
            # allocates a fresh tensor so there's no overlapping-memory issue.
            n_tiles = (audio_length + noise_length - 1) // noise_length
            noise = noise.repeat(n_tiles)[:audio_length]
        else:
            start = int(np.random.randint(0, noise_length - audio_length))
            noise = noise[start:start + audio_length].contiguous()

        noise_dB = self.compute_dB(noise)
        scale = torch.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10))
        noise = scale * noise
        return (waveform + noise).type_as(waveform)

    def add_reverberate(self, waveform: torch.Tensor) -> torch.Tensor:
        audio_length = waveform.shape[-1]
        idx = np.random.randint(0, len(self.reverb_paths))
        rir, _ = torchaudio.load(self.reverb_paths[idx])
        if rir.shape[0] > 1:
            rir = rir.mean(dim=0, keepdim=True)
        rir = rir.to(dtype=torch.float32).squeeze(0)  # (T_rir,)

        rir_norm = torch.sqrt(torch.sum(rir ** 2)).clamp(min=1e-9)
        rir = rir / rir_norm

        # scipy.signal.convolve on numpy is faster than torch fft conv
        out = signal.convolve(waveform.cpu().numpy(), rir.cpu().numpy(), mode='full')
        out = torch.from_numpy(out)
        return out[..., :audio_length].type_as(waveform)

    def drop_chunk_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        drop_count_low, drop_count_high = 1, 3
        drop_length_low, drop_length_high = 1000, 2000

        dropped = waveform.clone()
        drop_times = random.randint(drop_count_low, drop_count_high)
        if drop_times == 0:
            return dropped

        lengths = torch.randint(drop_length_low, drop_length_high + 1, (drop_times,))
        max_start = waveform.shape[-1] - int(lengths.max().item())
        if max_start <= 0:
            return dropped
        starts = torch.randint(0, max_start + 1, (drop_times,))

        for j in range(drop_times):
            s = int(starts[j])
            e = s + int(lengths[j])
            dropped[s:e] = 0.0
        return dropped.type_as(waveform)

    # ──────────────────────────────────────────────────────────────────
    def __call__(self, x: torch.Tensor, sr: int = 16000) -> torch.Tensor:  # noqa: ARG002
        # `sr` is currently unused (none of the active augmentations need it),
        # but kept in the signature so dataset.py can pass sr=16000 without
        # caring about the implementation detail. Useful if we re-enable
        # sample-rate-dependent augmentations later.
        # Reverb first (room acoustics happens before noise mixes in).
        if self.add_reverb:
            x = self.add_reverberate(x)
        if self.add_noise:
            x = self.add_real_noise(x)
        if self.drop_chunk:
            x = self.drop_chunk_waveform(x)
        # drop_freq removed — see module docstring.
        return x
