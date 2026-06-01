import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(
            inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)


class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop=160,
                 n_mels=80, coef=0.97, requires_grad=False,
                 spec_aug: bool = False,
                 freq_mask_param: int = 15,
                 time_mask_param: int = 20,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2):
        super(Mel_Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop = hop

        self.pre_emphasis = PreEmphasis(coef)
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.mel_basis = nn.Parameter(
            torch.FloatTensor(mel_basis), requires_grad=requires_grad)
        self.instance_norm = nn.InstanceNorm1d(num_features=n_mels)
        window = torch.hamming_window(self.win_length)
        self.window = nn.Parameter(
            torch.FloatTensor(window), requires_grad=False)

        # ── SpecAugment on the mel-spectrogram (no external data needed) ──
        # Applied only when self.training is True; turned on via config flag.
        self.spec_aug = spec_aug
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        if spec_aug:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, x):
        x = self.pre_emphasis(x)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window, win_length=self.win_length, return_complex=True)
        x = torch.abs(x)
        x += 1e-9
        x = torch.log(x)
        x = torch.matmul(self.mel_basis, x)        # (B, n_mels, T)
        x = self.instance_norm(x)                   # (B, n_mels, T)

        # SpecAugment: random frequency + time masks, training only.
        # FrequencyMasking / TimeMasking expect (..., freq, time) which is
        # exactly the layout here. Masking happens BEFORE the permute so the
        # downstream encoder receives the augmented spectrogram.
        if self.spec_aug and self.training:
            for _ in range(self.num_freq_masks):
                x = self.freq_masking(x)
            for _ in range(self.num_time_masks):
                x = self.time_masking(x)

        x = x.permute(0, 2, 1)                      # (B, T, n_mels)
        x = x.unsqueeze(1)                          # (B, 1, T, n_mels)
        return x
