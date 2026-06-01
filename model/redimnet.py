"""ReDimNet-b6 encoder wrapper for CAARMA pipeline.

ReDimNet (IDRnD, INTERSPEECH 2024) is a SOTA speaker encoder family.
b6 is the largest variant (~14.7M params); on VoxCeleb1-O the paper reports
EER 0.89% when used alone. Here we want to test if it improves over the
MFA-Conformer baseline (3.48% under CAARMA's full L_syn + AT + MD pipeline).

The official implementation lives at IDRnD/ReDimNet on GitHub. We load it
via torch.hub so we get the up-to-date model definition without vendoring.

Pipeline contract this wrapper enforces:
  input  : (B, 1, T_frames, n_mels=80)  — same shape Mel_Spectrogram emits
  output : (B, 192)                      — same shape MFA-Conformer emits

ReDimNet's native forward expects (B, n_mels, T) so we squeeze/permute.
ReDimNet-b6 native feature_dim is 256, so we add a Linear(256 → 192)
projection so the rest of the pipeline (AM-Softmax W = 192×1211, mixup,
discriminator adapter) can stay unchanged.

From-scratch training note
==========================
IDRnD's hubconf entry ReDimNet(model_name, train_type, dataset) ALWAYS
loads the pretrained checkpoint via load_custom() — it doesn't expose a
`pretrained=False` flag. For a fair apples-to-apples comparison against
MFA-Conformer (which we trained from scratch), we load the model and then
call reset_parameters() on every submodule that supports it. This wastes
the checkpoint download bandwidth on first run, but gives us clean
random-init weights for training.
"""
import torch
import torch.nn as nn


def _reset_weights_recursive(module: nn.Module) -> int:
    """Call reset_parameters() on every submodule that has it. Returns
    the number of modules reset (for sanity logging)."""
    n = 0
    for m in module.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
            n += 1
    return n


class ReDimNetB6(nn.Module):
    """ReDimNet-b6 wrapped to match the (B, 1, T, n_mels) → (B, 192) contract."""

    def __init__(self, n_mels: int = 80, embedding_dim: int = 192,
                 pretrained: bool = False):
        super().__init__()
        # Hub entry signature: ReDimNet(model_name, train_type='ptn',
        # dataset='vox2') — does NOT accept a pretrained kwarg, always loads
        # the checkpoint for (model_name, train_type, dataset).
        self.backbone = torch.hub.load(
            'IDRnD/ReDimNet',
            'ReDimNet',
            model_name='b6',
            train_type='ptn',
            dataset='vox2',
            source='github',
            trust_repo=True,
        )

        # If pretrained=False (default), wipe the loaded weights so we train
        # from random init like MFA-Conformer baseline.
        if not pretrained:
            n_reset = _reset_weights_recursive(self.backbone)
            print(f'[ReDimNetB6] reset_parameters() called on {n_reset} '
                  f'modules → from-scratch random init')

        # ReDimNet-b6's native output dim. The hub model's `.feat_dim` attr
        # exposes this; we read it instead of hard-coding for robustness.
        native_dim = getattr(self.backbone, 'feat_dim', 256)
        if native_dim != embedding_dim:
            self.proj = nn.Linear(native_dim, embedding_dim)
        else:
            self.proj = nn.Identity()

        self.n_mels = n_mels
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 1, T, n_mels) from Mel_Spectrogram. Squeeze channel and
        # transpose to (B, n_mels, T) which is what ReDimNet expects.
        if x.dim() == 4:
            x = x.squeeze(1)                  # (B, T, n_mels)
        x = x.transpose(1, 2).contiguous()    # (B, n_mels, T)

        emb = self.backbone(x)                # (B, native_dim)
        emb = self.proj(emb)                  # (B, embedding_dim)
        return emb
