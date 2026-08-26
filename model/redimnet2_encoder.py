"""CAARMA adapter for the official ReDimNet2 speaker encoder."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn


OFFICIAL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "redimnet2_official"
)
if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

from redimnet2.redimnet2 import ReDimNet2Wrap  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ReDimNet2B6Encoder(nn.Module):
    """Load the official B6 VoxCeleb2 large-margin checkpoint exactly."""

    def __init__(
        self,
        checkpoint_path: str,
        expected_sha256: str,
        embedding_dim: int = 192,
    ):
        super().__init__()
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ReDimNet2 checkpoint not found: {path}")

        actual_sha256 = _sha256(path)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise RuntimeError(
                "ReDimNet2 checkpoint SHA-256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )

        payload = _load_checkpoint(path)
        if "model_config" not in payload or "state_dict" not in payload:
            raise RuntimeError(
                "ReDimNet2 checkpoint must contain model_config and state_dict"
            )

        model_config = dict(payload["model_config"])
        checkpoint_dim = int(model_config.get("embed_dim", embedding_dim))
        if checkpoint_dim != int(embedding_dim):
            raise ValueError(
                "ReDimNet2 embedding dimension mismatch: "
                f"checkpoint={checkpoint_dim}, CAARMA={embedding_dim}"
            )

        self.encoder = ReDimNet2Wrap(**model_config)
        result = self.encoder.load_state_dict(payload["state_dict"])
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Official ReDimNet2 checkpoint did not load exactly: "
                f"missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )

        parameters = sum(parameter.numel() for parameter in self.parameters())
        print(
            "Loaded ReDimNet2-B6 Vox2-LM encoder "
            f"params={parameters} embedding_dim={checkpoint_dim} "
            f"sha256={actual_sha256}"
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        if waveform.ndim != 2:
            raise ValueError(
                "ReDimNet2 expects waveform shaped [batch, samples], "
                f"got {tuple(waveform.shape)}"
            )
        return self.encoder(waveform)
