from pathlib import Path

import torchaudio


_FFMPEG_AUDIO_SUFFIXES = frozenset({".aac", ".m4a", ".mp4"})


def _required_backend(filename):
    suffix = Path(str(filename)).suffix.lower()
    return "ffmpeg" if suffix in _FFMPEG_AUDIO_SUFFIXES else None


def load_audio_file(filename):
    """Load audio while routing AAC containers through TorchAudio FFmpeg."""
    backend = _required_backend(filename)
    try:
        if backend is None:
            return torchaudio.load(filename)
        return torchaudio.load(filename, backend=backend)
    except Exception as exc:
        if backend is None:
            raise
        try:
            available = torchaudio.list_audio_backends()
        except Exception:
            available = []
        raise RuntimeError(
            f"Failed to decode {filename!s} with TorchAudio's FFmpeg backend. "
            f"Available backends: {available}. Load the PSC ffmpeg/4.3.1 module "
            "before importing torchaudio."
        ) from exc
