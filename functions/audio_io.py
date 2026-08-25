from pathlib import Path

import torchaudio


_PYAV_AUDIO_SUFFIXES = frozenset({".aac", ".m4a", ".mp4"})


def _load_with_pyav(filename):
    try:
        import av
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to decode AAC/M4A audio. Run the PSC "
            "M4A decoder probe before starting VoxCeleb1+2 training."
        ) from exc

    chunks = []
    with av.open(str(filename), mode="r") as container:
        if not container.streams.audio:
            raise RuntimeError(f"No audio stream found in {filename!s}")

        stream = container.streams.audio[0]
        sample_rate = int(stream.codec_context.sample_rate or stream.rate)
        resampler = av.AudioResampler(
            format="fltp", layout="mono", rate=sample_rate
        )

        for frame in container.decode(stream):
            for output_frame in resampler.resample(frame):
                chunk = np.asarray(
                    output_frame.to_ndarray(), dtype=np.float32
                )
                if chunk.ndim == 1:
                    chunk = chunk[None, :]
                chunks.append(chunk)

        for output_frame in resampler.resample(None):
            chunk = np.asarray(
                output_frame.to_ndarray(), dtype=np.float32
            )
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError(f"PyAV decoded no samples from {filename!s}")

    waveform = np.ascontiguousarray(np.concatenate(chunks, axis=1))
    return torch.from_numpy(waveform), sample_rate


def load_audio_file(filename):
    """Load WAV/FLAC with TorchAudio and AAC containers with PyAV."""
    suffix = Path(str(filename)).suffix.lower()
    if suffix in _PYAV_AUDIO_SUFFIXES:
        return _load_with_pyav(filename)
    return torchaudio.load(filename)
