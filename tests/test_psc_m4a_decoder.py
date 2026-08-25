import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


REPO = Path(__file__).resolve().parents[1]


def load_audio_io(fake_torchaudio):
    module_path = REPO / "functions" / "audio_io.py"
    spec = importlib.util.spec_from_file_location(
        "test_audio_io", module_path
    )
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("torchaudio")
    sys.modules["torchaudio"] = fake_torchaudio
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("torchaudio", None)
        else:
            sys.modules["torchaudio"] = previous
    return module


class TestPSCM4ADecoder(unittest.TestCase):
    def setUp(self):
        self.fake_torchaudio = types.ModuleType("torchaudio")
        self.fake_torchaudio.load = Mock(return_value=("waveform", 16000))
        self.fake_torchaudio.list_audio_backends = Mock(
            return_value=["ffmpeg", "soundfile", "sox_io"]
        )
        self.audio_io = load_audio_io(self.fake_torchaudio)

    def test_m4a_uses_pyav_decoder(self):
        self.audio_io._load_with_pyav = Mock(
            return_value=("waveform", 16000)
        )
        result = self.audio_io.load_audio_file("speaker/sample.M4A")
        self.assertEqual(result, ("waveform", 16000))
        self.audio_io._load_with_pyav.assert_called_once_with(
            "speaker/sample.M4A"
        )
        self.fake_torchaudio.load.assert_not_called()

    def test_wav_preserves_default_backend(self):
        result = self.audio_io.load_audio_file("speaker/sample.wav")
        self.assertEqual(result, ("waveform", 16000))
        self.fake_torchaudio.load.assert_called_once_with(
            "speaker/sample.wav"
        )

    def test_vox1v2_launcher_uses_validated_pyav_vendor(self):
        script = (
            REPO / "scripts" / "psc" / "train_vox1v2.slurm"
        ).read_text(encoding="utf-8")
        self.assertNotIn("module load ffmpeg/4.3.1", script)
        self.assertIn("CAARMA_PYAV_VENDOR", script)
        self.assertIn("import av", script)
        self.assertIn("load_audio_file(probe_path)", script)
        self.assertIn("#SBATCH --exclude=v010", script)

    def test_probe_installs_binary_pyav_wheel_without_ffmpeg_module(self):
        script = (
            REPO / "scripts" / "psc" / "probe_m4a_decoder.slurm"
        ).read_text(encoding="utf-8")
        self.assertNotIn("module load ffmpeg/4.3.1", script)
        self.assertIn("--only-binary=:all:", script)
        self.assertIn('"av==12.3.0"', script)
        self.assertIn("load_audio_file(path)", script)


if __name__ == "__main__":
    unittest.main()
