import os
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from helper.config_utils import load_experiment_config
from tools.audit_voxceleb_protocol import audit_protocol
from tools.build_voxceleb_csv import build_manifest, speakers_from_trials


class PscDataAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.train_root = self.root / "vox1" / "dev" / "wav"
        for speaker in ("id0001", "id0002"):
            path = self.train_root / speaker / "video1" / "00001.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        self.eval_root = self.root / "vox1" / "test" / "wav"
        for speaker in ("id1001", "id1002"):
            path = self.eval_root / speaker / "video1" / "00001.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        self.trials = self.root / "veri_test2.txt"
        self.trials.write_text(
            "1 id1001/video1/00001.wav id1001/video1/00001.wav\n"
            "0 id1001/video1/00001.wav id1002/video1/00001.wav\n",
            encoding="utf-8",
        )
        self.manifest = self.root / "manifests" / "vox1.csv"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_manifest_and_protocol_audit(self):
        report = build_manifest([("vox1", self.train_root)], self.manifest)
        self.assertEqual(report["speakers"], 2)
        self.assertEqual(report["utterances"], 2)

        protocol, errors = audit_protocol(
            self.manifest, self.trials, self.eval_root
        )
        self.assertEqual(errors, [])
        self.assertEqual(protocol["train_speakers"], 2)
        self.assertEqual(protocol["trials"], 2)

    def test_unified_wav_root_excludes_trial_speakers(self):
        unified = self.root / "unified" / "wav"
        for speaker in ("id0001", "id0002", "id1001", "id1002"):
            path = unified / speaker / "video1" / "00001.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        excluded = speakers_from_trials(self.trials, re.compile(r"id\d+"))
        report = build_manifest(
            [("vox1", unified)],
            self.manifest,
            exclude_speakers=excluded,
        )
        self.assertEqual(excluded, {"id1001", "id1002"})
        self.assertEqual(report["speakers"], 2)
        self.assertEqual(report["excluded_speaker_count"], 2)

    def test_protocol_audit_detects_train_test_leakage(self):
        build_manifest([("vox1", self.train_root)], self.manifest)
        leaked = self.train_root / "id0001" / "video1" / "00001.wav"
        self.trials.write_text(f"1 {leaked} {leaked}\n", encoding="utf-8")
        _, errors = audit_protocol(self.manifest, self.trials, self.eval_root)
        self.assertIn("training and evaluation share utterance files", errors)
        self.assertIn("training and evaluation contain overlapping speaker IDs", errors)

    def test_environment_config_expansion_and_auto_speaker_count(self):
        build_manifest([("vox1", self.train_root)], self.manifest)
        save_root = self.root / "runs"
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "dataset": "${TEST_MANIFEST}",
            "trial_path": "${TEST_TRIALS}",
            "root": "${TEST_EVAL_ROOT}",
            "save_dir": "${TEST_SAVE_ROOT}/checkpoints",
            "num_spk": "auto",
        }), encoding="utf-8")

        values = {
            "TEST_MANIFEST": str(self.manifest),
            "TEST_TRIALS": str(self.trials),
            "TEST_EVAL_ROOT": str(self.eval_root),
            "TEST_SAVE_ROOT": str(save_root),
        }
        previous = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        try:
            config = load_experiment_config(str(config_path))
        finally:
            for key, old in previous.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

        self.assertEqual(config["num_spk"], 2)
        self.assertTrue(Path(config["save_dir"]).is_dir())


if __name__ == "__main__":
    unittest.main()
