import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".agents/skills/psc-caarma-runner/scripts/psc_job.py"
)
SPEC = importlib.util.spec_from_file_location("psc_job", SCRIPT)
psc_job = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(psc_job)


def completed(command, stdout="", returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


class PscJobRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_run_root = os.environ.get("CAARMA_RUN_ROOT")
        os.environ["CAARMA_RUN_ROOT"] = self.tempdir.name

    def tearDown(self):
        if self.previous_run_root is None:
            os.environ.pop("CAARMA_RUN_ROOT", None)
        else:
            os.environ["CAARMA_RUN_ROOT"] = self.previous_run_root
        self.tempdir.cleanup()

    def test_audit_script_is_allowed_and_production_requires_confirmation(self):
        audit = psc_job.REPO_ROOT / "scripts/psc/audit_voxceleb.slurm"
        metadata = psc_job.validate_script(audit, confirm_production=False)
        self.assertEqual(metadata["partition"], "GPU-shared")
        self.assertFalse(metadata["production"])
        self.assertEqual(metadata["maximum_su"], 1.0)

        train = psc_job.REPO_ROOT / "scripts/psc/train_vox1.slurm"
        with self.assertRaises(PermissionError):
            psc_job.validate_script(train, confirm_production=False)
        production = psc_job.validate_script(train, confirm_production=True)
        self.assertEqual(production["gpu_count"], 4)
        self.assertEqual(production["maximum_su"], 48.0)

    def test_metric_parser_preserves_same_evaluation_dcf(self):
        text = """
Epoch 7: 100% complete
cosine EER: 3.61%
cosine minDCF(10-2): 0.3424
cosine minDCF(10-3): 0.4626
Epoch 8: 100% complete
cosine EER: 3.45%
cosine minDCF(10-2): 0.3266
cosine minDCF(10-3): 0.4338
Trainer.fit stopped: `max_steps=12` reached.
"""
        metrics = psc_job.parse_metrics(text)
        self.assertEqual(metrics["best"]["epoch_zero_based"], 8)
        self.assertEqual(metrics["best"]["eer_percent"], 3.45)
        self.assertEqual(metrics["best"]["min_dcf_1e2"], 0.3266)
        self.assertEqual(metrics["best"]["min_dcf_1e3"], 0.4338)
        self.assertTrue(metrics["smoke_steps_complete"])

    def test_environment_fix_uses_available_gpu_and_frontend_probe(self):
        script = psc_job.REPO_ROOT / "scripts/psc/fix_pkg_resources.slurm"
        text = script.read_text()
        self.assertIn("#SBATCH --gpus=v100-32:1", text)
        self.assertIn("setuptools==75.8.0", text)
        self.assertIn("Mel_Spectrogram()", text)
        self.assertIn("PSC audio frontend OK", text)

    def test_submit_records_job_and_cancel_requires_explicit_yes(self):
        audit = psc_job.REPO_ROOT / "scripts/psc/audit_voxceleb.slurm"
        calls = []

        def fake_run(command, check=True):
            calls.append(command)
            if command[:3] == ["git", "-C", str(psc_job.REPO_ROOT)]:
                if command[-2:] == ["branch", "--show-current"]:
                    return completed(command, stdout=psc_job.REQUIRED_BRANCH + "\n")
                if command[-2:] == ["status", "--porcelain"]:
                    return completed(command, stdout="")
                return completed(command, stdout="deadbeef\n")
            if command[0] == "sbatch":
                return completed(command, stdout="12345\n")
            if command[0] == "squeue":
                return completed(command, stdout="12345|RUNNING|00:01|01:00:00|node\n")
            if command[0] == "scancel":
                return completed(command)
            raise AssertionError(command)

        args = argparse.Namespace(
            env_file=None,
            script=str(audit),
            confirm_production=False,
            afterok=None,
            require_completed=[],
            dry_run=False,
            label="audit-test",
        )
        with mock.patch.object(psc_job, "run", side_effect=fake_run):
            self.assertEqual(psc_job.cmd_submit(args), 0)
            state = json.loads(psc_job.state_path("12345").read_text())
            self.assertEqual(state["label"], "audit-test")
            self.assertEqual(state["git_commit"], "deadbeef")

            with self.assertRaises(PermissionError):
                psc_job.cmd_cancel(argparse.Namespace(job_id="12345", yes=False))
            self.assertEqual(
                psc_job.cmd_cancel(argparse.Namespace(job_id="12345", yes=True)), 0
            )
        self.assertTrue(any(command[0] == "scancel" for command in calls))


if __name__ == "__main__":
    unittest.main()
