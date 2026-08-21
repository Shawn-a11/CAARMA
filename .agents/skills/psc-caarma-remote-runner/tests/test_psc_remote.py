import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/psc_remote.py"
SPEC = importlib.util.spec_from_file_location("psc_remote", SCRIPT)
psc_remote = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(psc_remote)


class PscRemoteTest(unittest.TestCase):
    def test_rejects_unsafe_ids_and_paths(self):
        self.assertEqual(psc_remote.validate_job_id("44072499"), "44072499")
        with self.assertRaises(ValueError):
            psc_remote.validate_job_id("44072499;scancel")
        with self.assertRaises(ValueError):
            psc_remote.validate_remote_path("/jet/home/sge2/../other")

    def test_parse_key_values(self):
        payload = psc_remote.parse_kv(
            "JOB_ID=44072499\nJOB_NAME=caarma-mlpd361\nCOMMIT=abc123\n"
        )
        self.assertEqual(payload["JOB_ID"], "44072499")
        self.assertEqual(payload["JOB_NAME"], "caarma-mlpd361")

    def test_cancel_requires_explicit_yes(self):
        args = mock.Mock(job_id="44072499", yes=False, host="bridges2", timeout=5)
        with self.assertRaises(PermissionError):
            psc_remote.cmd_cancel(args)

    def test_submit_records_remote_job(self):
        stdout = "\n".join([
            "REPO=/jet/home/sge2/CAARMA",
            "COMMIT=abcdef123456",
            "SCRIPT=scripts/psc/train_vox1.slurm",
            "JOB_NAME=caarma-vox1",
            "GPUS=v100-32:4",
            "WALLTIME=12:00:00",
            "JOB_ID=44070000",
            "OUTPUT=/ocean/projects/cis220031p/sge2/logs/caarma-vox1-%j.out",
            "ERROR_LOG=/ocean/projects/cis220031p/sge2/logs/caarma-vox1-%j.err",
        ])
        completed = mock.Mock(stdout=stdout, returncode=0)
        args = mock.Mock(
            host="bridges2", repo="/jet/home/sge2/CAARMA",
            script="scripts/psc/train_vox1.slurm", expected_commit="abcdef1",
            afterok=None, confirm_production=True, allow_launcher_dirty=False,
            dry_run=False, timeout=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(psc_remote, "STATE_ROOT", Path(directory)):
                with mock.patch.object(psc_remote, "ssh_run", return_value=completed):
                    self.assertEqual(psc_remote.cmd_submit(args), 0)
                    state = psc_remote.load_state("44070000")
        self.assertEqual(
            state["OUTPUT"],
            "/ocean/projects/cis220031p/sge2/logs/caarma-vox1-44070000.out",
        )

    def test_logs_can_use_explicit_paths_without_local_state(self):
        args = mock.Mock(
            job_id="44072499", host="bridges2", lines=10, timeout=5,
            output="/ocean/projects/cis220031p/sge2/logs/job.out",
            error="/ocean/projects/cis220031p/sge2/logs/job.err",
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(psc_remote, "STATE_ROOT", Path(directory)):
                with mock.patch.object(psc_remote, "remote_tail", return_value="ok\n"):
                    self.assertEqual(psc_remote.cmd_logs(args), 0)

    def test_remote_tail_compacts_progress_lines(self):
        completed = mock.Mock(
            stdout="Epoch 0: 1/10\rEpoch 0: 2/10\x1b[A\nmetric\n",
            returncode=0,
        )
        with mock.patch.object(psc_remote, "ssh_run", return_value=completed):
            output = psc_remote.remote_tail(
                "bridges2",
                "/ocean/projects/cis220031p/sge2/logs/job.out",
                2,
                5,
            )
        self.assertEqual(output, "Epoch 0: 2/10\nmetric\n")


if __name__ == "__main__":
    unittest.main()
