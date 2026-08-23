import os
import re
import unittest


CONFIG = "config_psc_vox1_paper_exact.yaml"
SLURM = "scripts/psc/train_vox1_paper_exact.slurm"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), "r") as f:
        return f.read()


class TestPaperExactBs13Port(unittest.TestCase):
    def test_config_batch_size_is_13(self):
        cfg = _read(CONFIG)
        m = re.search(r"^batch_size:\s*(\d+)", cfg, re.MULTILINE)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 13)

    def test_config_matches_source_hyperparameters(self):
        cfg = _read(CONFIG)
        checks = {
            "init_lr": 0.001,
            "discriminator_lr": 0.0002,
            "am_margin": 0.2,
            "am_scale": 30,
            "epochs": 30,
            "seed": 42,
        }
        for key, expected in checks.items():
            with self.subTest(key=key):
                m = re.search(rf"^{key}:\s*([0-9.eE+-]+)", cfg, re.MULTILINE)
                self.assertIsNotNone(m)
                self.assertEqual(float(m.group(1)), float(expected))

    def test_slurm_references_config_jobname_and_gpus(self):
        slurm = _read(SLURM)
        self.assertIn(f"--config {CONFIG}", slurm)
        self.assertIn("#SBATCH --job-name=caarma-pexact13", slurm)
        self.assertIn("#SBATCH --gpus=v100-32:4", slurm)


if __name__ == "__main__":
    unittest.main()
