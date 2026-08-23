import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPO_ROOT / "config_psc_projection_crp.yaml"
SLURM_PATH = REPO_ROOT / "scripts" / "psc" / "train_projection_crp.slurm"


class ProjectionCrpBs13PortTest(unittest.TestCase):
    def test_config_batch_size_is_13(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["batch_size"], 13)

    def test_config_method_hyperparameters_match_source(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["model"], "MFA-CONFORMER")
        self.assertEqual(cfg["features"], "Fbank")
        self.assertEqual(cfg["criterion"], "AMSoftmaxGAN")
        self.assertEqual(cfg["init_lr"], 0.001)
        self.assertEqual(cfg["epochs"], 30)
        self.assertEqual(cfg["num_spk"], 1211)
        self.assertEqual(cfg["embedding_dim"], 192)
        self.assertTrue(cfg["persistence"])
        self.assertEqual(cfg["pair_strategy"], "crp")
        self.assertEqual(cfg["crp_topk"], 4)
        self.assertEqual(cfg["discriminator_type"], "projection")
        self.assertEqual(cfg["slerp_t"], 0.5)

    def test_slurm_references_expected_config_gpus_and_jobname(self):
        text = SLURM_PATH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=caarma-proj13", text)
        self.assertIn("--config config_psc_projection_crp.yaml", text)
        self.assertIn("#SBATCH --gpus=v100-32:4", text)


if __name__ == "__main__":
    unittest.main()
