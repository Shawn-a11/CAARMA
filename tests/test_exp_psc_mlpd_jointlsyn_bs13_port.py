import re
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config_psc_joint_lsyn_mlpd.yaml"
SLURM = ROOT / "scripts/psc/train_joint_lsyn_mlpd.slurm"
BUILD_CRITERION = ROOT / "criterion/build_criterion.py"


class ExpPscMlpdJointlsynBs13PortTest(unittest.TestCase):
    def test_config_batch_size_is_13(self):
        config = yaml.safe_load(CONFIG.read_text())
        self.assertEqual(config["batch_size"], 13)

    def test_config_preserves_source_method_hyperparameters(self):
        config = yaml.safe_load(CONFIG.read_text())
        self.assertEqual(config["model"], "MFA-CONFORMER")
        self.assertEqual(config["criterion"], "AMSoftmaxGAN")
        self.assertEqual(config["init_lr"], 0.001)
        self.assertEqual(config["epochs"], 30)
        self.assertEqual(config["num_spk"], 1211)
        self.assertTrue(config["mixup"])
        self.assertEqual(config["embedding_dim"], 192)

    def test_criterion_uses_am_margin_0_2_scale_30(self):
        source = BUILD_CRITERION.read_text()
        self.assertIn("m=0.2", source)
        self.assertIn("s=30", source)

    def test_config_paths_match_branch(self):
        config = yaml.safe_load(CONFIG.read_text())
        self.assertIn("mlpd_joint_lsyn_bs13/checkpoints", config["save_dir"])
        self.assertEqual(config["title"], "mlpd_joint_lsyn_bs13_psc")

    def test_slurm_uses_branch_entrypoint_and_resources(self):
        script = SLURM.read_text()
        self.assertIn("#SBATCH --job-name=caarma-mlpd13", script)
        self.assertIn("#SBATCH --gpus=v100-32:4", script)
        self.assertIn("config_psc_joint_lsyn_mlpd.yaml", script)
        self.assertIn("srun python -u train.py", script)
        self.assertIn('"$CAARMA_CONFIG"', script)


if __name__ == "__main__":
    unittest.main()
