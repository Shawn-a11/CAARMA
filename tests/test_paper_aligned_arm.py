import ast
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class PaperAlignedArmTest(unittest.TestCase):
    def test_config_matches_explicit_paper_values(self):
        config = yaml.safe_load((ROOT / "config_psc_vox1_paper_aligned.yaml").read_text())
        self.assertEqual(config["init_lr"], 0.001)
        self.assertEqual(config["discriminator_lr"], 0.0002)
        self.assertEqual(config["batch_size"], 50)
        self.assertEqual(config["epochs"], 30)
        self.assertFalse(config["freeze_hubert"])

    def test_every_batch_updates_both_optimizers(self):
        tree = ast.parse((ROOT / "train_paper_aligned.py").read_text())
        task = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Task")
        method = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == "training_step")
        calls = [node.func.attr for node in ast.walk(method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertIn("_d_step", calls)
        self.assertIn("_g_step", calls)

    def test_slurm_uses_paper_entrypoint(self):
        script = (ROOT / "scripts/psc/train_vox1_paper_aligned.slurm").read_text()
        self.assertIn("train_paper_aligned.py", script)
        self.assertIn("#SBATCH --gpus=v100-32:4", script)

if __name__ == "__main__":
    unittest.main()
