import ast
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class PscPaperExactPortTest(unittest.TestCase):
    def test_method_config(self):
        config = yaml.safe_load((ROOT / "config_psc_vox1_paper_exact.yaml").read_text())
        self.assertEqual(config["am_margin"], 0.2)
        self.assertEqual(config["am_scale"], 30)
        self.assertEqual(config["init_lr"], 0.001)
        self.assertEqual(config["discriminator_lr"], 0.0002)
        self.assertEqual(config["weight_decay"], 0.0000001)
        self.assertEqual(config["warmup_step"], 2000)
        self.assertEqual(config["epochs"], 30)
        self.assertEqual(config["batch_size"], 50)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["devices"], 4)
        self.assertFalse(config["freeze_hubert"])
        self.assertFalse(config["do_augmentation"])

    def test_no_lr_decay_in_training_code(self):
        # Paper specifies warmup only; the source repo's lr decay must be gone.
        source = (ROOT / "train_paper_aligned.py").read_text()
        self.assertNotIn("lr_scheduler", source)
        tree = ast.parse(source)
        task = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Task"
        )
        names = {node.name for node in task.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue(
            {"training_step", "configure_optimizers", "on_validation_epoch_end"} <= names
        )
        self.assertNotIn("on_train_epoch_end", names)

    def test_slurm_entrypoint(self):
        script = (ROOT / "scripts/psc/train_vox1_paper_exact.slurm").read_text()
        self.assertIn("config_psc_vox1_paper_exact.yaml", script)
        self.assertIn("#SBATCH --gpus=v100-32:4", script)
        self.assertIn("train_paper_aligned.py", script)
        self.assertNotIn("BASH_SOURCE", script)


if __name__ == "__main__":
    unittest.main()
