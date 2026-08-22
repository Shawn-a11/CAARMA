import unittest
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]


class TestPSCSourceFaithfulPort(unittest.TestCase):
    def test_method_config(self):
        config = yaml.safe_load(
            (REPO / "config_psc_vox1.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["am_margin"], 0.2)
        self.assertEqual(config["am_scale"], 30)
        self.assertEqual(config["epochs"], 30)
        self.assertEqual(config["batch_size"], 50)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["devices"], 4)

    def test_training_methods_remain_present(self):
        import ast

        tree = ast.parse(
            (REPO / "train_source_faithful.py").read_text(encoding="utf-8")
        )
        task = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Task"
        )
        methods = {
            node.name for node in task.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in ("training_step", "configure_optimizers", "on_validation_epoch_end"):
            self.assertIn(name, methods)

    def test_slurm_entrypoint(self):
        script = (REPO / "scripts/psc/train_vox1.slurm").read_text(encoding="utf-8")
        self.assertIn("config_psc_vox1.yaml", script)
        self.assertIn("#SBATCH --gpus=v100-32:4", script)
        # Regression guard for job 44074599: the launcher must never
        # self-locate via BASH_SOURCE (it resolves to the Slurm spool dir).
        self.assertNotIn("BASH_SOURCE", script)


if __name__ == "__main__":
    unittest.main()
