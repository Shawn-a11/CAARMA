import ast
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class PscConcatCrpPortTest(unittest.TestCase):
    def test_method_config(self):
        config = yaml.safe_load((ROOT / "config_psc_concat_crp.yaml").read_text())
        self.assertEqual(config["discriminator_type"], "concat")
        self.assertTrue(config["persistence"])
        self.assertEqual(config["pair_strategy"], "crp")
        self.assertEqual(config["slerp_t"], 0.5)
        self.assertEqual(config["synth_bank_size"], 10)
        self.assertEqual(config["synth_max_factor"], 4)
        self.assertEqual(config["crp_alpha"], 1.0)
        self.assertEqual(config["crp_topk"], 4)

    def test_training_methods_remain_present(self):
        tree = ast.parse((ROOT / "train.py").read_text())
        task = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Task")
        names = {node.name for node in task.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"training_step", "configure_optimizers", "on_validation_epoch_end"} <= names)

    def test_slurm_entrypoint(self):
        script = (ROOT / "scripts/psc/train_concat_crp.slurm").read_text()
        self.assertIn("config_psc_concat_crp.yaml", script)
        self.assertIn("#SBATCH --gpus=v100-32:4", script)

if __name__ == "__main__":
    unittest.main()
