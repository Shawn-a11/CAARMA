import ast
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class ExpPscConcatCrpBs13PortTest(unittest.TestCase):
    def test_batch_size_is_13(self):
        config = yaml.safe_load((ROOT / "config_psc_concat_crp.yaml").read_text())
        self.assertEqual(config["batch_size"], 13)

    def test_source_arm_hyperparameters_unchanged(self):
        config = yaml.safe_load((ROOT / "config_psc_concat_crp.yaml").read_text())
        self.assertEqual(config["init_lr"], 0.001)
        self.assertEqual(config["epochs"], 30)
        self.assertEqual(config["weight_decay"], 0.0000001)
        self.assertEqual(config["warmup_step"], 2000)
        self.assertEqual(config["num_spk"], 1211)
        self.assertTrue(config["mixup"])
        self.assertEqual(config["embedding_dim"], 192)
        self.assertTrue(config["persistence"])
        self.assertEqual(config["slerp_t"], 0.5)
        self.assertEqual(config["synth_bank_size"], 10)
        self.assertEqual(config["synth_max_factor"], 4)
        self.assertEqual(config["pair_strategy"], "crp")
        self.assertEqual(config["crp_alpha"], 1.0)
        self.assertEqual(config["crp_topk"], 4)
        self.assertEqual(config["discriminator_type"], "concat")

    def test_run_dir_and_title_updated(self):
        config = yaml.safe_load((ROOT / "config_psc_concat_crp.yaml").read_text())
        self.assertIn("concat_crp_persistent_bs13", config["save_dir"])
        self.assertEqual(config["title"], "concat_crp_persistent_bs13_psc")

    def test_training_methods_remain_present(self):
        tree = ast.parse((ROOT / "train.py").read_text())
        task = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Task")
        names = {node.name for node in task.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"training_step", "configure_optimizers", "on_validation_epoch_end"} <= names)

    def test_slurm_entrypoint(self):
        script = (ROOT / "scripts/psc/train_concat_crp.slurm").read_text()
        self.assertIn("config_psc_concat_crp.yaml", script)
        self.assertIn("#SBATCH --gpus=v100-32:4", script)
        self.assertIn("#SBATCH --job-name=caarma-concat13", script)
        self.assertIn("concat_crp_persistent_bs13/checkpoints", script)


if __name__ == "__main__":
    unittest.main()
