import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_helper(name):
    source = (ROOT / "train_paper_aligned.py").read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<helper>", "exec"),
        namespace,
    )
    return namespace[name]


class Table2Id6OptimizerMatchedTest(unittest.TestCase):
    def test_full_method_and_schedule_are_locked(self):
        config = yaml.safe_load(
            (ROOT / "config_psc_vox1_paper_aligned.yaml").read_text()
        )
        self.assertEqual(config["criterion"], "AMSoftmaxGAN")
        self.assertTrue(config["mixup"])
        self.assertEqual(config["gan_schedule"], "paired")
        self.assertEqual(config["lambda_adv_mode"], "dynamic")
        self.assertEqual(config["init_lr"], 0.002)
        self.assertEqual(config["discriminator_lr"], 0.0002)
        self.assertEqual(config["warmup_step"], 2000)
        self.assertEqual(config["lr_decay_after_epoch"], 16)
        self.assertEqual(config["lr_decay_gamma"], 0.5)
        self.assertFalse(config["sync_batchnorm"])

    def test_main_and_discriminator_schedules(self):
        helper = _load_helper("optimizer_learning_rate")
        common = dict(decay_after_epoch=16, decay_gamma=0.5)
        self.assertEqual(
            helper(0.002, 0, 2000, 0, **common),
            0.000001,
        )
        self.assertEqual(
            helper(0.002, 3000, 2000, 16, **common),
            0.001,
        )
        self.assertEqual(
            helper(0.0002, 3000, 0, 16, **common),
            0.0001,
        )
        self.assertEqual(
            helper(0.002, 9999, 2000, 29, **common),
            0.001,
        )

    def test_hubert_layers_and_invalid_scheduler(self):
        source = (ROOT / "train_paper_aligned.py").read_text()
        discriminator = (ROOT / "model/discriminator_mix.py").read_text()
        self.assertNotIn("StepLR", source)
        self.assertNotIn("def on_train_epoch_end", source)
        self.assertIn("main_update_count", source)
        for layer in (7, 9, 11, 12):
            self.assertIn(f"({layer}, self.projection_{layer})", discriminator)

    def test_psc_launcher_is_guarded(self):
        script = (
            ROOT / "scripts/psc/train_vox1_paper_aligned.slurm"
        ).read_text()
        self.assertIn("#SBATCH --exclude=v003,v007,v008,v010", script)
        self.assertIn("#SBATCH --time=09:00:00", script)
        self.assertIn("verify_nccl_allreduce.py", script)
        self.assertIn("Refusing to overwrite", script)


if __name__ == "__main__":
    unittest.main()
