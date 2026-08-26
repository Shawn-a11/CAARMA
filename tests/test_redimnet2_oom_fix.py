import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReDimNet2OomFixTest(unittest.TestCase):
    def test_effective_global_batch_is_preserved(self):
        config = (
            ROOT / "config_psc_vox1_redimnet2_caarma.yaml"
        ).read_text()

        def integer(key):
            match = re.search(rf"^{key}:\s*(\d+)\s*$", config, re.MULTILINE)
            self.assertIsNotNone(match, key)
            return int(match.group(1))

        devices = integer("devices")
        batch_size = integer("batch_size")
        accumulation = integer("manual_accumulate_grad_batches")
        self.assertEqual((devices, batch_size, accumulation), (4, 25, 2))
        self.assertEqual(devices * batch_size * accumulation, 200)

    def test_manual_optimization_accumulates_both_optimizers(self):
        tree = ast.parse((ROOT / "train_paper_aligned.py").read_text())
        task = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Task"
        )
        training_step = next(
            node for node in task.body
            if isinstance(node, ast.FunctionDef) and node.name == "training_step"
        )
        source = ast.unparse(training_step)
        self.assertIn("self.manual_accumulate_grad_batches", source)
        self.assertIn("zero_grad=zero_grad", source)
        self.assertIn("step_now=step_now", source)
        self.assertIn("loss_divisor=group_size", source)

    def test_launcher_uses_isolated_output_and_excludes_v010(self):
        launcher = (
            ROOT / "scripts/psc/train_redimnet2_b6_caarma.slurm"
        ).read_text()
        self.assertIn("#SBATCH --exclude=v010", launcher)
        self.assertIn("#SBATCH --gpus=v100-32:4", launcher)
        self.assertIn("slerpinit_mb25_acc2", launcher)


if __name__ == "__main__":
    unittest.main()
