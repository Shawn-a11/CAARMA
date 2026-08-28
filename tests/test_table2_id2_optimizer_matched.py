import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_helper(name):
    source = (ROOT / 'train_lsyn_only.py').read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<helper>', 'exec'), namespace)
    return namespace[name]


class Table2Id2OptimizerMatchedTest(unittest.TestCase):
    def test_method_and_schedule_are_isolated(self):
        config = yaml.safe_load(
            (ROOT / 'config_psc_vox1_table2_lsyn_only.yaml').read_text()
        )
        self.assertEqual(config['criterion'], 'AMSoftmaxGAN')
        self.assertTrue(config['mixup'])
        self.assertFalse(config['adversarial_training'])
        self.assertEqual(config['init_lr'], 0.002)
        self.assertEqual(config['warmup_step'], 2000)
        self.assertEqual(config['lr_decay_after_epoch'], 16)
        self.assertEqual(config['lr_decay_gamma'], 0.5)
        self.assertFalse(config['sync_batchnorm'])

    def test_single_decay_is_not_repeated(self):
        helper = _load_helper('warmup_then_single_decay_multiplier')
        args = dict(step=3000, warmup_steps=2000, decay_after_epoch=16, decay_gamma=0.5)
        self.assertEqual(helper(current_epoch=15, **args), 1.0)
        self.assertEqual(helper(current_epoch=16, **args), 0.5)
        self.assertEqual(helper(current_epoch=29, **args), 0.5)

    def test_automatic_optimizer_and_launcher_guards(self):
        source = (ROOT / 'train_lsyn_only.py').read_text()
        self.assertIn('LambdaLR', source)
        self.assertNotIn('StepLR', source)
        self.assertNotIn('automatic_optimization = False', source)
        script = (
            ROOT / 'scripts/psc/train_vox1_table2_lsyn_only.slurm'
        ).read_text()
        self.assertIn('#SBATCH --exclude=v003,v007,v008,v010', script)
        self.assertIn('verify_nccl_allreduce.py', script)
        self.assertIn('Refusing to overwrite', script)


if __name__ == '__main__':
    unittest.main()
