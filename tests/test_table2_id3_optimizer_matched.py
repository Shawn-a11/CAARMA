import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_helper(name):
    source = (ROOT / 'train_at_only.py').read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<helper>', 'exec'), namespace)
    return namespace[name]


class Table2Id3OptimizerMatchedTest(unittest.TestCase):
    def test_method_and_schedule_are_isolated(self):
        config = yaml.safe_load(
            (ROOT / 'config_psc_vox1_table2_at_only.yaml').read_text()
        )
        self.assertTrue(config['adversarial_training'])
        self.assertFalse(config['synthetic_loss'])
        self.assertEqual(config['discriminator_type'], 'spectral')
        self.assertEqual(config['init_lr'], 0.002)
        self.assertEqual(config['discriminator_lr'], 0.0002)
        self.assertEqual(config['warmup_step'], 2000)
        self.assertEqual(config['lr_decay_after_epoch'], 16)
        self.assertEqual(config['lr_decay_gamma'], 0.5)
        self.assertFalse(config['sync_batchnorm'])

    def test_pre_update_schedule(self):
        helper = _load_helper('optimizer_learning_rate')
        args = dict(base_lr=0.002, warmup_steps=2000, decay_after_epoch=16, decay_gamma=0.5)
        self.assertEqual(helper(update_step=0, current_epoch=0, **args), 0.000001)
        self.assertEqual(helper(update_step=1999, current_epoch=15, **args), 0.002)
        self.assertEqual(helper(update_step=3000, current_epoch=16, **args), 0.001)
        self.assertEqual(helper(update_step=9999, current_epoch=29, **args), 0.001)

    def test_invalid_scheduler_and_syncbn_are_removed(self):
        source = (ROOT / 'train_at_only.py').read_text()
        self.assertNotIn('StepLR', source)
        self.assertNotIn('def on_train_epoch_end', source)
        self.assertNotIn('self.trainer.global_step <', source)
        self.assertIn('main_update_count', source)
        self.assertIn(
            "sync_batchnorm=bool(config.get('sync_batchnorm', False))",
            source,
        )
        script = (
            ROOT / 'scripts/psc/train_vox1_table2_at_only.slurm'
        ).read_text()
        self.assertIn('#SBATCH --exclude=v003,v007,v008,v010', script)
        self.assertIn('verify_nccl_allreduce.py', script)
        self.assertIn('Refusing to overwrite', script)


if __name__ == '__main__':
    unittest.main()
