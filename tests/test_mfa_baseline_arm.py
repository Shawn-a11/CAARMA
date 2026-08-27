import ast
import unittest
from pathlib import Path

import torch
import yaml

from criterion.amsoftmax import amsoftmax


ROOT = Path(__file__).resolve().parents[1]


class MfaBaselineArmTest(unittest.TestCase):
    def test_config_matches_paper_table2_id1_control(self):
        config = yaml.safe_load(
            (ROOT / 'config_psc_vox1_mfa_baseline.yaml').read_text()
        )
        self.assertEqual(config['model'], 'MFA-CONFORMER')
        self.assertEqual(config['criterion'], 'AMSoftmax')
        self.assertFalse(config['mixup'])
        self.assertFalse(config['do_augmentation'])
        self.assertEqual(config['am_margin'], 0.2)
        self.assertEqual(config['am_scale'], 30)
        self.assertEqual(config['batch_size'], 50)
        self.assertEqual(config['warmup_step'], 2000)
        self.assertFalse(config['sync_batchnorm'])
        self.assertNotIn('lr_scheduler_step_size', config)
        self.assertNotIn('lr_scheduler_gamma', config)

    def test_optimizer_matches_job_44493754_control(self):
        source = (ROOT / 'train_mfa_baseline.py').read_text()
        self.assertIn('from torch.optim.lr_scheduler import LambdaLR', source)
        self.assertIn("'interval': 'step'", source)
        self.assertIn("'frequency': 1", source)
        self.assertNotIn('StepLR', source)
        self.assertNotIn('automatic_optimization = False', source)
        self.assertNotIn('manual_backward', source)
        self.assertNotIn('def on_train_epoch_end', source)
        self.assertIn(
            "sync_batchnorm=bool(config.get('sync_batchnorm', False))",
            source,
        )

    def test_entrypoint_has_no_synthetic_or_adversarial_training(self):
        source = (ROOT / 'train_mfa_baseline.py').read_text()
        tree = ast.parse(source)
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn('MixupDiscriminator', imported_names)
        self.assertNotIn('mixup_data_euc_avg', imported_names)
        self.assertNotIn('loss_syn', source)
        self.assertNotIn('BCEWithLogitsLoss', source)
        self.assertNotIn('AMSoftmaxGAN', source)

    def test_plain_amsoftmax_rejects_synthetic_mode(self):
        loss = amsoftmax(embedding_dim=4, num_classes=3)
        embedding = torch.randn(2, 4)
        labels = torch.tensor([0, 2])
        value, accuracy, synthetic = loss(embedding, labels)
        self.assertEqual(value.ndim, 0)
        self.assertEqual(accuracy.numel(), 1)
        self.assertIsNone(synthetic)
        with self.assertRaises(ValueError):
            loss(embedding, labels, flagSyn=True)

    def test_slurm_is_isolated_and_excludes_bad_node(self):
        script = (
            ROOT / 'scripts/psc/train_vox1_mfa_baseline.slurm'
        ).read_text()
        self.assertIn('#SBATCH --gpus=v100-32:4', script)
        self.assertIn('#SBATCH --exclude=v008,v010', script)
        self.assertIn('#SBATCH --time=04:00:00', script)
        self.assertNotIn('HubertModel', script)
        self.assertIn('train_mfa_baseline.py', script)


if __name__ == '__main__':
    unittest.main()
