from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Table2LsynOnlyContractTest(unittest.TestCase):
    def test_method_is_lsyn_only(self):
        config = (
            ROOT / 'config_psc_vox1_table2_lsyn_only.yaml'
        ).read_text()
        source = (ROOT / 'train_lsyn_only.py').read_text()

        self.assertIn('criterion: "AMSoftmaxGAN"', config)
        self.assertIn('mixup: true', config)
        self.assertIn('adversarial_training: false', config)
        self.assertIn('flagSyn=True', source)
        self.assertIn("1.0 / float(self.config['num_spk'])", source)
        self.assertIn('scaled_syn_loss = self.syn_loss_scale * syn_loss', source)
        self.assertIn('from torch.optim.lr_scheduler import LambdaLR', source)
        self.assertIn("'interval': 'step'", source)
        self.assertIn('sync_batchnorm=bool(', source)
        self.assertIn('sync_batchnorm: false', config)
        self.assertNotIn('StepLR', source)
        self.assertNotIn('manual_backward', source)
        self.assertNotIn('automatic_optimization', source)
        self.assertNotIn('lr_scheduler_step_size', config)
        self.assertNotIn('lr_scheduler_gamma', config)
        self.assertNotIn('self.loss_syn = loss', source)
        self.assertNotIn('Discriminator', source)
        self.assertNotIn('BCEWithLogitsLoss', source)

    def test_psc_job_is_isolated(self):
        script = (
            ROOT / 'scripts/psc/train_vox1_table2_lsyn_only.slurm'
        ).read_text()
        self.assertIn('#SBATCH --gpus=v100-32:4', script)
        self.assertIn('#SBATCH --exclude=v010,v013', script)
        self.assertIn('paper_table2_id2_lsyn_only_matched_optimizer', script)
        self.assertIn('train_lsyn_only.py', script)


if __name__ == '__main__':
    unittest.main()
