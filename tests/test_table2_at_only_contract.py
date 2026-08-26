from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Table2AtOnlyContractTest(unittest.TestCase):
    def test_method_is_at_only(self):
        config = (
            ROOT / 'config_psc_vox1_table2_at_only.yaml'
        ).read_text()
        source = (ROOT / 'train_at_only.py').read_text()

        self.assertIn('criterion: "AMSoftmaxGAN"', config)
        self.assertIn('mixup: true', config)
        self.assertIn('adversarial_training: true', config)
        self.assertIn('synthetic_loss: false', config)
        self.assertIn('discriminator_type: "spectral"', config)
        self.assertIn('Discriminator_spectral', source)
        self.assertNotIn('MixupDiscriminator', source)
        self.assertNotIn('flagSyn=True', source)
        self.assertIn('am_loss + self.lambda_adv * generator_loss', source)

    def test_psc_job_is_isolated(self):
        script = (
            ROOT / 'scripts/psc/train_vox1_table2_at_only.slurm'
        ).read_text()
        self.assertIn('#SBATCH --gpus=v100-32:4', script)
        self.assertIn('#SBATCH --exclude=v010', script)
        self.assertIn('paper_table2_id3_at_only', script)
        self.assertIn('train_at_only.py', script)


if __name__ == '__main__':
    unittest.main()
