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
        self.assertIn("(1.0 / self.config['num_spk']) * syn_loss", source)
        self.assertNotIn('Discriminator', source)
        self.assertNotIn('BCEWithLogitsLoss', source)

    def test_psc_job_is_isolated(self):
        script = (
            ROOT / 'scripts/psc/train_vox1_table2_lsyn_only.slurm'
        ).read_text()
        self.assertIn('#SBATCH --gpus=v100-32:4', script)
        self.assertIn('#SBATCH --exclude=v010', script)
        self.assertIn('paper_table2_id2_lsyn_only', script)
        self.assertIn('train_lsyn_only.py', script)


if __name__ == '__main__':
    unittest.main()
