import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PscJointLsynPortTest(unittest.TestCase):
    def test_method_config_is_preserved(self):
        config = yaml.safe_load(
            (ROOT / "config_psc_joint_lsyn_mlpd.yaml").read_text()
        )
        self.assertEqual(config["model"], "MFA-CONFORMER")
        self.assertEqual(config["criterion"], "AMSoftmaxGAN")
        self.assertEqual(config["batch_size"], 50)
        self.assertEqual(config["num_spk"], 1211)
        self.assertTrue(config["mixup"])

    def test_discriminator_is_plain_mlp(self):
        source = (ROOT / "model/discriminator_mix.py").read_text()
        tree = ast.parse(source)
        mixup = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MixupDiscriminator"
        )
        class_source = ast.get_source_segment(source, mixup)
        self.assertIn("nn.Linear(emb_dim, hidden_dim)", class_source)
        self.assertNotIn("HubertModel.from_pretrained", class_source)

    def test_slurm_uses_branch_entrypoint(self):
        script = (
            ROOT / "scripts/psc/train_joint_lsyn_mlpd.slurm"
        ).read_text()
        self.assertIn("#SBATCH --gpus=v100-32:4", script)
        self.assertIn("srun python -u train.py", script)
        self.assertNotIn("train_source_faithful.py", script)


if __name__ == "__main__":
    unittest.main()
