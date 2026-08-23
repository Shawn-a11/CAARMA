import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPO_ROOT / "config_psc_crp_persistent.yaml"
SLURM_PATH = REPO_ROOT / "scripts" / "psc" / "train_crp_persistent.slurm"


class PSCPersistentBS13PortTest(unittest.TestCase):
    def test_config_batch_size_is_13(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(cfg["batch_size"], 13)

    def test_config_matches_source_arm_hyperparameters(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        # Architecture / training backbone (unchanged from source arm)
        self.assertEqual(cfg["model"], "MFA-CONFORMER")
        self.assertEqual(cfg["features"], "Fbank")
        self.assertEqual(cfg["criterion"], "AMSoftmaxGAN")
        self.assertEqual(cfg["init_lr"], 0.001)
        self.assertEqual(cfg["epochs"], 30)
        self.assertEqual(cfg["weight_decay"], 0.0000001)
        self.assertEqual(cfg["warmup_step"], 2000)
        self.assertEqual(cfg["num_workers"], 4)
        self.assertEqual(cfg["second"], 3)
        self.assertFalse(cfg["do_augmentation"])
        self.assertEqual(cfg["num_spk"], 1211)
        self.assertTrue(cfg["mixup"])
        self.assertEqual(cfg["embedding_dim"], 192)
        # No data augmentation in source arm
        self.assertFalse(cfg["augmentations"]["add_noise"])
        self.assertFalse(cfg["augmentations"]["add_reverb"])
        self.assertFalse(cfg["augmentations"]["drop_freq"])
        self.assertFalse(cfg["augmentations"]["drop_chunk"])
        # CRP persistent-synth method hyperparameters (unchanged)
        self.assertTrue(cfg["persistence"])
        self.assertEqual(cfg["slerp_t"], 0.5)
        self.assertEqual(cfg["synth_bank_size"], 10)
        self.assertEqual(cfg["synth_max_factor"], 4)
        self.assertEqual(cfg["pair_strategy"], "crp")
        self.assertEqual(cfg["crp_alpha"], 1.0)
        self.assertEqual(cfg["crp_topk"], 4)
        self.assertEqual(cfg["discriminator_type"], "spectral")

    def test_config_uses_bs13_run_dir_and_title(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIn("crp_persistent_synth_bs13/checkpoints", cfg["save_dir"])
        self.assertEqual(cfg["title"], "crp_persistent_synth_bs13_psc")

    def test_slurm_references_config_gpus_and_jobname(self):
        slurm = SLURM_PATH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=caarma-crp13", slurm)
        self.assertIn("#SBATCH --gpus=v100-32:4", slurm)
        self.assertIn("--config config_psc_crp_persistent.yaml", slurm)
        self.assertIn("$CAARMA_RUN_ROOT/crp_persistent_synth_bs13/checkpoints", slurm)


if __name__ == "__main__":
    unittest.main()
