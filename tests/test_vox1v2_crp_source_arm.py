import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Vox1V2CrpSourceArmTest(unittest.TestCase):
    def test_source_schedule_updates_crp_only_in_main_step(self):
        source = (ROOT / "train_source_faithful.py").read_text()
        for required in (
            "rebuild_persistent_pairing",
            "embedding_d, label, update_state=False",
            "embedding, label, update_state=True",
            "synchronize_synth_state",
            "Ns = synthetic_embeddings.size(0)",
            "preds[:Ns], preds[Ns:]",
        ):
            self.assertIn(required, source)
        self.assertNotIn("cache_selection=True", source)
        self.assertNotIn("reuse_selection=True", source)

    def test_config_changes_only_the_persistent_crp_method_surface(self):
        base = (ROOT / "config_psc_vox1v2.yaml").read_text()
        crp = (ROOT / "config_psc_vox1v2_crp.yaml").read_text()
        fixed_keys = (
            "model", "features", "dataset", "trial_path", "root",
            "hubert_model_name", "init_lr", "epochs", "weight_decay",
            "warmup_step", "batch_size", "num_workers", "second",
            "num_spk", "embedding_dim", "am_margin", "am_scale",
            "devices", "precision", "seed",
        )
        for key in fixed_keys:
            pattern = re.compile(rf"^{re.escape(key)}:\s*(.+)$", re.MULTILINE)
            self.assertEqual(pattern.search(base).group(1), pattern.search(crp).group(1))
        for expected in (
            'persistence: true',
            'interpolation: "lerp"',
            'pair_strategy: "crp"',
            'reuse_policy: "popularity"',
            'candidate_pool: "topk"',
            'synth_init: "xavier"',
        ):
            self.assertIn(expected, crp)

        criterion = (ROOT / "criterion/amsoftmax_mix_gan.py").read_text()
        self.assertIn('self.interpolation == "lerp"', criterion)
        self.assertIn("return interpolation(p0, p1, self.slerp_t)", criterion)

    def test_launcher_is_isolated_and_excludes_v010(self):
        launcher = (
            ROOT / "scripts/psc/train_vox1v2_crp.slurm"
        ).read_text()
        self.assertIn("#SBATCH --exclude=v010,v013", launcher)
        self.assertIn("#SBATCH --gpus=v100-32:4", launcher)
        self.assertIn("config_psc_vox1v2_crp.yaml", launcher)
        self.assertIn("caarma_vox1v2_crp_persistent_hubert_ddp_retry", launcher)
        self.assertIn('--save-dir "$RUN_DIR/checkpoints"', launcher)

    def test_8gpu_launcher_preserves_global_batch(self):
        source = (ROOT / "train_source_faithful.py").read_text()
        launcher = (
            ROOT / "scripts/psc/train_vox1v2_crp_8gpu.slurm"
        ).read_text()
        self.assertIn("dist.get_world_size()", source)
        self.assertIn("--batch-size", source)
        self.assertIn("#SBATCH --exclude=v010,v013", launcher)
        self.assertIn("#SBATCH --gpus=v100-32:8", launcher)
        self.assertIn("#SBATCH --ntasks-per-node=8", launcher)
        self.assertIn("--devices 8", launcher)
        self.assertIn("--batch-size 25", launcher)
        self.assertIn('--save-dir "$RUN_DIR/checkpoints"', launcher)


if __name__ == "__main__":
    unittest.main()
