import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReDimNet2ResumeWiringTest(unittest.TestCase):
    def test_training_entrypoint_restores_full_lightning_state(self):
        source = (ROOT / "train_paper_aligned.py").read_text()
        self.assertIn('"--resume-checkpoint"', source)
        self.assertIn("ckpt_path=resume_checkpoint", source)

    def test_slurm_job_uses_existing_checkpoint_and_isolated_output(self):
        source = (
            ROOT / "scripts/psc/train_redimnet2_b6_caarma.slurm"
        ).read_text()
        self.assertIn('test -f "$CAARMA_RESUME_CHECKPOINT"', source)
        self.assertIn('--resume-checkpoint "$CAARMA_RESUME_CHECKPOINT"', source)
        self.assertIn("redimnet2_b6_caarma_resume_epoch5_to30", source)


if __name__ == "__main__":
    unittest.main()
