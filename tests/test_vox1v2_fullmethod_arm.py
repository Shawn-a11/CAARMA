import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Vox1V2FullMethodArmTest(unittest.TestCase):
    def test_isolation_guard(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "verify_vox1v2_fullmethod_isolation.py"),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Vox1+2 full-method isolation OK", completed.stdout)

    def test_projection_source_is_shape_safe(self):
        source = (ROOT / "train_source_faithful.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("len(synth_cols_d) != synth_for_d.size(0)", source)
        self.assertIn(
            "len(synth_cols_g) != synthetic_embeddings.size(0)", source
        )
        self.assertIn("Ns = synthetic_embeddings.size(0)", source)


if __name__ == "__main__":
    unittest.main()
