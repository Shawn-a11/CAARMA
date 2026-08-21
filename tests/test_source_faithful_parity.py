import ast
import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REFERENCE = "1c860a3:train.py"
CRITICAL_TASK_METHODS = (
    "adjust_weight",
    "_d_step",
    "_g_step",
    "training_step",
    "configure_optimizers",
    "on_train_epoch_end",
    "compute_eer",
    "compute_minDCF",
    "on_validation_epoch_end",
)


def task_methods(source: str):
    tree = ast.parse(source)
    task = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Task"
    )
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in task.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class SourceFaithfulParityTest(unittest.TestCase):
    def test_critical_training_and_validation_methods_match_completed_reference(self):
        reference = subprocess.run(
            ["git", "-C", str(REPO), "show", REFERENCE],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        psc = (REPO / "train_source_faithful.py").read_text(encoding="utf-8")
        reference_methods = task_methods(reference)
        psc_methods = task_methods(psc)
        for method in CRITICAL_TASK_METHODS:
            self.assertEqual(
                psc_methods[method],
                reference_methods[method],
                f"PSC runtime drifted from source-faithful method: {method}",
            )

    def test_psc_entry_contains_required_ddp_guards(self):
        source = (REPO / "train_source_faithful.py").read_text(encoding="utf-8")
        required = (
            "toggle_optimizer(opt_d)",
            "with torch.no_grad():",
            "toggle_optimizer(opt_main)",
            "torch.cat(",
            "_ddp_params_and_buffers_to_ignore",
            'precision="16-mixed"',
        )
        for token in required:
            self.assertIn(token, source)

    def test_production_slurm_uses_source_faithful_entry(self):
        script = (REPO / "scripts/psc/train_vox1.slurm").read_text(encoding="utf-8")
        self.assertIn("train_source_faithful.py", script)
        self.assertNotIn("python -u train.py", script)
        self.assertIn("CAARMA source-faithful import OK", script)
        self.assertIn("HubertModel.from_pretrained", script)


if __name__ == "__main__":
    unittest.main()
