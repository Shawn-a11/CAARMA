import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parents[1]
RUNNER_PATH = PROJECT_ROOT / "autoresearch" / "run_trial.py"
SPEC = importlib.util.spec_from_file_location("run_trial", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class AutoresearchToolsTest(unittest.TestCase):
    def test_metric_parser_keeps_matching_dcf_values(self):
        log = """
Epoch 4: 100% complete
cosine EER: 4.12%
cosine minDCF(10-2): 0.4012
cosine minDCF(10-3): 0.5123
Epoch 5: 100% complete
cosine EER: 3.98%
cosine minDCF(10-2): 0.3722
cosine minDCF(10-3): 0.4888
"""
        records = runner.parse_validation_metrics(log)
        best = runner.best_validation_record(records)
        self.assertEqual(len(records), 2)
        self.assertEqual(best["eer"], 3.98)
        self.assertEqual(best["min_dcf_1e2"], 0.3722)
        self.assertEqual(best["epoch_zero_based"], 5)

    def test_override_whitelist(self):
        space = runner.load_yaml(PROJECT_ROOT / "autoresearch" / "search_space.yaml")
        allowed = runner.flatten_search_space(space)
        runner.validate_overrides({"init_lr": 0.001}, allowed)
        runner.validate_overrides({"discriminator_lr": 0.0002}, allowed)
        runner.validate_overrides({"joint_lsyn_scale": 2.0}, allowed)
        with self.assertRaises(ValueError):
            runner.validate_overrides({"batch_size": 100}, allowed)
        with self.assertRaises(ValueError):
            runner.validate_overrides({"discriminator_backbone_lr_factor": 0.01}, allowed)
        with self.assertRaises(ValueError):
            runner.validate_overrides({"init_lr": 0.003}, allowed)

    def test_dry_run_generates_isolated_config_and_blocks_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "dev_trials.txt"
            trials.write_text("1 enroll.wav test.wav\n", encoding="utf-8")
            output = root / "runs"
            command = [
                sys.executable,
                str(RUNNER_PATH),
                "--tag",
                "dry-one",
                "--budget",
                "8",
                "--seed",
                "42",
                "--trial-path",
                str(trials),
                "--output-root",
                str(output),
                "--set",
                "init_lr=0.0005",
                "--dry-run",
            ]
            first = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            generated = yaml.safe_load(
                (output / "dry-one" / "config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(generated["init_lr"], 0.0005)
            self.assertEqual(generated["epochs"], 8)
            self.assertEqual(generated["seed"], 42)

            command[command.index("dry-one")] = "dry-two"
            second = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to rerun", second.stderr)

    def test_existing_log_can_be_registered_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "dev_trials.txt"
            trials.write_text("1 enroll.wav test.wav\n", encoding="utf-8")
            existing_log = root / "existing.log"
            existing_log.write_text(
                "\n".join(
                    [
                        "Epoch 14: 100% complete",
                        "cosine EER: 3.51%",
                        "cosine minDCF(10-2): 0.3405",
                        "cosine minDCF(10-3): 0.4483",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "runs"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--tag",
                    "imported",
                    "--budget",
                    "30",
                    "--seed",
                    "42",
                    "--trial-path",
                    str(trials),
                    "--output-root",
                    str(output),
                    "--import-log",
                    str(existing_log),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result_text = (output / "results.tsv").read_text(encoding="utf-8")
            self.assertIn("3.51", result_text)
            self.assertIn("0.3405", result_text)
            self.assertIn("\timported\t", result_text)


if __name__ == "__main__":
    unittest.main()
