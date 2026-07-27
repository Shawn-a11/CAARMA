import importlib.util
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parents[1]
RUNNER_PATH = PROJECT_ROOT / "autoresearch" / "run_trial.py"
SPLIT_BUILDER_PATH = PROJECT_ROOT / "autoresearch" / "build_dev_split.py"
SPEC = importlib.util.spec_from_file_location("run_trial", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class AutoresearchToolsTest(unittest.TestCase):
    @staticmethod
    def write_runner_inputs(root):
        trials = root / "dev_trials.txt"
        trials.write_text("1 enroll.wav test.wav\n", encoding="utf-8")
        train_csv = root / "train.csv"
        train_csv.write_text(
            "utt_paths,utt_spk_int_labels\n/train.wav,0\n",
            encoding="utf-8",
        )
        eval_root = root / "dev_wav"
        eval_root.mkdir()
        return trials, train_csv, eval_root

    def test_experiment_family_and_default_output_are_isolated(self):
        config = runner.load_yaml(PROJECT_ROOT / "config.yaml")
        space = runner.load_yaml(PROJECT_ROOT / "autoresearch" / "search_space.yaml")
        runner.validate_frozen_config(config, space)
        self.assertEqual(config["experiment_family"], "joint_lsyn_mlpd_v1")
        self.assertEqual(
            runner.DEFAULT_OUTPUT_ROOT.name,
            "autoresearch_runs_joint_lsyn_mlpd",
        )

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

    def test_final_test_eval_root_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "renamed_trials.txt"
            trials.write_text("1 enroll.wav test.wav\n", encoding="utf-8")
            train_csv = root / "train.csv"
            train_csv.write_text(
                "utt_paths,utt_spk_int_labels\n/train.wav,0\n",
                encoding="utf-8",
            )
            eval_root = root / "test" / "wav"
            eval_root.mkdir(parents=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--tag",
                    "blocked-final-test",
                    "--budget",
                    "8",
                    "--seed",
                    "42",
                    "--trial-path",
                    str(trials),
                    "--train-csv",
                    str(train_csv),
                    "--eval-root",
                    str(eval_root),
                    "--output-root",
                    str(root / "runs"),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("final test asset", result.stderr)

    def test_dry_run_generates_isolated_config_and_blocks_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials, train_csv, eval_root = self.write_runner_inputs(root)
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
                "--train-csv",
                str(train_csv),
                "--eval-root",
                str(eval_root),
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
            self.assertEqual(generated["dataset"], str(train_csv.resolve()))
            self.assertEqual(generated["root"], str(eval_root.resolve()) + "/")

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
            trials, train_csv, eval_root = self.write_runner_inputs(root)
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
                    "--train-csv",
                    str(train_csv),
                    "--eval-root",
                    str(eval_root),
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

    def test_dev_split_is_deterministic_and_has_no_utterance_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_root = root / "dev" / "wav"
            rows = []
            for speaker in range(3):
                for utterance in range(5):
                    audio = (
                        eval_root
                        / f"id{speaker:05d}"
                        / f"recording{utterance:02d}"
                        / "00001.wav"
                    )
                    audio.parent.mkdir(parents=True, exist_ok=True)
                    audio.touch()
                    rows.append(
                        {
                            "utt_paths": str(audio),
                            "utt_spk_int_labels": str(speaker),
                        }
                    )

            input_csv = root / "full.csv"
            with input_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["utt_paths", "utt_spk_int_labels"],
                )
                writer.writeheader()
                writer.writerows(rows)

            outputs = []
            for name in ("split_a", "split_b"):
                output = root / name
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SPLIT_BUILDER_PATH),
                        "--input-csv",
                        str(input_csv),
                        "--eval-root",
                        str(eval_root),
                        "--output-dir",
                        str(output),
                        "--seed",
                        "42",
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(output)

            self.assertEqual(
                (outputs[0] / "dev_trials.txt").read_bytes(),
                (outputs[1] / "dev_trials.txt").read_bytes(),
            )
            manifest = json.loads(
                (outputs[0] / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["speakers"], 3)
            self.assertEqual(manifest["input_utterances"], 15)
            self.assertEqual(manifest["training_utterances"], 6)
            self.assertEqual(manifest["heldout_utterances"], 9)
            self.assertEqual(manifest["positive_trials"], 9)
            self.assertEqual(manifest["negative_trials"], 9)

            with (outputs[0] / "train.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                train_paths = {
                    row["utt_paths"] for row in csv.DictReader(handle)
                }
            with (outputs[0] / "heldout.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                heldout_paths = {
                    row["utt_paths"] for row in csv.DictReader(handle)
                }
            self.assertFalse(train_paths.intersection(heldout_paths))


if __name__ == "__main__":
    unittest.main()
