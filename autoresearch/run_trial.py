#!/usr/bin/env python3
"""Run one restricted joint-Lsyn MLP-D baseline-tuning trial.

This script never edits tracked source files. It validates requested overrides
against search_space.yaml, creates a unique run directory, launches the existing
DDP training entry point, parses validation metrics, and appends an untracked
TSV record.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_SPACE = Path(__file__).with_name("search_space.yaml")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "autoresearch_runs_joint_lsyn_mlpd"
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EER_PATTERN = re.compile(r"cosine EER:\s*([0-9]+(?:\.[0-9]+)?)%")
DCF2_PATTERN = re.compile(r"cosine minDCF\(10-2\):\s*([0-9]+(?:\.[0-9]+)?)")
DCF3_PATTERN = re.compile(r"cosine minDCF\(10-3\):\s*([0-9]+(?:\.[0-9]+)?)")
EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RESULT_COLUMNS = [
    "timestamp_utc",
    "tag",
    "status",
    "best_eer",
    "min_dcf_1e2",
    "min_dcf_1e3",
    "best_epoch_zero_based",
    "runtime_seconds",
    "seed",
    "budget_epochs",
    "commit",
    "fingerprint",
    "description",
    "overrides",
    "run_dir",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must use key=value syntax: {text!r}")
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key is empty: {text!r}")
    return key, yaml.safe_load(raw_value)


def flatten_search_space(space: dict[str, Any]) -> dict[str, list[Any]]:
    allowed: dict[str, list[Any]] = {}
    stages = space.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("search_space.yaml: stages must be a mapping")
    for stage, parameters in stages.items():
        if not isinstance(parameters, dict):
            raise ValueError(f"search stage {stage!r} must be a mapping")
        for key, values in parameters.items():
            if key in allowed:
                raise ValueError(f"Duplicate search-space key: {key}")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Search-space values for {key} must be a non-empty list")
            allowed[key] = values
    return allowed


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def validate_overrides(
    overrides: dict[str, Any],
    allowed: dict[str, list[Any]],
) -> None:
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(
                f"{key!r} is not tunable. Allowed keys: {', '.join(sorted(allowed))}"
            )
        if not any(values_equal(value, candidate) for candidate in allowed[key]):
            raise ValueError(
                f"{key}={value!r} is outside the approved values {allowed[key]!r}"
            )


def validate_frozen_config(config: dict[str, Any], space: dict[str, Any]) -> None:
    frozen = space.get("frozen_config", {})
    if not isinstance(frozen, dict):
        raise ValueError("search_space.yaml: frozen_config must be a mapping")
    mismatches = []
    for key, expected in frozen.items():
        actual = config.get(key)
        if not values_equal(actual, expected):
            mismatches.append(f"{key}: expected {expected!r}, found {actual!r}")
    if mismatches:
        raise ValueError(
            "Base config no longer matches the frozen reproduction method:\n  "
            + "\n  ".join(mismatches)
        )


def parse_validation_metrics(log_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_epoch: int | None = None
    current: dict[str, Any] | None = None
    for raw_line in log_text.splitlines():
        line = ANSI_PATTERN.sub("", raw_line)
        epoch_matches = EPOCH_PATTERN.findall(line)
        if epoch_matches:
            current_epoch = int(epoch_matches[-1])

        eer_match = EER_PATTERN.search(line)
        if eer_match:
            if current is not None:
                records.append(current)
            current = {
                "eer": float(eer_match.group(1)),
                "min_dcf_1e2": None,
                "min_dcf_1e3": None,
                "epoch_zero_based": current_epoch,
            }
            continue

        if current is None:
            continue
        dcf2_match = DCF2_PATTERN.search(line)
        if dcf2_match:
            current["min_dcf_1e2"] = float(dcf2_match.group(1))
        dcf3_match = DCF3_PATTERN.search(line)
        if dcf3_match:
            current["min_dcf_1e3"] = float(dcf3_match.group(1))

    if current is not None:
        records.append(current)
    return records


def best_validation_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return min(
        records,
        key=lambda row: (
            row["eer"],
            row["min_dcf_1e2"] if row["min_dcf_1e2"] is not None else math.inf,
        ),
    )


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def fingerprint_config(config: dict[str, Any], commit: str) -> str:
    ignored = {"save_dir", "checkpoint_path"}
    payload = {
        "commit": commit,
        "config": {key: value for key, value in config.items() if key not in ignored},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def find_duplicate(output_root: Path, fingerprint: str) -> Path | None:
    if not output_root.exists():
        return None
    for metadata_path in output_root.glob("*/trial.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("fingerprint") == fingerprint:
            return metadata_path.parent
    return None


def safe_tsv_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ")


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        empty = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        if empty:
            writer.writeheader()
        writer.writerow({key: safe_tsv_value(row.get(key)) for key in RESULT_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_training(
    config_path: Path,
    log_path: Path,
    timeout_minutes: float,
) -> tuple[str, float]:
    command = [sys.executable, "train.py", "--config", str(config_path)]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_minutes * 60)
            status = "complete" if return_code == 0 else "crash"
        except subprocess.TimeoutExpired:
            terminate_process_group(process)
            status = "timeout"
    return status, time.monotonic() - started


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one whitelist-restricted CAARMA autoresearch trial."
    )
    parser.add_argument("--tag", required=True, help="Unique trial identifier.")
    parser.add_argument("--budget", required=True, type=int, help="Total max epochs.")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--trial-path", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--description", default="")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--search-space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("CAARMA_AUTORESEARCH_ROOT", DEFAULT_OUTPUT_ROOT)),
    )
    parser.add_argument("--timeout-minutes", type=float, default=360.0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--import-log",
        type=Path,
        help="Register an existing run instead of launching training.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help="Required when trial-path appears to be a final test list.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not TAG_PATTERN.fullmatch(args.tag):
        raise SystemExit("tag may contain only letters, numbers, dot, underscore, and dash")
    if args.budget <= 0:
        raise SystemExit("budget must be positive")
    if args.timeout_minutes <= 0:
        raise SystemExit("timeout-minutes must be positive")
    if not args.trial_path.expanduser().is_file():
        raise SystemExit(f"trial-path not found: {args.trial_path}")
    if args.resume_from and not args.resume_from.expanduser().is_file():
        raise SystemExit(f"resume checkpoint not found: {args.resume_from}")
    if args.import_log and not args.import_log.expanduser().is_file():
        raise SystemExit(f"existing log not found: {args.import_log}")
    if "test" in str(args.trial_path).lower() and not args.allow_final_test:
        raise SystemExit(
            "trial-path looks like a final test list. Use a development trial list, "
            "or pass --allow-final-test explicitly for a controlled reproduction run."
        )

    base_config = load_yaml(args.base_config)
    space = load_yaml(args.search_space)
    validate_frozen_config(base_config, space)
    allowed = flatten_search_space(space)

    overrides: dict[str, Any] = {}
    for text in args.overrides:
        key, value = parse_override(text)
        if key in overrides:
            raise SystemExit(f"Duplicate override: {key}")
        overrides[key] = value
    validate_overrides(overrides, allowed)

    commit = git_commit()
    run_dir = args.output_root.expanduser().resolve() / args.tag
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")

    config = dict(base_config)
    config.update(overrides)
    config["epochs"] = int(args.budget)
    config["seed"] = int(args.seed)
    config["trial_path"] = str(args.trial_path.expanduser().resolve())
    config["mode"] = "fit"
    config["checkpoint_path"] = "None"
    config["resume_from_checkpoint"] = (
        str(args.resume_from.expanduser().resolve()) if args.resume_from else "None"
    )
    config["save_dir"] = str(run_dir / "checkpoints")

    fingerprint = fingerprint_config(config, commit)
    duplicate = find_duplicate(args.output_root.expanduser().resolve(), fingerprint)
    if duplicate is not None:
        raise SystemExit(
            f"Equivalent trial already registered at {duplicate}; refusing to rerun it."
        )

    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    metadata = {
        "tag": args.tag,
        "fingerprint": fingerprint,
        "commit": commit,
        "description": args.description,
        "overrides": overrides,
        "budget_epochs": args.budget,
        "seed": args.seed,
        "trial_path": str(args.trial_path),
        "resume_from": str(args.resume_from) if args.resume_from else None,
    }
    metadata_path = run_dir / "trial.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    log_path = run_dir / "train.log"
    if args.dry_run:
        print(f"Dry run prepared: {run_dir}")
        print(f"Command: {sys.executable} train.py --config {config_path}")
        return 0

    if args.import_log:
        shutil.copy2(args.import_log, log_path)
        status = "imported"
        runtime_seconds = 0.0
    else:
        print(f"Starting trial {args.tag}; log: {log_path}", flush=True)
        status, runtime_seconds = run_training(
            config_path=config_path,
            log_path=log_path,
            timeout_minutes=args.timeout_minutes,
        )

    records = parse_validation_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
    best = best_validation_record(records)
    if best is None and status in {"complete", "imported"}:
        status = "no_metric"

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "status": status,
        "best_eer": best["eer"] if best else None,
        "min_dcf_1e2": best["min_dcf_1e2"] if best else None,
        "min_dcf_1e3": best["min_dcf_1e3"] if best else None,
        "best_epoch_zero_based": best["epoch_zero_based"] if best else None,
        "runtime_seconds": round(runtime_seconds, 1),
        "seed": args.seed,
        "budget_epochs": args.budget,
        "commit": commit,
        "fingerprint": fingerprint,
        "description": args.description,
        "overrides": json.dumps(overrides, sort_keys=True),
        "run_dir": str(run_dir),
    }
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_result(args.output_root.expanduser().resolve() / "results.tsv", result)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if best is not None and status in {"complete", "imported"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
