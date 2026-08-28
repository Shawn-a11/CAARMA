#!/usr/bin/env python3
"""Reject MFA tuning jobs that drift from the Job 44493754 protocol."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL_TRAIN_SHA256 = (
    "18f296baf0cae333c6a7b1ddd10186bf1a00ead1b36a6c94d12a4997d9eff804"
)
NCCL_PREFLIGHT_SHA256 = (
    "a57910753a909a146ec93d90326650322024afd2ea2872ebd6fc85d2f0811c9c"
)


def _scalar(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_top_level_yaml(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_line in path.read_text().splitlines():
        if not raw_line or raw_line[0].isspace() or raw_line.lstrip().startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_]+):\s*(.*?)\s*$", raw_line)
        if match and match.group(2):
            result[match.group(1)] = _scalar(match.group(2))
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"MFA isolation check failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--axis",
        choices=("control", "init_lr", "am_margin", "lr2e3_weight_decay"),
        required=True,
    )
    parser.add_argument("--value", type=float)
    args = parser.parse_args()
    if args.axis != "control" and args.value is None:
        parser.error("--value is required for a tuning arm")

    train_path = ROOT / "train_mfa_baseline.py"
    preflight_path = ROOT / "tools/verify_nccl_allreduce.py"
    config_path = ROOT / "config_psc_vox1_mfa_baseline.yaml"
    slurm_path = ROOT / "scripts/psc/train_vox1_mfa_baseline.slurm"
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    _require(
        train_hash == CONTROL_TRAIN_SHA256,
        f"train entrypoint changed ({train_hash}); expected {CONTROL_TRAIN_SHA256}",
    )
    preflight_hash = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    _require(
        preflight_hash == NCCL_PREFLIGHT_SHA256,
        f"NCCL preflight changed ({preflight_hash}); expected {NCCL_PREFLIGHT_SHA256}",
    )

    config = _read_top_level_yaml(config_path)
    invariants = {
        "model": "MFA-CONFORMER",
        "criterion": "AMSoftmax",
        "epochs": 30,
        "warmup_step": 2000,
        "batch_size": 50,
        "num_workers": 4,
        "second": 3,
        "do_augmentation": False,
        "mixup": False,
        "embedding_dim": 192,
        "am_scale": 30,
        "devices": 4,
        "num_nodes": 1,
        "precision": "16-mixed",
        "sync_batchnorm": False,
        "seed": 42,
    }
    for key, expected in invariants.items():
        _require(config.get(key) == expected, f"{key}={config.get(key)!r}, expected {expected!r}")
    _require("lr_scheduler_step_size" not in config, "StepLR step_size is prohibited")
    _require("lr_scheduler_gamma" not in config, "StepLR gamma is prohibited")

    expected_lr = (
        args.value
        if args.axis == "init_lr"
        else 0.002
        if args.axis == "lr2e3_weight_decay"
        else 0.001
    )
    expected_margin = args.value if args.axis == "am_margin" else 0.2
    expected_weight_decay = (
        args.value if args.axis == "lr2e3_weight_decay" else 1e-7
    )
    _require(config.get("init_lr") == expected_lr, "unexpected init_lr change")
    _require(config.get("am_margin") == expected_margin, "unexpected am_margin change")
    _require(
        config.get("weight_decay") == expected_weight_decay,
        "unexpected weight_decay change",
    )
    _require(config.get("tuning_axis", "control") == args.axis, "tuning_axis mismatch")
    if args.axis != "control":
        _require(config.get("tuning_value") == args.value, "tuning_value mismatch")
        _require(config.get("fixed_control_job") == 44493754, "control JobID is not locked")

    slurm = slurm_path.read_text()
    for required in (
        "#SBATCH --partition=GPU-shared",
        "#SBATCH --gpus=v100-32:4",
        "#SBATCH --exclude=v003,v007,v008,v010",
        "#SBATCH --ntasks-per-node=4",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
        "TORCH_NCCL_BLOCKING_WAIT=1",
        "tools/verify_nccl_allreduce.py",
        "--numel 19991936",
        "--timeout-seconds 180",
        "train_mfa_baseline.py",
    ):
        _require(required in slurm, f"launcher missing {required}")

    print(
        "MFA control isolation OK:",
        f"axis={args.axis}",
        f"value={args.value}",
        f"train_sha256={train_hash}",
    )


if __name__ == "__main__":
    main()
