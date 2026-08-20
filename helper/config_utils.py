"""Portable configuration helpers for the PSC reproduction branch."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


_UNRESOLVED_ENV = re.compile(r"\$\{[^}]+\}")


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        unresolved = _UNRESOLVED_ENV.findall(expanded)
        if unresolved:
            raise ValueError(
                f"Unresolved environment variable(s) in {value!r}: "
                + ", ".join(unresolved)
            )
        return expanded
    return value


def _speaker_count(manifest_path: str) -> int:
    frame = pd.read_csv(manifest_path, usecols=["utt_spk_int_labels"])
    if frame.empty:
        raise ValueError(f"Training manifest is empty: {manifest_path}")
    labels = sorted(int(value) for value in frame["utt_spk_int_labels"].unique())
    if labels != list(range(len(labels))):
        raise ValueError(
            "utt_spk_int_labels must be contiguous integers starting at zero; "
            f"found min={labels[0]}, max={labels[-1]}, count={len(labels)}"
        )
    return len(labels)


def load_experiment_config(path: str) -> dict[str, Any]:
    """Load YAML, expand environment variables, and validate core paths."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    config = _expand(raw)

    for key in ("dataset", "trial_path", "root", "save_dir", "num_spk"):
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    for key in ("dataset", "trial_path"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key} does not exist: {config[key]}")
    if not Path(config["root"]).is_dir():
        raise NotADirectoryError(f"Evaluation root does not exist: {config['root']}")

    manifest_count = _speaker_count(config["dataset"])
    configured_count = config["num_spk"]
    if isinstance(configured_count, str) and configured_count.lower() == "auto":
        config["num_spk"] = manifest_count
    elif int(configured_count) != manifest_count:
        raise ValueError(
            "num_spk does not match manifest labels: "
            f"config={configured_count}, manifest={manifest_count}"
        )
    else:
        config["num_spk"] = int(configured_count)

    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    return config
