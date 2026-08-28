#!/usr/bin/env python3
"""Reject Vox1+2 full-method runs that drift from control Job 44529360."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL_CONFIG = ROOT / "config_psc_vox1v2_crp.yaml"
METHOD_CONFIG = ROOT / "config_psc_vox1v2_fullmethod.yaml"
TRAINER = ROOT / "train_source_faithful.py"
DISCRIMINATOR = ROOT / "model" / "discriminator_mix.py"
LAUNCHER = ROOT / "scripts" / "psc" / "train_vox1v2_fullmethod.slurm"

ALLOWED_CONFIG_DIFFERENCES = {
    "save_dir",
    "title",
    "fixed_control_job",
    "interpolation",
    "synth_init",
    "discriminator_type",
}


def parse_scalar(value):
    value = value.strip()
    if not value:
        return {}
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if value[:1] in {'"', "'"}:
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_yaml(path):
    """Parse the scalar and one-level mapping subset used by CAARMA configs."""
    data = {}
    current_mapping = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()
        if ":" not in line:
            raise AssertionError(f"Unsupported YAML line in {path}: {raw_line}")
        key, value = line.split(":", 1)
        parsed = parse_scalar(value)
        if indentation == 0:
            data[key] = parsed
            current_mapping = key if isinstance(parsed, dict) else None
        elif indentation == 2 and current_mapping is not None:
            data[current_mapping][key] = parsed
        else:
            raise AssertionError(f"Unsupported YAML nesting in {path}: {raw_line}")
    return data


def require(text, needle, source):
    if needle not in text:
        raise AssertionError(f"Missing {needle!r} in {source}")


def main():
    control = load_yaml(CONTROL_CONFIG)
    method = load_yaml(METHOD_CONFIG)
    all_keys = set(control) | set(method)
    changed = {
        key
        for key in all_keys
        if control.get(key) != method.get(key)
    }
    unexpected = changed - ALLOWED_CONFIG_DIFFERENCES
    if unexpected:
        raise AssertionError(
            "Unexpected control drift in config fields: "
            + ", ".join(sorted(unexpected))
        )

    expected = {
        "fixed_control_job": 44529360,
        "interpolation": "slerp",
        "synth_init": "slerp_parents",
        "discriminator_type": "projection",
        "pair_strategy": "crp",
        "reuse_policy": "popularity",
        "candidate_pool": "topk",
        "crp_alpha": 1.0,
        "crp_topk": 4,
        "batch_size": 50,
        "devices": 4,
        "epochs": 30,
        "seed": 42,
    }
    for key, value in expected.items():
        if method.get(key) != value:
            raise AssertionError(
                f"Expected {key}={value!r}, found {method.get(key)!r}"
            )

    if control.get("interpolation") != "lerp":
        raise AssertionError("Control interpolation is no longer LERP")
    if control.get("synth_init") != "xavier":
        raise AssertionError("Control W_syn initialization is no longer Xavier")
    if "discriminator_type" in control:
        raise AssertionError(
            "Control unexpectedly declares a discriminator override"
        )

    trainer = TRAINER.read_text(encoding="utf-8")
    for needle in (
        "self.pretrain_eps = 15",
        "lambda_adv_value=0.0005",
        "StepLR(embedding_optimizer, step_size=4, gamma=0.5)",
        "sync_batchnorm=True",
        "ProjectionDiscriminator_spectral",
        "self._real_conditions(label, combined.device)",
        "self._synth_conditions(synth_cols_g, combined.device)",
    ):
        require(trainer, needle, TRAINER)

    discriminator = DISCRIMINATOR.read_text(encoding="utf-8")
    for needle in (
        "class ProjectionDiscriminator_spectral",
        "requires_condition = True",
        "return self.head(phi) + compatibility",
    ):
        require(discriminator, needle, DISCRIMINATOR)

    launcher = LAUNCHER.read_text(encoding="utf-8")
    for needle in (
        "#SBATCH --partition=GPU-shared",
        "#SBATCH --exclude=v008,v010,v013",
        "#SBATCH --gpus=v100-32:4",
        "#SBATCH --ntasks-per-node=4",
        "#SBATCH --time=24:00:00",
        "config_psc_vox1v2_fullmethod.yaml",
        "caarma_vox1v2_full_crp_slerpinit_projection_source",
        "Projection-D forward ready:",
    ):
        require(launcher, needle, LAUNCHER)

    print("Vox1+2 full-method isolation OK")
    print("Control Job: 44529360")
    print("Changed method fields:", ", ".join(sorted(changed)))


if __name__ == "__main__":
    main()
