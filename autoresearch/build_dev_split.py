#!/usr/bin/env python3
"""Build a deterministic utterance-held-out development split.

The split keeps every speaker identity in training but removes a small set of
utterances per speaker from the training CSV. Those held-out utterances become
speaker-verification trials under the supplied evaluation root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


PATH_COLUMN = "utt_paths"
LABEL_COLUMN = "utt_spk_int_labels"


def speaker_sort_key(value: str) -> tuple[int, int | str, str]:
    try:
        return (0, int(value), value)
    except ValueError:
        return (1, value, value)


def relative_audio_path(path: str, eval_root: Path) -> str:
    absolute = Path(path).expanduser().resolve()
    if not absolute.is_file():
        raise ValueError(f"Audio file not found: {absolute}")
    try:
        relative = absolute.relative_to(eval_root)
    except ValueError as error:
        raise ValueError(
            f"Audio path {absolute} is not under eval root {eval_root}"
        ) from error
    return relative.as_posix()


def select_heldout(rows: list[dict[str, str]], count: int, rng: random.Random):
    """Prefer different recording directories, then fill from remaining rows."""
    shuffled = list(rows)
    rng.shuffle(shuffled)
    selected: list[dict[str, str]] = []
    selected_paths: set[str] = set()
    recording_dirs: set[str] = set()

    for row in shuffled:
        recording_dir = str(Path(row[PATH_COLUMN]).parent)
        if recording_dir in recording_dirs:
            continue
        selected.append(row)
        selected_paths.add(row[PATH_COLUMN])
        recording_dirs.add(recording_dir)
        if len(selected) == count:
            return selected

    for row in shuffled:
        if row[PATH_COLUMN] in selected_paths:
            continue
        selected.append(row)
        selected_paths.add(row[PATH_COLUMN])
        if len(selected) == count:
            return selected
    return selected


def build_split(
    input_csv: Path,
    eval_root: Path,
    output_dir: Path,
    heldout_per_speaker: int,
    positive_pairs_per_speaker: int,
    negative_ratio: float,
    seed: int,
) -> dict[str, Any]:
    input_csv = input_csv.expanduser().resolve()
    eval_root = eval_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_csv.is_file():
        raise ValueError(f"Input CSV not found: {input_csv}")
    if not eval_root.is_dir():
        raise ValueError(f"Evaluation root is not a directory: {eval_root}")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if PATH_COLUMN not in fieldnames or LABEL_COLUMN not in fieldnames:
            raise ValueError(
                f"CSV must contain {PATH_COLUMN!r} and {LABEL_COLUMN!r}"
            )
        rows = list(reader)

    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_speaker[row[LABEL_COLUMN]].append(row)

    rng = random.Random(seed)
    heldout_by_speaker: dict[str, list[dict[str, str]]] = {}
    heldout_paths: set[str] = set()
    for speaker in sorted(by_speaker, key=speaker_sort_key):
        speaker_rows = by_speaker[speaker]
        if len(speaker_rows) <= heldout_per_speaker:
            raise ValueError(
                f"Speaker {speaker} has only {len(speaker_rows)} utterances; "
                f"cannot hold out {heldout_per_speaker} and retain training data"
            )
        selected = select_heldout(speaker_rows, heldout_per_speaker, rng)
        if len(selected) != heldout_per_speaker:
            raise ValueError(f"Could not select enough held-out rows for {speaker}")
        heldout_by_speaker[speaker] = selected
        heldout_paths.update(row[PATH_COLUMN] for row in selected)

    train_rows = [row for row in rows if row[PATH_COLUMN] not in heldout_paths]
    positives: list[tuple[int, str, str]] = []
    eval_items: list[tuple[str, str]] = []
    for speaker in sorted(heldout_by_speaker, key=speaker_sort_key):
        paths = [
            relative_audio_path(row[PATH_COLUMN], eval_root)
            for row in heldout_by_speaker[speaker]
        ]
        eval_items.extend((speaker, path) for path in paths)
        combinations = list(itertools.combinations(paths, 2))
        rng.shuffle(combinations)
        for first, second in combinations[:positive_pairs_per_speaker]:
            positives.append((1, first, second))

    target_negatives = int(round(len(positives) * negative_ratio))
    negatives: list[tuple[int, str, str]] = []
    used_negative_pairs: set[tuple[str, str]] = set()
    max_attempts = max(1000, target_negatives * 100)
    for _ in range(max_attempts):
        if len(negatives) >= target_negatives:
            break
        left, right = rng.sample(eval_items, 2)
        if left[0] == right[0]:
            continue
        key = tuple(sorted((left[1], right[1])))
        if key in used_negative_pairs:
            continue
        used_negative_pairs.add(key)
        negatives.append((0, left[1], right[1]))
    if len(negatives) != target_negatives:
        raise ValueError(
            f"Generated {len(negatives)} negatives, expected {target_negatives}"
        )

    trials = positives + negatives
    rng.shuffle(trials)

    train_paths = {row[PATH_COLUMN] for row in train_rows}
    if train_paths.intersection(heldout_paths):
        raise AssertionError("Training and development utterances overlap")

    output_dir.mkdir(parents=True, exist_ok=False)
    train_csv = output_dir / "train.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    heldout_csv = output_dir / "heldout.csv"
    with heldout_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for speaker in sorted(heldout_by_speaker, key=speaker_sort_key):
            writer.writerows(heldout_by_speaker[speaker])

    trial_path = output_dir / "dev_trials.txt"
    with trial_path.open("w", encoding="utf-8") as handle:
        for label, first, second in trials:
            handle.write(f"{label} {first} {second}\n")

    digest = hashlib.sha256(trial_path.read_bytes()).hexdigest()
    manifest = {
        "input_csv": str(input_csv),
        "eval_root": str(eval_root),
        "seed": seed,
        "heldout_per_speaker": heldout_per_speaker,
        "positive_pairs_per_speaker": positive_pairs_per_speaker,
        "negative_ratio": negative_ratio,
        "speakers": len(by_speaker),
        "input_utterances": len(rows),
        "training_utterances": len(train_rows),
        "heldout_utterances": len(heldout_paths),
        "positive_trials": len(positives),
        "negative_trials": len(negatives),
        "trial_sha256": digest,
        "train_csv": str(train_csv),
        "heldout_csv": str(heldout_csv),
        "dev_trials": str(trial_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic utterance-held-out VoxCeleb dev split."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--heldout-per-speaker", type=int, default=3)
    parser.add_argument("--positive-pairs-per-speaker", type=int, default=3)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.heldout_per_speaker < 2:
        parser.error("--heldout-per-speaker must be at least 2")
    max_positive_pairs = (
        args.heldout_per_speaker * (args.heldout_per_speaker - 1) // 2
    )
    if not 1 <= args.positive_pairs_per_speaker <= max_positive_pairs:
        parser.error(
            "--positive-pairs-per-speaker must be between 1 and "
            f"{max_positive_pairs}"
        )
    if args.negative_ratio <= 0:
        parser.error("--negative-ratio must be positive")

    manifest = build_split(
        input_csv=args.input_csv,
        eval_root=args.eval_root,
        output_dir=args.output_dir,
        heldout_per_speaker=args.heldout_per_speaker,
        positive_pairs_per_speaker=args.positive_pairs_per_speaker,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
