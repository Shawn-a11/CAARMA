#!/usr/bin/env python3
"""Audit CAARMA train manifests and VoxCeleb verification trials."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def infer_speaker(path: Path, pattern: re.Pattern[str]) -> str | None:
    for part in reversed(path.parts):
        if pattern.fullmatch(part):
            return part
    return None


def resolve_trial_path(eval_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (eval_root / value.lstrip("/\\")).resolve()


def read_trials(path: Path) -> list[tuple[int, str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 3:
                raise ValueError(
                    f"Trial line {line_number} must contain LABEL PATH1 PATH2: {stripped!r}"
                )
            label = int(fields[0])
            if label not in (0, 1):
                raise ValueError(f"Trial label must be 0 or 1 at line {line_number}")
            rows.append((label, fields[1], fields[2]))
    if not rows:
        raise ValueError(f"Trial file is empty: {path}")
    return rows


def audit_protocol(
    train_csv: Path,
    trial_path: Path,
    eval_root: Path,
    speaker_regex: str = r"id\d+",
    check_all_train_files: bool = True,
) -> tuple[dict, list[str]]:
    required = {"utt_paths", "utt_spk_int_labels"}
    frame = pd.read_csv(train_csv)
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "This is not a CAARMA audio manifest; missing column(s): "
            + ", ".join(missing_columns)
        )

    labels = sorted(int(value) for value in frame["utt_spk_int_labels"].unique())
    contiguous = labels == list(range(len(labels)))
    train_paths = [Path(value).expanduser().resolve() for value in frame["utt_paths"]]
    duplicate_train_paths = len(train_paths) - len(set(train_paths))
    if check_all_train_files:
        missing_train = [str(path) for path in train_paths if not path.is_file()]
    else:
        missing_train = [str(path) for path in train_paths[:1000] if not path.is_file()]

    trials = read_trials(trial_path)
    trial_paths = {
        resolve_trial_path(eval_root, value)
        for _, first, second in trials
        for value in (first, second)
    }
    missing_eval = [str(path) for path in sorted(trial_paths) if not path.is_file()]
    utterance_overlap = sorted(str(path) for path in set(train_paths) & trial_paths)

    pattern = re.compile(speaker_regex)
    train_speakers = {speaker for path in train_paths
                      if (speaker := infer_speaker(path, pattern)) is not None}
    eval_speakers = {speaker for path in trial_paths
                     if (speaker := infer_speaker(path, pattern)) is not None}
    speaker_overlap = sorted(train_speakers & eval_speakers)
    label_counts = Counter(label for label, _, _ in trials)

    report = {
        "train_csv": str(train_csv.resolve()),
        "trial_path": str(trial_path.resolve()),
        "eval_root": str(eval_root.resolve()),
        "train_utterances": len(frame),
        "train_speakers": len(labels),
        "labels_contiguous_zero_based": contiguous,
        "duplicate_train_paths": duplicate_train_paths,
        "missing_train_files": len(missing_train),
        "missing_train_examples": missing_train[:20],
        "trials": len(trials),
        "positive_trials": label_counts[1],
        "negative_trials": label_counts[0],
        "evaluation_utterances": len(trial_paths),
        "missing_evaluation_files": len(missing_eval),
        "missing_evaluation_examples": missing_eval[:20],
        "train_test_utterance_overlap": len(utterance_overlap),
        "train_test_utterance_overlap_examples": utterance_overlap[:20],
        "inferred_train_speakers": len(train_speakers),
        "inferred_evaluation_speakers": len(eval_speakers),
        "train_test_speaker_overlap": len(speaker_overlap),
        "train_test_speaker_overlap_examples": speaker_overlap[:20],
    }
    errors = []
    if not contiguous:
        errors.append("speaker labels are not contiguous and zero-based")
    if duplicate_train_paths:
        errors.append("training manifest contains duplicate utterance paths")
    if missing_train:
        errors.append("training manifest contains missing audio files")
    if missing_eval:
        errors.append("trial paths do not resolve under eval_root")
    if utterance_overlap:
        errors.append("training and evaluation share utterance files")
    if speaker_overlap:
        errors.append("training and evaluation contain overlapping speaker IDs")
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--trial-path", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--speaker-regex", default=r"id\d+")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--sample-train-files", action="store_true")
    args = parser.parse_args()

    report, errors = audit_protocol(
        args.train_csv,
        args.trial_path,
        args.eval_root,
        speaker_regex=args.speaker_regex,
        check_all_train_files=not args.sample_train_files,
    )
    payload = {"report": report, "errors": errors}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
