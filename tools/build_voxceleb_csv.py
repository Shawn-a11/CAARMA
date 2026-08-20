#!/usr/bin/env python3
"""Build a CAARMA audio manifest from explicit VoxCeleb training roots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_EXTENSIONS = (".wav", ".flac", ".m4a")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must use NAME=/absolute/path")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid source {value!r}")
    return name, path


def infer_speaker_id(path: Path, root: Path, pattern: re.Pattern[str]) -> str | None:
    relative = path.relative_to(root)
    for part in reversed(relative.parts[:-1]):
        if pattern.fullmatch(part):
            return part
    return None


def build_manifest(
    sources: list[tuple[str, Path]],
    output: Path,
    speaker_pattern: str = r"id\d+",
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> dict:
    pattern = re.compile(speaker_pattern)
    normalized_ext = {ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                      for ext in extensions}
    rows = []
    unresolved = []
    seen_paths = set()
    speaker_sources = defaultdict(set)

    for source_name, root in sources:
        root = root.expanduser().resolve()
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in normalized_ext:
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            speaker_id = infer_speaker_id(resolved, root, pattern)
            if speaker_id is None:
                if len(unresolved) < 20:
                    unresolved.append(str(resolved))
                continue
            speaker_sources[speaker_id].add(source_name)
            rows.append((str(resolved), speaker_id, source_name))

    if unresolved:
        raise ValueError(
            "Could not infer speaker IDs from some audio paths. "
            f"Adjust --speaker-regex. Examples: {unresolved[:5]}"
        )
    if not rows:
        raise ValueError("No audio files were discovered under the supplied roots")

    speakers = sorted({speaker_id for _, speaker_id, _ in rows})
    label_by_speaker = {speaker_id: index for index, speaker_id in enumerate(speakers)}
    rows.sort(key=lambda row: row[0])

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("utt_paths", "utt_spk_int_labels", "utt_spk_id", "source"))
        for path, speaker_id, source_name in rows:
            label = label_by_speaker[speaker_id]
            writer.writerow((path, label, speaker_id, source_name))
            digest.update(f"{path}\t{label}\t{speaker_id}\t{source_name}\n".encode())

    overlap = {
        speaker: sorted(source_names)
        for speaker, source_names in speaker_sources.items()
        if len(source_names) > 1
    }
    report = {
        "output": str(output),
        "sources": {name: str(root) for name, root in sources},
        "speaker_regex": speaker_pattern,
        "extensions": sorted(normalized_ext),
        "utterances": len(rows),
        "speakers": len(speakers),
        "label_min": 0,
        "label_max": len(speakers) - 1,
        "cross_source_speaker_count": len(overlap),
        "cross_source_speaker_examples": dict(list(sorted(overlap.items()))[:20]),
        "rows_sha256": digest.hexdigest(),
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--speaker-regex", default=r"id\d+")
    parser.add_argument("--extension", action="append", dest="extensions")
    args = parser.parse_args()
    report = build_manifest(
        args.source,
        args.output,
        speaker_pattern=args.speaker_regex,
        extensions=tuple(args.extensions or DEFAULT_EXTENSIONS),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
