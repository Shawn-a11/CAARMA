#!/usr/bin/env python3
"""Audit persistent synthetic-class occupancy saved in CAARMA checkpoints.

The audit is read-only. It distinguishes:

* allocated pairs: entries in ``pair_col`` that consume registry capacity;
* occupied classes: entries in ``created_pairs`` with at least one visit.

For DDP runs, the Python CRP registry is rank-local. A Lightning checkpoint
therefore normally contains the state from rank 0 only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_synth_states(value: Any, path: str = "checkpoint") -> list[tuple[str, Mapping]]:
    matches: list[tuple[str, Mapping]] = []
    if isinstance(value, Mapping):
        keys = set(value)
        if {"pair_visits", "pair_col", "max_cols"}.issubset(keys):
            matches.append((path, value))
        for key, child in value.items():
            if isinstance(child, (Mapping, list, tuple)):
                matches.extend(_find_synth_states(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            if isinstance(child, (Mapping, list, tuple)):
                matches.extend(_find_synth_states(child, f"{path}[{index}]"))
    return matches


def _pair_key(value: Any) -> tuple[int, ...]:
    if isinstance(value, (set, frozenset, list, tuple)):
        return tuple(sorted(int(item) for item in value))
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+", value)
        if numbers:
            return tuple(sorted(int(item) for item in numbers))
    raise ValueError(f"Unsupported pair key: {value!r}")


def _pair_counts(value: Any) -> dict[tuple[int, ...], int]:
    if isinstance(value, Mapping):
        rows = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = value
    else:
        raise ValueError("pair_visits must be a mapping or a sequence of pairs")

    counts: dict[tuple[int, ...], int] = {}
    for row in rows:
        if not isinstance(row, Sequence) or len(row) != 2:
            raise ValueError(f"Malformed pair_visits row: {row!r}")
        key, count = row
        counts[_pair_key(key)] = int(count)
    return counts


def _pair_col_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    raise ValueError("pair_col must be a mapping or a sequence")


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return int(ordered[index])


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    total = sum(ordered)
    if total <= 0:
        return 0.0
    n = len(ordered)
    weighted = sum((2 * index - n - 1) * value for index, value in enumerate(ordered, 1))
    return weighted / (n * total)


def _effective_classes(values: list[int]) -> float:
    total = float(sum(values))
    if total <= 0:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in values if value > 0)
    return math.exp(entropy)


def _share_of_largest(values: list[int], fraction: float) -> float:
    if not values or sum(values) <= 0:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    return sum(sorted(values, reverse=True)[:count]) / sum(values)


def _histogram(values: list[int]) -> dict[str, int]:
    return {
        "1": sum(value == 1 for value in values),
        "2": sum(value == 2 for value in values),
        "3-4": sum(3 <= value <= 4 for value in values),
        "5-9": sum(5 <= value <= 9 for value in values),
        "10-19": sum(10 <= value <= 19 for value in values),
        "20-49": sum(20 <= value <= 49 for value in values),
        "50+": sum(value >= 50 for value in values),
    }


def audit_state(state: Mapping) -> dict[str, Any]:
    pair_visits = _pair_counts(state.get("pair_visits", []))
    created_raw = state.get("created_pairs", [])
    created = {_pair_key(key) for key in created_raw}

    # Older checkpoints may omit created_pairs. Positive visit counts are the
    # strongest evidence that a pair was actually used for synthetic training.
    occupied_keys = created | {key for key, count in pair_visits.items() if count > 0}
    visits = [pair_visits.get(key, 0) for key in occupied_keys]
    positive_visits = [value for value in visits if value > 0]
    zero_visit_created = sum(value == 0 for value in visits)

    occupied = len(positive_visits)
    total_visits = sum(positive_visits)
    allocated = _pair_col_count(state.get("pair_col", []))
    max_cols = int(state.get("max_cols", 0))
    singleton_count = sum(value == 1 for value in positive_visits)
    doubleton_count = sum(value == 2 for value in positive_visits)
    effective = _effective_classes(positive_visits)

    return {
        "pair_strategy": state.get("pair_strategy"),
        "crp_alpha": state.get("crp_alpha"),
        "crp_topk": state.get("crp_topk"),
        "max_cols": max_cols,
        "allocated_pairs": allocated,
        "allocated_fraction": allocated / max_cols if max_cols else None,
        "registry_capacity_reached": bool(max_cols and allocated >= max_cols),
        "occupied_classes": occupied,
        "occupied_fraction_of_capacity": occupied / max_cols if max_cols else None,
        "created_pairs_with_zero_visits": zero_visit_created,
        "total_visits": total_visits,
        "singleton_count": singleton_count,
        "singleton_ratio": singleton_count / occupied if occupied else 0.0,
        "doubleton_count": doubleton_count,
        "doubleton_ratio": doubleton_count / occupied if occupied else 0.0,
        "mean_occupancy": statistics.fmean(positive_visits) if positive_visits else 0.0,
        "median_occupancy": statistics.median(positive_visits) if positive_visits else 0.0,
        "p90_occupancy": _nearest_rank(positive_visits, 0.90),
        "p99_occupancy": _nearest_rank(positive_visits, 0.99),
        "maximum_occupancy": max(positive_visits, default=0),
        "gini_coefficient": _gini(positive_visits),
        "effective_number_of_classes": effective,
        "effective_class_ratio": effective / occupied if occupied else 0.0,
        "top_1pct_visit_share": _share_of_largest(positive_visits, 0.01),
        "top_10pct_visit_share": _share_of_largest(positive_visits, 0.10),
        "occupancy_histogram": _histogram(positive_visits),
    }


def audit_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = _load_checkpoint(path)
    matches = _find_synth_states(checkpoint)
    if not matches:
        raise ValueError(
            "No persistent CRP state found. Expected pair_visits, pair_col, and max_cols. "
            "This may be a one-shot run or an older checkpoint saved before extra state."
        )
    if len(matches) > 1:
        paths = ", ".join(path for path, _ in matches)
        raise ValueError(f"Multiple CRP states found ({paths}); audit them separately.")

    state_path, state = matches[0]
    result = audit_state(state)
    raw_epoch = checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
    result.update(
        {
            "checkpoint": str(path),
            "checkpoint_epoch_zero_based": int(raw_epoch) if raw_epoch is not None else None,
            "display_epoch_one_based": int(raw_epoch) + 1 if raw_epoch is not None else None,
            "state_path": state_path,
            "ddp_scope_warning": (
                "Python CRP state is rank-local; a standard Lightning checkpoint normally "
                "reports rank 0 only, not a merged multi-rank registry."
            ),
        }
    )
    return result


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(?:epoch[=_-]?)?(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, str(path))


def audit_directory(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    checkpoints = sorted(path.rglob("*.ckpt"), key=_checkpoint_sort_key)
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        try:
            reports.append(audit_checkpoint(checkpoint))
        except (OSError, RuntimeError, ValueError) as error:
            errors.append({"checkpoint": str(checkpoint), "error": str(error)})
    reports.sort(
        key=lambda report: (
            report["checkpoint_epoch_zero_based"]
            if report["checkpoint_epoch_zero_based"] is not None
            else 10**9,
            report["checkpoint"],
        )
    )
    return reports, errors


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_report(report: Mapping[str, Any]) -> None:
    print(f"Checkpoint: {report['checkpoint']}")
    print(f"State path: {report['state_path']}")
    print(
        "Epoch: "
        f"{_format_number(report['display_epoch_one_based'])} "
        f"(checkpoint raw={_format_number(report['checkpoint_epoch_zero_based'])})"
    )
    print(
        "CRP: "
        f"strategy={_format_number(report['pair_strategy'])}, "
        f"alpha={_format_number(report['crp_alpha'])}, "
        f"top-k={_format_number(report['crp_topk'])}"
    )
    print(
        "Registry: "
        f"allocated={report['allocated_pairs']}/{report['max_cols']} "
        f"({_format_number(report['allocated_fraction'])}), "
        f"capacity reached={_format_number(report['registry_capacity_reached'])}"
    )
    print(
        "Occupied: "
        f"classes={report['occupied_classes']}, visits={report['total_visits']}, "
        f"zero-visit created={report['created_pairs_with_zero_visits']}"
    )
    print(
        "Singleton / doubleton: "
        f"{report['singleton_count']} ({report['singleton_ratio']:.4f}) / "
        f"{report['doubleton_count']} ({report['doubleton_ratio']:.4f})"
    )
    print(
        "Occupancy: "
        f"mean={report['mean_occupancy']:.2f}, "
        f"median={report['median_occupancy']:.2f}, "
        f"p90={report['p90_occupancy']}, p99={report['p99_occupancy']}, "
        f"max={report['maximum_occupancy']}"
    )
    print(
        "Inequality: "
        f"Gini={report['gini_coefficient']:.4f}, "
        f"effective classes={report['effective_number_of_classes']:.2f} "
        f"({report['effective_class_ratio']:.4f} of occupied), "
        f"top-1% share={report['top_1pct_visit_share']:.4f}, "
        f"top-10% share={report['top_10pct_visit_share']:.4f}"
    )
    histogram = ", ".join(f"{key}:{value}" for key, value in report["occupancy_histogram"].items())
    print(f"Histogram: {histogram}")
    print(f"Scope warning: {report['ddp_scope_warning']}")


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"reports": [], "errors": []}
    if args.checkpoint:
        try:
            payload["reports"].append(audit_checkpoint(args.checkpoint))
        except (OSError, RuntimeError, ValueError) as error:
            payload["errors"].append({"checkpoint": str(args.checkpoint), "error": str(error)})
    if args.checkpoint_dir:
        reports, errors = audit_directory(args.checkpoint_dir)
        payload["reports"].extend(reports)
        payload["errors"].extend(errors)

    saturated = [
        report for report in payload["reports"] if report["registry_capacity_reached"]
    ]
    payload["timeline"] = {
        "checkpoints_scanned": len(payload["reports"]),
        "first_observed_saturation": saturated[0] if saturated else None,
        "interpretation": (
            "This is the first saved checkpoint observed at capacity, not necessarily the exact "
            "training epoch when capacity was first reached. Exact recovery requires a checkpoint "
            "or allocated-pair log for every epoch."
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only audit of CRP synthetic-class occupancy in CAARMA checkpoints."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="A Lightning .ckpt file.")
    source.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory recursively scanned for .ckpt files.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    payload = _build_payload(args)
    for index, report in enumerate(payload["reports"]):
        if index:
            print("\n" + "-" * 80)
        print_report(report)

    timeline = payload["timeline"]
    print("\nTimeline")
    print(f"Checkpoints successfully scanned: {timeline['checkpoints_scanned']}")
    first = timeline["first_observed_saturation"]
    if first:
        print(
            "First observed registry saturation: "
            f"display epoch {_format_number(first['display_epoch_one_based'])}, "
            f"{first['checkpoint']}"
        )
    else:
        print("First observed registry saturation: not observed in the supplied checkpoints")
    print(f"Note: {timeline['interpretation']}")

    for error in payload["errors"]:
        print(f"\nSkipped {error['checkpoint']}: {error['error']}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report written to: {args.json_output}")

    return 0 if payload["reports"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
