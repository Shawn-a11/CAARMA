#!/usr/bin/env python3
"""Print completed CAARMA autoresearch trials ranked by development EER."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    with args.results.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    valid = [
        row
        for row in rows
        if row.get("status") in {"complete", "imported"} and row.get("best_eer")
    ]
    valid.sort(
        key=lambda row: (
            float(row["best_eer"]),
            float(row["min_dcf_1e2"]) if row.get("min_dcf_1e2") else float("inf"),
        )
    )

    print("rank\ttag\tEER\tminDCF(1e-2)\tepoch\tseed\tbudget\tdescription")
    for rank, row in enumerate(valid[: args.top], 1):
        print(
            "\t".join(
                [
                    str(rank),
                    row["tag"],
                    row["best_eer"],
                    row.get("min_dcf_1e2", ""),
                    row.get("best_epoch_zero_based", ""),
                    row.get("seed", ""),
                    row.get("budget_epochs", ""),
                    row.get("description", ""),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
