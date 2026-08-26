"""Diagnose how injected corruption maps onto reconciliation exceptions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose synthetic batch coverage")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    truth = read_csv(args.data_dir / "ground_truth.csv")
    settlements = read_csv(args.data_dir / "settlements.csv")
    settlement_results = read_csv(args.results_dir / "settlement_reconciliation.csv")

    truth_by_settlement: dict[str, list[str]] = defaultdict(list)
    for row in truth:
        truth_by_settlement[row["entity_id"]].append(row["corruption_type"])

    exception_ids = {
        row["settlement_id"] for row in settlement_results if row["status"] == "EXCEPTION"
    }
    corrupted_ids = set(truth_by_settlement)
    clean_exception_ids = exception_ids - corrupted_ids
    corrupted_but_reconciled = corrupted_ids - exception_ids

    type_counts = Counter(row["corruption_type"] for row in truth)
    corrupted_exception_overlap = Counter()
    for settlement_id in exception_ids & corrupted_ids:
        corrupted_exception_overlap.update(truth_by_settlement[settlement_id])

    summary = {
        "total_settlements": len(settlements),
        "settlement_exceptions": len(exception_ids),
        "unique_corrupted_settlements": len(corrupted_ids),
        "corrupted_and_exception": len(exception_ids & corrupted_ids),
        "clean_settlement_exceptions": len(clean_exception_ids),
        "corrupted_but_reconciled": len(corrupted_but_reconciled),
        "ground_truth_type_counts": dict(type_counts),
        "corruption_types_on_exception_settlements": dict(corrupted_exception_overlap),
        "warning": (
            "Large clean-exception count indicates false positives in the deterministic settlement check."
            if clean_exception_ids else None
        ),
    }

    output = args.results_dir / "batch_diagnostics.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
