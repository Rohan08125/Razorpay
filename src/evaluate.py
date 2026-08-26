"""Evaluate deterministic reconciliation against settlement-level ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# Ground-truth corruption -> observable reconciliation symptom.
OBSERVABLES = {
    "MISSING_SETTLEMENT_ITEM": {"PAYMENT_COUNT_MISMATCH"},
    "DUPLICATE_SETTLEMENT_ITEM": {"PAYMENT_COUNT_MISMATCH"},
    "FEE_MISMATCH": {"FEE_TOTAL_MISMATCH", "NET_TOTAL_MISMATCH"},
    "AMOUNT_MISMATCH": {"GROSS_AMOUNT_MISMATCH"},
    "MISSING_BANK_TRANSACTION": {"MISSING_BANK_TRANSACTION"},
    "DUPLICATE_BANK_TRANSACTION": {"DUPLICATE_BANK_TRANSACTION"},
    "BANK_AMOUNT_MISMATCH": {"BANK_AMOUNT_MISMATCH"},
    "WRONG_BANK_REFERENCE": {"MISSING_BANK_TRANSACTION"},
    "PARTIAL_SETTLEMENT": {"BANK_AMOUNT_MISMATCH"},
    "UNEXPLAINED_VARIANCE": {"BANK_AMOUNT_MISMATCH"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic financial reconciliation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    truth = read_csv(args.data_dir / "ground_truth.csv")
    settlement_results = read_csv(args.results_dir / "settlement_reconciliation.csv")
    bank_results = read_csv(args.results_dir / "bank_reconciliation.csv")

    expected: dict[str, set[str]] = defaultdict(set)
    for row in truth:
        expected[row["entity_id"]].add(row["corruption_type"])

    detected: dict[str, set[str]] = defaultdict(set)
    for row in settlement_results:
        if row["issues"]:
            detected[row["settlement_id"]].update(row["issues"].split("|"))
    for row in bank_results:
        if row["issues"]:
            detected[row["settlement_id"]].update(row["issues"].split("|"))

    per_type: dict[str, dict[str, int | float]] = {}
    for corruption_type in sorted(OBSERVABLES):
        expected_cases = [
            row for row in truth if row["corruption_type"] == corruption_type
        ]
        observable_labels = OBSERVABLES[corruption_type]
        tp = fn = 0

        for row in expected_cases:
            entity_id = row["entity_id"]
            if detected.get(entity_id, set()) & observable_labels:
                tp += 1
            else:
                fn += 1

        fp = 0
        for entity_id, observed in detected.items():
            if not observed & observable_labels:
                continue
            expected_types = expected.get(entity_id, set())
            expected_observables = set().union(
                *(OBSERVABLES.get(t, set()) for t in expected_types)
            ) if expected_types else set()
            if not (observed & observable_labels & expected_observables):
                fp += 1

        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        per_type[corruption_type] = {
            "ground_truth_cases": len(expected_cases),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    summary = {
        "ground_truth_cases": len(truth),
        "unique_corrupted_settlements": len(expected),
        "observed_exception_settlements": len(detected),
        "types": per_type,
        "notes": [
            "Ground truth is evaluation-only and is never consumed by reconciliation.",
            "Each corruption type is scored against the reconciliation layer where its observable symptom appears.",
        ],
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "evaluation_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
