"""Evaluate reconciliation results against hidden corruption ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic reconciliation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    truth = read_csv(args.data_dir / "ground_truth.csv")
    payment_results = read_csv(args.results_dir / "payment_reconciliation.csv")
    settlement_results = read_csv(args.results_dir / "settlement_reconciliation.csv")
    bank_results = read_csv(args.results_dir / "bank_reconciliation.csv")

    # Ground truth is an evaluation-only oracle. We map each injected issue to
    # the reconciliation layer where it should be observable.
    expected: dict[str, set[str]] = defaultdict(set)
    for row in truth:
        expected[row["entity_id"]].add(row["corruption_type"])

    detected: dict[str, set[str]] = defaultdict(set)
    for row in payment_results:
        if row["issues"]:
            detected[row["payment_id"]].update(row["issues"].split("|"))
    for row in settlement_results:
        if row["issues"]:
            detected[row["settlement_id"]].update(row["issues"].split("|"))
    for row in bank_results:
        if row["issues"]:
            detected[row["settlement_id"]].update(row["issues"].split("|"))

    # Directly comparable labels. Some ground-truth types intentionally map to
    # the engine's observable symptom rather than having identical names.
    mappings = {
        "MISSING_SETTLEMENT_ITEM": "MISSING_SETTLEMENT_ITEM",
        "DUPLICATE_SETTLEMENT_ITEM": "DUPLICATE_SETTLEMENT_ITEM",
        "FEE_MISMATCH": "FEE_MISMATCH",
        "AMOUNT_MISMATCH": "AMOUNT_MISMATCH",
        "MISSING_BANK_TRANSACTION": "MISSING_BANK_TRANSACTION",
        "DUPLICATE_BANK_TRANSACTION": "DUPLICATE_BANK_TRANSACTION",
        "BANK_AMOUNT_MISMATCH": "BANK_AMOUNT_MISMATCH",
        "WRONG_BANK_REFERENCE": "MISSING_BANK_TRANSACTION",
        "PARTIAL_SETTLEMENT": "BANK_AMOUNT_MISMATCH",
        "UNEXPLAINED_VARIANCE": "BANK_AMOUNT_MISMATCH",
    }

    per_type: dict[str, dict[str, int | float]] = {}
    for corruption_type in sorted({row["corruption_type"] for row in truth}):
        tp = fp = fn = 0
        for row in truth:
            if row["corruption_type"] != corruption_type:
                continue
            entity_id = row["entity_id"]
            observable = mappings[corruption_type]
            if observable in detected.get(entity_id, set()):
                tp += 1
            else:
                fn += 1

        # Count false positives only for this observable label on entities with
        # no matching ground-truth label.
        observable = mappings[corruption_type]
        for entity_id, labels in detected.items():
            if observable in labels and observable not in {
                mappings.get(label) for label in expected.get(entity_id, set())
            }:
                fp += 1

        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        per_type[corruption_type] = {
            "ground_truth_cases": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    summary = {
        "ground_truth_cases": len(truth),
        "unique_corrupted_entities": len(expected),
        "types": per_type,
        "notes": [
            "Evaluation uses ground_truth.csv only and is never consumed by reconciliation.",
            "Some corruption types are evaluated against observable symptoms produced by the deterministic engine.",
        ],
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
