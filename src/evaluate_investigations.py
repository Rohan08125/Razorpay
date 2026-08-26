"""Evaluate investigation root causes against the corruption ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate investigation findings")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    truth = read_csv(args.data_dir / "ground_truth.csv")
    settlement_truth: dict[str, set[str]] = {}
    for row in truth:
        if row["entity_type"] == "settlement":
            settlement_truth.setdefault(row["entity_id"], set()).add(row["corruption_type"])

    cases = json.loads(
        (args.results_dir / "controller_cases.json").read_text(encoding="utf-8")
    )
    finding_counts = Counter(case["root_cause"] for case in cases)

    normalized = {
        "BANK_AMOUNT_MISMATCH": {"BANK_AMOUNT_VARIANCE"},
        "MISSING_BANK_TRANSACTION": {"MISSING_BANK_CREDIT"},
        "DUPLICATE_BANK_TRANSACTION": {"DUPLICATE_OR_CONFLICTING_BANK_CREDIT"},
        "WRONG_BANK_REFERENCE": {
            "MISSING_BANK_CREDIT",
            "DUPLICATE_OR_CONFLICTING_BANK_CREDIT",
            "BANK_AMOUNT_VARIANCE",
        },
        "PARTIAL_SETTLEMENT": {"BANK_AMOUNT_VARIANCE"},
        "UNEXPLAINED_VARIANCE": {"BANK_AMOUNT_VARIANCE"},
    }

    matched = 0
    applicable = 0
    for case in cases:
        expected = settlement_truth.get(case["case_id"], set())
        if not expected:
            continue
        applicable += 1
        root = case["root_cause"]
        if any(root in normalized.get(label, {label}) for label in expected):
            matched += 1

    summary = {
        "cases": len(cases),
        "finding_counts": dict(finding_counts),
        "ground_truth_linked_cases": applicable,
        "ground_truth_root_cause_matches": matched,
        "ground_truth_linked_accuracy": round(matched / applicable, 4) if applicable else 0.0,
        "average_confidence": round(
            sum(case["confidence"] for case in cases) / len(cases), 4
        ) if cases else 0.0,
        "escalation_rate": round(
            sum(case["recommended_action"] == "ESCALATE_FOR_MANUAL_REVIEW" for case in cases)
            / len(cases),
            4,
        ) if cases else 0.0,
    }

    output = args.results_dir / "investigation_evaluation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
