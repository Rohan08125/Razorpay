"""Evaluate investigation findings against settlement-level corruption truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def expected_root_causes(labels: set[str]) -> set[str]:
    mapping = {
        "MISSING_SETTLEMENT_ITEM": "SETTLEMENT_ITEM_COUNT_MISMATCH",
        "DUPLICATE_SETTLEMENT_ITEM": "SETTLEMENT_ITEM_COUNT_MISMATCH",
        "FEE_MISMATCH": "SETTLEMENT_TOTAL_MISMATCH",
        "AMOUNT_MISMATCH": "SETTLEMENT_TOTAL_MISMATCH",
        "BANK_AMOUNT_MISMATCH": "BANK_AMOUNT_VARIANCE",
        "MISSING_BANK_TRANSACTION": "MISSING_BANK_CREDIT",
        "DUPLICATE_BANK_TRANSACTION": "DUPLICATE_OR_CONFLICTING_BANK_CREDIT",
        "WRONG_BANK_REFERENCE": "MISSING_BANK_CREDIT",
        "PARTIAL_SETTLEMENT": "BANK_AMOUNT_VARIANCE",
        "UNEXPLAINED_VARIANCE": "BANK_AMOUNT_VARIANCE",
    }
    return {mapping[label] for label in labels if label in mapping}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate investigation findings")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    truth = read_csv(args.data_dir / "ground_truth.csv")
    settlement_truth: dict[str, set[str]] = {}
    for row in truth:
        settlement_truth.setdefault(row["entity_id"], set()).add(row["corruption_type"])

    cases = json.loads(
        (args.results_dir / "controller_cases.json").read_text(encoding="utf-8")
    )
    finding_counts = Counter(case["root_cause"] for case in cases)

    matched = 0
    linked = 0
    for case in cases:
        labels = settlement_truth.get(case["case_id"])
        if labels is None:
            continue
        linked += 1
        if case["root_cause"] in expected_root_causes(labels):
            matched += 1

    unique_truth_settlements = len(settlement_truth)
    coverage = linked / len(cases) if cases else 0.0
    escalation_rate = (
        sum(case.get("recommended_action") != "AUTO_RECONCILE" for case in cases) / len(cases)
        if cases else 0.0
    )

    summary = {
        "cases": len(cases),
        "finding_counts": dict(finding_counts),
        "ground_truth_rows": len(truth),
        "unique_ground_truth_settlements": unique_truth_settlements,
        "ground_truth_linked_cases": linked,
        "ground_truth_coverage": round(coverage, 4),
        "ground_truth_root_cause_matches": matched,
        "ground_truth_linked_accuracy": round(matched / linked, 4) if linked else None,
        "average_confidence": round(
            sum(case["confidence"] for case in cases) / len(cases), 4
        ) if cases else 0.0,
        "escalation_rate": round(escalation_rate, 4),
        "warning": (
            "Do not report linked accuracy as batch accuracy unless coverage is near 100%."
            if coverage < 0.95 else None
        ),
    }

    output = args.results_dir / "investigation_evaluation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
