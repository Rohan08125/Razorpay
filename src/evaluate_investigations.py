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
    bank_truth = {
        row["entity_id"]: row["corruption_type"]
        for row in truth
        if row["entity_type"] == "bank_transaction"
    }

    # Ground truth bank transaction IDs are mapped to settlements by the
    # corruption generator's details/reference structure in the generated data.
    bank_rows = read_csv(args.data_dir / "bank_transactions.csv")
    bank_to_settlement = {row["bank_transaction_id"]: row["reference"] for row in bank_rows}
    settlement_truth: dict[str, set[str]] = {}
    for bank_id, corruption in bank_truth.items():
        settlement_id = bank_to_settlement.get(bank_id)
        if settlement_id:
            settlement_truth.setdefault(settlement_id, set()).add(corruption)

    cases = json.loads((args.results_dir / "ai_case_packets.json").read_text(encoding="utf-8"))
    finding_counts = Counter(case["root_cause"] for case in cases)

    matched = 0
    applicable = 0
    for case in cases:
        expected = settlement_truth.get(case["case_id"], set())
        if not expected:
            continue
        applicable += 1
        root = case["root_cause"]
        normalized = {
            "BANK_AMOUNT_MISMATCH": {"BANK_AMOUNT_VARIANCE"},
            "MISSING_BANK_TRANSACTION": {"MISSING_BANK_CREDIT"},
            "DUPLICATE_BANK_TRANSACTION": {"DUPLICATE_OR_CONFLICTING_BANK_CREDIT"},
            "WRONG_BANK_REFERENCE": {"MISSING_BANK_CREDIT", "DUPLICATE_OR_CONFLICTING_BANK_CREDIT"},
            "PARTIAL_SETTLEMENT": {"BANK_AMOUNT_VARIANCE"},
            "UNEXPLAINED_VARIANCE": {"BANK_AMOUNT_VARIANCE"},
        }
        if any(root in normalized.get(label, {label}) for label in expected):
            matched += 1

    summary = {
        "cases": len(cases),
        "finding_counts": dict(finding_counts),
        "ground_truth_linked_cases": applicable,
        "ground_truth_root_cause_matches": matched,
        "ground_truth_linked_accuracy": round(matched / applicable, 4) if applicable else 0.0,
        "average_confidence": round(sum(case["confidence"] for case in cases) / len(cases), 4) if cases else 0.0,
        "escalation_rate": round(
            sum(case["recommended_action"] == "ESCALATE_FOR_MANUAL_REVIEW" for case in cases) / len(cases), 4
        ) if cases else 0.0,
    }

    output = args.results_dir / "investigation_evaluation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
