"""Evaluate reconciliation quality against injected ground truth and Ollama consistency."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ai_controller import build_case
from investigate_exception import ExceptionInvestigator


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"

CORRUPTION_TO_ROOT_CAUSE = {
    "MISSING_SETTLEMENT_ITEM": "SETTLEMENT_ITEM_COUNT_MISMATCH",
    "DUPLICATE_SETTLEMENT_ITEM": "SETTLEMENT_ITEM_COUNT_MISMATCH",
    "FEE_MISMATCH": "SETTLEMENT_TOTAL_MISMATCH",
    "AMOUNT_MISMATCH": "SETTLEMENT_ITEM_AMOUNT_MISMATCH",
    "MISSING_BANK_TRANSACTION": "MISSING_BANK_CREDIT",
    "DUPLICATE_BANK_TRANSACTION": "DUPLICATE_OR_CONFLICTING_BANK_CREDIT",
    "BANK_AMOUNT_MISMATCH": "BANK_AMOUNT_VARIANCE",
    "WRONG_BANK_REFERENCE": "MISSING_BANK_CREDIT",
    "PARTIAL_SETTLEMENT": "BANK_AMOUNT_VARIANCE",
    "UNEXPLAINED_VARIANCE": "BANK_AMOUNT_VARIANCE",
}


def normalize_action(action: str | None) -> str:
    mapping = {
        "AUTO_RECONCILE": "AUTO_RECONCILE",
        "Auto reconcile": "AUTO_RECONCILE",
        "REVIEW_SETTLEMENT_ITEM": "REVIEW_SETTLEMENT_ITEM",
        "Review settlement item": "REVIEW_SETTLEMENT_ITEM",
        "ESCALATE_FOR_MANUAL_REVIEW": "ESCALATE_FOR_MANUAL_REVIEW",
        "Escalate for manual review": "ESCALATE_FOR_MANUAL_REVIEW",
        "Escalate for missing bank credit investigation.": "ESCALATE_FOR_MANUAL_REVIEW",
        "Escalate for manual bank-side review.": "ESCALATE_FOR_MANUAL_REVIEW",
        "Investigate partial or unexplained settlement variance.": "ESCALATE_FOR_MANUAL_REVIEW",
    }
    return mapping.get(action or "", action or "UNKNOWN")


def load_ground_truth() -> list[dict[str, str]]:
    if not GROUND_TRUTH_PATH.exists():
        return []
    with GROUND_TRUTH_PATH.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ollama_path = RESULTS_DIR / "ollama_agent_decisions.json"
    ollama_cases = json.loads(ollama_path.read_text(encoding="utf-8")) if ollama_path.exists() else []

    investigator = ExceptionInvestigator(DATA_DIR)
    expected_by_id = {
        settlement_id: build_case(settlement_id, investigator)
        for settlement_id in investigator.data.settlement_by_id
    }

    truth = load_ground_truth()
    truth_by_id = {row["entity_id"]: row for row in truth}

    predicted_exception_ids = {
        case_id
        for case_id, case in expected_by_id.items()
        if case.get("root_cause") != "NONE"
    }
    actual_corrupt_ids = set(truth_by_id)
    true_positive_ids = predicted_exception_ids & actual_corrupt_ids
    false_positive_ids = predicted_exception_ids - actual_corrupt_ids
    false_negative_ids = actual_corrupt_ids - predicted_exception_ids

    tp = len(true_positive_ids)
    fp = len(false_positive_ids)
    fn = len(false_negative_ids)
    tn = len(expected_by_id) - len(true_positive_ids | false_positive_ids | false_negative_ids)

    detection_accuracy = (tp + tn) / len(expected_by_id) if expected_by_id else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    root_matches = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in truth:
        case_id = row["entity_id"]
        corruption_type = row["corruption_type"]
        expected_root = CORRUPTION_TO_ROOT_CAUSE.get(corruption_type, "UNKNOWN")
        actual_root = expected_by_id.get(case_id, {}).get("root_cause", "MISSING_CASE")
        confusion[corruption_type][actual_root] += 1
        if actual_root == expected_root:
            root_matches += 1

    root_compared = len(truth)
    ground_truth_root_accuracy = root_matches / root_compared if root_compared else 0.0
    by_type: dict[str, dict[str, float | int]] = {}
    for corruption_type, counts in confusion.items():
        total_type = sum(counts.values())
        expected_root = CORRUPTION_TO_ROOT_CAUSE.get(corruption_type, "UNKNOWN")
        correct = counts.get(expected_root, 0)
        by_type[corruption_type] = {
            "cases": total_type,
            "root_cause_matches": correct,
            "root_cause_accuracy": round(correct / total_type, 4) if total_type else 0.0,
        }

    compared = 0
    root_matches_ai = 0
    action_matches_ai = 0
    confidence_sum = 0.0
    fallback_cases = 0
    disagreements = []

    for ai_case in ollama_cases:
        case_id = ai_case.get("case_id")
        expected = expected_by_id.get(case_id)
        if expected is None:
            continue

        compared += 1
        ai_root = ai_case.get("root_cause")
        expected_root = expected.get("root_cause")
        ai_action = normalize_action(ai_case.get("action"))
        expected_action = normalize_action(expected.get("recommended_action"))
        confidence = float(ai_case.get("confidence", 0.0))
        confidence_sum += confidence
        if confidence == 0.0:
            fallback_cases += 1

        root_match = ai_root == expected_root
        action_match = ai_action == expected_action
        root_matches_ai += root_match
        action_matches_ai += action_match

        if not root_match or not action_match:
            disagreements.append({
                "case_id": case_id,
                "ai_root_cause": ai_root,
                "expected_root_cause": expected_root,
                "ai_action": ai_action,
                "expected_action": expected_action,
                "confidence": confidence,
            })

    ollama_root_accuracy = root_matches_ai / compared if compared else 0.0
    ollama_action_accuracy = action_matches_ai / compared if compared else 0.0
    ollama_fallback_rate = fallback_cases / compared if compared else 0.0
    ollama_average_confidence = confidence_sum / compared if compared else 0.0

    result = {
        "ground_truth_cases": len(truth),
        "deterministic_cases_available": len(expected_by_id),
        "corrupted_cases_detected": tp,
        "false_positive_cases": fp,
        "missed_corrupted_cases": fn,
        "clean_cases_correctly_reconciled": tn,
        "detection_accuracy": round(detection_accuracy, 4),
        "detection_precision": round(precision, 4),
        "detection_recall": round(recall, 4),
        "detection_f1": round(f1, 4),
        "ground_truth_root_cause_accuracy": round(ground_truth_root_accuracy, 4),
        "ground_truth_root_cause_matches": root_matches,
        "ground_truth_root_cause_cases": root_compared,
        "by_corruption_type": by_type,
        "ollama_cases": len(ollama_cases),
        "ollama_compared_cases": compared,
        "ollama_root_cause_accuracy": round(ollama_root_accuracy, 4),
        "ollama_action_accuracy": round(ollama_action_accuracy, 4),
        "ollama_fallback_cases": fallback_cases,
        "ollama_fallback_rate": round(ollama_fallback_rate, 4),
        "ollama_average_confidence": round(ollama_average_confidence, 4),
        "ollama_disagreements": disagreements,
        "note": (
            "Ground-truth metrics measure the deterministic reconciliation engine "
            "against controlled corruption in data/ground_truth.csv. Ollama metrics "
            "measure whether the local explanation layer preserves deterministic decisions."
        ),
    }

    output_path = RESULTS_DIR / "ollama_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
