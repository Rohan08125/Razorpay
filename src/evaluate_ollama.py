"""Evaluate local Ollama decisions against the deterministic controller."""

from __future__ import annotations

import json
from pathlib import Path

from ai_controller import build_case
from investigate_exception import ExceptionInvestigator


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")


def normalize_action(action: str | None) -> str:
    mapping = {
        "AUTO_RECONCILE": "AUTO_RECONCILE",
        "Auto reconcile": "AUTO_RECONCILE",
        "REVIEW_SETTLEMENT_ITEM": "REVIEW_SETTLEMENT_ITEM",
        "Review settlement item": "REVIEW_SETTLEMENT_ITEM",
        "ESCALATE_FOR_MANUAL_REVIEW": "ESCALATE_FOR_MANUAL_REVIEW",
        "Escalate for manual review": "ESCALATE_FOR_MANUAL_REVIEW",
    }
    return mapping.get(action or "", action or "UNKNOWN")


def main() -> None:
    ollama_path = RESULTS_DIR / "ollama_agent_decisions.json"
    ollama_cases = json.loads(ollama_path.read_text(encoding="utf-8"))

    # Build the expected cases fresh from the deterministic engine rather than
    # comparing against an older/stale controller_cases.json result file.
    investigator = ExceptionInvestigator(DATA_DIR)
    expected_by_id = {}
    for settlement_id in investigator.data.settlement_by_id:
        expected_by_id[settlement_id] = build_case(
            settlement_id,
            investigator,
        )

    compared = 0
    root_matches = 0
    action_matches = 0
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
        expected_action = normalize_action(
            expected.get("recommended_action")
        )

        confidence = float(ai_case.get("confidence", 0.0))
        confidence_sum += confidence

        if confidence == 0.0:
            fallback_cases += 1

        root_match = ai_root == expected_root
        action_match = ai_action == expected_action

        if root_match:
            root_matches += 1
        if action_match:
            action_matches += 1

        if not root_match or not action_match:
            disagreements.append(
                {
                    "case_id": case_id,
                    "ai_root_cause": ai_root,
                    "expected_root_cause": expected_root,
                    "ai_action": ai_action,
                    "expected_action": expected_action,
                    "confidence": confidence,
                }
            )

    if compared:
        root_accuracy = root_matches / compared
        action_accuracy = action_matches / compared
        average_confidence = confidence_sum / compared
        fallback_rate = fallback_cases / compared
    else:
        root_accuracy = 0.0
        action_accuracy = 0.0
        average_confidence = 0.0
        fallback_rate = 0.0

    result = {
        "ollama_cases": len(ollama_cases),
        "deterministic_cases_available": len(expected_by_id),
        "compared_cases": compared,
        "root_cause_matches": root_matches,
        "root_cause_accuracy": round(root_accuracy, 4),
        "action_matches": action_matches,
        "action_accuracy": round(action_accuracy, 4),
        "fallback_cases": fallback_cases,
        "fallback_rate": round(fallback_rate, 4),
        "average_confidence": round(average_confidence, 4),
        "disagreements": disagreements,
        "note": (
            "Expected decisions are rebuilt from the deterministic engine; "
            "ground_truth.csv is not consumed by the AI controller."
        ),
    }

    output_path = RESULTS_DIR / "ollama_evaluation.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
