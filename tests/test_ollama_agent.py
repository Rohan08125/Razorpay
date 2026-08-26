import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ollama_agent import (
    _deterministic_decision,
    _evidence_confidence,
    _fallback_explanation,
    _parse_explanation,
)


def test_deterministic_decision_cannot_be_overridden_by_unknown_action():
    root_cause, action = _deterministic_decision(
        {"root_cause": "SETTLEMENT_TOTAL_MISMATCH", "action": "hallucinated_action"}
    )
    assert root_cause == "SETTLEMENT_TOTAL_MISMATCH"
    assert action == "ESCALATE_FOR_MANUAL_REVIEW"


def test_none_case_is_auto_reconcile():
    assert _deterministic_decision({"root_cause": "NONE", "action": "anything"}) == (
        "NONE",
        "AUTO_RECONCILE",
    )


def test_parse_explanation_accepts_fenced_json():
    parsed = _parse_explanation('```json\n{"rationale":"ok","evidence_used":["fact"]}\n```')
    assert parsed == {"rationale": "ok", "evidence_used": ["fact"]}


def test_parse_explanation_rejects_non_json():
    assert _parse_explanation("not json") is None


def test_fallback_explanation_is_deterministic_and_evidence_based():
    rationale, evidence = _fallback_explanation(
        "SETTLEMENT_TOTAL_MISMATCH",
        {
            "settlement": {"net_amount": "123.45"},
            "item_count": 4,
            "bank_transaction_count": 1,
        },
    )
    assert "settlement total mismatch" in rationale
    assert len(evidence) == 3
    assert "₹123.45" in evidence[0]


def test_evidence_confidence_is_bounded():
    score = _evidence_confidence(
        "SETTLEMENT_TOTAL_MISMATCH",
        {
            "settlement": {"net_amount": "100.00"},
            "item_count": 5,
            "bank_transaction_count": 1,
        },
    )
    assert 0.0 <= score <= 0.99
