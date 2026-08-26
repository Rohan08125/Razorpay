from src.evaluate_ollama import normalize_action


def test_normalize_action_accepts_display_labels():
    assert normalize_action("Auto reconcile") == "AUTO_RECONCILE"
    assert normalize_action("Review settlement item") == "REVIEW_SETTLEMENT_ITEM"
    assert normalize_action("Escalate for manual bank-side review.") == "ESCALATE_FOR_MANUAL_REVIEW"


def test_normalize_action_preserves_unknown_values():
    assert normalize_action("unexpected") == "unexpected"
    assert normalize_action(None) == "UNKNOWN"
