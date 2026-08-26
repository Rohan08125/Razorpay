"""Fast local Ollama/Qwen reasoning layer for finance exceptions."""

from __future__ import annotations

import json
import os
from typing import Any

from finance_tools import FinanceData


SYSTEM_PROMPT = """You are a finance reconciliation decision assistant.

The deterministic reconciliation engine is the SOURCE OF TRUTH.

Your job is ONLY to:
1. Explain the supplied deterministic finding.
2. Identify the evidence supporting it.
3. Produce a concise rationale.

Do NOT change the deterministic root cause.
Do NOT change the deterministic action.
Do NOT invent facts.
Do NOT perform tool calls.
Do NOT explain your thinking.

Return ONLY valid JSON.

The confidence field will be calculated by the deterministic controller.
"""


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string"},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "financial_impact": {"type": "number"},
        "action": {
            "type": "string",
            "enum": [
                "AUTO_RECONCILE",
                "REVIEW_SETTLEMENT_ITEM",
                "ESCALATE_FOR_MANUAL_REVIEW",
            ],
        },
        "rationale": {"type": "string"},
        "evidence_used": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "root_cause",
        "confidence",
        "financial_impact",
        "action",
        "rationale",
        "evidence_used",
    ],
    "additionalProperties": False,
}


def _compact_evidence(
    data: FinanceData,
    settlement_id: str,
) -> dict[str, Any]:
    """Build a small evidence packet."""

    settlement = data.get_settlement(settlement_id)

    if "error" in settlement:
        return settlement

    items = settlement.get("items", [])
    bank = settlement.get("bank_transactions", [])

    payment_ids = list(
        dict.fromkeys(
            item.get("payment_id")
            for item in items
            if item.get("payment_id")
        )
    )

    payments = []

    for payment_id in payment_ids[:12]:
        payment = data.get_payment(payment_id)
        expected = data.calculate_expected_net(payment_id)

        payments.append(
            {
                "payment_id": payment_id,
                "amount": payment.get("payment", {}).get("amount"),
                "expected_net": expected.get("expected_net"),
            }
        )

    return {
        "settlement": settlement.get("settlement") or {},
        "item_count": len(items),
        "bank_transaction_count": len(bank),
        "bank_transactions": bank[:6],
        "payments_checked": payments,
    }


def _deterministic_decision(
    case: dict[str, Any],
) -> tuple[str, str]:
    """Extract the deterministic decision already produced by the controller."""

    root_cause = case.get(
        "root_cause",
        "DATA_ERROR",
    )

    action = (
        case.get("action")
        or case.get("recommended_action")
    )

    if root_cause == "NONE":
        return root_cause, "AUTO_RECONCILE"

    if action in {
        "AUTO_RECONCILE",
        "REVIEW_SETTLEMENT_ITEM",
        "ESCALATE_FOR_MANUAL_REVIEW",
    }:
        return root_cause, action

    return root_cause, "ESCALATE_FOR_MANUAL_REVIEW"


def _evidence_confidence(
    root_cause: str,
    evidence: dict[str, Any],
) -> float:
    """Calculate confidence from observable deterministic evidence."""

    if root_cause == "NONE":
        return 0.95

    settlement = evidence.get("settlement", {})
    item_count = evidence.get("item_count", 0)
    bank_count = evidence.get("bank_transaction_count", 0)
    recorded_net = settlement.get("net_amount")

    evidence_score = 0.70

    if item_count > 0:
        evidence_score += 0.05

    if bank_count > 0:
        evidence_score += 0.05

    if recorded_net is not None:
        evidence_score += 0.05

    if root_cause in {
        "SETTLEMENT_ITEM_COUNT_MISMATCH",
        "MISSING_SETTLEMENT_ITEM",
        "DUPLICATE_SETTLEMENT_ITEM",
    }:
        if item_count > 0:
            evidence_score += 0.10

    elif root_cause in {
        "SETTLEMENT_TOTAL_MISMATCH",
        "AMOUNT_MISMATCH",
        "FEE_MISMATCH",
    }:
        if recorded_net is not None:
            evidence_score += 0.10

    elif root_cause in {
        "MISSING_BANK_CREDIT",
        "BANK_AMOUNT_MISMATCH",
        "WRONG_BANK_REFERENCE",
        "DUPLICATE_BANK_TRANSACTION",
    }:
        if bank_count > 0:
            evidence_score += 0.10

    return round(min(evidence_score, 0.99), 2)


def investigate(
    case: dict[str, Any],
    data_dir: str = "data",
) -> dict[str, Any]:
    from ollama import Client

    data = FinanceData(data_dir)

    model = os.getenv(
        "OLLAMA_MODEL",
        "qwen3:0.6b",
    )

    settlement_id = (
        case.get("settlement_id")
        or case.get("case_id")
    )

    if not settlement_id:
        raise ValueError(
            "Case is missing settlement_id/case_id"
        )

    evidence = _compact_evidence(
        data,
        settlement_id,
    )

    if "error" in evidence:
        raise RuntimeError(evidence["error"])

    deterministic_root_cause, deterministic_action = (
        _deterministic_decision(case)
    )

    confidence = _evidence_confidence(
        deterministic_root_cause,
        evidence,
    )

    decision_input = {
        "case_id": settlement_id,
        "deterministic_root_cause": deterministic_root_cause,
        "deterministic_action": deterministic_action,
        "evidence": evidence,
    }

    prompt = (
        "Return ONLY JSON.\n"
        "Explain the deterministic finding using ONLY the supplied evidence.\n"
        "Do not change the root cause or action.\n"
        "CASE:\n"
        + json.dumps(
            decision_input,
            ensure_ascii=False,
        )
    )

    client = Client(
        host=os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )
    )

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "/no_think\n" + prompt,
            },
        ],
        think=False,
        format=DECISION_SCHEMA,
        options={
            "temperature": 0,
            "num_predict": 160,
            "num_ctx": 4096,
        },
        keep_alive="5m",
    )

    content = response.message.content or ""

    if not content:
        raise RuntimeError(
            "Ollama returned no final decision."
        )

    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        decision = None

        if start != -1 and end > start:
            try:
                decision = json.loads(
                    content[start:end + 1]
                )
            except json.JSONDecodeError:
                decision = None

        if decision is None:
            decision = {
                "rationale": (
                    "Local model returned malformed JSON; "
                    "deterministic evidence retained."
                ),
                "evidence_used": [],
            }

    # FINAL SAFETY GUARD: the LLM cannot override financial decisions.
    decision["root_cause"] = deterministic_root_cause
    decision["action"] = deterministic_action
    decision["confidence"] = confidence
    decision["financial_impact"] = float(
        decision.get("financial_impact", 0.0)
    )
    decision["provider"] = "ollama"
    decision["model"] = model
    decision["tool_calls"] = 1

    return decision
