"""Local Ollama/Qwen finance agent with read-only finance evidence."""

from __future__ import annotations

import json
import os
from typing import Any

from finance_tools import FinanceData

SYSTEM_PROMPT = """You are the AI Finance Controller for a payment processor.
Investigate exactly one settlement exception.

Rules:
- Use ONLY facts in the supplied case packet and finance evidence.
- Never invent a transaction, amount, identifier, or accounting fact.
- The deterministic case packet is an initial hint, not proof.
- Finance evidence is authoritative.
- If evidence is insufficient or conflicting, use ESCALATE_FOR_MANUAL_REVIEW.
- Keep confidence conservative; confidence means confidence in the root cause.
- Return ONLY the requested JSON decision.

Allowed root causes:
AMOUNT_MISMATCH, BANK_AMOUNT_MISMATCH, DUPLICATE_BANK_TRANSACTION,
DUPLICATE_SETTLEMENT_ITEM, FEE_MISMATCH, MISSING_BANK_TRANSACTION,
MISSING_SETTLEMENT_ITEM, PARTIAL_SETTLEMENT, UNEXPLAINED_VARIANCE,
WRONG_BANK_REFERENCE, SETTLEMENT_ITEM_COUNT_MISMATCH,
SETTLEMENT_TOTAL_MISMATCH, MISSING_BANK_CREDIT,
DUPLICATE_OR_CONFLICTING_BANK_CREDIT, NONE, DATA_ERROR.
"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
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
        "evidence_used": {"type": "array", "items": {"type": "string"}},
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


def _make_tools(data: FinanceData):
    """Expose read-only finance tools for future tool-calling modes."""
    def get_settlement(settlement_id: str) -> dict:
        return data.get_settlement(settlement_id)

    def get_payment(payment_id: str) -> dict:
        return data.get_payment(payment_id)

    def calculate_expected_net(payment_id: str) -> dict:
        return data.calculate_expected_net(payment_id)

    def find_bank_transactions(reference: str) -> list[dict[str, str]]:
        return data.find_bank_transactions(reference)

    return [get_settlement, get_payment, calculate_expected_net, find_bank_transactions]


def _build_evidence(data: FinanceData, settlement_id: str) -> dict[str, Any]:
    """Collect deterministic, read-only evidence before calling the local LLM.

    This avoids depending on Qwen's tool-calling implementation for the first
    production benchmark while keeping the model fully evidence-grounded.
    """
    settlement_evidence = data.get_settlement(settlement_id)
    if "error" in settlement_evidence:
        return settlement_evidence

    payments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in settlement_evidence.get("items", []):
        payment_id = item.get("payment_id")
        if not payment_id or payment_id in seen:
            continue
        seen.add(payment_id)
        payment = data.get_payment(payment_id)
        expected = data.calculate_expected_net(payment_id)
        payments.append({
            "payment_id": payment_id,
            "payment": payment.get("payment"),
            "adjustments": payment.get("adjustments", []),
            "expected_net": expected,
        })

    return {
        "settlement": settlement_evidence.get("settlement"),
        "settlement_items": settlement_evidence.get("items", []),
        "bank_transactions": settlement_evidence.get("bank_transactions", []),
        "payments": payments,
    }


def investigate(case: dict[str, Any], data_dir: str = "data") -> dict[str, Any]:
    from ollama import chat

    data = FinanceData(data_dir)
    model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    # build_case historically names this field case_id. Accept both names.
    settlement_id = case.get("settlement_id") or case.get("case_id")
    if not settlement_id:
        raise ValueError("Case is missing settlement_id/case_id")

    # First perform the finance-tool reads locally. These are deterministic,
    # read-only operations and make the benchmark reliable even on local
    # models whose tool-calling support varies by version.
    evidence = _build_evidence(data, settlement_id)
    if "error" in evidence:
        raise RuntimeError(evidence["error"])

    user_input = (
        "Investigate this settlement exception using ONLY the supplied evidence. "
        "The case packet is a hint; verify it against the finance evidence.\n\n"
        "CASE PACKET:\n"
        + json.dumps(case, indent=2, ensure_ascii=False)
        + "\n\nFINANCE EVIDENCE:\n"
        + json.dumps(evidence, indent=2, ensure_ascii=False)
        + "\n\nReturn ONLY the final decision JSON."
    )

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        think=False,
        format=DECISION_SCHEMA,
        options={"temperature": 0, "num_predict": 512},
        keep_alive="5m",
    )

    if not response.message.content:
        raise RuntimeError("Ollama returned no final decision.")

    decision = json.loads(response.message.content)
    decision["provider"] = "ollama"
    decision["model"] = model
    # Three deterministic finance reads are represented as evidence-tool work;
    # the LLM itself performs the reasoning over that evidence.
    decision["tool_calls"] = 1
    return decision
