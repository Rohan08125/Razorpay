"""Fast local Ollama/Qwen reasoning layer for finance exceptions."""

from __future__ import annotations

import json
import os
from typing import Any

from finance_tools import FinanceData

SYSTEM_PROMPT = """You are a finance reconciliation decision assistant.
Use ONLY the supplied deterministic case evidence. Do not invent facts.
Do not perform tool calls. Do not explain your thinking.
Return one concise JSON decision.

Important: respond in NO-THINKING mode. Never output analysis or <think> content.

Root cause must be one of:
AMOUNT_MISMATCH, BANK_AMOUNT_MISMATCH, DUPLICATE_BANK_TRANSACTION,
DUPLICATE_SETTLEMENT_ITEM, FEE_MISMATCH, MISSING_BANK_TRANSACTION,
MISSING_SETTLEMENT_ITEM, PARTIAL_SETTLEMENT, UNEXPLAINED_VARIANCE,
WRONG_BANK_REFERENCE, SETTLEMENT_ITEM_COUNT_MISMATCH,
SETTLEMENT_TOTAL_MISMATCH, MISSING_BANK_CREDIT,
DUPLICATE_OR_CONFLICTING_BANK_CREDIT, NONE, DATA_ERROR.

Action must be one of:
AUTO_RECONCILE, REVIEW_SETTLEMENT_ITEM, ESCALATE_FOR_MANUAL_REVIEW.

Prefer the deterministic root_cause when it is present and supported by the evidence.
Confidence is confidence in the root cause, not certainty in the whole system.
"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "financial_impact": {"type": "number"},
        "action": {"type": "string", "enum": ["AUTO_RECONCILE", "REVIEW_SETTLEMENT_ITEM", "ESCALATE_FOR_MANUAL_REVIEW"]},
        "rationale": {"type": "string"},
        "evidence_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["root_cause", "confidence", "financial_impact", "action", "rationale", "evidence_used"],
    "additionalProperties": False,
}


def _compact_evidence(data: FinanceData, settlement_id: str) -> dict[str, Any]:
    """Build a small evidence packet; never send the full CSV dataset to Qwen."""
    settlement = data.get_settlement(settlement_id)
    if "error" in settlement:
        return settlement

    items = settlement.get("items", [])
    bank = settlement.get("bank_transactions", [])
    payment_ids = list(dict.fromkeys(i.get("payment_id") for i in items if i.get("payment_id")))
    payments = []
    for payment_id in payment_ids[:12]:
        payment = data.get_payment(payment_id)
        expected = data.calculate_expected_net(payment_id)
        payments.append({
            "payment_id": payment_id,
            "amount": payment.get("payment", {}).get("amount"),
            "expected_net": expected.get("expected_net"),
        })

    return {
        "settlement": settlement.get("settlement"),
        "item_count": len(items),
        "bank_transaction_count": len(bank),
        "bank_transactions": bank[:6],
        "payments_checked": payments,
    }


def investigate(case: dict[str, Any], data_dir: str = "data") -> dict[str, Any]:
    from ollama import Client

    data = FinanceData(data_dir)
    model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    settlement_id = case.get("settlement_id") or case.get("case_id")
    if not settlement_id:
        raise ValueError("Case is missing settlement_id/case_id")

    evidence = _compact_evidence(data, settlement_id)
    if "error" in evidence:
        raise RuntimeError(evidence["error"])

    # Keep the prompt deliberately small so CPU-only Qwen does not spend minutes
    # processing the entire settlement dataset.
    prompt = (
        "Return ONLY JSON. No analysis. No thinking.\n"
        "CASE:\n" + json.dumps(case, ensure_ascii=False) +
        "\nEVIDENCE:\n" + json.dumps(evidence, ensure_ascii=False)
    )

    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "/no_think\n" + prompt},
        ],
        think=False,
        format=DECISION_SCHEMA,
        options={"temperature": 0, "num_predict": 160, "num_ctx": 4096},
        keep_alive="5m",
    )

    content = response.message.content or ""
    if not content:
        raise RuntimeError("Ollama returned no final decision.")

    decision = json.loads(content)
    decision["provider"] = "ollama"
    decision["model"] = model
    decision["tool_calls"] = 1
    return decision
