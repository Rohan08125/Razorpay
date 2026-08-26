"""Fast local Ollama/Qwen reasoning layer for finance exceptions."""

from __future__ import annotations

import json
import os
from typing import Any

from finance_tools import FinanceData


SYSTEM_PROMPT = """You are a finance reconciliation explanation assistant.

The deterministic reconciliation engine is the SOURCE OF TRUTH.
Your job is ONLY to explain the supplied decision using the supplied evidence.

Do NOT change the root cause.
Do NOT change the action.
Do NOT invent facts.
Do NOT perform tool calls.
Do NOT explain your thinking.

Return ONLY valid JSON with exactly two fields:
- rationale: one short factual sentence
- evidence_used: a list of 1 to 3 short evidence statements

Do not include any other fields.
"""


EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "evidence_used": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
    },
    "required": ["rationale", "evidence_used"],
    "additionalProperties": False,
}


def _compact_evidence(data: FinanceData, settlement_id: str) -> dict[str, Any]:
    """Build a small evidence packet; never send the full dataset to Qwen."""
    settlement = data.get_settlement(settlement_id)
    if "error" in settlement:
        return settlement

    items = settlement.get("items", [])
    bank = settlement.get("bank_transactions", [])
    settlement_record = settlement.get("settlement") or {}

    return {
        "settlement": settlement_record,
        "item_count": len(items),
        "bank_transaction_count": len(bank),
        "bank_transactions": bank[:6],
    }


def _deterministic_decision(case: dict[str, Any]) -> tuple[str, str]:
    """Extract the financial decision already produced by the deterministic layer."""
    root_cause = case.get("root_cause", "DATA_ERROR")
    action = case.get("action") or case.get("recommended_action")

    if root_cause == "NONE":
        return root_cause, "AUTO_RECONCILE"

    if action in {
        "AUTO_RECONCILE",
        "REVIEW_SETTLEMENT_ITEM",
        "ESCALATE_FOR_MANUAL_REVIEW",
    }:
        return root_cause, action

    return root_cause, "ESCALATE_FOR_MANUAL_REVIEW"


def _evidence_confidence(root_cause: str, evidence: dict[str, Any]) -> float:
    """Calculate confidence from observable deterministic evidence."""
    if root_cause == "NONE":
        return 0.95

    settlement = evidence.get("settlement", {})
    item_count = evidence.get("item_count", 0)
    bank_count = evidence.get("bank_transaction_count", 0)
    recorded_net = settlement.get("net_amount")

    score = 0.70
    if item_count > 0:
        score += 0.05
    if bank_count > 0:
        score += 0.05
    if recorded_net is not None:
        score += 0.05

    if root_cause in {
        "SETTLEMENT_ITEM_COUNT_MISMATCH",
        "MISSING_SETTLEMENT_ITEM",
        "DUPLICATE_SETTLEMENT_ITEM",
    } and item_count > 0:
        score += 0.10
    elif root_cause in {
        "SETTLEMENT_TOTAL_MISMATCH",
        "AMOUNT_MISMATCH",
        "FEE_MISMATCH",
    } and recorded_net is not None:
        score += 0.10
    elif root_cause in {
        "MISSING_BANK_CREDIT",
        "BANK_AMOUNT_MISMATCH",
        "WRONG_BANK_REFERENCE",
        "DUPLICATE_BANK_TRANSACTION",
    } and bank_count > 0:
        score += 0.10

    return round(min(score, 0.99), 2)


def investigate(case: dict[str, Any], data_dir: str = "data") -> dict[str, Any]:
    from ollama import Client

    data = FinanceData(data_dir)
    model = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    settlement_id = case.get("settlement_id") or case.get("case_id")

    if not settlement_id:
        raise ValueError("Case is missing settlement_id/case_id")

    evidence = _compact_evidence(data, settlement_id)
    if "error" in evidence:
        raise RuntimeError(evidence["error"])

    deterministic_root_cause, deterministic_action = _deterministic_decision(case)
    confidence = _evidence_confidence(deterministic_root_cause, evidence)

    decision_input = {
        "case_id": settlement_id,
        "root_cause": deterministic_root_cause,
        "action": deterministic_action,
        "evidence": evidence,
    }

    prompt = (
        "Return ONLY the two-field JSON object.\n"
        "Explain the deterministic decision using only the evidence.\n"
        "CASE:\n"
        + json.dumps(decision_input, ensure_ascii=False)
    )

    client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "/no_think\n" + prompt},
        ],
        think=False,
        format=EXPLANATION_SCHEMA,
        options={
            "temperature": 0,
            "num_predict": 100,
            "num_ctx": 2048,
        },
        keep_alive="5m",
    )

    content = response.message.content or ""
    if not content:
        raise RuntimeError("Ollama returned no explanation.")

    try:
        explanation = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        explanation = None
        if start != -1 and end > start:
            try:
                explanation = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

        if explanation is None:
            explanation = {
                "rationale": "Deterministic decision retained because the local model returned malformed JSON.",
                "evidence_used": [],
            }

    # FINAL SAFETY GUARD: the LLM can explain the decision, but cannot make it.
    return {
        "root_cause": deterministic_root_cause,
        "confidence": confidence,
        "financial_impact": float(case.get("financial_impact", 0.0)),
        "action": deterministic_action,
        "rationale": explanation.get("rationale", "Deterministic decision retained."),
        "evidence_used": explanation.get("evidence_used", []),
        "provider": "ollama",
        "model": model,
        "tool_calls": 1,
    }
