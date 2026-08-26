"""LLM interface for evidence-first finance investigation.

The LLM never receives write access to financial data. It receives a structured
case packet and must return a strict JSON decision grounded in the supplied
 evidence. If no provider is configured, the deterministic controller remains
 the safe fallback.
"""

from __future__ import annotations

import json
import os
from typing import Any

SYSTEM_PROMPT = """You are an AI Finance Controller investigating payment and settlement exceptions.
Use only the supplied evidence. Never invent transactions, amounts, IDs, or accounting facts.
Return JSON with exactly these keys:
root_cause, confidence, financial_impact, action, rationale, evidence_used.
Allowed action values: AUTO_RECONCILE, REVIEW_SETTLEMENT_ITEM, ESCALATE_FOR_MANUAL_REVIEW.
If evidence is insufficient, choose ESCALATE_FOR_MANUAL_REVIEW and lower confidence.
"""


def build_prompt(case: dict[str, Any]) -> str:
    return SYSTEM_PROMPT + "\n\nCASE EVIDENCE:\n" + json.dumps(case, indent=2)


def deterministic_fallback(case: dict[str, Any]) -> dict[str, Any]:
    root = case.get("root_cause", "DATA_ERROR")
    confidence = float(case.get("confidence", 0.0))
    if root == "NONE":
        action = "AUTO_RECONCILE"
    elif root == "SETTLEMENT_ITEM_COUNT_MISMATCH":
        action = "REVIEW_SETTLEMENT_ITEM"
    else:
        action = "ESCALATE_FOR_MANUAL_REVIEW"
    return {
        "root_cause": root,
        "confidence": confidence,
        "financial_impact": case.get("financial_impact", 0.0),
        "action": action,
        "rationale": "Deterministic evidence-first fallback; no LLM provider configured.",
        "evidence_used": case.get("evidence", []),
        "provider": "deterministic_fallback",
    }


def investigate(case: dict[str, Any]) -> dict[str, Any]:
    # Keep the application runnable without requiring a paid provider.
    # A provider adapter can be enabled later through environment configuration.
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    if provider == "none":
        return deterministic_fallback(case)
    raise RuntimeError(
        f"LLM_PROVIDER={provider!r} is not configured. Set LLM_PROVIDER=none for the safe fallback."
    )
