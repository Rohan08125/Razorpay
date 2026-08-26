"""Local Ollama/Qwen finance agent with read-only finance tools."""

from __future__ import annotations

import json
import os
from typing import Any

from finance_tools import FinanceData

SYSTEM_PROMPT = """You are the AI Finance Controller for a payment processor.
Investigate exactly one settlement exception.

Rules:
- Use ONLY facts returned by the supplied finance tools and the case packet.
- Never invent a transaction, amount, identifier, or accounting fact.
- The case packet is an initial hint, not proof.
- Your FIRST action MUST be a call to get_settlement for the settlement under investigation.
- Do not produce a final decision until get_settlement has returned evidence.
- If the settlement evidence references a payment that needs deeper verification, call get_payment and calculate_expected_net.
- If bank evidence needs independent verification, call find_bank_transactions.
- You may make multiple tool calls over multiple turns.
- If evidence is insufficient or conflicting, use ESCALATE_FOR_MANUAL_REVIEW.
- Keep confidence conservative; confidence means confidence in the root cause, not certainty that the system is correct.
- After tool investigation, return ONLY the requested JSON decision.

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
    def get_settlement(settlement_id: str) -> dict:
        """Read one settlement, its settlement items, and bank transactions referencing it."""
        return data.get_settlement(settlement_id)

    def get_payment(payment_id: str) -> dict:
        """Read one payment, its settlement items, and its adjustments."""
        return data.get_payment(payment_id)

    def calculate_expected_net(payment_id: str) -> dict:
        """Calculate the expected net amount for a payment from its adjustments."""
        return data.calculate_expected_net(payment_id)

    def find_bank_transactions(reference: str) -> list[dict[str, str]]:
        """Find bank transactions using a settlement reference."""
        return data.find_bank_transactions(reference)

    return [get_settlement, get_payment, calculate_expected_net, find_bank_transactions]


def investigate(case: dict[str, Any], data_dir: str = "data") -> dict[str, Any]:
    from ollama import chat

    data = FinanceData(data_dir)
    model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    max_tool_rounds = int(os.getenv("MAX_TOOL_ROUNDS", "6"))
    tools = _make_tools(data)
    available = {fn.__name__: fn for fn in tools}

    settlement_id = case.get("settlement_id")
    if not settlement_id:
        raise ValueError("Case is missing settlement_id")

    user_input = (
        "Investigate this settlement exception. The case packet is only an initial hint. "
        "Your first response MUST call get_settlement with settlement_id="
        + json.dumps(settlement_id)
        + ". Do not answer yet.\n\n"
        + json.dumps(case, indent=2, ensure_ascii=False)
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    tool_calls = 0
    response = None

    # Phase 1: tool-driven investigation. Do NOT constrain this turn to JSON,
    # because the model must be allowed to emit a tool call.
    for _ in range(max_tool_rounds):
        response = chat(
            model=model,
            messages=messages,
            tools=tools,
            think=False,
            options={"temperature": 0},
        )
        messages.append(response.message)

        calls = response.message.tool_calls or []
        if not calls:
            break

        for call in calls:
            name = call.function.name
            args = dict(call.function.arguments or {})
            function = available.get(name)
            if function is None:
                result: Any = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = function(**args)
                except Exception as exc:
                    result = {"error": f"Tool {name} failed: {exc}"}
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            tool_calls += 1

        # Once evidence has been collected, let the model decide whether it
        # needs another tool. If it stops calling tools, we proceed to a
        # separate structured final-decision turn below.

    if tool_calls == 0:
        raise RuntimeError("Ollama did not call a finance tool before deciding.")

    # Phase 2: structured final decision. Keeping format here (rather than on
    # the tool-calling turns) prevents JSON formatting from suppressing tools.
    messages.append(
        {
            "role": "user",
            "content": "Investigation is complete. Now return ONLY the final decision using exactly this JSON schema: "
            + json.dumps(DECISION_SCHEMA),
        }
    )
    response = chat(
        model=model,
        messages=messages,
        think=False,
        format=DECISION_SCHEMA,
        options={"temperature": 0},
    )

    if not response.message.content:
        raise RuntimeError("Ollama returned no final decision.")

    decision = json.loads(response.message.content)
    decision["provider"] = "ollama"
    decision["model"] = model
    decision["tool_calls"] = tool_calls
    return decision
