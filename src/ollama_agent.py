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
- You MUST call get_settlement for the settlement under investigation before making a decision.
- If the settlement evidence references a payment that needs deeper verification, call get_payment and calculate_expected_net.
- If bank evidence needs independent verification, call find_bank_transactions.
- You may make multiple tool calls over multiple turns.
- If evidence is insufficient or conflicting, use ESCALATE_FOR_MANUAL_REVIEW.
- Keep confidence conservative; confidence means confidence in the root cause, not certainty that the system is correct.
- Return ONLY the requested JSON decision after investigation.

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

    user_input = (
        "Investigate this settlement exception. Verify the case packet with tools before deciding.\n\n"
        + json.dumps(case, indent=2, ensure_ascii=False)
        + "\n\nReturn the final decision using exactly this JSON schema: "
        + json.dumps(DECISION_SCHEMA)
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    tool_calls = 0
    response = None
    for _ in range(max_tool_rounds):
        response = chat(
            model=model,
            messages=messages,
            tools=tools,
            # Qwen3's internal thinking can be very slow on CPU. Keep the
            # benchmark responsive; the evidence/tool calls remain intact.
            think=False,
            format=DECISION_SCHEMA,
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

    if response is None or not response.message.content:
        raise RuntimeError("Ollama returned no final decision.")

    decision = json.loads(response.message.content)
    decision["provider"] = "ollama"
    decision["model"] = model
    decision["tool_calls"] = tool_calls
    return decision
