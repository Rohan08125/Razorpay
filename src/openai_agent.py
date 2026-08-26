"""Real OpenAI Responses API agent with read-only finance tools."""

from __future__ import annotations

import json
import os
from typing import Any

from finance_tools import FinanceData

SYSTEM_PROMPT = """You are the AI Finance Controller for a payment processor.
Investigate exactly one settlement exception.
Use only facts returned by the supplied finance tools and the case packet.
Never invent a transaction, amount, identifier, or accounting fact.
Before deciding, call get_settlement for the settlement under investigation.
If the settlement evidence references a payment that needs deeper verification,
call get_payment and calculate_expected_net for that payment. Use find_bank_transactions
when bank evidence needs an independent lookup.
Return a concise, evidence-grounded decision. If evidence is insufficient, escalate.
"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "financial_impact": {"type": "number"},
        "action": {
            "type": "string",
            "enum": ["AUTO_RECONCILE", "REVIEW_SETTLEMENT_ITEM", "ESCALATE_FOR_MANUAL_REVIEW"],
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

TOOLS = [
    {
        "type": "function",
        "name": "get_settlement",
        "description": "Read a settlement and all settlement items and bank transactions referencing it.",
        "parameters": {
            "type": "object",
            "properties": {"settlement_id": {"type": "string"}},
            "required": ["settlement_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_payment",
        "description": "Read one payment, its settlement items, and adjustments.",
        "parameters": {
            "type": "object",
            "properties": {"payment_id": {"type": "string"}},
            "required": ["payment_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "calculate_expected_net",
        "description": "Calculate expected payment net from the payment and its adjustments.",
        "parameters": {
            "type": "object",
            "properties": {"payment_id": {"type": "string"}},
            "required": ["payment_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "find_bank_transactions",
        "description": "Find all bank transactions by settlement reference.",
        "parameters": {
            "type": "object",
            "properties": {"reference": {"type": "string"}},
            "required": ["reference"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _tool_result(data: FinanceData, name: str, args: dict[str, Any]) -> Any:
    if name == "get_settlement":
        return data.get_settlement(args["settlement_id"])
    if name == "get_payment":
        return data.get_payment(args["payment_id"])
    if name == "calculate_expected_net":
        return data.calculate_expected_net(args["payment_id"])
    if name == "find_bank_transactions":
        return data.find_bank_transactions(args["reference"])
    return {"error": f"Unknown tool: {name}"}


def investigate(case: dict[str, Any], data_dir: str = "data") -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    data = FinanceData(data_dir)
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    max_tool_rounds = int(os.getenv("MAX_TOOL_ROUNDS", "5"))

    user_input = (
        "Investigate this exception. The case packet is an initial hint only; verify facts with tools.\n\n"
        + json.dumps(case, indent=2)
    )

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_input,
        tools=TOOLS,
        tool_choice={"type": "function", "name": "get_settlement"},
        text={"format": {"type": "json_schema", "name": "finance_decision", "schema": DECISION_SCHEMA, "strict": True}},
        store=False,
    )

    tool_calls = 0
    while tool_calls < max_tool_rounds:
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break

        outputs = []
        for call in calls:
            args = json.loads(call.arguments)
            result = _tool_result(data, call.name, args)
            outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            })
            tool_calls += 1

        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            previous_response_id=response.id,
            input=outputs,
            tools=TOOLS,
            text={"format": {"type": "json_schema", "name": "finance_decision", "schema": DECISION_SCHEMA, "strict": True}},
            store=False,
        )

    if not response.output_text:
        raise RuntimeError("Model returned no final decision.")

    decision = json.loads(response.output_text)
    decision["provider"] = "openai"
    decision["model"] = model
    decision["tool_calls"] = tool_calls
    return decision
