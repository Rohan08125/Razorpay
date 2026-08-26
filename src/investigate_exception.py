"""Investigate reconciliation exceptions using auditable finance evidence.

This module is intentionally deterministic. The LLM will be added later as a
reasoning interface over these tools, not as the source of accounting truth.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from finance_tools import FinanceData


class ExceptionInvestigator:
    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data = FinanceData(data_dir)

    def investigate_settlement(self, settlement_id: str) -> dict:
        context = self.data.get_settlement(settlement_id)
        if "error" in context:
            return context

        settlement = context["settlement"]
        items = context["items"]
        bank = context["bank_transactions"]
        evidence: list[str] = []
        findings: list[str] = []
        recommendations: list[str] = []

        recorded_net = float(settlement["net_amount"])
        item_net = round(sum(float(x["net_amount"]) for x in items), 2)
        item_count = len(items)
        expected_count = int(settlement["payment_count"])

        if item_count != expected_count:
            findings.append("Settlement item count does not match recorded payment count.")
            evidence.append(f"Expected {expected_count} items; found {item_count}.")

        if abs(item_net - recorded_net) > 0.01:
            findings.append("Settlement net amount does not equal the sum of settlement items.")
            evidence.append(f"Recorded net ₹{recorded_net:.2f}; item net ₹{item_net:.2f}.")

        if len(bank) == 0:
            findings.append("No bank transaction references this settlement.")
            recommendations.append("Escalate for missing bank credit investigation.")
        elif len(bank) > 1:
            amounts = [float(x["credit_amount"]) for x in bank]
            if len(set(round(x, 2) for x in amounts)) == 1:
                findings.append("Multiple bank credits have the same settlement reference and amount.")
                evidence.append(f"Found {len(bank)} identical referenced credits.")
                recommendations.append("Escalate as probable duplicate bank credit.")
            else:
                findings.append("Multiple bank transactions reference the same settlement with differing amounts.")
                recommendations.append("Escalate for manual bank-side review.")
        else:
            actual = float(bank[0]["credit_amount"])
            variance = round(actual - recorded_net, 2)
            if abs(variance) > 0.01:
                findings.append("Bank credit does not match the recorded settlement net amount.")
                evidence.append(f"Expected ₹{recorded_net:.2f}; bank credit ₹{actual:.2f}; variance ₹{variance:.2f}.")
                recommendations.append("Investigate partial or unexplained settlement variance.")
            else:
                evidence.append(f"Bank credit ₹{actual:.2f} matches settlement net ₹{recorded_net:.2f}.")

        if not findings:
            status = "RECONCILED"
            root_cause = "NONE"
            recommendations.append("No exception requiring investigation.")
        elif any("Multiple bank credits" in x for x in findings):
            status = "ESCALATE"
            root_cause = "DUPLICATE_OR_CONFLICTING_BANK_CREDIT"
        elif any("No bank transaction" in x for x in findings):
            status = "ESCALATE"
            root_cause = "MISSING_BANK_CREDIT"
        elif any("Bank credit does not match" in x for x in findings):
            status = "ESCALATE"
            root_cause = "BANK_AMOUNT_VARIANCE"
        elif any("item count" in x for x in findings):
            status = "ESCALATE"
            root_cause = "SETTLEMENT_ITEM_COUNT_MISMATCH"
        else:
            status = "ESCALATE"
            root_cause = "SETTLEMENT_TOTAL_MISMATCH"

        return {
            "settlement_id": settlement_id,
            "merchant_id": settlement["merchant_id"],
            "status": status,
            "root_cause": root_cause,
            "findings": findings,
            "evidence": evidence,
            "recommendations": recommendations,
        }
