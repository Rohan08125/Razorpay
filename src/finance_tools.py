"""Read-only tools used by the finance investigation layer."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


class FinanceData:
    def __init__(self, data_dir: Path | str = Path("data")) -> None:
        self.data_dir = Path(data_dir)
        self.payments = self._read("payments.csv")
        self.adjustments = self._read("adjustments.csv")
        self.settlements = self._read("settlements.csv")
        self.items = self._read("settlement_items.csv")
        self.bank = self._read("bank_transactions.csv")

        self.payment_by_id = {r["payment_id"]: r for r in self.payments}
        self.settlement_by_id = {r["settlement_id"]: r for r in self.settlements}
        self.items_by_payment = defaultdict(list)
        self.items_by_settlement = defaultdict(list)
        self.adjustments_by_payment = defaultdict(list)
        self.bank_by_reference = defaultdict(list)

        for row in self.items:
            self.items_by_payment[row["payment_id"]].append(row)
            self.items_by_settlement[row["settlement_id"]].append(row)
        for row in self.adjustments:
            self.adjustments_by_payment[row["payment_id"]].append(row)
        for row in self.bank:
            self.bank_by_reference[row["reference"]].append(row)

    def _read(self, filename: str) -> list[dict[str, str]]:
        with (self.data_dir / filename).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def get_payment(self, payment_id: str) -> dict:
        payment = self.payment_by_id.get(payment_id)
        if not payment:
            return {"error": f"Unknown payment_id: {payment_id}"}
        return {
            "payment": payment,
            "settlement_items": self.items_by_payment.get(payment_id, []),
            "adjustments": self.adjustments_by_payment.get(payment_id, []),
        }

    def get_settlement(self, settlement_id: str) -> dict:
        settlement = self.settlement_by_id.get(settlement_id)
        if not settlement:
            return {"error": f"Unknown settlement_id: {settlement_id}"}
        return {
            "settlement": settlement,
            "items": self.items_by_settlement.get(settlement_id, []),
            "bank_transactions": self.bank_by_reference.get(settlement_id, []),
        }

    def calculate_expected_net(self, payment_id: str) -> dict:
        payment = self.payment_by_id.get(payment_id)
        if not payment:
            return {"error": f"Unknown payment_id: {payment_id}"}
        amount = float(payment["amount"])
        totals = defaultdict(float)
        for row in self.adjustments_by_payment.get(payment_id, []):
            totals[row["adjustment_type"]] += float(row["amount"])
        expected = round(
            amount - totals["FEE"] - totals["TAX"] - totals["REFUND"] - totals["CHARGEBACK"],
            2,
        )
        return {
            "payment_id": payment_id,
            "gross_amount": round(amount, 2),
            "fee": round(totals["FEE"], 2),
            "tax": round(totals["TAX"], 2),
            "refund": round(totals["REFUND"], 2),
            "chargeback": round(totals["CHARGEBACK"], 2),
            "expected_net": expected,
        }

    def find_bank_transactions(self, reference: str) -> list[dict[str, str]]:
        return self.bank_by_reference.get(reference, [])
