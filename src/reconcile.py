"""Deterministic reconciliation engine for the synthetic finance dataset.

This is intentionally AI-free. Rules establish a trustworthy baseline before the
AI controller is introduced.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

TOLERANCE = 0.01


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE


def money(value: float) -> str:
    return f"{value:.2f}"


def aggregate_adjustments(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        result[row["payment_id"]][row["adjustment_type"]] += float(row["amount"])
    return result


def reconcile_payments(
    payments: list[dict[str, str]],
    items: list[dict[str, str]],
    adjustments: list[dict[str, str]],
) -> list[dict[str, str]]:
    items_by_payment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        items_by_payment[item["payment_id"]].append(item)

    adj = aggregate_adjustments(adjustments)
    results: list[dict[str, str]] = []

    for payment in payments:
        payment_id = payment["payment_id"]
        amount = float(payment["amount"])
        payment_items = items_by_payment.get(payment_id, [])
        issues: list[str] = []

        if not payment_items:
            issues.append("MISSING_SETTLEMENT_ITEM")
        elif len(payment_items) > 1:
            issues.append("DUPLICATE_SETTLEMENT_ITEM")

        # If there is exactly one item, compare it against the payment and
        # independently calculated adjustment totals.
        if payment_items:
            item = payment_items[0]
            item_gross = float(item["gross_amount"])
            item_fee = float(item["fee_amount"])
            item_tax = float(item["tax_amount"])
            item_refund = float(item["refund_amount"])
            item_net = float(item["net_amount"])

            if not close(item_gross, amount):
                issues.append("AMOUNT_MISMATCH")

            expected_fee = adj[payment_id]["FEE"]
            expected_tax = adj[payment_id]["TAX"]
            expected_refund = adj[payment_id]["REFUND"] + adj[payment_id]["CHARGEBACK"]

            if not close(item_fee, expected_fee):
                issues.append("FEE_MISMATCH")
            if not close(item_tax, expected_tax):
                issues.append("TAX_MISMATCH")
            if not close(item_refund, expected_refund):
                issues.append("REFUND_MISMATCH")

            expected_net = round(amount - expected_fee - expected_tax - expected_refund, 2)
            if not close(item_net, expected_net):
                issues.append("NET_AMOUNT_MISMATCH")

            expected_net_for_report = expected_net
        else:
            expected_net_for_report = round(
                amount
                - adj[payment_id]["FEE"]
                - adj[payment_id]["TAX"]
                - adj[payment_id]["REFUND"]
                - adj[payment_id]["CHARGEBACK"],
                2,
            )

        status = "RECONCILED" if not issues else "EXCEPTION"
        results.append(
            {
                "payment_id": payment_id,
                "merchant_id": payment["merchant_id"],
                "payment_amount": money(amount),
                "expected_net_amount": money(expected_net_for_report),
                "settlement_item_count": str(len(payment_items)),
                "status": status,
                "issues": "|".join(dict.fromkeys(issues)),
            }
        )

    return results


def reconcile_settlements(
    settlements: list[dict[str, str]], items: list[dict[str, str]]
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        grouped[item["settlement_id"]].append(item)

    results: list[dict[str, str]] = []
    for settlement in settlements:
        settlement_id = settlement["settlement_id"]
        batch = grouped.get(settlement_id, [])
        gross = sum(float(row["gross_amount"]) for row in batch)
        fees = sum(float(row["fee_amount"]) for row in batch)
        taxes = sum(float(row["tax_amount"]) for row in batch)
        refunds = sum(float(row["refund_amount"]) for row in batch)
        net = sum(float(row["net_amount"]) for row in batch)

        issues: list[str] = []
        if len(batch) != int(settlement["payment_count"]):
            issues.append("PAYMENT_COUNT_MISMATCH")
        if not close(gross, float(settlement["gross_amount"])):
            issues.append("GROSS_AMOUNT_MISMATCH")
        if not close(fees, float(settlement["fee_amount"])):
            issues.append("FEE_TOTAL_MISMATCH")
        if not close(taxes, float(settlement["tax_amount"])):
            issues.append("TAX_TOTAL_MISMATCH")
        if not close(refunds, float(settlement["refund_amount"])):
            issues.append("REFUND_TOTAL_MISMATCH")
        if not close(net, float(settlement["net_amount"])):
            issues.append("NET_TOTAL_MISMATCH")

        results.append(
            {
                "settlement_id": settlement_id,
                "merchant_id": settlement["merchant_id"],
                "status": "RECONCILED" if not issues else "EXCEPTION",
                "item_count": str(len(batch)),
                "expected_item_net": money(net),
                "recorded_net": settlement["net_amount"],
                "issues": "|".join(issues),
            }
        )
    return results


def reconcile_bank(
    settlements: list[dict[str, str]], bank: list[dict[str, str]]
) -> list[dict[str, str]]:
    by_reference: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bank:
        by_reference[row["reference"]].append(row)

    results: list[dict[str, str]] = []
    for settlement in settlements:
        settlement_id = settlement["settlement_id"]
        matches = by_reference.get(settlement_id, [])
        issues: list[str] = []

        if not matches:
            issues.append("MISSING_BANK_TRANSACTION")
        elif len(matches) > 1:
            issues.append("DUPLICATE_BANK_TRANSACTION")
        else:
            bank_row = matches[0]
            expected = float(settlement["net_amount"])
            actual = float(bank_row["credit_amount"])
            if not close(expected, actual):
                issues.append("BANK_AMOUNT_MISMATCH")
            if bank_row["merchant_id"] != settlement["merchant_id"]:
                issues.append("MERCHANT_REFERENCE_MISMATCH")

        results.append(
            {
                "settlement_id": settlement_id,
                "merchant_id": settlement["merchant_id"],
                "expected_bank_credit": settlement["net_amount"],
                "matched_bank_count": str(len(matches)),
                "status": "RECONCILED" if not issues else "EXCEPTION",
                "issues": "|".join(issues),
            }
        )

    return results


def summarize(payment_results: list[dict[str, str]], settlement_results: list[dict[str, str]], bank_results: list[dict[str, str]]) -> dict:
    def stats(rows: list[dict[str, str]]) -> dict[str, int | float]:
        total = len(rows)
        reconciled = sum(row["status"] == "RECONCILED" for row in rows)
        return {
            "total": total,
            "reconciled": reconciled,
            "exceptions": total - reconciled,
            "reconciliation_rate": round(reconciled / total, 4) if total else 0.0,
        }

    return {
        "payment_level": stats(payment_results),
        "settlement_level": stats(settlement_results),
        "bank_level": stats(bank_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic financial reconciliation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    data = args.data_dir
    payments = read_csv(data / "payments.csv")
    adjustments = read_csv(data / "adjustments.csv")
    settlements = read_csv(data / "settlements.csv")
    items = read_csv(data / "settlement_items.csv")
    bank = read_csv(data / "bank_transactions.csv")

    payment_results = reconcile_payments(payments, items, adjustments)
    settlement_results = reconcile_settlements(settlements, items)
    bank_results = reconcile_bank(settlements, bank)

    write_csv(payment_results, args.results_dir / "payment_reconciliation.csv")
    write_csv(settlement_results, args.results_dir / "settlement_reconciliation.csv")
    write_csv(bank_results, args.results_dir / "bank_reconciliation.csv")

    summary = summarize(payment_results, settlement_results, bank_results)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "reconciliation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
