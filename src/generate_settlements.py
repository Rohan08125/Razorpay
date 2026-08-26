"""Generate batch settlements and settlement items from payments."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def add_business_days(start: datetime, days: int) -> datetime:
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def adjustment_map(adjustments: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in adjustments:
        result[row["payment_id"]][row["adjustment_type"]] += float(row["amount"])
    return result


def generate_settlements(
    merchants: list[dict[str, str]],
    payments: list[dict[str, str]],
    adjustments: list[dict[str, str]],
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    merchant_cycles = {m["merchant_id"]: int(m["settlement_cycle"][2:]) for m in merchants}
    adj = adjustment_map(adjustments)

    by_batch: dict[tuple[str, date], list[dict[str, str]]] = defaultdict(list)
    for payment in payments:
        if payment["payment_status"] != "captured":
            continue
        paid_at = datetime.fromisoformat(payment["payment_timestamp"])
        cycle = merchant_cycles[payment["merchant_id"]]
        settlement_at = add_business_days(paid_at, cycle)
        # Group payments by merchant and settlement date to model batch settlements.
        by_batch[(payment["merchant_id"], settlement_at.date())].append(payment)

    settlements: list[dict[str, str]] = []
    items: list[dict[str, str]] = []

    for settlement_index, ((merchant_id, settlement_date), batch) in enumerate(
        sorted(by_batch.items()), start=1
    ):
        gross = 0.0
        fees = 0.0
        taxes = 0.0
        refunds = 0.0
        net = 0.0
        settlement_id = f"set_{settlement_index:05d}"

        for payment in batch:
            payment_id = payment["payment_id"]
            amount = float(payment["amount"])
            values = adj[payment_id]
            fee = values["FEE"]
            tax = values["TAX"]
            refund = values["REFUND"] + values["CHARGEBACK"]
            item_net = round(amount - fee - tax - refund, 2)

            gross += amount
            fees += fee
            taxes += tax
            refunds += refund
            net += item_net

            items.append(
                {
                    "settlement_id": settlement_id,
                    "payment_id": payment_id,
                    "gross_amount": f"{amount:.2f}",
                    "fee_amount": f"{fee:.2f}",
                    "tax_amount": f"{tax:.2f}",
                    "refund_amount": f"{refund:.2f}",
                    "net_amount": f"{item_net:.2f}",
                }
            )

        settlements.append(
            {
                "settlement_id": settlement_id,
                "merchant_id": merchant_id,
                "settlement_date": settlement_date.isoformat(),
                "payment_count": str(len(batch)),
                "gross_amount": f"{gross:.2f}",
                "fee_amount": f"{fees:.2f}",
                "tax_amount": f"{taxes:.2f}",
                "refund_amount": f"{refunds:.2f}",
                "net_amount": f"{net:.2f}",
                "settlement_status": rng.choice(("processed", "processed", "processed")),
            }
        )

    return settlements, items


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic settlement data")
    parser.add_argument("--merchants", type=Path, default=Path("data/merchants.csv"))
    parser.add_argument("--payments", type=Path, default=Path("data/payments.csv"))
    parser.add_argument("--adjustments", type=Path, default=Path("data/adjustments.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settlements-output", type=Path, default=Path("data/settlements.csv"))
    parser.add_argument("--items-output", type=Path, default=Path("data/settlement_items.csv"))
    args = parser.parse_args()

    merchants = load_csv(args.merchants)
    payments = load_csv(args.payments)
    adjustments = load_csv(args.adjustments)
    settlements, items = generate_settlements(merchants, payments, adjustments, args.seed)
    write_csv(settlements, args.settlements_output)
    write_csv(items, args.items_output)
    print(f"Generated {len(settlements)} settlements and {len(items)} settlement items")


if __name__ == "__main__":
    main()
