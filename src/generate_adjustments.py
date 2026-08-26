"""Generate synthetic fees, taxes, refunds, and chargebacks."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

ADJUSTMENT_TYPES = ("FEE", "TAX", "REFUND", "CHARGEBACK")


def load_payments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def generate_adjustments(
    payments: list[dict[str, str]], seed: int = 42
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    adjustments: list[dict[str, str]] = []

    for payment in payments:
        amount = float(payment["amount"])
        payment_time = datetime.fromisoformat(payment["payment_timestamp"])
        payment_id = payment["payment_id"]

        # Every captured payment has a processing fee.
        fee_rate = rng.choice((0.015, 0.018, 0.02, 0.022, 0.025))
        fee = round(amount * fee_rate, 2)
        adjustments.append(
            {
                "adjustment_id": f"adj_{len(adjustments) + 1:06d}",
                "payment_id": payment_id,
                "adjustment_type": "FEE",
                "amount": f"{fee:.2f}",
                "adjustment_timestamp": (
                    payment_time + timedelta(hours=rng.randint(1, 24))
                ).isoformat(sep=" "),
            }
        )

        # GST-like tax on the processing fee.
        tax = round(fee * 0.18, 2)
        adjustments.append(
            {
                "adjustment_id": f"adj_{len(adjustments) + 1:06d}",
                "payment_id": payment_id,
                "adjustment_type": "TAX",
                "amount": f"{tax:.2f}",
                "adjustment_timestamp": (
                    payment_time + timedelta(hours=rng.randint(1, 24))
                ).isoformat(sep=" "),
            }
        )

        # Refunds are less common and vary in size.
        if rng.random() < 0.08:
            refund_fraction = rng.choice((0.25, 0.5, 1.0))
            refund = round(amount * refund_fraction, 2)
            adjustments.append(
                {
                    "adjustment_id": f"adj_{len(adjustments) + 1:06d}",
                    "payment_id": payment_id,
                    "adjustment_type": "REFUND",
                    "amount": f"{refund:.2f}",
                    "adjustment_timestamp": (
                        payment_time + timedelta(days=rng.randint(1, 5))
                    ).isoformat(sep=" "),
                }
            )

        # Chargebacks are rare and generally represent the full payment amount.
        if rng.random() < 0.01:
            adjustments.append(
                {
                    "adjustment_id": f"adj_{len(adjustments) + 1:06d}",
                    "payment_id": payment_id,
                    "adjustment_type": "CHARGEBACK",
                    "amount": f"{amount:.2f}",
                    "adjustment_timestamp": (
                        payment_time + timedelta(days=rng.randint(2, 10))
                    ).isoformat(sep=" "),
                }
            )

    return adjustments


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic adjustment data")
    parser.add_argument("--payments", type=Path, default=Path("data/payments.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("data/adjustments.csv")
    )
    args = parser.parse_args()

    payments = load_payments(args.payments)
    if not payments:
        raise ValueError("Payment file is empty.")

    adjustments = generate_adjustments(payments, args.seed)
    write_csv(adjustments, args.output)
    print(f"Generated {len(adjustments)} adjustments -> {args.output}")


if __name__ == "__main__":
    main()
