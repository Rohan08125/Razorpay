"""Generate deterministic synthetic payment data for the AI Finance Controller."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

PAYMENT_METHODS = ["UPI", "CARD", "NETBANKING", "WALLET"]
CURRENCY = "INR"
PAYMENT_STATUS = "captured"

# Weighted amounts make the synthetic data more realistic than uniform integers.
AMOUNT_BUCKETS = [
    (100, 999, 0.30),
    (1000, 4999, 0.35),
    (5000, 9999, 0.20),
    (10000, 24999, 0.10),
    (25000, 100000, 0.05),
]


def generate_amount(rng: random.Random) -> float:
    """Generate a payment amount from a weighted set of realistic buckets."""
    pick = rng.random()
    cumulative = 0.0

    for low, high, weight in AMOUNT_BUCKETS:
        cumulative += weight
        if pick <= cumulative:
            return round(rng.uniform(low, high), 2)

    low, high, _ = AMOUNT_BUCKETS[-1]
    return round(rng.uniform(low, high), 2)


def random_timestamp(rng: random.Random, start: datetime, end: datetime) -> datetime:
    """Return a random timestamp between start and end."""
    seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, seconds))


def generate_payments(
    merchants: list[dict[str, str]],
    count: int,
    seed: int,
    start_date: datetime,
    end_date: datetime,
) -> list[dict[str, str]]:
    """Generate deterministic synthetic captured payments."""
    if not merchants:
        raise ValueError("At least one merchant is required.")
    if count <= 0:
        raise ValueError("Payment count must be greater than zero.")
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date.")

    rng = random.Random(seed)
    payments: list[dict[str, str]] = []

    for index in range(1, count + 1):
        merchant = rng.choice(merchants)
        timestamp = random_timestamp(rng, start_date, end_date)
        amount = generate_amount(rng)
        payment_method = rng.choices(
            PAYMENT_METHODS,
            weights=[0.55, 0.25, 0.15, 0.05],
            k=1,
        )[0]

        payments.append(
            {
                "payment_id": f"pay_{index:06d}",
                "merchant_id": merchant["merchant_id"],
                "order_id": f"ord_{rng.randint(10000000, 99999999)}",
                "payment_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": f"{amount:.2f}",
                "currency": CURRENCY,
                "payment_method": payment_method,
                "payment_status": PAYMENT_STATUS,
            }
        )

    return payments


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic payment data")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-08-01 00:00:00",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-08-20 23:59:59",
    )
    parser.add_argument(
        "--merchants",
        type=Path,
        default=Path("data/merchants.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/payments.csv"),
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d %H:%M:%S")

    with args.merchants.open("r", newline="", encoding="utf-8") as file:
        merchants = list(csv.DictReader(file))

    payments = generate_payments(
        merchants=merchants,
        count=args.count,
        seed=args.seed,
        start_date=start_date,
        end_date=end_date,
    )
    write_csv(payments, args.output)
    print(f"Generated {len(payments)} payments -> {args.output}")


if __name__ == "__main__":
    main()
