"""Generate synthetic bank credits from settlement batches."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def generate_bank_transactions(
    settlements: list[dict[str, str]], seed: int = 42
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for index, settlement in enumerate(settlements, start=1):
        settlement_date = date.fromisoformat(settlement["settlement_date"])
        # Normal bank credits arrive on the settlement date or the next business day.
        lag = rng.choice((0, 0, 0, 1))
        bank_date = settlement_date + timedelta(days=lag)
        while bank_date.weekday() >= 5:
            bank_date += timedelta(days=1)

        rows.append(
            {
                "bank_transaction_id": f"bank_{index:06d}",
                "merchant_id": settlement["merchant_id"],
                "transaction_date": bank_date.isoformat(),
                "reference": settlement["settlement_id"],
                "credit_amount": settlement["net_amount"],
                "debit_amount": "0.00",
                "description": f"Settlement {settlement['settlement_id']}",
            }
        )

    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic bank transactions")
    parser.add_argument("--settlements", type=Path, default=Path("data/settlements.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/bank_transactions.csv"))
    args = parser.parse_args()

    settlements = load_csv(args.settlements)
    if not settlements:
        raise ValueError("Settlement file is empty.")

    rows = generate_bank_transactions(settlements, args.seed)
    write_csv(rows, args.output)
    print(f"Generated {len(rows)} bank transactions -> {args.output}")


if __name__ == "__main__":
    main()
