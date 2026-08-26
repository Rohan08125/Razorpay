"""Inject known reconciliation issues into clean synthetic financial data."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

CORRUPTION_RATES = {
    "MISSING_SETTLEMENT_ITEM": 0.018,
    "DUPLICATE_SETTLEMENT_ITEM": 0.012,
    "FEE_MISMATCH": 0.018,
    "AMOUNT_MISMATCH": 0.012,
    "MISSING_BANK_TRANSACTION": 0.018,
    "DUPLICATE_BANK_TRANSACTION": 0.010,
    "BANK_AMOUNT_MISMATCH": 0.018,
    "WRONG_BANK_REFERENCE": 0.010,
    "PARTIAL_SETTLEMENT": 0.012,
    "UNEXPLAINED_VARIANCE": 0.012,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty dataset: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def money(value: float) -> str:
    return f"{value:.2f}"


def inject(settlements, items, bank, seed: int = 42):
    rng = random.Random(seed)
    truth = []

    candidates = list(items)
    rng.shuffle(candidates)
    used_payment_ids = set()

    def record(corruption_type, settlement_id, details):
        truth.append(
            {
                "entity_type": "settlement",
                "entity_id": settlement_id,
                "corruption_type": corruption_type,
                "details": details,
            }
        )

    offset = 0
    for corruption_type, rate in CORRUPTION_RATES.items():
        if corruption_type not in {
            "MISSING_SETTLEMENT_ITEM",
            "DUPLICATE_SETTLEMENT_ITEM",
            "FEE_MISMATCH",
            "AMOUNT_MISMATCH",
        }:
            continue

        target_count = max(1, int(len(candidates) * rate))
        selected = []
        while len(selected) < target_count and offset < len(candidates):
            row = candidates[offset]
            offset += 1
            if row["payment_id"] not in used_payment_ids:
                used_payment_ids.add(row["payment_id"])
                selected.append(row)

        for row in selected:
            payment_id = row["payment_id"]
            settlement_id = row["settlement_id"]
            if corruption_type == "MISSING_SETTLEMENT_ITEM":
                items.remove(row)
                record(corruption_type, settlement_id, f"Settlement item for {payment_id} removed")
            elif corruption_type == "DUPLICATE_SETTLEMENT_ITEM":
                items.append(dict(row))
                record(corruption_type, settlement_id, f"Settlement item for {payment_id} duplicated")
            elif corruption_type == "FEE_MISMATCH":
                old = float(row["fee_amount"])
                new = round(old * 1.35, 2)
                row["fee_amount"] = money(new)
                row["net_amount"] = money(float(row["gross_amount"]) - new - float(row["tax_amount"]) - float(row["refund_amount"]))
                record(corruption_type, settlement_id, f"Fee changed from {money(old)} to {money(new)}")
            elif corruption_type == "AMOUNT_MISMATCH":
                old = float(row["gross_amount"])
                new = round(old * rng.choice((0.95, 1.05)), 2)
                row["gross_amount"] = money(new)
                record(corruption_type, settlement_id, f"Gross amount changed from {money(old)} to {money(new)}")

    bank_candidates = list(bank)
    rng.shuffle(bank_candidates)
    used_bank_ids = set()

    for corruption_type in (
        "MISSING_BANK_TRANSACTION",
        "DUPLICATE_BANK_TRANSACTION",
        "BANK_AMOUNT_MISMATCH",
        "WRONG_BANK_REFERENCE",
        "PARTIAL_SETTLEMENT",
        "UNEXPLAINED_VARIANCE",
    ):
        target_count = max(1, int(len(bank_candidates) * CORRUPTION_RATES[corruption_type]))
        selected = []
        for row in bank_candidates:
            if len(selected) >= target_count:
                break
            if row["bank_transaction_id"] not in used_bank_ids:
                used_bank_ids.add(row["bank_transaction_id"])
                selected.append(row)

        for row in selected:
            bank_id = row["bank_transaction_id"]
            settlement_id = row["reference"]
            if corruption_type == "MISSING_BANK_TRANSACTION":
                bank.remove(row)
                record(corruption_type, settlement_id, f"Bank credit {bank_id} removed")
            elif corruption_type == "DUPLICATE_BANK_TRANSACTION":
                bank.append(dict(row))
                record(corruption_type, settlement_id, f"Bank credit {bank_id} duplicated")
            elif corruption_type == "BANK_AMOUNT_MISMATCH":
                old = float(row["credit_amount"])
                new = round(old - max(10.0, old * 0.01), 2)
                row["credit_amount"] = money(new)
                record(corruption_type, settlement_id, f"Credit changed from {money(old)} to {money(new)}")
            elif corruption_type == "WRONG_BANK_REFERENCE":
                old = row["reference"]
                other = rng.choice(bank_candidates)
                row["reference"] = other["reference"]
                record(corruption_type, old, f"Bank reference changed from {old} to {row['reference']}")
            elif corruption_type == "PARTIAL_SETTLEMENT":
                old = float(row["credit_amount"])
                new = round(old * 0.75, 2)
                row["credit_amount"] = money(new)
                record(corruption_type, settlement_id, f"Credit reduced from {money(old)} to {money(new)}")
            elif corruption_type == "UNEXPLAINED_VARIANCE":
                old = float(row["credit_amount"])
                new = round(old - rng.uniform(50, 500), 2)
                row["credit_amount"] = money(max(new, 0))
                record(corruption_type, settlement_id, f"Unexplained variance introduced from {money(old)} to {row['credit_amount']}")

    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject controlled reconciliation corruption")
    parser.add_argument("--settlements", type=Path, default=Path("data/settlements.csv"))
    parser.add_argument("--items", type=Path, default=Path("data/settlement_items.csv"))
    parser.add_argument("--bank", type=Path, default=Path("data/bank_transactions.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settlements = read_csv(args.settlements)
    items = read_csv(args.items)
    bank = read_csv(args.bank)
    truth = inject(settlements, items, bank, args.seed)

    write_csv(settlements, args.settlements)
    write_csv(items, args.items)
    write_csv(bank, args.bank)
    write_csv(truth, Path("data/ground_truth.csv"))
    print(f"Injected {len(truth)} known issues")
    print("Ground truth -> data/ground_truth.csv")


if __name__ == "__main__":
    main()
