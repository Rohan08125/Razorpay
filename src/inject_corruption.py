"""Inject known reconciliation issues into clean synthetic financial data."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

CORRUPTION_COUNTS = {
    "MISSING_SETTLEMENT_ITEM": 180,
    "DUPLICATE_SETTLEMENT_ITEM": 120,
    "FEE_MISMATCH": 180,
    "AMOUNT_MISMATCH": 120,
    "MISSING_BANK_TRANSACTION": 15,
    "DUPLICATE_BANK_TRANSACTION": 8,
    "BANK_AMOUNT_MISMATCH": 15,
    "WRONG_BANK_REFERENCE": 8,
    "PARTIAL_SETTLEMENT": 10,
    "UNEXPLAINED_VARIANCE": 10,
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
    """Inject one controlled issue into a unique settlement per ground-truth row.

    The previous implementation selected individual item rows and relied on their
    settlement IDs implicitly. This version selects settlement IDs from the
    canonical settlements table first, guaranteeing that every ground-truth case
    refers to an existing settlement and that item-level corruption is observable
    by the settlement reconciliation layer.
    """
    rng = random.Random(seed)
    truth = []

    settlement_ids = [row["settlement_id"] for row in settlements]
    items_by_settlement: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in items:
        items_by_settlement[row["settlement_id"]].append(row)

    eligible = [sid for sid in settlement_ids if items_by_settlement.get(sid)]
    rng.shuffle(eligible)

    total_item_cases = sum(
        CORRUPTION_COUNTS[name]
        for name in (
            "MISSING_SETTLEMENT_ITEM",
            "DUPLICATE_SETTLEMENT_ITEM",
            "FEE_MISMATCH",
            "AMOUNT_MISMATCH",
        )
    )
    total_bank_cases = sum(
        CORRUPTION_COUNTS[name]
        for name in (
            "MISSING_BANK_TRANSACTION",
            "DUPLICATE_BANK_TRANSACTION",
            "BANK_AMOUNT_MISMATCH",
            "WRONG_BANK_REFERENCE",
            "PARTIAL_SETTLEMENT",
            "UNEXPLAINED_VARIANCE",
        )
    )
    if len(eligible) < total_item_cases + total_bank_cases:
        raise ValueError("Not enough unique settlements for corruption benchmark.")

    def record(corruption_type, settlement_id, details):
        truth.append(
            {
                "entity_type": "settlement",
                "entity_id": settlement_id,
                "corruption_type": corruption_type,
                "details": details,
            }
        )

    # Item-level corruption is deliberately assigned to stable settlement IDs.
    cursor = 0
    for corruption_type in (
        "MISSING_SETTLEMENT_ITEM",
        "DUPLICATE_SETTLEMENT_ITEM",
        "FEE_MISMATCH",
        "AMOUNT_MISMATCH",
    ):
        target_ids = eligible[cursor : cursor + CORRUPTION_COUNTS[corruption_type]]
        cursor += CORRUPTION_COUNTS[corruption_type]
        for settlement_id in target_ids:
            row = items_by_settlement[settlement_id][0]
            payment_id = row["payment_id"]
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
                row["net_amount"] = money(
                    float(row["gross_amount"])
                    - new
                    - float(row["tax_amount"])
                    - float(row["refund_amount"])
                )
                record(corruption_type, settlement_id, f"Fee changed from {money(old)} to {money(new)}")
            elif corruption_type == "AMOUNT_MISMATCH":
                old = float(row["gross_amount"])
                new = round(old * rng.choice((0.95, 1.05)), 2)
                row["gross_amount"] = money(new)
                record(corruption_type, settlement_id, f"Gross amount changed from {money(old)} to {money(new)}")

    # Bank-level corruption uses the same canonical settlement IDs.
    bank_by_settlement: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bank:
        bank_by_settlement[row["reference"]].append(row)

    bank_ids = eligible[cursor : cursor + total_bank_cases]
    cursor += total_bank_cases
    bank_cursor = 0

    for corruption_type in (
        "MISSING_BANK_TRANSACTION",
        "DUPLICATE_BANK_TRANSACTION",
        "BANK_AMOUNT_MISMATCH",
        "WRONG_BANK_REFERENCE",
        "PARTIAL_SETTLEMENT",
        "UNEXPLAINED_VARIANCE",
    ):
        target_ids = bank_ids[bank_cursor : bank_cursor + CORRUPTION_COUNTS[corruption_type]]
        bank_cursor += CORRUPTION_COUNTS[corruption_type]
        for settlement_id in target_ids:
            matches = bank_by_settlement.get(settlement_id, [])
            if not matches:
                raise ValueError(f"No bank transaction for settlement {settlement_id}")
            row = matches[0]
            bank_id = row["bank_transaction_id"]

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
                other_id = rng.choice([sid for sid in settlement_ids if sid != settlement_id])
                row["reference"] = other_id
                record(corruption_type, settlement_id, f"Bank reference changed from {old} to {other_id}")
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

    # Hard invariants for the benchmark: every case must point to one canonical
    # settlement, and every case ID is unique so case-level metrics are unambiguous.
    canonical = set(settlement_ids)
    truth_ids = [row["entity_id"] for row in truth]
    if not set(truth_ids).issubset(canonical):
        raise AssertionError("Ground truth contains an unknown settlement ID.")
    if len(truth_ids) != len(set(truth_ids)):
        raise AssertionError("Ground truth contains duplicate settlement IDs.")

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
