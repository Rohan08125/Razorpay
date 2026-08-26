"""Generate synthetic merchant master data for the AI Finance Controller."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

CATEGORIES = [
    "SaaS",
    "E-commerce",
    "Education",
    "Food & Restaurant",
    "Travel",
    "Healthcare",
    "Retail",
    "Services",
]

SETTLEMENT_CYCLES = ["T+0", "T+1", "T+2", "T+3"]

NAME_PREFIXES = [
    "Apex",
    "Bright",
    "Cloud",
    "Crest",
    "Ever",
    "Green",
    "Nova",
    "Prime",
    "Swift",
    "Urban",
]

NAME_SUFFIXES = [
    "Labs",
    "Mart",
    "Works",
    "Hub",
    "Solutions",
    "Foods",
    "Systems",
    "Retail",
    "Services",
    "Ventures",
]


def generate_merchants(count: int, seed: int) -> list[dict[str, str]]:
    """Generate deterministic synthetic merchants."""
    rng = random.Random(seed)
    merchants: list[dict[str, str]] = []
    used_names: set[str] = set()

    for index in range(1, count + 1):
        while True:
            name = f"{rng.choice(NAME_PREFIXES)} {rng.choice(NAME_SUFFIXES)}"
            if name not in used_names:
                used_names.add(name)
                break

        merchants.append(
            {
                "merchant_id": f"mer_{index:03d}",
                "merchant_name": name,
                "business_category": rng.choice(CATEGORIES),
                "settlement_cycle": rng.choice(SETTLEMENT_CYCLES),
            }
        )

    return merchants


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic merchant data")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/merchants.csv"),
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("Merchant count must be greater than zero.")

    merchants = generate_merchants(args.count, args.seed)
    write_csv(merchants, args.output)
    print(f"Generated {len(merchants)} merchants -> {args.output}")


if __name__ == "__main__":
    main()
