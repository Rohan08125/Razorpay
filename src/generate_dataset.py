"""Run the complete clean-data generation pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(SRC / script), *args]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("generate_merchants.py", "--count", "50", "--seed", "42")
    run("generate_payments.py", "--count", "10000", "--seed", "42")
    run("generate_adjustments.py", "--seed", "42")
    run("generate_settlements.py", "--seed", "42")
    run("generate_bank_transactions.py", "--seed", "42")
    run("inject_corruption.py", "--seed", "42")
    print("Synthetic financial dataset generated successfully.")


if __name__ == "__main__":
    main()
