"""Run the investigation layer over all settlement exceptions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from investigate_exception import ExceptionInvestigator


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate settlement exceptions")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    reconciliation = read_csv(args.results_dir / "settlement_reconciliation.csv")
    investigator = ExceptionInvestigator(args.data_dir)
    results: list[dict] = []

    for row in reconciliation:
        if row["status"] != "EXCEPTION":
            continue
        results.append(investigator.investigate_settlement(row["settlement_id"]))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "investigation_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    counts = {}
    for row in results:
        counts[row["root_cause"]] = counts.get(row["root_cause"], 0) + 1

    summary = {
        "exceptions_investigated": len(results),
        "root_cause_counts": counts,
        "escalated": sum(row["status"] == "ESCALATE" for row in results),
    }
    with (args.results_dir / "investigation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
