"""Run the evidence-first controller across settlement exceptions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ai_controller import build_case
from investigate_exception import ExceptionInvestigator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI-ready controller over settlement exceptions")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    investigator = ExceptionInvestigator(args.data_dir)
    reconciliation = json.loads(
        (args.results_dir / "reconciliation_summary.json").read_text(encoding="utf-8")
    )

    with (args.results_dir / "settlement_reconciliation.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    exception_ids = [row["settlement_id"] for row in rows if row["status"] == "EXCEPTION"]
    cases = [build_case(settlement_id, investigator) for settlement_id in exception_ids]

    output = args.results_dir / "controller_cases.json"
    output.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    escalated = sum(
        case["recommended_action"] != "AUTO_RECONCILE" for case in cases
    )
    summary = {
        "settlement_exceptions": len(exception_ids),
        "cases_generated": len(cases),
        "escalated": escalated,
        "escalation_rate": round(escalated / len(cases), 4) if cases else 0.0,
        "payment_reconciliation_rate": reconciliation["payment_level"]["reconciliation_rate"],
        "bank_reconciliation_rate": reconciliation["bank_level"]["reconciliation_rate"],
    }
    (args.results_dir / "controller_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
