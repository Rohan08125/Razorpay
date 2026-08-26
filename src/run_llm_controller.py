"""Run the evidence-first controller and emit auditable decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ai_controller import build_case
from investigate_exception import ExceptionInvestigator
from llm_controller import investigate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run finance controller over exception batch")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    investigator = ExceptionInvestigator(args.data_dir)
    with (args.results_dir / "settlement_reconciliation.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    exception_ids = [r["settlement_id"] for r in rows if r["status"] == "EXCEPTION"]
    decisions = []
    for settlement_id in exception_ids:
        case = build_case(settlement_id, investigator)
        decision = investigate(case)
        decision["case_id"] = settlement_id
        decisions.append(decision)

    output = args.results_dir / "llm_controller_decisions.json"
    output.write_text(json.dumps(decisions, indent=2), encoding="utf-8")

    summary = {
        "cases": len(decisions),
        "provider": decisions[0]["provider"] if decisions else "none",
        "auto_reconcile": sum(x["action"] == "AUTO_RECONCILE" for x in decisions),
        "review_settlement_item": sum(x["action"] == "REVIEW_SETTLEMENT_ITEM" for x in decisions),
        "escalate": sum(x["action"] == "ESCALATE_FOR_MANUAL_REVIEW" for x in decisions),
        "average_confidence": round(sum(x["confidence"] for x in decisions) / len(decisions), 4) if decisions else 0.0,
    }
    (args.results_dir / "llm_controller_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
