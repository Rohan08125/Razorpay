"""Evaluate controller coverage and decision safety without claiming LLM accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate finance controller output")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    decisions = json.loads((args.results_dir / "llm_controller_decisions.json").read_text(encoding="utf-8"))
    summary = json.loads((args.results_dir / "reconciliation_summary.json").read_text(encoding="utf-8"))

    cases = len(decisions)
    low_confidence = sum(float(x["confidence"]) < 0.75 for x in decisions)
    escalated = sum(x["action"] == "ESCALATE_FOR_MANUAL_REVIEW" for x in decisions)
    review = sum(x["action"] == "REVIEW_SETTLEMENT_ITEM" for x in decisions)
    auto = sum(x["action"] == "AUTO_RECONCILE" for x in decisions)

    result = {
        "input_records": summary["payment_level"]["total"],
        "payment_reconciliation_rate": summary["payment_level"]["reconciliation_rate"],
        "bank_reconciliation_rate": summary["bank_level"]["reconciliation_rate"],
        "exception_cases_investigated": cases,
        "controller_actions": {
            "auto_reconcile": auto,
            "review_settlement_item": review,
            "escalate": escalated,
        },
        "escalation_rate": round(escalated / cases, 4) if cases else 0.0,
        "low_confidence_cases": low_confidence,
        "average_confidence": round(sum(float(x["confidence"]) for x in decisions) / cases, 4) if cases else 0.0,
        "throughput_cases_per_run": cases,
        "benchmark_note": "Decision output is evidence-grounded. LLM accuracy is not claimed until a real provider is configured and independently evaluated.",
    }

    output = args.results_dir / "controller_evaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
