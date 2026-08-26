"""Run the local Ollama/Qwen finance agent on settlement exceptions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ai_controller import build_case
from investigate_exception import ExceptionInvestigator
from ollama_agent import investigate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Ollama finance agent")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, default=10, help="Number of exceptions to send to the local LLM")
    args = parser.parse_args()

    investigator = ExceptionInvestigator(args.data_dir)
    with (args.results_dir / "settlement_reconciliation.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    exception_ids = [r["settlement_id"] for r in rows if r["status"] == "EXCEPTION"][: args.limit]
    decisions = []
    failures = []

    for index, settlement_id in enumerate(exception_ids, 1):
        try:
            case = build_case(settlement_id, investigator)
            decision = investigate(case, str(args.data_dir))
            decision["case_id"] = settlement_id
            decisions.append(decision)
            print(
                f"[{index}/{len(exception_ids)}] {settlement_id} -> "
                f"{decision['root_cause']} ({float(decision['confidence']):.2f}) "
                f"tools={decision['tool_calls']} action={decision['action']}"
            )
        except Exception as exc:
            failures.append({"case_id": settlement_id, "error": str(exc)})
            print(f"[{index}/{len(exception_ids)}] {settlement_id} -> ERROR: {exc}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "ollama_agent_decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.results_dir / "ollama_agent_failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "requested_cases": len(exception_ids),
        "completed_cases": len(decisions),
        "failed_cases": len(failures),
        "provider": "ollama",
        "model": decisions[0]["model"] if decisions else None,
        "average_tool_calls": round(sum(x["tool_calls"] for x in decisions) / len(decisions), 2) if decisions else 0.0,
        "average_confidence": round(sum(float(x["confidence"]) for x in decisions) / len(decisions), 4) if decisions else 0.0,
        "actions": {
            "auto_reconcile": sum(x["action"] == "AUTO_RECONCILE" for x in decisions),
            "review_settlement_item": sum(x["action"] == "REVIEW_SETTLEMENT_ITEM" for x in decisions),
            "escalate": sum(x["action"] == "ESCALATE_FOR_MANUAL_REVIEW" for x in decisions),
        },
    }
    (args.results_dir / "ollama_agent_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
