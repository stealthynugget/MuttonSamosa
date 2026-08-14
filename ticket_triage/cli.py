"""Command-line entry point.

Usage:
    python -m ticket_triage.cli demo
    python -m ticket_triage.cli ingest path/to/tickets.json
    python -m ticket_triage.cli dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dashboard.build import build_dashboard
from .models import Ticket
from .pipeline import TriagePipeline

_DATA_DIR = Path(__file__).parent / "data"


def _print_result(result) -> None:
    t, c = result.ticket, result.classification
    print(f"\n[{t.ticket_id}] {t.subject!r}")
    print(f"  category={c.category.value} urgency={c.urgency.value} "
          f"confidence={c.confidence:.2f} ({c.classifier_name})")
    print(f"  -> {result.action.value}")
    if result.action.value == "auto_responded":
        print(f"     draft response ({len(result.kb_matches)} KB match(es)):")
        print("     " + result.draft_response.replace("\n", "\n     "))
    else:
        print(f"     assigned_team={result.assigned_team}")
        print("     " + result.escalation_summary.replace("\n", "\n     "))


def _load_tickets(path: Path) -> list[Ticket]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Ticket(**item) for item in raw]


def cmd_demo(args: argparse.Namespace) -> None:
    pipeline = TriagePipeline()
    tickets = _load_tickets(_DATA_DIR / "sample_tickets.json")
    results = pipeline.process_batch(tickets)
    for r in results:
        _print_result(r)

    summary = pipeline.store.summary_counts()
    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))

    out_path = build_dashboard(pipeline.store)
    print(f"\nDashboard written to {out_path}")


def cmd_ingest(args: argparse.Namespace) -> None:
    pipeline = TriagePipeline()
    tickets = _load_tickets(Path(args.file))
    results = pipeline.process_batch(tickets)
    for r in results:
        _print_result(r)
    out_path = build_dashboard(pipeline.store)
    print(f"\nDashboard written to {out_path}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    pipeline = TriagePipeline()
    out_path = build_dashboard(pipeline.store)
    print(f"Dashboard written to {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ticket_triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run the pipeline over the bundled sample tickets")
    p_demo.set_defaults(func=cmd_demo)

    p_ingest = sub.add_parser("ingest", help="Run the pipeline over a JSON file of tickets")
    p_ingest.add_argument("file", help="Path to a JSON array of ticket objects")
    p_ingest.set_defaults(func=cmd_ingest)

    p_dash = sub.add_parser("dashboard", help="Regenerate the HTML dashboard from the log")
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
