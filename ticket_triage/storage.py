"""Persists triage outcomes for reporting, auditing, and the dashboard.

Uses SQLite (stdlib, zero setup) so the project runs anywhere with no
external database. Swap in Postgres/etc. by reimplementing this module's
interface (log_result / fetch_all / summary_counts) unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .models import TriageResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS triage_log (
    ticket_id TEXT PRIMARY KEY,
    received_at TEXT,
    processed_at TEXT,
    customer_id TEXT,
    channel TEXT,
    subject TEXT,
    category TEXT,
    urgency TEXT,
    confidence REAL,
    classifier_name TEXT,
    action TEXT,
    assigned_team TEXT,
    draft_response TEXT,
    escalation_summary TEXT,
    kb_matches TEXT,
    notes TEXT,
    outcome_status TEXT DEFAULT 'open'
);
"""


class TriageStore:
    def __init__(self, db_path: str | Path = "triage_log.db"):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_result(self, result: TriageResult) -> None:
        t, c = result.ticket, result.classification
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO triage_log (
                    ticket_id, received_at, processed_at, customer_id, channel, subject,
                    category, urgency, confidence, classifier_name, action, assigned_team,
                    draft_response, escalation_summary, kb_matches, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    processed_at=excluded.processed_at,
                    category=excluded.category,
                    urgency=excluded.urgency,
                    confidence=excluded.confidence,
                    action=excluded.action,
                    assigned_team=excluded.assigned_team,
                    draft_response=excluded.draft_response,
                    escalation_summary=excluded.escalation_summary,
                    kb_matches=excluded.kb_matches,
                    notes=excluded.notes
                """,
                (
                    t.ticket_id,
                    t.received_at.isoformat(),
                    result.processed_at.isoformat(),
                    t.customer_id,
                    t.channel,
                    t.subject,
                    c.category.value,
                    c.urgency.value,
                    c.confidence,
                    c.classifier_name,
                    result.action.value,
                    result.assigned_team,
                    result.draft_response,
                    result.escalation_summary,
                    json.dumps([m.__dict__ for m in result.kb_matches]),
                    result.notes,
                ),
            )

    def set_outcome_status(self, ticket_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE triage_log SET outcome_status = ? WHERE ticket_id = ?",
                (status, ticket_id),
            )

    def fetch_all(self) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM triage_log ORDER BY processed_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def summary_counts(self) -> dict:
        rows = self.fetch_all()
        total = len(rows)
        by_category: dict[str, int] = {}
        by_urgency: dict[str, int] = {}
        by_action: dict[str, int] = {}
        for r in rows:
            by_category[r["category"]] = by_category.get(r["category"], 0) + 1
            by_urgency[r["urgency"]] = by_urgency.get(r["urgency"], 0) + 1
            by_action[r["action"]] = by_action.get(r["action"], 0) + 1
        auto_rate = (by_action.get("auto_responded", 0) / total) if total else 0.0
        return {
            "total": total,
            "by_category": by_category,
            "by_urgency": by_urgency,
            "by_action": by_action,
            "auto_response_rate": round(auto_rate, 3),
        }
