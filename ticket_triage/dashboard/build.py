"""Builds a single self-contained HTML dashboard file from the triage log.

Data is embedded directly in the HTML (rather than fetched at runtime) so the
file works when opened directly from disk -- no local server, no CORS issues.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..storage import TriageStore

_OUT_DIR = Path(__file__).parent
_TEMPLATE_PATH = Path(__file__).parent / "template.html"


def build_dashboard(store: TriageStore, out_path: str | Path | None = None) -> Path:
    rows = store.fetch_all()
    summary = store.summary_counts()

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__TRIAGE_DATA__", json.dumps(rows)).replace(
        "__TRIAGE_SUMMARY__", json.dumps(summary)
    )

    out = Path(out_path) if out_path else _OUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    return out
