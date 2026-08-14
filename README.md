# Ticket Triage

An automated support-ticket triage system: it classifies incoming tickets by **intent and urgency**, drafts **auto-responses** for common, well-understood issues grounded in a knowledge base, and **escalates** complex or high-priority tickets to a human agent with a pre-filled summary. Every outcome is logged for reporting via a built-in dashboard.

Built for the workflow:

```
Ingest → Classify → Decide Path → Auto-Respond (or) Escalate → Log & Track
```

## Why it's built this way

- **No required external dependencies.** The default classifier is deterministic keyword scoring (`RuleBasedClassifier`), and knowledge-base search is a pure-Python TF-IDF-style cosine similarity — no numpy/sklearn. This means the pipeline runs anywhere, immediately, with zero API keys or setup, and is fully unit-testable/deterministic in CI.
- **LLM classification is a drop-in upgrade, not a requirement.** Set `ANTHROPIC_API_KEY` and the pipeline automatically switches to `LLMClassifier` for higher-quality, context-aware classification — and falls back to the rule-based classifier if the key is missing or the call fails, so triage never silently drops a ticket.
- **Auto-responses are template-grounded, not freely generated.** Every draft reply traces back to a specific knowledge-base entry (`KB-004`, etc.), which keeps the channel auditable and avoids sending a hallucinated answer straight to a customer with no human in the loop.
- **A confidence threshold gates automation.** Low-confidence classifications, high-risk categories (security, complaints, unknown), and tickets above a configurable urgency ceiling always escalate — even if the category is normally auto-response-eligible. See `TriageConfig` in `pipeline.py`.

## Project layout

```
ticket_triage/
  models.py        Ticket, ClassificationResult, KBMatch, TriageResult dataclasses
  classifier.py     RuleBasedClassifier (default) + LLMClassifier (optional)
  knowledge_base.py  KB loader + pure-Python TF-IDF search
  responder.py      Grounded auto-response drafting
  escalation.py     Escalation summary + human-team routing + SLA
  pipeline.py       Orchestrates the full ingest → log workflow
  storage.py        SQLite-backed triage log (zero setup)
  cli.py            `demo`, `ingest`, `dashboard` commands
  dashboard/        Self-contained HTML dashboard (data embedded, no server)
  data/
    faq_kb.json        Sample knowledge base (15 entries across 9 categories)
    sample_tickets.json  10 sample tickets for the demo
tests/
  test_pipeline.py  Classifier, KB search, and pipeline decision-logic tests
examples/
  sample_dashboard.html  A pre-generated dashboard from the sample tickets
```

## Quickstart

```bash
pip install -r requirements.txt   # optional: only pytest/anthropic are used

# Run the pipeline over the bundled sample tickets and build the dashboard
python -m ticket_triage.cli demo

# Run it over your own tickets (JSON array of {subject, body, customer_id, channel})
python -m ticket_triage.cli ingest path/to/tickets.json

# Regenerate the dashboard from whatever's already logged
python -m ticket_triage.cli dashboard

# Run the test suite
pytest
```

Open `ticket_triage/dashboard/index.html` in a browser after running any of the above — it's a self-contained file, no server needed. A pre-built example lives at `examples/sample_dashboard.html`.

## Using the library directly

```python
from ticket_triage import Ticket, TriagePipeline

pipeline = TriagePipeline()

ticket = Ticket(
    subject="Can't log in",
    body="I keep getting an invalid password error, please help.",
    customer_id="cust_123",
    channel="email",
)

result = pipeline.process(ticket)

print(result.action)              # Action.AUTO_RESPONDED or Action.ESCALATED
print(result.classification)      # category, urgency, confidence, rationale
print(result.draft_response)      # if auto-responded
print(result.escalation_summary)  # if escalated
```

## Enabling the LLM classifier

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic
```

No code changes needed — `TriagePipeline()` picks up `get_default_classifier()`, which prefers `LLMClassifier` whenever a key is present and falls back automatically otherwise. To force one explicitly:

```python
from ticket_triage import TriagePipeline
from ticket_triage.classifier import RuleBasedClassifier, LLMClassifier

pipeline = TriagePipeline(classifier=RuleBasedClassifier())   # force deterministic
pipeline = TriagePipeline(classifier=LLMClassifier())          # force LLM
```

## Tuning the auto-response boundary

All of the decision logic lives in `TriageConfig` (`pipeline.py`):

```python
from ticket_triage import TriagePipeline, TriageConfig
from ticket_triage.models import Urgency

pipeline = TriagePipeline(config=TriageConfig(
    confidence_threshold=0.7,        # raise the bar for auto-response
    max_auto_urgency=Urgency.LOW,    # only fully-calm tickets get auto-responded
))
```

## Extending it

- **Real ticket ingestion**: replace `cli.py`'s JSON-file reader with a webhook/email/API listener that constructs `Ticket` objects the same way.
- **Bigger knowledge base**: `KnowledgeBase` reads any JSON file shaped like `data/faq_kb.json`; swap in a vector DB by reimplementing `.search()` with the same signature.
- **Real agent routing**: `escalation.assigned_team()` currently returns a team name string — point it at your ticketing system's API (Zendesk, Freshdesk, Jira Service Management, etc.) to actually create/assign the ticket.
- **Outcome tracking**: `TriageStore.set_outcome_status(ticket_id, status)` exists so you can later mark tickets resolved/reopened once a human closes the loop, and have that reflected in the dashboard.
