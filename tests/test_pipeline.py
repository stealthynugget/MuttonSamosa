import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from ticket_triage.classifier import RuleBasedClassifier
from ticket_triage.knowledge_base import KnowledgeBase
from ticket_triage.models import Action, Category, Ticket, Urgency
from ticket_triage.pipeline import TriageConfig, TriagePipeline
from ticket_triage.storage import TriageStore

DATA_DIR = Path(__file__).parent.parent / "ticket_triage" / "data"


@pytest.fixture
def classifier():
    return RuleBasedClassifier()


@pytest.fixture
def kb():
    return KnowledgeBase.from_json_file(DATA_DIR / "faq_kb.json")


@pytest.fixture
def pipeline(tmp_path):
    store = TriageStore(tmp_path / "test_log.db")
    return TriagePipeline(store=store)


def test_classifier_detects_billing(classifier):
    t = Ticket(subject="Refund request", body="I was charged twice on my credit card this month, please refund.")
    result = classifier.classify(t)
    assert result.category == Category.BILLING
    assert 0.0 <= result.confidence <= 1.0


def test_classifier_detects_account_access(classifier):
    t = Ticket(subject="Locked out", body="I can't login, it says my password is wrong and now I'm locked out.")
    result = classifier.classify(t)
    assert result.category == Category.ACCOUNT_ACCESS


def test_classifier_detects_critical_urgency(classifier):
    t = Ticket(subject="PRODUCTION DOWN", body="This is a critical emergency, production is down and all users are affected.")
    result = classifier.classify(t)
    assert result.urgency == Urgency.CRITICAL


def test_classifier_detects_low_urgency(classifier):
    t = Ticket(subject="Quick question", body="Just curious, no rush at all, whenever you get a chance let me know.")
    result = classifier.classify(t)
    assert result.urgency == Urgency.LOW


def test_classifier_unknown_category_on_empty_signal(classifier):
    t = Ticket(subject="hi", body="ok")
    result = classifier.classify(t)
    assert result.category == Category.UNKNOWN
    assert result.confidence < 0.5


def test_kb_search_returns_relevant_match(kb):
    matches = kb.search("How do I reset my password", category=Category.ACCOUNT_ACCESS)
    assert matches
    assert matches[0].kb_id == "KB-001"


def test_kb_search_respects_category_filter(kb):
    matches = kb.search("cancel my plan please", category=Category.BILLING)
    # cancellation-specific KB entries shouldn't leak into billing-filtered search
    assert all(m.kb_id != "KB-010" for m in matches)


def test_kb_search_no_match_returns_empty(kb):
    matches = kb.search("zzz completely unrelated gibberish qqq", min_score=0.5)
    assert matches == []


def test_pipeline_auto_responds_to_clear_low_risk_ticket(pipeline):
    t = Ticket(subject="Forgot password", body="How do I reset my password? I can't remember it.")
    result = pipeline.process(t)
    assert result.action == Action.AUTO_RESPONDED
    assert result.draft_response is not None
    assert result.kb_matches


def test_pipeline_escalates_critical_ticket_even_if_category_eligible(pipeline):
    t = Ticket(
        subject="PRODUCTION DOWN - all users affected",
        body="This is a critical emergency, every user is getting 500 errors right now.",
    )
    result = pipeline.process(t)
    assert result.action == Action.ESCALATED
    assert result.assigned_team is not None


def test_pipeline_escalates_security_category_regardless_of_confidence(pipeline):
    t = Ticket(subject="Account hacked", body="My account was hacked, I see a suspicious login I don't recognize.")
    result = pipeline.process(t)
    assert result.action == Action.ESCALATED
    assert result.classification.category == Category.SECURITY


def test_pipeline_escalates_low_confidence_classification(pipeline):
    t = Ticket(subject="asdkjh test", body="not sure what happened but something is weird")
    result = pipeline.process(t)
    assert result.action == Action.ESCALATED


def test_pipeline_logs_every_ticket(pipeline):
    tickets = [
        Ticket(subject="Reset password", body="How do I reset my password?"),
        Ticket(subject="Angry complaint", body="This is the worst, unacceptable, terrible service ever."),
    ]
    pipeline.process_batch(tickets)
    rows = pipeline.store.fetch_all()
    assert len(rows) == 2


def test_confidence_threshold_forces_escalation(tmp_path):
    store = TriageStore(tmp_path / "strict.db")
    strict_config = TriageConfig(confidence_threshold=0.99)
    pipeline = TriagePipeline(store=store, config=strict_config)
    t = Ticket(subject="Refund", body="I was charged twice, please refund me.")
    result = pipeline.process(t)
    # Threshold is unreachable, so even a clear billing ticket must escalate.
    assert result.action == Action.ESCALATED


def test_escalation_summary_includes_key_fields(pipeline):
    t = Ticket(subject="I'm furious", body="This is unacceptable and I am extremely frustrated with your terrible service.")
    result = pipeline.process(t)
    assert result.action == Action.ESCALATED
    assert result.ticket.ticket_id in result.escalation_summary
    assert result.assigned_team in result.escalation_summary
